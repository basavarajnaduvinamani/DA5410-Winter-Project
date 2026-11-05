"""POPE end-to-end run (primary benchmark). GPU for inference; CPU for --write-splits/--prepare-only.

  python -m scripts.run_pope --output RUN --profile quick --timing-probe 20   # estimate, then stop
  python -m scripts.run_pope --output RUN --profile quick                      # baseline..test..report
Re-running the same command resumes from RUN/predictions.jsonl.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

from src.pope import image_disjoint_split, limit_split, load_pope
from src.pope_pipeline import Pipeline, append_jsonl, plan_counts, write_json

STAGES = ('baseline', 'calibrate', 'sweep', 'test', 'report')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_config(path, profile=None, overrides=None):
    config = json.loads(Path(path).read_text())
    profiles = config.pop('profiles', {})
    if profile:
        if profile not in profiles:
            raise ValueError(f'Unknown profile {profile!r}; choose from {sorted(profiles)}')
        config.update(profiles[profile]); config['profile'] = profile
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return config


def split_manifest(items, config):
    split = image_disjoint_split(sorted({i['image'] for i in items}), config['seed'], config['split_fractions'])
    return dict(seed=config['seed'], fractions=config['split_fractions'], unit='COCO image (all POPE questions about it)',
                ordering='sha256(f"{seed}:{image}") ascending', counts={k: len(v) for k, v in split.items()}, **split)


def identity_for(config, items):
    pope_hashes = {s: digest(Path(config['pope_dir']) / f'coco_pope_{s}.json') for s in config['pope_splits']}
    ident = dict(config={k: v for k, v in config.items() if k not in ('run_id', 'coco_image_dir')}, pope_sha256=pope_hashes,
                 n_items=len(items))
    ident['run_id'] = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    return ident


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'UNKNOWN'


def estimate(out, config, split, items, seconds_per_question):
    counts = plan_counts(config, split, items)
    hours = {k: round(v * seconds_per_question / 3600, 2) for k, v in counts.items() if isinstance(v, int)}
    hours['total'] = round(sum(hours.values()), 2)
    record = dict(seconds_per_question=seconds_per_question, counts=counts, estimated_hours=hours,
                  note='Calibration CPU fitting (minutes) and report generation are not included.')
    write_json(out / 'timing.json', record)
    print(json.dumps(record, indent=2))
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/pope_llava.json')
    p.add_argument('--output', help='Run directory (use Google Drive on Colab so it survives disconnects)')
    p.add_argument('--profile', default='full', help='quick | full (see config "profiles")')
    p.add_argument('--adapter', help='Override adapter: llava (default) | qwen2_5_vl | internvl')
    p.add_argument('--model-id'); p.add_argument('--revision')
    p.add_argument('--quantization', choices=['4bit', '8bit', 'none'])
    p.add_argument('--coco-dir', help='Directory holding COCO_val2014_*.jpg')
    p.add_argument('--stages', default=','.join(STAGES))
    p.add_argument('--timing-probe', type=int, default=0, help='Answer N questions, print runtime estimate, stop')
    p.add_argument('--write-splits', help='Write the image-disjoint split manifest to this path and exit (no model)')
    p.add_argument('--prepare-only', action='store_true', help='Write identity/splits/plan, no model')
    args = p.parse_args(argv)
    overrides = dict(adapter=args.adapter, model_id=args.model_id, revision=args.revision,
                     quantization=args.quantization, coco_image_dir=args.coco_dir)
    if args.adapter and args.adapter != 'llava' and not args.model_id:
        from src.adapters import DEFAULT_CHECKPOINTS
        overrides.update(model_id=DEFAULT_CHECKPOINTS[args.adapter], revision=args.revision)
    config = resolve_config(args.config, args.profile, overrides)
    items = load_pope(config['pope_dir'], config['pope_splits'])
    if args.write_splits:
        write_json(args.write_splits, split_manifest(items, config)); print('wrote', args.write_splits); return
    if not args.output:
        p.error('--output is required')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ident = identity_for(config, items)
    if (out / 'identity.json').exists():
        previous = json.loads((out / 'identity.json').read_text())
        if previous['run_id'] != ident['run_id']:
            raise ValueError('Output directory belongs to a different config/data; choose a new --output')
    write_json(out / 'identity.json', ident)
    config['run_id'] = ident['run_id']
    split = limit_split(image_disjoint_split(sorted({i['image'] for i in items}), config['seed'],
                                             config['split_fractions']), config.get('max_images'))
    write_json(out / 'splits.json', dict(split_manifest(items, config), used=split,
                                         used_counts={k: len(v) for k, v in split.items()}))
    append_jsonl(out / 'sessions.jsonl', [dict(started=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                                               commit=git_commit(), python=sys.version.split()[0])])
    if args.prepare_only:
        print(json.dumps(plan_counts(config, split, items), indent=2)); return
    import torch
    if not torch.cuda.is_available() and os.environ.get('COUNTERPRIOR_ALLOW_CPU') != '1':
        write_json(out / 'status.json', dict(status='blocked', reason='CUDA GPU required', inference_executed=False))
        raise RuntimeError('Select a GPU runtime (Runtime > Change runtime type > T4 GPU)')
    if not (out / 'environment.json').exists():
        write_json(out / 'environment.json', dict(python=sys.version, cuda=torch.cuda.is_available(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()}))
    from PIL import Image
    from src.adapters import load_adapter
    coco = Path(config['coco_image_dir'])
    needed = sorted({i['image'] for i in items if any(i['image'] in v for v in split.values())})
    missing = [n for n in needed if not (coco / n).exists()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} COCO images missing from {coco} (e.g. {missing[0]}); run scripts/fetch_pope.py')
    adapter = load_adapter(config['adapter'], config['model_id'], config.get('revision'),
                           config['quantization'], config['attn_implementation'])
    pipe = Pipeline(adapter, items, lambda n: Image.open(coco / n).convert('RGB'), config, out)
    if args.timing_probe:
        n, secs = pipe.baseline(budget=args.timing_probe)
        if n == 0:
            raise RuntimeError('Timing probe found no unanswered questions; estimate from timing.json instead')
        estimate(out, config, split, items, secs / n)
        return
    run_stages(pipe, out, config, args.stages.split(','))


def run_stages(pipe, out, config, stages):
    status = dict(stages_done=[], inference_executed=True)
    for stage in STAGES:
        if stage not in stages:
            continue
        print(f'== {stage}')
        if stage == 'baseline':
            pipe.baseline()
        elif stage == 'calibrate':
            pipe.calibrate()
        elif stage == 'sweep':
            pipe.sweep()
        elif stage == 'test':
            pipe.test()
        elif stage == 'report':
            from scripts.make_report import build_report
            build_report(out, out / 'REPORT.md', out / 'figures')
        status['stages_done'].append(stage)
        write_json(out / 'status.json', dict(status, last_update=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if '--output' in sys.argv:
            out = Path(sys.argv[sys.argv.index('--output') + 1]); out.mkdir(parents=True, exist_ok=True)
            with (out / 'errors.log').open('a', encoding='utf8') as f:
                f.write(datetime.now(timezone.utc).isoformat() + '\n' + traceback.format_exc() + '\n')
        raise
