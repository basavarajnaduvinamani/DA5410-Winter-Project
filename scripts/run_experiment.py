"""Colab experiment. Raw outputs are append-only; full manual adjudication required."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
import numpy as np
import torch
from PIL import Image
from src.adjudication import adjudicate
from src.protocol import digest, make_splits, select_calibration, direction


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf8')


def load_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()] if Path(path).exists() else []


def append(path, data):
    with Path(path).open('a', encoding='utf8') as f:
        f.write(json.dumps(data, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/llava.json')
    p.add_argument('--output', required=True, help='New or matching existing run directory')
    p.add_argument('--reviews', help='Human reviews JSONL; calibration baseline reviews required for selection')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--ablations', action='store_true')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    seed = config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cases = json.loads(Path(config['dataset']).read_text())
    for c in cases:
        if digest(Path('dataset/images') / c['image_file']) != c['image_sha256']:
            raise ValueError('Image hash mismatch: '+c['id'])
    splits = make_splits(cases, seed)
    source_paths = sorted([*Path('src').glob('*.py'), *Path('scripts').glob('*.py')])
    identity = dict(config=config, dataset_sha256=digest(config['dataset']),
                    environment_lock_sha256=digest('requirements-colab.lock'),
                    source_sha256={str(p):digest(p) for p in source_paths},
                    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    identity['run_id'] = __import__('hashlib').sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text()) != identity:
        raise ValueError('Existing run belongs to different source/config; choose a new output directory')
    write_json(out/'identity.json', identity); write_json(out/'splits.json', splits)
    environment = dict(command=sys.argv, python=sys.version, timestamp=datetime.now(timezone.utc).isoformat(),
                       cuda_available=torch.cuda.is_available(),
                       packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
    if not (out/'environment.json').exists():
        write_json(out/'environment.json', environment)
    if args.prepare_only:
        write_json(out/'status.json', dict(status='prepared_only', inference_executed=False))
        print('Prepared image-disjoint splits and provenance; NO model inference executed.')
        return
    if not torch.cuda.is_available():
        write_json(out/'status.json', dict(status='blocked', reason='CUDA GPU required', inference_executed=False))
        raise RuntimeError('Select a Colab GPU runtime; no model inference executed')
    from src.model_loader import load_llava_model, find_language_decoder_layers
    from src.probe import extract_prompt_terminal_state
    from src.rotational_steering import RotationalSteeringHook, SteeringContext
    model, processor = load_llava_model(model_id=config['model_id'], revision=config['revision'],
                                        attn_implementation=config['attn_implementation'],load_in_4bit=config['load_in_4bit'])
    layers = find_language_decoder_layers(model)
    predpath = out/'predictions.jsonl'
    records = load_jsonl(predpath); done = {r['prediction_id'] for r in records}
    case_by_id = {c['id']:c for c in cases}
    states = []
    def generate(c, method, hook=None, layer=None, probability=None):
        pid = c['id']+':'+method
        if pid in done:
            return
        image = Image.open(Path('dataset/images')/c['image_file']).convert('RGB')
        inputs = processor(text=f"USER: <image>\n{c['prompt']}\nASSISTANT:",images=image,return_tensors='pt').to(model.device)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        from contextlib import nullcontext
        with SteeringContext(layers[layer],hook) if hook is not None else nullcontext():
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=config['max_new_tokens'], do_sample=False, use_cache=True)
        answer = processor.tokenizer.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        record = dict(prediction_id=pid, id=c['id'], method=method, split=splits[c['id']],
                      category=c['category'], image_group=c['image_group'], image_sha256=c['image_sha256'],
                      prompt=c['prompt'], answer=answer, automatic_adjudication=adjudicate(answer,c),
                      seed=seed, config=config, dataset_sha256=identity['dataset_sha256'],
                      run_id=identity['run_id'], commit=identity['commit'], probe_probability=probability,
                      intervention=dict(layer=layer,alpha=hook.alpha,threshold=hook.threshold,mode=hook.mode,
                                        operation=hook.operation,vector_sha256=__import__('hashlib').sha256(hook.steering_vector.cpu().numpy().tobytes()).hexdigest()) if hook else None)
        append(predpath,record); records.append(record); done.add(pid)
    for c in cases:
        statefile = out/(c['id']+'.npy')
        if not statefile.exists():
            features = extract_prompt_terminal_state(model,processor,str(Path('dataset/images')/c['image_file']),c['prompt']).numpy()
            with statefile.open('xb') as f:
                np.save(f,features)
        states.append(np.load(statefile,allow_pickle=False))
        generate(c,'baseline')
    states = np.stack(states)
    state_hashes={c['id']:digest(out/(c['id']+'.npy')) for c in cases}
    if (out/'activation_hashes.json').exists() and json.loads((out/'activation_hashes.json').read_text())!=state_hashes:
        raise ValueError('Saved activation bytes changed; do not resume this run')
    write_json(out/'activation_hashes.json',state_hashes)
    from scripts.blind_review import export_review
    export_review(out, seed)
    review_rows = load_jsonl(args.reviews) if args.reviews else []
    if len({r['prediction_id'] for r in review_rows}) != len(review_rows):
        raise ValueError('Duplicate human reviews')
    reviews = {r['prediction_id']:r for r in review_rows}
    cal = [i for i,c in enumerate(cases) if splits[c['id']] == 'calibration']
    labels = []
    for i in cal:
        pid = cases[i]['id']+':baseline'; review = reviews.get(pid)
        if not review or review.get('label') not in {'correct','incorrect','hallucinated'} or not review.get('reviewer') or not review.get('reason'):
            write_json(out/'status.json',dict(status='needs_calibration_reviews', inference_executed=True, instruction='Review blind_review_<count>.jsonl; use blind_key.private.json to prepare reviews.jsonl; rerun with --reviews'))
            print('Baseline saved. Review all calibration answers before model selection; see status.json.')
            return
        labels.append(int(review['label']=='hallucinated'))
    groups = [cases[i]['image_group'] for i in cal]
    calibration_reviews=[reviews[cases[i]['id']+':baseline'] for i in cal]
    calibration_hash=__import__('hashlib').sha256(json.dumps(calibration_reviews,sort_keys=True).encode()).hexdigest()
    if (out/'selection.json').exists() and json.loads((out/'selection.json').read_text())['calibration_reviews_sha256']!=calibration_hash:
        raise ValueError('Calibration labels changed after selection; frozen runs cannot be retuned')
    selection, vec, probe = select_calibration(states[cal],labels,groups,seed)
    selection['calibration_prediction_ids']=[cases[i]['id']+':baseline' for i in cal]
    selection['calibration_reviews_sha256']=calibration_hash
    write_json(out/'selection.json',selection)
    np.save(out/'direction.npy',vec)
    np.savez(out/'probe.npz',coef=probe.coef_,intercept=probe.intercept_,classes=probe.classes_)
    for c,features in zip(cases,states):
        if splits[c['id']] == 'calibration':
            continue
        probability=float(probe.predict_proba(features[selection['layer']][None,:])[0,1])
        hook = RotationalSteeringHook(torch.tensor(vec),alpha=config['alpha'],threshold=selection['threshold'])
        generate(c,'steered',hook,selection['layer'],probability)
        if args.ablations:
            for method in config['ablation_methods']:
                altvec=vec; mode='adaptive'; operation='slerp'
                if method in {'random_direction','mean_difference'}:
                    altvec,_=direction(states[cal,selection['layer']],np.asarray(labels),method,seed)
                elif method=='shuffled_labels':
                    altvec,_=direction(states[cal,selection['layer']],np.random.default_rng(seed).permutation(labels))
                elif method=='constant':
                    mode='constant'
                elif method=='addition':
                    operation='addition'
                hook=RotationalSteeringHook(torch.tensor(altvec),alpha=config['alpha'],threshold=selection['threshold'],mode=mode,operation=operation)
                generate(c,method,hook,selection['layer'],probability)
    export_review(out,seed)
    write_json(out/'status.json',dict(status='inference_complete_needs_all_answer_reviews',inference_executed=True))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Preserve errors in the selected run folder, even on partial runs.
        if '--output' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
            with (out/'errors.log').open('a',encoding='utf8') as f:
                f.write(traceback.format_exc()+'\n')
            write_json(out/'failure.json',dict(status='failed',error=str(sys.exc_info()[1]),predictions_saved=(out/'predictions.jsonl').exists()))
        raise
