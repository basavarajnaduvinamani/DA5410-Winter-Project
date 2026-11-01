"""Validate a saved POPE run before generating public aggregates. Never runs a model."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile
from src.pope import load_pope, image_disjoint_split, limit_split, partition_of, parse_yes_no

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'identity.json','splits.json','predictions.jsonl','detection_scores.jsonl','selection.json','steering_plan.json','validation_sweep.json','test_plan.json','directions.npz','layer_scores_test.npz','status.json','environment.json','timing.json','sessions.jsonl','environment-freeze.txt'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def extract_bundle(bundle,destination):
    destination=Path(destination).resolve()
    if not destination.is_relative_to(ROOT/'incoming'):raise ValueError('ZIP destination must be inside ignored incoming/')
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(bundle) as z:
        for item in z.infolist():
            parts=PurePosixPath(item.filename).parts
            if not parts or '..' in parts or PurePosixPath(item.filename).is_absolute() or ':' in item.filename or stat.S_ISLNK(item.external_attr>>16):
                raise ValueError('Unsafe ZIP member: '+item.filename)
            if parts[0]!='run' or len(parts)!=2 or parts[1] not in ALLOWED:continue
            if item.file_size>512*1024*1024:raise ValueError('Unexpectedly large run artifact')
            target=destination/parts[1]
            payload=z.read(item)
            if target.exists() and target.read_bytes()!=payload:raise ValueError('Refusing to overwrite a different saved artifact: '+target.name)
            target.write_bytes(payload)
    return destination


def validate_run(run):
    run=Path(run)
    for name in ['identity.json','splits.json','predictions.jsonl','detection_scores.jsonl','selection.json','status.json','environment.json']:
        if not (run/name).exists():raise ValueError('Incomplete run: missing '+name)
    identity=read(run/'identity.json');cfg=identity['config']
    identity_base={k:v for k,v in identity.items() if k!='run_id'}
    expected=hashlib.sha256(json.dumps(identity_base,sort_keys=True).encode()).hexdigest()[:16]
    if identity.get('run_id')!=expected:raise ValueError('Run identity hash mismatch')
    known=[read(ROOT/'configs/pope_llava.json'),read(ROOT/'configs/pope_qwen_replication.json')]
    reference=next((k for k in known if k['model_id']==cfg.get('model_id') and k['revision']==cfg.get('revision')),None)
    if not reference:raise ValueError('Unknown or unpinned model checkpoint')
    if cfg.get('profile') not in ('quick','full'):raise ValueError('Unregistered profile')
    for key in ['adapter','quantization','max_new_tokens','seed','split_fractions','detection','steering','bootstrap_iterations','state_layers','attn_implementation']:
        if cfg.get(key)!=reference.get(key):raise ValueError('Config differs from registered protocol: '+key)
    max_images=reference['profiles'][cfg['profile']].get('max_images',reference.get('max_images'))
    if cfg.get('max_images')!=max_images:raise ValueError('Profile sample limit differs')
    for name,h in identity['pope_sha256'].items():
        if sha(ROOT/'dataset/pope'/('coco_pope_'+name+'.json'))!=h:raise ValueError('Question hash mismatch: '+name)
    if set(identity['pope_sha256'])!=set(reference['pope_splits']):raise ValueError('Question split coverage differs')
    items=load_pope(ROOT/'dataset/pope',reference['pope_splits'])
    split=limit_split(image_disjoint_split(sorted({i['image'] for i in items}),cfg['seed'],cfg['split_fractions']),max_images)
    stored=read(run/'splits.json')
    if stored.get('used')!=split:raise ValueError('Stored image split differs from frozen protocol')
    part=partition_of(split);selected={i['uid']:dict(i,partition=part[i['image']]) for i in items if i['image'] in part}
    raw=rows(run/'predictions.jsonl');seen=set();by_method={}
    for r in raw:
        if r['prediction_id'] in seen:raise ValueError('Duplicate prediction ID: '+r['prediction_id'])
        seen.add(r['prediction_id']);i=selected.get(r['uid'])
        if i is None:raise ValueError('Prediction is outside the selected image set')
        for key in ['image','question','label','pope_split','partition']:
            if r.get(key)!=i[key]:raise ValueError('Prediction/case mismatch: '+key)
        if r.get('run_id')!=expected or r.get('seed')!=cfg['seed']:raise ValueError('Prediction provenance mismatch')
        pred=parse_yes_no(r['answer'])
        if pred!=r['pred'] or (pred==r['label'])!=r['correct']:raise ValueError('Official POPE score mismatch')
        if r['prediction_id']!=r['method']+'|'+r['uid']:raise ValueError('Malformed prediction identity')
        by_method.setdefault(r['method'],{})[r['uid']]=r
    if set(by_method.get('baseline',{}))!=set(selected):raise ValueError('Baseline coverage is incomplete')
    modern=cfg['adapter']!='llava';required={'baseline','calibrate','report'} if modern else {'baseline','calibrate','sweep','test','report'}
    if not required.issubset(read(run/'status.json').get('stages_done',[])):raise ValueError('Run did not complete all required stages')
    env=read(run/'environment.json')
    if not env.get('cuda') or not env.get('gpu'):raise ValueError('Pretrained-model GPU execution is not recorded')
    scores=rows(run/'detection_scores.jsonl');wanted={u for u,i in selected.items() if i['partition'] in {'validation','test'}}
    if {r['uid'] for r in scores}!=wanted or len(scores)!=len(wanted):raise ValueError('Detection coverage or uniqueness mismatch')
    for r in scores:
        base=by_method['baseline'][r['uid']];target=int(not base['correct']) if cfg['detection']['target']=='error' else int(base['label']=='no' and base['pred']=='yes')
        if r.get('y')!=target or r['image']!=base['image'] or r['partition']!=base['partition']:raise ValueError('Detection label mismatch')
    if not modern:
        for name in ['validation_sweep.json','steering_plan.json','test_plan.json','layer_scores_test.npz']:
            if not (run/name).exists():raise ValueError('Incomplete intervention evidence: '+name)
        sweep=read(run/'validation_sweep.json');test_uids={u for u,i in selected.items() if i['partition']=='test'}
        strength={}
        for method,grid in [('slerp_pca',cfg['steering']['slerp_alphas']),('add_mean_difference',cfg['steering']['addition_betas'])]:
            saved=sweep['methods'][method]
            if [r['strength'] for r in saved['curve']]!=grid:raise ValueError('Validation grid changed')
            best=max(saved['curve'],key=lambda c:(c['accuracy'],-c['strength']))['strength']
            if saved['selected_strength']!=best:raise ValueError('Strength was not selected by the frozen validation rule')
            strength[method]=best
        strength.update({k:strength['slerp_pca'] for k in cfg['steering']['controls']})
        for method,alpha in strength.items():
            got=by_method.get(method,{})
            if set(got)!=test_uids:raise ValueError('Test coverage incomplete: '+method)
            if any(r['strength']!=alpha or r['partition']!='test' for r in got.values()):raise ValueError('Test intervention differs from validation selection')
    return dict(status='verified',run_id=expected,profile=cfg['profile'],test_images=len(split['test']),model_id=cfg['model_id'],revision=cfg['revision'],target=cfg['detection']['target'],hardware=env['gpu'],runtime=None,verification_scope='Question bytes, registered configuration, image partition IDs, official scoring and saved output coverage; source COCO image bytes and model weights are not independently rehashed by this bundle check',config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),artifacts={p.name:sha(p) for p in run.iterdir() if p.is_file() and p.name in ALLOWED})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--zip');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if bool(a.run)==bool(a.zip):p.error('Supply exactly one of --run or --zip')
    run=Path(a.run) if a.run else extract_bundle(a.zip,ROOT/'incoming'/('bundle-'+sha(a.zip)[:12]))
    manifest=validate_run(run)
    if a.dry_run:print(json.dumps(manifest,indent=2));return
    from scripts.make_report import build_report
    modern=manifest['model_id']!='llava-hf/llava-1.5-7b-hf'
    target=ROOT/('results/REPORT_QWEN.md' if modern else 'results/REPORT.md')
    if (ROOT/'results/published_run.json').exists() and not modern:
        old=read(ROOT/'results/published_run.json')
        if old['run_id']!=manifest['run_id']:raise ValueError('Preserve the earlier published run; archive it before selecting another')
    build_report(run,target,ROOT/('results/figures-qwen' if modern else 'results/figures'),None if modern else ROOT/'README.md')
    manifest['metrics_sha256']=sha(target.with_name(target.stem+'_metrics.json'))
    record=ROOT/('results/published_qwen_run.json' if modern else 'results/published_run.json')
    record.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    if not modern:
        status_file=ROOT/'results/2026-11-01/gpu_status.json'
        previous=read(status_file) if status_file.exists() else {}
        initial=previous.get('initial_attempt',previous)
        status_file.write_text(json.dumps(dict(status='verified_saved_run',run_id=manifest['run_id'],profile=manifest['profile'],test_images=manifest['test_images'],hardware=manifest['hardware'],source='results/published_run.json',initial_attempt=initial),indent=2)+'\n',encoding='utf8')
        subprocess.run(['node','site/scripts/export-data.mjs'],cwd=ROOT,check=True)
        subprocess.run([__import__('sys').executable,'-m','scripts.build_public_report'],cwd=ROOT,check=True)
    print('Verified aggregates generated; raw run artifacts stay outside Git:',record)

if __name__=='__main__':main()
