"""Report held-out metrics only after complete human adjudication."""
import argparse
import json
from pathlib import Path
from src.protocol import cluster_bootstrap, probe_metrics, digest
from scripts.run_experiment import load_jsonl, write_json


def analyze(run, review_file):
    run=Path(run); rows=load_jsonl(run/'predictions.jsonl'); reviews=load_jsonl(review_file)
    identity=json.loads((run/'identity.json').read_text()); config=identity['config']
    splits=json.loads((run/'splits.json').read_text())
    if len({r['prediction_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate predictions')
    if len({r['prediction_id'] for r in reviews})!=len(reviews):
        raise ValueError('Duplicate reviews')
    review={r['prediction_id']:r for r in reviews}
    expected_ids=set(splits)
    if {r['id'] for r in rows if r['method']=='baseline'} != expected_ids:
        raise ValueError('Incomplete baseline coverage')
    held_ids={k for k,v in splits.items() if v!='calibration'}
    if {r['id'] for r in rows if r['method']=='steered'} != held_ids:
        raise ValueError('Incomplete steered coverage')
    for r in rows:
        if r['run_id'] != identity['run_id'] or r['split']!=splits[r['id']]:
            raise ValueError('Provenance mismatch')
        rev=review.get(r['prediction_id'],{})
        if rev.get('label') not in {'correct','incorrect','hallucinated'} or not rev.get('reviewer') or not rev.get('reason'):
            raise ValueError('Complete manual review required: '+r['prediction_id'])
        r['label']=rev['label']
    output=dict(run_id=identity['run_id'],commit=identity['commit'],seed=config['seed'],config=config,
                reviews_sha256=digest(review_file),dataset_sha256=identity['dataset_sha256'],
                ci_method='paired image-cluster percentile bootstrap; conditional on frozen calibration',metrics={})
    seed=config['seed']; iterations=config['bootstrap_iterations']
    def boot(rs,metric):
        return cluster_bootstrap(rs,metric,seed,iterations)
    for split in ['validation','test']:
        group=[r for r in rows if r['split']==split]
        metrics={}
        for method in sorted({r['method'] for r in group}):
            for category in ['overall','counter_attribute','counter_existence','positive_control']:
                subset=[r for r in group if r['method']==method and (category=='overall' or r['category']==category)]
                entry=dict(n=len(subset),correct=sum(r['label']=='correct' for r in subset),
                           hallucinated=sum(r['label']=='hallucinated' for r in subset),
                           incorrect=sum(r['label']=='incorrect' for r in subset))
                for key,label in [('accuracy','correct'),('hallucination_rate','hallucinated')]:
                    entry[key]=boot(subset,lambda rs:sum(r['label']==label for r in rs)/len(rs))
                metrics[method+':'+category]=entry
        for method in sorted({r['method'] for r in group}-{'baseline'}):
            matched=[r for r in group if r['method'] in {'baseline',method}]
            if {r['id'] for r in matched if r['method']==method}!=held_ids.intersection(r['id'] for r in group):
                raise ValueError('Incomplete ablation coverage')
            for category in ['overall','counter_attribute','counter_existence','positive_control']:
                pairs=[r for r in matched if category=='overall' or r['category']==category]
                def delta(rs):
                    b=[r for r in rs if r['method']=='baseline']; s=[r for r in rs if r['method']==method]
                    if not b or not s:return None
                    return sum(r['label']=='hallucinated' for r in s)/len(s)-sum(r['label']=='hallucinated' for r in b)/len(b)
                metrics[method+':'+category+':hallucination_delta']=boot(pairs,delta)
            controls=[r for r in matched if r['category']=='positive_control']
            def retention(rs):
                correct={r['id'] for r in rs if r['method']=='baseline' and r['label']=='correct'}
                selected=[r for r in rs if r['method']==method and r['id'] in correct]
                return sum(r['label']=='correct' for r in selected)/len(selected) if selected else None
            metrics[method+':paired_sightedness_retention']=boot(controls,retention)
        base={r['id']:r for r in group if r['method']=='baseline'}
        probes=[dict(r, y=int(base[r['id']]['label']=='hallucinated')) for r in group if r['method']=='steered']
        for name in ['auroc','precision','recall','f1']:
            metrics['probe:'+name]=boot(probes,lambda rs:probe_metrics([r['y'] for r in rs],[r['probe_probability'] for r in rs])[name])
        output['metrics'][split]=metrics
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reviews',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.run,a.reviews)
    if Path(a.output).exists():raise FileExistsError('Preserve previous analysis; choose a new output')
    write_json(a.output,result)


if __name__=='__main__':main()
