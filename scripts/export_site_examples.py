"""Export at most the allowed licensed COCO examples; never downloads or invents attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from scripts.ingest_run import read,rows,sha,validate_run,ROOT


def build_examples(run,annotations,images,limit=24):
    if not 0<=limit<=24:raise ValueError('Example limit must stay within the site policy')
    proof=validate_run(run);meta=read(annotations)
    licences={l['id']:l for l in meta.get('licenses',[])};by_name={i['file_name']:i for i in meta.get('images',[])}
    raw=rows(Path(run)/'predictions.jsonl');base={r['uid']:r for r in raw if r['method']=='baseline' and r['partition']=='test'}
    steered={r['uid']:r for r in raw if r['method']=='slerp_pca'};scores={r['uid']:r for r in rows(Path(run)/'detection_scores.jsonl')}
    skipped=[];examples=[];used=set();dest=ROOT/'site/public/examples';dest.mkdir(exist_ok=True)
    # Regressions first, then fixes, then unchanged answers; examples are illustrative,
    # never the denominator for aggregate research claims.
    def rank(r):
        b=base[r['uid']];s=steered.get(r['uid'])
        return (0 if s and b['correct'] and not s['correct'] else 1 if s and not b['correct'] and s['correct'] else 2,r['pope_split'],r['uid'])
    for b in sorted(base.values(),key=rank):
        if len(examples)>=limit:break
        if b['image'] in used or b['uid'] not in steered:continue
        image=by_name.get(b['image']);licence=licences.get((image or {}).get('license'),{})
        # COCO's licence field alone is not a photographer credit. If author/creator
        # is missing, exclude the photograph instead of claiming complete attribution.
        creator=(image or {}).get('author') or (image or {}).get('creator')
        url=licence.get('url','').replace('http://','https://');source=(image or {}).get('flickr_url','').replace('http://','https://')
        permitted=any(x in url for x in ['creativecommons.org/licenses/by/','creativecommons.org/licenses/by-sa/','creativecommons.org/licenses/by-nd/','creativecommons.org/publicdomain/','flickr.com/commons/usage/','usa.gov/government-copyright'])
        if not image or not creator or not permitted or not source.startswith('https://'):
            skipped.append(dict(image=b['image'],reason='Redistribution terms or complete photographer attribution not verified from supplied annotations'));continue
        original=Path(images)/b['image']
        if not original.is_file():skipped.append(dict(image=b['image'],reason='Original image bytes not supplied'));continue
        s=steered[b['uid']];trace=s.get('gate_trace') or [];first=trace[0] if trace else {};gate=s.get('intervention') or {}
        alignment=(first.get('alignment') or [None])[0];triggered=(first.get('steered') or [None])[0]
        shutil.copyfile(original,dest/b['image']);used.add(b['image'])
        examples.append(dict(id=b['uid'],image=b['image'],imageId=image['id'],imageSrc='/examples/'+b['image'],imageSha256=sha(original),imageAlt='COCO val2014 image for the saved question: '+b['question'],attribution=dict(creator=creator,source=source,license=licence['name'],licenseUrl=url),split=b['pope_split'],question=b['question'],truth=b['label'],baseline=b['answer'],baselinePred=b['pred'],steered=s['answer'],steeredPred=s['pred'],probeScore=scores.get(b['uid'],{}).get('scores',{}).get('probe'),probeThreshold=.5,gate=dict(triggered=triggered,alignment=alignment,threshold=gate.get('threshold'),source='saved prefill hook trace' if trace else 'not recorded')))
    for e in examples:
        e['attribution']['title']=by_name[e['image']].get('photo_title')
    return dict(run_id=proof['run_id'],annotation_sha256=sha(annotations),selection_rule='Illustrative, deterministic: regressions, fixes, unchanged; one question per unique image; all aggregate results use the full held-out denominator',examples=examples,excluded=skipped)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--annotations',required=True);p.add_argument('--images',required=True);p.add_argument('--limit',type=int,default=24);a=p.parse_args()
    data=build_examples(a.run,a.annotations,a.images,a.limit)
    (ROOT/'results/examples_manifest.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    print('Licensed examples:',len(data['examples']),'Excluded:',len(data['excluded']))

if __name__=='__main__':main()
