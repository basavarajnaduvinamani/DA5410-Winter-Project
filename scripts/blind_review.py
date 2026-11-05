"""Export method/split-blinded records; mapping stays with experiment operator."""
import hashlib
import json
from pathlib import Path


def export_review(out, seed):
    out=Path(out)
    rows=[json.loads(s) for s in (out/'predictions.jsonl').read_text(encoding='utf8').splitlines() if s]
    key={}
    blinded=[]
    for r in rows:
        review_id=hashlib.sha256((str(seed)+r['prediction_id']).encode()).hexdigest()[:20]
        key[review_id]=r['prediction_id']
        blinded.append(dict(review_id=review_id,image_file='dataset/images/'+next(c['image_file'] for c in json.loads(Path('dataset/counterprior_corrected.json').read_text()) if c['id']==r['id']),prompt=r['prompt'],answer=r['answer'],label='',reviewer='',reason=''))
    blinded.sort(key=lambda r:r['review_id'])
    # Export snapshots never replace filled reviews.
    target=out/('blind_review_'+str(len(rows))+'.jsonl')
    if not target.exists():
        target.write_text(''.join(json.dumps(r)+'\n' for r in blinded),encoding='utf8')
    (out/'blind_key.private.json').write_text(json.dumps(key,indent=2),encoding='utf8')
    return target


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--filled',required=True);p.add_argument('--key',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    key=json.loads(Path(a.key).read_text());rows=[json.loads(s) for s in Path(a.filled).read_text().splitlines() if s]
    with Path(a.output).open('x',encoding='utf8') as f:
        for r in rows:
            if r.get('label'):
                f.write(json.dumps(dict(prediction_id=key[r['review_id']],label=r['label'],reviewer=r['reviewer'],reason=r['reason']))+'\n')


if __name__=='__main__':
    main()
