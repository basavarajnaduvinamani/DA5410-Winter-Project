"""Resolve public attribution for a bounded illustrative sample, keeping photos out of Git."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile
from scripts.ingest_run import ROOT, read, rows, validate_run
from scripts.export_site_examples import build_examples

ANNOTATION_SHA='031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'
def get(url, limit):
    request=urllib.request.Request(url,headers={'User-Agent':'CounterPrior-research-attribution/1.0'})
    with urllib.request.urlopen(request,timeout=25) as response:
        payload=response.read(limit+1)
    if len(payload)>limit:raise ValueError('Response exceeds expected size')
    return payload

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--annotations-zip',required=True);a=p.parse_args()
    validate_run(a.run)
    archive=Path(a.annotations_zip)
    with archive.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=ANNOTATION_SHA:raise ValueError('Official COCO annotation archive checksum mismatch')
    # Read only the metadata prefix, excluding hundreds of thousands of captions.
    with zipfile.ZipFile(archive) as z, z.open('annotations/captions_val2014.json') as f:
        prefix=b''
        while len(prefix)<64*1024*1024:
            chunk=f.read(1024*1024)
            if not chunk:raise ValueError('COCO metadata layout differs from the verified archive')
            prefix+=chunk
            boundary=re.search(rb',\s*"annotations"\s*:',prefix)
            if boundary:
                meta=json.loads(prefix[:boundary.start()]+b'}');break
        else:raise ValueError('COCO metadata prefix exceeds expected size')
    licenses={x['id']:x for x in meta['licenses']};images={x['file_name']:x for x in meta['images']}
    raw=rows(Path(a.run)/'predictions.jsonl');base={x['uid']:x for x in raw if x['method']=='baseline' and x['partition']=='test'}
    steered={x['uid']:x for x in raw if x['method']=='slerp_pca'}
    def rank(b):
        s=steered.get(b['uid']);return (0 if s and b['correct'] and not s['correct'] else 1 if s and not b['correct'] and s['correct'] else 2,b['pope_split'],b['uid'])
    output=ROOT/'data/site-examples';output.mkdir(parents=True,exist_ok=True)
    selected=[];seen=set();excluded=[]
    for b in sorted(base.values(),key=rank):
        name=b['image']
        if name in seen or len(selected)>=24:continue
        seen.add(name);image=dict(images[name]);licence=licenses[image['license']]
        url=licence['url'].replace('http://','https://')
        if not any(v in url for v in ['creativecommons.org/licenses/by/','creativecommons.org/licenses/by-sa/','creativecommons.org/licenses/by-nd/','creativecommons.org/publicdomain/']):
            excluded.append({'image':name,'reason':'Outside the BY/BY-SA/BY-ND/public-domain redistribution policy'});continue
        match=re.search(r'/(\d+)_',image.get('flickr_url',''))
        if not match:
            excluded.append({'image':name,'reason':'No Flickr photo identity in original annotation'});continue
        photo_id=match.group(1);number=int(photo_id);encoded=''
        alphabet='123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ'
        while number:
            number,digit=divmod(number,58);encoded=alphabet[digit]+encoded
        source='https://flic.kr/p/'+encoded
        endpoint='https://www.flickr.com/services/oembed/?'+urllib.parse.urlencode({'url':source,'format':'json'})
        try:
            credit=json.loads(get(endpoint,1024*1024))
            if not credit.get('author_name') or not credit.get('author_url','').startswith('https://www.flickr.com/'):
                raise ValueError('Photographer attribution is incomplete')
            image.update(author=credit['author_name'],photo_title=credit.get('title'),flickr_url=credit['author_url'].rstrip('/')+'/'+photo_id+'/',attribution_evidence={'url':endpoint,'short_url':source,'author_url':credit['author_url'],'payload_sha256':hashlib.sha256(json.dumps(credit,sort_keys=True).encode()).hexdigest()})
            target=output/name
            if not target.exists():target.write_bytes(get('https://s3.amazonaws.com/images.cocodataset.org/val2014/'+name,8*1024*1024))
            selected.append(image)
        except Exception as error:excluded.append({'image':name,'reason':str(error)})
        print('Attribution checked:',name,'eligible:',len(selected),flush=True)
    enriched=output/'verified_metadata.json'
    enriched.write_text(json.dumps({'images':selected,'licenses':meta['licenses'],'archive_sha256':ANNOTATION_SHA},indent=2)+'\n',encoding='utf8')
    manifest=build_examples(a.run,enriched,output)
    manifest['attribution_checks']=excluded
    manifest['attribution_sources']={x['file_name']:x['attribution_evidence'] for x in selected}
    (ROOT/'results/examples_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print('Published eligible examples:',len(manifest['examples']))
if __name__=='__main__':main()
