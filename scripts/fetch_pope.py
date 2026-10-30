"""Verify the vendored POPE question files and download only the COCO val2014 images they use.

  python -m scripts.fetch_pope --coco-dir data/coco/val2014
Already-present images are kept (point --coco-dir at an existing val2014 folder to skip downloads).
Writes <coco-dir>/pope_images_sha256.json so a run can record exactly which bytes it saw.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from src.pope import load_pope


def verify_questions(pope_dir):
    sources = json.loads((Path(pope_dir) / 'sources.json').read_text())
    for name, meta in sources['files'].items():
        got = hashlib.sha256((Path(pope_dir) / name).read_bytes()).hexdigest()
        if got != meta['sha256']:
            raise ValueError(f'{name} sha256 {got} != pinned {meta["sha256"]}')
    return sources


def download(url, dest, retries=4):
    for attempt in range(retries + 1):
        try:
            tmp = dest.with_suffix('.part')
            with urllib.request.urlopen(url, timeout=60) as r:
                tmp.write_bytes(r.read())
            tmp.rename(dest)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt + 1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pope-dir', default='dataset/pope')
    p.add_argument('--coco-dir', required=True)
    a = p.parse_args(argv)
    sources = verify_questions(a.pope_dir)
    images = sorted({i['image'] for i in load_pope(a.pope_dir)})
    coco = Path(a.coco_dir); coco.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(images):
        dest = coco / name
        if not dest.exists():
            download(sources['images']['per_image_url'].format(image=name), dest)
        if (k + 1) % 50 == 0:
            print(f'{k+1}/{len(images)} images ready')
    from PIL import Image
    manifest = {}
    for name in images:
        with Image.open(coco / name) as im:
            im.verify()
        manifest[name] = hashlib.sha256((coco / name).read_bytes()).hexdigest()
    (coco / 'pope_images_sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{len(images)} POPE images verified in {coco}')


if __name__ == '__main__':
    main()
