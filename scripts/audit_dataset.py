"""Verify the manually reviewed manifest against current bytes and pixels."""
import hashlib
import json
from pathlib import Path
import subprocess
from collections import Counter, defaultdict
from PIL import Image


def main():
    records = json.loads(Path('dataset/audit_2026-10-30.json').read_text())
    cases = json.loads(Path('dataset/counterprior_corrected.json').read_text())
    groups = defaultdict(list)
    for row in records:
        p = Path('dataset/images') / row['image_file']
        im = Image.open(p).convert('RGB')
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['file_sha256'], p
        pixel = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        assert pixel == row['pixel_sha256'], p
        groups[pixel].append(row['id'])
    assert {r['id'] for r in records if r['decision'] == 'retain_provisional'} == {c['id'] for c in cases}
    for c in cases:
        r = next(r for r in records if r['id'] == c['id'])
        assert c['image_group'] == r['pixel_sha256'] and c['image_sha256'] == r['file_sha256']
    summary = dict(command='python -m scripts.audit_dataset', seed=None,
                   commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   original_rows=len(records), retained_rows=len(cases), excluded_rows=len(records)-len(cases),
                   category_counts=dict(Counter(c['category'] for c in cases)),
                   exact_duplicate_image_groups=[v for v in groups.values() if len(v)>1],
                   manifest_sha256=hashlib.sha256(Path('dataset/counterprior_corrected.json').read_bytes()).hexdigest())
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
