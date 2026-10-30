"""POPE loader, official yes/no parsing, metrics and image-disjoint splits.

POPE (Li et al., EMNLP 2023) asks "Is there a <object> in the image?" about
COCO val2014 images. Question files are vendored in dataset/pope (MIT licence,
pinned commit in dataset/pope/sources.json); COCO images are supplied by the
user (scripts/fetch_pope.py downloads only the 500 referenced images).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

POPE_SPLITS = ('random', 'popular', 'adversarial')
PARTITIONS = ('calibration', 'validation', 'test')


def load_pope(pope_dir, splits=POPE_SPLITS):
    """Return one dict per question with a stable uid '<split>:<question_id>'."""
    items = []
    for split in splits:
        path = Path(pope_dir) / f'coco_pope_{split}.json'
        if not path.exists():
            raise FileNotFoundError(f'{path} missing; see dataset/pope/sources.json for the download URL')
        for line in path.read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            label = row['label'].strip().lower()
            if label not in {'yes', 'no'}:
                raise ValueError(f'Unexpected POPE label {label!r} in {path}')
            items.append(dict(uid=f"{split}:{row['question_id']}", pope_split=split,
                              question_id=row['question_id'], image=row['image'],
                              question=row['text'], label=label))
    if len({i['uid'] for i in items}) != len(items):
        raise ValueError('Duplicate POPE question ids')
    return items


def parse_yes_no(text):
    """Official POPE rule (AoiDragon/POPE evaluate.py): first sentence, 'no'/'not' => no, else yes."""
    first = text.split('.')[0].replace(',', '')
    words = first.split(' ')
    return 'no' if ('No' in words or 'not' in words or 'no' in words) else 'yes'


def pope_metrics(labels, predictions):
    """Accuracy, precision, recall, F1 with 'yes' as the positive class, plus yes-ratio."""
    y = np.asarray(labels) == 'yes'
    p = np.asarray(predictions) == 'yes'
    if len(y) == 0:
        return dict(n=0, accuracy=None, precision=None, recall=None, f1=None, yes_ratio=None)
    tp = int(np.sum(p & y)); fp = int(np.sum(p & ~y)); fn = int(np.sum(~p & y)); tn = int(np.sum(~p & ~y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(n=int(len(y)), accuracy=(tp + tn) / len(y), precision=precision, recall=recall, f1=f1,
                yes_ratio=float(p.mean()), tp=tp, fp=fp, fn=fn, tn=tn,
                # Object hallucination: answering yes about an absent object.
                false_yes_rate=fp / (fp + tn) if fp + tn else None)


def image_disjoint_split(images, seed, fractions):
    """Deterministic image-level partition; every question about an image shares its partition.

    Ordering uses sha256(seed:image), so it depends only on image ids and seed,
    never on labels, answers or model outputs.
    """
    if set(fractions) != set(PARTITIONS) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError('fractions must cover calibration/validation/test and sum to 1')
    ordered = sorted(set(images), key=lambda im: hashlib.sha256(f'{seed}:{im}'.encode()).hexdigest())
    n = len(ordered)
    n_cal = int(round(n * fractions['calibration']))
    n_val = int(round(n * fractions['validation']))
    if min(n_cal, n_val, n - n_cal - n_val) < 1:
        raise ValueError('Too few images for a three-way split')
    return dict(calibration=ordered[:n_cal], validation=ordered[n_cal:n_cal + n_val],
                test=ordered[n_cal + n_val:])


def limit_split(split, max_images):
    """Take the first k images per partition (the same deterministic order) for quick runs."""
    if not max_images:
        return split
    return {k: v[:max_images[k]] if max_images.get(k) else v for k, v in split.items()}


def partition_of(split):
    return {im: part for part, ims in split.items() for im in ims}
