"""Resumable POPE pipeline: baseline -> calibration fits -> validation sweep -> test.

Every stage appends raw rows to <out>/predictions.jsonl (one JSON object per
model answer) and never overwrites earlier rows, so a Colab disconnect only
loses the in-flight question. Selection uses calibration (layers, detectors,
directions, gates) and validation (steering strength) only; test rows are
generated once per method with frozen settings.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .detection import fit_detectors, fit_direction, gate_threshold, score_logits, score_states
from .pope import PARTITIONS, image_disjoint_split, limit_split, parse_yes_no, partition_of, pope_metrics
from .rotational_steering import RotationalSteeringHook

SHARD_SIZE = 256


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf8')


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def append_jsonl(path, rows):
    with Path(path).open('a', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def vector_sha256(vec):
    return hashlib.sha256(np.ascontiguousarray(vec, dtype=np.float64).tobytes()).hexdigest()


def plan_counts(config, split, items):
    part = partition_of(split)
    n = {p: sum(part.get(i['image']) == p for i in items) for p in PARTITIONS}
    s = config['steering']
    return dict(baseline=sum(n.values()),
                validation_sweep=n['validation'] * (len(s['slerp_alphas']) + len(s['addition_betas'])),
                test_steering=n['test'] * (2 + len(s['controls'])), questions_per_partition=n)


class Pipeline:
    def __init__(self, adapter, items, load_image, config, out, log=print):
        self.adapter, self.config, self.log = adapter, config, log
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'states').mkdir(exist_ok=True)
        self.load_image = load_image
        self.seed = config['seed']
        full = image_disjoint_split(sorted({i['image'] for i in items}), self.seed, config['split_fractions'])
        self.split = limit_split(full, config.get('max_images'))
        part = partition_of(self.split)
        self.items = sorted((dict(i, partition=part[i['image']]) for i in items if i['image'] in part),
                            key=lambda i: (PARTITIONS.index(i['partition']), i['image'], i['uid']))
        self.pred_path = self.out / 'predictions.jsonl'
        self.rows = {r['prediction_id']: r for r in read_jsonl(self.pred_path)}
        self.layers = config.get('state_layers') or list(range(adapter.num_layers))
        if self.layers == 'all':
            self.layers = list(range(adapter.num_layers))
        self.run_id = config.get('run_id')
        self._cached = (None, None)

    # ---------------------------------------------------------------- utils
    def part(self, name):
        return [i for i in self.items if i['partition'] == name]

    def _image(self, name):
        # Items are ordered by image, so a one-image cache loads each image once per pass.
        if self._cached[0] != name:
            self._cached = (name, self.load_image(name))
        return self._cached[1]

    def _row(self, item, method, strength, result, intervention=None):
        pred = parse_yes_no(result['answer'])
        return dict(prediction_id=f'{method}|{item["uid"]}', uid=item['uid'], method=method, strength=strength,
                    partition=item['partition'], pope_split=item['pope_split'], image=item['image'],
                    question_id=item['question_id'], question=item['question'], label=item['label'],
                    answer=result['answer'], pred=pred, correct=pred == item['label'],
                    first_token_id=result['first_token_id'], logprob_first=result['logprob_first'],
                    entropy_first=result['entropy_first'], p_yes=result['p_yes'], p_no=result['p_no'],
                    intervention=intervention, seed=self.seed, run_id=self.run_id, gate_trace=result.get('gate_trace'))

    def _generate(self, items, method, strength=None, steering=None, intervention=None, budget=None):
        todo = [i for i in items if f'{method}|{i["uid"]}' not in self.rows]
        if budget is not None:
            todo = todo[:budget]
        start = time.time()
        for k, item in enumerate(todo):
            hook = steering() if steering else None
            torch.manual_seed(self.seed)
            result = self.adapter.run(self._image(item['image']), item['question'],
                                      max_new_tokens=self.config['max_new_tokens'],
                                      steering=(intervention['layer'], hook) if hook else None)
            row = self._row(item, method, strength, result, intervention)
            append_jsonl(self.pred_path, [row]); self.rows[row['prediction_id']] = row
            if (k + 1) % 100 == 0:
                self.log(f'{method}: {k+1}/{len(todo)} ({(time.time()-start)/(k+1):.2f}s/question)')
        return len(todo), time.time() - start

    # ------------------------------------------------------------- baseline
    def _shards(self):
        index = {}
        for meta in sorted((self.out / 'states').glob('shard_*.json')):
            for row, uid in enumerate(json.loads(meta.read_text())['uids']):
                index[uid] = (meta.with_suffix('.npy'), row)
        return index

    def baseline(self, budget=None):
        """Answer every selected question once, saving prefill terminal states of all probed layers."""
        index = self._shards()
        todo = [i for i in self.items if f'baseline|{i["uid"]}' not in self.rows or i['uid'] not in index]
        if budget is not None:
            todo = todo[:budget]
        buffer_rows, buffer_states, start = [], [], time.time()
        shard_id = len(list((self.out / 'states').glob('shard_*.npy')))

        def flush():
            nonlocal shard_id, buffer_rows, buffer_states
            if not buffer_rows:
                return
            # States first, rows second: a crash between the two only orphans a shard.
            name = self.out / 'states' / f'shard_{shard_id:05d}'
            np.save(name.with_suffix('.npy'), np.stack(buffer_states))
            write_json(name.with_suffix('.json'), dict(uids=[r['uid'] for r in buffer_rows], layers=self.layers))
            fresh = [r for r in buffer_rows if r['prediction_id'] not in self.rows]
            append_jsonl(self.pred_path, fresh)
            self.rows.update({r['prediction_id']: r for r in fresh})
            shard_id += 1; buffer_rows, buffer_states = [], []
        for k, item in enumerate(todo):
            torch.manual_seed(self.seed)
            result = self.adapter.run(self._image(item['image']), item['question'],
                                      max_new_tokens=self.config['max_new_tokens'], capture_layers=self.layers)
            buffer_states.append(result.pop('states'))
            buffer_rows.append(self._row(item, 'baseline', 0.0, result))
            if len(buffer_rows) >= SHARD_SIZE:
                flush()
            if (k + 1) % 100 == 0:
                self.log(f'baseline: {k+1}/{len(todo)} ({(time.time()-start)/(k+1):.2f}s/question)')
        flush()
        return len(todo), time.time() - start

    def states(self, items):
        index = self._shards(); cache = {}
        out = []
        for i in items:
            path, row = index[i['uid']]
            if path not in cache:
                cache[path] = np.load(path, mmap_mode='r')
            out.append(np.asarray(cache[path][row]))  # fp16; detectors cast one layer at a time
        return np.stack(out)

    def baseline_rows(self, items):
        return [self.rows[f'baseline|{i["uid"]}'] for i in items]

    # ---------------------------------------------------------- calibration
    def target(self, rows):
        """1 = hallucination. 'error': any wrong answer; 'false_yes': yes about an absent object."""
        if self.config['detection']['target'] == 'false_yes':
            return np.array([int(r['label'] == 'no' and r['pred'] == 'yes') for r in rows])
        return np.array([int(not r['correct']) for r in rows])

    def calibrate(self):
        cal = self.part('calibration')
        rows = self.baseline_rows(cal)
        y = self.target(rows); X = self.states(cal); groups = [i['image'] for i in cal]
        det = self.config['detection']
        selection, fitted, per_layer = fit_detectors(X, y, groups, self.seed, det['cv_folds'], det['C'])
        selection['state_layers'] = self.layers
        selection['model_layers'] = {d: int(self.layers[i]) for d, i in selection['layers'].items()}
        selection['target'] = det['target']
        # Steering acts on the probe-selected layer, indexed into the captured layer list.
        k = selection['layers']['probe'] if self.config['steering'].get('layer', 'probe') == 'probe' \
            else self.layers.index(self.config['steering']['layer'])
        Xl = X[:, k].astype(np.float32)
        directions, plan = {}, {}
        for kind in ('pca', 'mean_difference', 'random_direction', 'shuffled_labels'):
            vec = fit_direction(Xl, y, kind, self.seed)
            directions[kind] = vec
            gate = self.config['steering']['gate'] == 'calibration_cosine'
            plan[kind] = dict(direction=kind, layer=int(self.layers[k]), state_index=int(k),
                              threshold=gate_threshold(Xl, y, vec) if gate else None,
                              mode='adaptive' if gate else 'constant', vector_sha256=vector_sha256(vec))
        plan['_mean_norm'] = float(np.linalg.norm(Xl, axis=-1).mean())
        write_json(self.out / 'selection.json', selection)
        write_json(self.out / 'steering_plan.json', plan)
        np.savez(self.out / 'directions.npz', **directions)
        # Held-out scores; nothing below is fitted.
        scores = []
        for name in ('validation', 'test'):
            items = self.part(name)
            if not items:
                continue
            Xh = self.states(items); brs = self.baseline_rows(items); yh = self.target(brs)
            s = score_states(Xh, selection, fitted)
            s.update(score_logits([r['logprob_first'] for r in brs], [r['entropy_first'] for r in brs]))
            for j, (item, r) in enumerate(zip(items, brs)):
                scores.append(dict(uid=item['uid'], partition=name, pope_split=item['pope_split'], image=item['image'],
                                   label=r['label'], pred=r['pred'], y=int(yh[j]),
                                   scores={k2: float(v[j]) for k2, v in s.items()}))
            if name == 'test':
                matrix = np.stack([m.predict_proba(Xh[:, l].astype(np.float32))[:, 1] for l, m in enumerate(per_layer)], 1)
                np.savez(self.out / 'layer_scores_test.npz', uids=np.array([i['uid'] for i in items]),
                         y=yh, images=np.array([i['image'] for i in items]), scores=matrix,
                         layers=np.array(self.layers))
        (self.out / 'detection_scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in scores), encoding='utf8')
        return selection, plan

    # ------------------------------------------------------------- steering
    def _method(self, name, strength, plan, directions):
        spec = dict(slerp_pca=('pca', 'slerp'), add_mean_difference=('mean_difference', 'addition'),
                    random_direction=('random_direction', 'slerp'), shuffled_labels=('shuffled_labels', 'slerp'))[name]
        p = plan[spec[0]]
        alpha = strength * plan['_mean_norm'] if spec[1] == 'addition' else strength
        vec = torch.tensor(directions[spec[0]], dtype=torch.float32)
        intervention = dict(p, operation=spec[1], strength=strength, alpha=alpha)
        threshold = p['threshold'] if p['threshold'] is not None else 0.0
        return (lambda: RotationalSteeringHook(vec, alpha=alpha, threshold=threshold, mode=p['mode'],
                                               operation=spec[1], record_trace=True)), intervention

    def _load_plan(self):
        plan = json.loads((self.out / 'steering_plan.json').read_text())
        directions = dict(np.load(self.out / 'directions.npz'))
        return plan, directions

    def sweep(self):
        """Validation-only strength selection: highest validation accuracy, ties to the smaller strength."""
        plan, directions = self._load_plan()
        val = self.part('validation'); s = self.config['steering']
        base = pope_metrics([r['label'] for r in self.baseline_rows(val)], [r['pred'] for r in self.baseline_rows(val)])
        result = dict(baseline=base, methods={})
        for name, grid in (('slerp_pca', s['slerp_alphas']), ('add_mean_difference', s['addition_betas'])):
            curve = []
            for strength in grid:
                method = f'{name}@{strength:g}'
                hook, intervention = self._method(name, strength, plan, directions)
                self._generate(val, method, strength, hook, intervention)
                rows = [self.rows[f'{method}|{i["uid"]}'] for i in val]
                curve.append(dict(strength=strength, **pope_metrics([r['label'] for r in rows], [r['pred'] for r in rows])))
            best = max(curve, key=lambda c: (c['accuracy'], -c['strength']))
            result['methods'][name] = dict(curve=curve, selected_strength=best['strength'],
                                           improves_on_validation_baseline=best['accuracy'] > base['accuracy'])
        write_json(self.out / 'validation_sweep.json', result)
        return result

    def test(self):
        plan, directions = self._load_plan()
        chosen = json.loads((self.out / 'validation_sweep.json').read_text())['methods']
        test = self.part('test')
        runs = [('slerp_pca', chosen['slerp_pca']['selected_strength']),
                ('add_mean_difference', chosen['add_mean_difference']['selected_strength'])]
        # Controls reuse the SLERP strength chosen on validation (matched dose, never tuned).
        runs += [(c, chosen['slerp_pca']['selected_strength']) for c in self.config['steering']['controls']]
        for name, strength in runs:
            hook, intervention = self._method(name, strength, plan, directions)
            self._generate(test, name, strength, hook, intervention)
        write_json(self.out / 'test_plan.json', dict(runs=[dict(method=m, strength=s) for m, s in runs]))
