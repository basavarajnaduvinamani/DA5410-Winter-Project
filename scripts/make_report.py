"""Build REPORT.md tables and figures from a run directory's saved files only.

  python -m scripts.make_report --run RUN --out results/REPORT.md --figures results/figures --readme README.md
  python -m scripts.make_report --placeholder --out results/REPORT.md --readme README.md   # before any run
No model is loaded and nothing is refitted; every number comes from
predictions.jsonl, detection_scores.jsonl, layer_scores_test.npz,
selection.json and validation_sweep.json, with image-clustered bootstrap CIs.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.evaluation import auprc, auroc, cluster_bootstrap_idx
from src.pope import POPE_SPLITS, pope_metrics

README_START, README_END = '<!-- RESULTS:START -->', '<!-- RESULTS:END -->'
DETECTORS = [('probe', 'Linear probe (hidden state)'), ('mean_difference', 'Mean-difference projection'),
             ('pca', 'PCA truth-axis projection'), ('neg_logprob', 'Answer-token −log p'),
             ('entropy', 'Answer-token entropy')]
METHODS = [('slerp_pca', 'SLERP → PCA truth axis'), ('add_mean_difference', 'Add mean-difference'),
           ('random_direction', 'Control: random direction (SLERP)'),
           ('shuffled_labels', 'Control: shuffled-label axis (SLERP)')]
# Categorical slots 1-4 of the dataviz reference palette, fixed order.
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
DASH = '—'


def read_jsonl(path):
    path = Path(path)
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()] if path.exists() else []


def fmt(entry, digits=3):
    if not entry or entry.get('estimate') is None:
        return DASH
    est = f"{entry['estimate']:.{digits}f}"
    if entry.get('ci95'):
        lo, hi = entry['ci95']
        return f'{est} [{lo:.{digits}f}, {hi:.{digits}f}]'
    return est


def f3(v):
    return DASH if v is None else f'{v:.3f}'


def nan(v):
    return np.nan if v is None else v


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join([':---'] + [':---:'] * (len(header) - 1)) + '|']
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


class Run:
    def __init__(self, run):
        self.dir = Path(run)
        self.identity = json.loads((self.dir / 'identity.json').read_text())
        self.config = self.identity['config']
        self.seed = self.config['seed']; self.iterations = self.config.get('bootstrap_iterations', 1000)
        rows = read_jsonl(self.dir / 'predictions.jsonl')
        self.rows = {}
        for r in rows:
            self.rows.setdefault(r['method'], {})[r['uid']] = r
        self.splits = json.loads((self.dir / 'splits.json').read_text()) if (self.dir / 'splits.json').exists() else {}

    def load(self, name):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else None

    def boot(self, groups, fn):
        return cluster_bootstrap_idx(groups, fn, self.seed, self.iterations)

    # ------------------------------------------------------------ metrics
    def pope_block(self, rows):
        groups = [r['image'] for r in rows]
        lab = np.array([r['label'] for r in rows]); pred = np.array([r['pred'] for r in rows])
        out = dict(n=len(rows), images=len(set(groups)))
        for key in ('accuracy', 'precision', 'recall', 'f1', 'yes_ratio', 'false_yes_rate'):
            out[key] = self.boot(groups, lambda idx, k=key: pope_metrics(lab[idx], pred[idx])[k])
        return out

    def baseline(self):
        base = self.rows.get('baseline', {})
        result = {}
        for scope in ('all', 'test'):
            for split in POPE_SPLITS + ('pooled',):
                rows = [r for r in base.values() if (scope == 'all' or r['partition'] == 'test')
                        and (split == 'pooled' or r['pope_split'] == split)]
                if rows:
                    result[f'{scope}:{split}'] = self.pope_block(rows)
        return result

    def detection(self):
        rows = [r for r in read_jsonl(self.dir / 'detection_scores.jsonl') if r['partition'] == 'test']
        if not rows:
            return None
        y = np.array([r['y'] for r in rows]); groups = [r['image'] for r in rows]
        result = dict(n=len(rows), positives=int(y.sum()), base_rate=float(y.mean()), detectors={}, per_split={})
        for key, _ in DETECTORS:
            s = np.array([r['scores'][key] for r in rows])
            result['detectors'][key] = dict(auroc=self.boot(groups, lambda i: auroc(y[i], s[i])),
                                            auprc=self.boot(groups, lambda i: auprc(y[i], s[i])))
        for split in POPE_SPLITS:
            idx = [j for j, r in enumerate(rows) if r['pope_split'] == split]
            if idx:
                ys = y[idx]; gs = [groups[j] for j in idx]
                arrays = {k: np.array([rows[j]['scores'][k] for j in idx]) for k in ('probe', 'neg_logprob')}
                result['per_split'][split] = {k: self.boot(gs, lambda i, a=a: auroc(ys[i], a[i])) for k, a in arrays.items()}
        # Paired detector difference on exactly the same image resamples.
        probe = np.array([r['scores']['probe'] for r in rows])
        confidence = np.array([r['scores']['neg_logprob'] for r in rows])
        def difference(idx):
            a, b = auroc(y[idx], probe[idx]), auroc(y[idx], confidence[idx])
            return None if a is None or b is None else a - b
        result['comparisons'] = {'probe_minus_neg_logprob': self.boot(groups, difference)}
        return result

    def steering(self):
        base = self.rows.get('baseline', {})
        test_uids = sorted(u for u, r in base.items() if r['partition'] == 'test')
        result = {}
        for method, _ in METHODS:
            got = self.rows.get(method, {})
            uids = [u for u in test_uids if u in got]
            if not uids:
                continue
            if len(uids) != len(test_uids):
                result[method] = dict(incomplete=f'{len(uids)}/{len(test_uids)} test questions answered')
                continue
            b = [base[u] for u in uids]; m = [got[u] for u in uids]; groups = [r['image'] for r in b]
            lab = np.array([r['label'] for r in b])
            bp = np.array([r['pred'] for r in b]); mp = np.array([r['pred'] for r in m])
            bc = bp == lab; mc = mp == lab

            def metric(pred, idx, key):
                return pope_metrics(lab[idx], pred[idx])[key]
            entry = dict(n=len(uids), strength=m[0]['strength'], intervention=m[0].get('intervention'))
            for key in ('accuracy', 'f1', 'yes_ratio', 'false_yes_rate', 'recall'):
                entry[key] = self.boot(groups, lambda i, k=key: metric(mp, i, k))
            entry['delta_accuracy'] = self.boot(groups, lambda i: mc[i].mean() - bc[i].mean())
            neg = lab == 'no'
            entry['delta_false_yes_rate'] = self.boot(groups, lambda i: (mp[i][neg[i]] == 'yes').mean() - (bp[i][neg[i]] == 'yes').mean() if neg[i].any() else None)
            pos = lab == 'yes'
            entry['delta_recall'] = self.boot(groups, lambda i: (mp[i][pos[i]] == 'yes').mean() - (bp[i][pos[i]] == 'yes').mean() if pos[i].any() else None)
            # Utility: of questions the baseline already got right, how many stay right.
            entry['retention'] = self.boot(groups, lambda i: mc[i][bc[i]].mean() if bc[i].any() else None)
            entry['per_split_delta_accuracy'] = {}
            entry['per_split_metrics'] = {}
            for split in POPE_SPLITS:
                sel = np.array([r['pope_split'] == split for r in b])
                if sel.any():
                    gs = [g for g, s in zip(groups, sel) if s]; mcs, bcs = mc[sel], bc[sel]
                    entry['per_split_delta_accuracy'][split] = self.boot(gs, lambda i: mcs[i].mean() - bcs[i].mean())
                    sl, sm = lab[sel], mp[sel]
                    entry['per_split_metrics'][split] = {key: self.boot(gs, lambda i, k=key: pope_metrics(sl[i], sm[i])[k]) for key in ('false_yes_rate', 'recall')}
            result[method] = entry
        if 'slerp_pca' in result and 'incomplete' not in result['slerp_pca']:
            b = [base[u] for u in test_uids]; groups = [r['image'] for r in b]
            lab = np.array([r['label'] for r in b]); neg = lab == 'no'
            sp = np.array([self.rows['slerp_pca'][u]['pred'] for u in test_uids])
            comparisons = {}
            for control in ('random_direction', 'shuffled_labels'):
                if control in result and 'incomplete' not in result[control]:
                    cp = np.array([self.rows[control][u]['pred'] for u in test_uids])
                    comparisons[control] = self.boot(groups, lambda i, cp=cp: (sp[i][neg[i]] == 'yes').mean() - (cp[i][neg[i]] == 'yes').mean() if neg[i].any() else None)
            result['slerp_pca']['versus_controls_false_yes'] = comparisons
        return result


# ---------------------------------------------------------------- figures
def figures(run, fig_dir, selection, sweep):
    # Reports are files, never inline notebook displays; set before importing.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#e6e5e0', 'grid.linewidth': .8, 'axes.edgecolor': '#52514e',
                         'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                         'font.size': 10, 'lines.linewidth': 2})
    if selection:
        layers = [selection['state_layers'][r['layer']] for r in selection['sweep']]
        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
        for (key, label), color in zip(DETECTORS[:3], COLORS):
            ax.plot(layers, [r[f'calibration_cv_auroc_{key}'] for r in selection['sweep']], color=color,
                    marker='o', markersize=3, label=f'{label}: calibration CV')
        lt = run.dir / 'layer_scores_test.npz'
        if lt.exists():
            d = np.load(lt, allow_pickle=False)
            ax.plot(list(d['layers']), [auroc(d['y'], d['scores'][:, j]) or np.nan for j in range(d['scores'].shape[1])],
                    color=COLORS[0], linestyle='--', label='Linear probe: held-out test (descriptive only)')
        chosen = selection['model_layers']['probe']
        ax.axvline(chosen, color='#52514e', linewidth=1, linestyle=':')
        ax.annotate(f'selected on calibration: layer {chosen}', (chosen, 1.0), xytext=(-4, -10), ha='right',
                    textcoords='offset points', fontsize=8, color='#52514e')
        ax.axhline(.5, color='#b5b4ad', linewidth=1)
        low = min([.45] + [v for line in ax.get_lines() for v in np.asarray(line.get_ydata(), float) if np.isfinite(v)])
        ax.set(xlabel='Decoder layer (zero-based raw block output)', ylabel='AUROC (hallucination)', ylim=(max(0., low - .05), 1.0))
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
        fig.tight_layout(); fig.savefig(fig_dir / 'layer_sweep.png'); plt.close(fig)
        made.append('layer_sweep.png')
    if sweep:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), dpi=150)
        for (name, label), color in zip(METHODS[:2], COLORS):
            curve = sweep['methods'].get(name, {}).get('curve', [])
            x = [0] + [c['strength'] for c in curve]
            axes[0].plot(x, [sweep['baseline']['accuracy']] + [c['accuracy'] for c in curve], color=color, marker='o', markersize=4, label=label)
            axes[1].plot([nan(sweep['baseline']['false_yes_rate'])] + [nan(c['false_yes_rate']) for c in curve],
                         [nan(sweep['baseline']['recall'])] + [nan(c['recall']) for c in curve], color=color, marker='o', markersize=4, label=label)
        axes[0].set(xlabel='Strength (SLERP α / addition β)', ylabel='Validation accuracy')
        axes[1].set(xlabel='False-yes rate on absent objects (↓)', ylabel='Recall on present objects (↑)')
        axes[0].scatter([0], [sweep['baseline']['accuracy']], color='#0b0b0b', zorder=3, s=20, label='Unsteered baseline')
        axes[1].scatter([nan(sweep['baseline']['false_yes_rate'])], [nan(sweep['baseline']['recall'])], color='#0b0b0b', zorder=3, s=20)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / 'steering_tradeoff_validation.png'); plt.close(fig)
        made.append('steering_tradeoff_validation.png')
    return made


# ----------------------------------------------------------------- markdown
def placeholder_sections():
    base = table(['POPE split', 'Questions', 'Accuracy', 'Precision', 'Recall', 'F1', 'Yes-ratio'],
                 [[s, DASH, DASH, DASH, DASH, DASH, DASH] for s in POPE_SPLITS])
    det = table(['Detector', 'Layer', 'Test AUROC', 'Test AUPRC'], [[n, DASH, DASH, DASH] for _, n in DETECTORS])
    steer = table(['Method', 'Strength', 'Accuracy', 'Δ accuracy', 'False-yes rate', 'Δ false-yes', 'Recall (utility)', 'Retention'],
                  [[n, DASH, DASH, DASH, DASH, DASH, DASH, DASH] for _, n in METHODS])
    return base, det, steer


def sections(run, base, det, steer, selection):
    def bl(scope):
        return table(['POPE split', 'Questions (images)', 'Accuracy', 'Precision', 'Recall', 'F1', 'Yes-ratio'],
                     [[s, f"{b['n']} ({b['images']})", fmt(b['accuracy']), fmt(b['precision']), fmt(b['recall']),
                       fmt(b['f1']), fmt(b['yes_ratio'])] for s in POPE_SPLITS + ('pooled',)
                      if (b := base.get(f'{scope}:{s}'))])
    layer = (selection or {}).get('model_layers', {})
    if det:
        dt = table(['Detector', 'Layer', 'Test AUROC', 'Test AUPRC'],
                   [[n, layer.get(k, 'n/a (no fit)'), fmt(det['detectors'][k]['auroc']), fmt(det['detectors'][k]['auprc'])]
                    for k, n in DETECTORS])
        dt += f"\n\nTest questions: {det['n']}; endpoint-positive: {det['positives']} (base rate {det['base_rate']:.3f}, the AUPRC of a random detector)."
    else:
        dt = placeholder_sections()[1]
    if steer:
        srows = []
        for k, n in METHODS:
            e = steer.get(k)
            if not e:
                srows.append([n] + [DASH] * 7)
            elif 'incomplete' in e:
                srows.append([n, e['incomplete']] + [DASH] * 6)
            else:
                srows.append([n, f"{e['strength']:g}", fmt(e['accuracy']), fmt(e['delta_accuracy']), fmt(e['false_yes_rate']),
                              fmt(e['delta_false_yes_rate']), fmt(e['recall']), fmt(e['retention'])])
        st = table(['Method', 'Strength', 'Accuracy', 'Δ accuracy', 'False-yes rate', 'Δ false-yes', 'Recall (utility)', 'Retention'], srows)
    else:
        st = placeholder_sections()[2]
    return bl('test') if base else placeholder_sections()[0], dt, st, bl('all') if base else None


def build_report(run_dir, out, fig_dir, readme=None, placeholder=False):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if placeholder or run_dir is None:
        b, d, s = placeholder_sections()
        body = ['# CounterPrior results', '',
                '**Status: no GPU run yet.** Every cell below is a placeholder. This file is regenerated by '
                '`python -m scripts.make_report --run RUN` from the saved run files; no number here is typed by hand.',
                '', '## 1. POPE baseline (LLaVA-1.5-7B, held-out test images)', '', b,
                '', '## 2. Hallucination detection (test images)', '', d,
                '', '## 3. Steering on test images (strength chosen on validation)', '', s]
        out.write_text('\n'.join(body) + '\n', encoding='utf8')
        summary = '\n\n'.join(['_No GPU run yet: placeholders. Filled automatically by `scripts/make_report.py`._',
                               '**POPE baseline (test images)**', b, '**Detection (test AUROC/AUPRC)**', d,
                               '**Steering (test)**', s])
        if readme:
            update_readme(readme, summary, out)
        return dict(status='placeholder')
    run = Run(run_dir)
    selection, sweep = run.load('selection.json'), run.load('validation_sweep.json')
    base, det, steer = run.baseline(), run.detection(), run.steering()
    made = figures(run, fig_dir, selection, sweep)
    b, d, s, ball = sections(run, base, det, steer, selection)
    rel = os.path.relpath(Path(fig_dir), out.parent)
    used = run.splits.get('used_counts', {})
    cfg = run.config
    body = ['# CounterPrior results', '',
            f"Run `{run.identity['run_id']}` · adapter `{cfg['adapter']}` · `{cfg['model_id']}` @ `{cfg.get('revision')}` · "
            f"quantization `{cfg['quantization']}` · profile `{cfg.get('profile', 'full')}` · seed {cfg['seed']}.",
            f"Images per partition (calibration / validation / test): {used.get('calibration', DASH)} / "
            f"{used.get('validation', DASH)} / {used.get('test', DASH)}. "
            f"CIs: 95% percentile bootstrap over images ({run.iterations} resamples), conditional on frozen calibration/validation choices.",
            '', 'Generated by `scripts/make_report.py` from saved run files only.', '',
            '## 1. POPE baseline on held-out test images', '', b, '',
            '<details><summary>Baseline on every selected image (no tuning involved, so comparable to published POPE numbers)</summary>', '',
            ball or DASH, '', '</details>', '',
            '## 2. Hallucination detection on test images', '',
            f"Target: `{(selection or {}).get('target', cfg['detection']['target'])}` (1 = baseline answer was wrong). "
            'Layers for the three hidden-state detectors were chosen by image-grouped K-fold AUROC on calibration images only; '
            'the logit baselines need no fitting.', '', d, '']
    if det and det['per_split']:
        body += ['Per POPE split (test AUROC):', '',
                 table(['POPE split', 'Linear probe', 'Answer-token −log p'],
                       [[sp, fmt(v['probe']), fmt(v['neg_logprob'])] for sp, v in det['per_split'].items()]), '']
    if 'layer_sweep.png' in made:
        body += [f'![Layer sweep]({rel}/layer_sweep.png)', '',
                 'Solid: calibration cross-validated AUROC (used for selection). Dashed: per-layer test AUROC, shown for description only.', '']
    body += ['## 3. Steering on test images', '',
             'Strength per method was the best validation accuracy on the grid; controls reuse the SLERP strength. '
             'Δ = method − baseline on the same questions, paired bootstrap. False-yes rate = "yes" about an absent object '
             '(object hallucination; lower is better). Recall on present objects and retention of baseline-correct answers are the utility check.', '', s, '']
    if steer:
        rows = [[n] + [fmt(steer[k]['per_split_delta_accuracy'].get(sp)) for sp in POPE_SPLITS]
                for k, n in METHODS if k in steer and 'per_split_delta_accuracy' in steer[k]]
        if rows:
            body += ['Δ accuracy by POPE split:', '', table(['Method'] + list(POPE_SPLITS), rows), '']
    if sweep:
        body += ['Validation sweep (selection data):', '',
                 table(['Method', 'Strength', 'Val. accuracy', 'Val. false-yes rate', 'Val. recall'],
                       [['Baseline', '0', f3(sweep['baseline']['accuracy']), f3(sweep['baseline']['false_yes_rate']), f3(sweep['baseline']['recall'])]] +
                       [[n, f"{c['strength']:g}" + (' (selected)' if c['strength'] == sweep['methods'][k]['selected_strength'] else ''),
                         f3(c['accuracy']), f3(c['false_yes_rate']), f3(c['recall'])]
                        for k, n in METHODS[:2] if k in sweep['methods'] for c in sweep['methods'][k]['curve']]), '']
        for k, n in METHODS[:2]:
            if k in sweep['methods'] and not sweep['methods'][k]['improves_on_validation_baseline']:
                body += [f'- {n}: no strength on the grid beat the unsteered baseline on validation; the test row reports the best grid point anyway.']
        body += ['']
    if 'steering_tradeoff_validation.png' in made:
        body += [f'![Validation trade-off]({rel}/steering_tradeoff_validation.png)', '']
    body += ['## Provenance', '', f"- Status: `{json.dumps(run.load('status.json'))}`",
             f"- Timing probe: `{json.dumps((run.load('timing.json') or {}).get('seconds_per_question'))}` s/question",
             f"- POPE question-file sha256: `{json.dumps(run.identity.get('pope_sha256'))}`",
             '- Raw files: predictions.jsonl, detection_scores.jsonl, layer_scores_test.npz, selection.json, steering_plan.json, validation_sweep.json']
    out.write_text('\n'.join(body) + '\n', encoding='utf8')
    metrics = dict(run_id=run.identity['run_id'], baseline=base, detection=det, steering=steer,
                   provenance=dict(config=cfg, used_counts=used, selected_layers=(selection or {}).get('model_layers', {}),
                                   calibration_curve=[dict(layer=(selection or {}).get('state_layers', [])[r['layer']], probe=r.get('calibration_cv_auroc_probe'), meanDifference=r.get('calibration_cv_auroc_mean_difference'), pca=r.get('calibration_cv_auroc_pca')) for r in (selection or {}).get('sweep', [])]))
    def judgement(m, direction):
        ci = (m or {}).get('ci95')
        if not ci: return 'inconclusive'
        if direction == 'positive': return 'supported' if ci[0] > 0 else 'not supported' if ci[1] < 0 else 'inconclusive'
        return 'supported' if ci[1] < 0 else 'not supported' if ci[0] > 0 else 'inconclusive'
    comparison = (det or {}).get('comparisons', {}).get('probe_minus_neg_logprob')
    pca = (det or {}).get('detectors', {}).get('pca', {}).get('auroc')
    pcaci = (pca or {}).get('ci95')
    geometry_status = 'inconclusive' if not pcaci else 'supported' if pcaci[0] > .5 else 'not supported' if pcaci[1] < .5 else 'inconclusive'
    sl = steer.get('slerp_pca', {})
    controls = sl.get('versus_controls_false_yes', {})
    mechanism_supported = len(controls) == 2 and all(judgement(v, 'negative') == 'supported' for v in controls.values())
    reduction_status = judgement(sl.get('delta_false_yes_rate'), 'negative')
    recall_delta = (sl.get('delta_recall') or {}).get('estimate')
    tradeoff = 'A measured reduction comes with lower recall.' if reduction_status == 'supported' and recall_delta is not None and recall_delta < 0 else 'Interpret the false-yes and recall intervals together.'
    metrics['hypotheses'] = [
        dict(id='detection', status=judgement(comparison, 'positive'), finding=f"For the configured {cfg['detection']['target']} target, the paired probe-minus-confidence AUROC difference is {fmt(comparison)}.", source='results/REPORT_metrics.json'),
        dict(id='geometry', status=geometry_status, finding=f"Held-out PCA-projection AUROC is {fmt(pca)}; this does not establish a universal truth direction.", source='results/REPORT_metrics.json'),
        dict(id='steering', status=reduction_status if mechanism_supported else 'inconclusive', reduction_status=reduction_status, finding=(f"SLERP false-yes delta is {fmt(sl.get('delta_false_yes_rate'))}; recall delta is {fmt(sl.get('delta_recall'))} (recall {fmt(sl.get('recall'))}). {tradeoff} Geometric superiority to both matched controls is {'supported' if mechanism_supported else 'inconclusive'}. No unqualified safety improvement is established." if sl else 'Steering was not executed for this run.'), source='results/REPORT_metrics.json')]
    metrics['summary'] = [
        f"The layer {(selection or {}).get('model_layers', {}).get('probe', DASH)} {cfg['detection']['target']}-detection probe reached AUROC {fmt((det or {}).get('detectors', {}).get('probe', {}).get('auroc'))}; its paired improvement over answer-token confidence is {judgement(comparison, 'positive')}.",
        (f"SLERP changed false-yes rate by {fmt(sl.get('delta_false_yes_rate'))} and recall by {fmt(sl.get('delta_recall'))}; superiority to both matched controls is {'supported' if mechanism_supported else 'inconclusive'}." if sl else 'Steering was not executed for this model.')]
    body += ['', '## Registered hypothesis interpretation', '',
             table(['Question', 'Status', 'Finding'], [[h['id'], h['status'], h['finding']] for h in metrics['hypotheses']]), '',
             'Paired false-yes differences (SLERP minus control; negative favours SLERP):', '',
             table(['Control', 'Paired difference [95% CI]'], [[name, fmt(m)] for name, m in controls.items()]), '']
    out.write_text('\n'.join(body) + '\n', encoding='utf8')

    (out.parent / (out.stem + '_metrics.json')).write_text(json.dumps(metrics, indent=2, default=lambda o: None) + '\n')
    if readme:
        report_ref = Path(os.path.relpath(out.resolve(), Path(readme).resolve().parent)).as_posix()
        summary = '\n\n'.join([f"_Auto-filled from run `{run.identity['run_id']}` by `scripts/make_report.py`; full report: [{report_ref}]({report_ref})._",
                               ' '.join(metrics['summary']),
                               '**POPE baseline (test images)**', b, '**Detection (test AUROC/AUPRC)**', d, '**Steering (test)**', s])
        update_readme(readme, summary, out)
    return metrics


def update_readme(readme, summary, out):
    text = Path(readme).read_text(encoding='utf8')
    if README_START not in text or README_END not in text:
        raise ValueError(f'{readme} lacks {README_START}/{README_END} markers')
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    Path(readme).write_text(head + README_START + '\n' + summary + '\n' + README_END + tail, encoding='utf8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run'); p.add_argument('--out', default='results/REPORT.md')
    p.add_argument('--figures', default='results/figures'); p.add_argument('--readme')
    p.add_argument('--placeholder', action='store_true')
    a = p.parse_args(argv)
    if not a.placeholder and not a.run:
        p.error('--run RUN or --placeholder')
    build_report(a.run, a.out, a.figures, a.readme, a.placeholder)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
