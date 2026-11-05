"""Calibration-only layer sweep, hallucination detectors and steering directions.

Label convention: y = 1 means the baseline answer was a hallucination/error,
y = 0 means it was correct. Every fit in this module sees calibration rows
only; callers pass validation/test states solely to `score_*` functions.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .protocol import direction as _direction, cosine


def _probe(C, seed):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))


def _balanced(y, seed):
    """Indices with equal numbers of each class, so PC1 is not dominated by the majority class."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = min(len(pos), len(neg))
    return np.sort(np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])


def fit_direction(X, y, kind, seed):
    """Unit direction pointing from hallucinated toward correct states (random/shuffled are controls)."""
    y = np.asarray(y)
    if kind == 'pca':
        idx = _balanced(y, seed)
        return _direction(X[idx], y[idx], 'pca', seed)[0]
    if kind == 'mean_difference':
        return _direction(X, y, 'mean_difference', seed)[0]
    if kind == 'random_direction':
        return _direction(X, y, 'random_direction', seed)[0]
    if kind == 'shuffled_labels':
        # Mean difference, not PCA: PC1 ignores labels except for its sign, so a
        # shuffled-label PCA axis would nearly equal the real one and test nothing.
        perm = np.random.default_rng(seed).permutation(y)
        return _direction(X, perm, 'mean_difference', seed)[0]
    raise ValueError(kind)


def gate_threshold(X, y, vec):
    """Cosine threshold maximising calibration balanced accuracy of (cos < t) as an error flag."""
    alignment = cosine(X, vec)
    thresholds = np.unique(np.r_[-1., np.quantile(alignment, np.linspace(0, 1, 101)), 1.])
    return float(max(thresholds, key=lambda t: (balanced_accuracy_score(y, alignment < t), -abs(t))))


def _folds(y, groups, k):
    n_groups = len(np.unique(groups))
    k = min(k, n_groups)
    if k < 2:
        raise ValueError('Calibration needs at least two image groups')
    folds = list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups))
    for train, held in folds:
        if len(np.unique(y[train])) != 2:
            raise ValueError('Calibration fold has only one class; use more calibration images')
    return folds


def layer_sweep(states, y, groups, seed, folds=5, C=1.0):
    """Image-grouped K-fold out-of-fold AUROC per layer for probe, mean-difference and PCA projections."""
    y = np.asarray(y); groups = np.asarray(groups)
    if len(np.unique(y)) != 2:
        raise ValueError('Calibration needs both hallucinated and correct baseline answers')
    splits = _folds(y, groups, folds)
    rows = []
    for layer in range(states.shape[1]):
        X = states[:, layer, :].astype(np.float32)
        oof = {k: np.zeros(len(y)) for k in ('probe', 'mean_difference', 'pca')}
        for train, held in splits:
            oof['probe'][held] = _probe(C, seed).fit(X[train], y[train]).predict_proba(X[held])[:, 1]
            for kind in ('mean_difference', 'pca'):
                # Low alignment with the "correct" direction => higher hallucination score.
                oof[kind][held] = -cosine(X[held], fit_direction(X[train], y[train], kind, seed))
        rows.append(dict(layer=layer, **{f'calibration_cv_auroc_{k}': float(roc_auc_score(y, v)) for k, v in oof.items()}))
    return rows


def select_layer(sweep, detector='probe'):
    key = f'calibration_cv_auroc_{detector}'
    return max(sweep, key=lambda r: (r[key], -r['layer']))['layer']


def fit_detectors(states, y, groups, seed, folds=5, C=1.0):
    """Sweep layers on calibration, pick each detector's layer there, then refit on all calibration rows."""
    y = np.asarray(y)
    sweep = layer_sweep(states, y, groups, seed, folds, C)
    chosen = {d: select_layer(sweep, d) for d in ('probe', 'mean_difference', 'pca')}
    fitted = dict(probe=_probe(C, seed).fit(states[:, chosen['probe']].astype(np.float32), y))
    for kind in ('mean_difference', 'pca'):
        fitted[kind] = fit_direction(states[:, chosen[kind]].astype(np.float32), y, kind, seed)
    # Per-layer probes are kept only to draw the descriptive held-out layer curve.
    per_layer = [_probe(C, seed).fit(states[:, l].astype(np.float32), y) for l in range(states.shape[1])]
    selection = dict(selection='calibration_only_group_kfold', folds=min(folds, len(np.unique(groups))),
                     layers=chosen, n_calibration=int(len(y)), n_calibration_hallucinated=int(y.sum()),
                     calibration_images=int(len(np.unique(groups))), sweep=sweep)
    return selection, fitted, per_layer


def score_states(states, selection, fitted):
    layers = selection['layers']
    return dict(
        probe=fitted['probe'].predict_proba(states[:, layers['probe']].astype(np.float32))[:, 1],
        mean_difference=-cosine(states[:, layers['mean_difference']].astype(np.float32), fitted['mean_difference']),
        pca=-cosine(states[:, layers['pca']].astype(np.float32), fitted['pca']))


def score_logits(logprob_first, entropy_first):
    """Training-free baselines: low answer-token log-probability / high entropy => hallucination."""
    return dict(neg_logprob=-np.asarray(logprob_first, float), entropy=np.asarray(entropy_first, float))
