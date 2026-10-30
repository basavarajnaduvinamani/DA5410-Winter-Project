"""Deterministic image grouping, calibration-only selection and held-out CIs."""
import hashlib
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def digest(path):
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_splits(cases, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for c in cases:
        groups.setdefault(c['image_group'], []).append(c)
    strata = {}
    for g, rows in sorted(groups.items()):
        strata.setdefault(tuple(sorted({r['category'] for r in rows})), []).append(g)
    assignment = {}
    for _, group_ids in sorted(strata.items()):
        rng.shuffle(group_ids)
        n = len(group_ids)
        # Sparse strata go to calibration; never pretend absent test strata exist.
        nc = max(1, n//2)
        nv = max(1, (n-nc)//2) if n-nc > 1 else 0
        for i, g in enumerate(group_ids):
            assignment[g] = 'calibration' if i < nc else 'validation' if i < nc+nv else 'test'
    return {c['id']: assignment[c['image_group']] for c in cases}


def direction(X, y, kind='pca', seed=0):
    if len(np.unique(y)) != 2:
        raise ValueError('Calibration needs both hallucinated and non-hallucinated answers; acquire more audited images')
    delta = X[y == 0].mean(0) - X[y == 1].mean(0)
    variance = None
    if kind == 'pca':
        pca = PCA(n_components=1, svd_solver='full').fit(X)
        vec = pca.components_[0]
        variance = float(pca.explained_variance_ratio_[0])
        if vec @ delta < 0:
            vec = -vec
    elif kind == 'mean_difference':
        vec = delta
    elif kind == 'random_direction':
        vec = np.random.default_rng(seed).normal(size=X.shape[-1])
    else:
        raise ValueError(kind)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError('Degenerate calibration direction')
    return vec/norm, variance


def cosine(X, vec):
    return X @ vec / np.maximum(np.linalg.norm(X, axis=-1), 1e-12)


def select_calibration(states, y, groups, seed):
    """Choose layer/gate exclusively by image-group leave-one-out calibration.

    No final-test data or validation labels enter this function.
    """
    y = np.asarray(y)
    logo = LeaveOneGroupOut()
    for train, _ in logo.split(states, y, groups):
        if len(np.unique(y[train])) != 2:
            raise ValueError('Calibration fold has only one class; more audited calibration images required')
    candidates = []
    for layer in range(states.shape[1]):
        X = states[:, layer, :]
        oof = np.zeros(len(y)); alignment = np.zeros(len(y))
        for train, held in logo.split(X, y, groups):
            if len(np.unique(y[train])) != 2:
                raise ValueError('Calibration fold has only one class; more audited calibration images required')
            clf = LogisticRegression(C=1., max_iter=2000, random_state=seed).fit(X[train], y[train])
            oof[held] = clf.predict_proba(X[held])[:, 1]
            vec, _ = direction(X[train], y[train])
            alignment[held] = cosine(X[held], vec)
        thresholds = np.unique(np.r_[-1., alignment, 1.])
        threshold = max(thresholds, key=lambda t: balanced_accuracy_score(y, alignment < t))
        candidates.append(dict(layer=layer, threshold=float(threshold),
                               calibration_oof_auroc=float(roc_auc_score(y, oof)),
                               calibration_oof_gate_balanced_accuracy=float(balanced_accuracy_score(y, alignment < threshold))))
    best = max(candidates, key=lambda r: (r['calibration_oof_auroc'], r['calibration_oof_gate_balanced_accuracy'], -r['layer']))
    layer = best['layer']
    vec, variance = direction(states[:, layer], y)
    probe = LogisticRegression(C=1., max_iter=2000, random_state=seed).fit(states[:, layer], y)
    return dict(best, pc1_variance=variance, selection='calibration_only', candidates=candidates), vec, probe


def probe_metrics(y, probability):
    y = np.asarray(y); pred = np.asarray(probability) >= .5
    return dict(auroc=float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
                precision=float(precision_score(y,pred,zero_division=0)),
                recall=float(recall_score(y,pred,zero_division=0)), f1=float(f1_score(y,pred,zero_division=0)))


def cluster_bootstrap(rows, metric, seed, iterations=1000):
    """Resample image clusters jointly across methods; never refit on test data."""
    clusters = {}
    for row in rows:
        clusters.setdefault(row['image_group'], []).append(row)
    if not clusters:
        return dict(estimate=None, ci95=None, valid_bootstrap=0)
    keys = sorted(clusters); rng = np.random.default_rng(seed); values = []
    for _ in range(iterations):
        sample = [r for g in rng.choice(keys, size=len(keys), replace=True) for r in clusters[g]]
        value = metric(sample)
        if value is not None and np.isfinite(value):
            values.append(value)
    return dict(estimate=metric(rows), ci95=np.quantile(values,[.025,.975]).tolist() if values else None,
                valid_bootstrap=len(values), image_clusters=len(keys))
