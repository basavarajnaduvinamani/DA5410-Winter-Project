"""Image-clustered bootstrap over aligned arrays and detection metrics."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y, score):
    y = np.asarray(y)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def auprc(y, score):
    y = np.asarray(y)
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None


def cluster_bootstrap_idx(groups, statistic, seed, iterations=1000):
    """Percentile CI resampling whole image clusters with replacement.

    `statistic(idx)` receives row indices into arrays aligned with `groups`;
    paired comparisons index baseline and method arrays with the same idx, so
    both arms see the identical resample. Nothing is refitted.
    """
    groups = np.asarray(groups)
    keys, inverse = np.unique(groups, return_inverse=True)
    if len(keys) == 0:
        return dict(estimate=None, ci95=None, valid_bootstrap=0, image_clusters=0, n=0)
    members = [np.flatnonzero(inverse == k) for k in range(len(keys))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([members[k] for k in pick])
        value = statistic(idx)
        if value is not None and np.isfinite(value):
            values.append(value)
    estimate = statistic(np.arange(len(groups)))
    return dict(estimate=None if estimate is None else float(estimate),
                ci95=[float(v) for v in np.quantile(values, [.025, .975])] if values else None,
                valid_bootstrap=len(values), image_clusters=int(len(keys)), n=int(len(groups)))
