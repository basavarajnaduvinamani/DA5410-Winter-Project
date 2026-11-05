import unittest

import numpy as np

from src.detection import fit_detectors, fit_direction, gate_threshold, layer_sweep, score_logits, score_states


def synthetic(n_images=30, per_image=4, layers=3, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_images), per_image)
    y = rng.integers(0, 2, size=len(groups))
    X = rng.normal(size=(len(groups), layers, dim))
    X[:, 1, 0] += np.where(y == 0, 3.0, -3.0)  # only layer 1 carries the signal
    return X, y, groups


class Detection(unittest.TestCase):
    def test_sweep_selects_informative_layer(self):
        X, y, g = synthetic()
        selection, fitted, per_layer = fit_detectors(X, y, g, 0, folds=5)
        self.assertEqual(selection['layers'], dict(probe=1, mean_difference=1, pca=1))
        self.assertEqual(selection['n_calibration'], len(y))
        self.assertEqual(len(per_layer), 3)
        self.assertGreater(selection['sweep'][1]['calibration_cv_auroc_probe'], .95)
        self.assertLess(abs(selection['sweep'][0]['calibration_cv_auroc_probe'] - .5), .2)
        Xt, yt, _ = synthetic(seed=1)
        s = score_states(Xt, selection, fitted)
        for key in ('probe', 'mean_difference', 'pca'):
            self.assertGreater(np.mean(s[key][yt == 1]), np.mean(s[key][yt == 0]), key)

    def test_requires_two_classes_and_groups(self):
        X, y, g = synthetic()
        with self.assertRaisesRegex(ValueError, 'both'):
            layer_sweep(X, np.zeros_like(y), g, 0)
        with self.assertRaisesRegex(ValueError, 'two image groups'):
            layer_sweep(X, y, np.zeros_like(g), 0)

    def test_directions_point_toward_correct_and_controls_differ(self):
        X, y, _ = synthetic()
        Xl = X[:, 1]
        delta = Xl[y == 0].mean(0) - Xl[y == 1].mean(0)
        for kind in ('pca', 'mean_difference'):
            v = fit_direction(Xl, y, kind, 0)
            self.assertAlmostEqual(np.linalg.norm(v), 1, places=6)
            self.assertGreater(v @ delta, 0)
        # Controls in a wide space where the signal is one weak dimension among many.
        Xw, yw, _ = synthetic(n_images=60, dim=64)
        Xw = Xw[:, 1] / np.r_[3.0, np.ones(63)]
        md = fit_direction(Xw, yw, 'mean_difference', 0)
        random, shuffled = fit_direction(Xw, yw, 'random_direction', 0), fit_direction(Xw, yw, 'shuffled_labels', 0)
        self.assertLess(abs(random @ md), .5); self.assertLess(abs(shuffled @ md), .5)
        md = fit_direction(Xl, y, 'mean_difference', 0)
        # PC1 is label-free up to sign, which is why the shuffled control uses a mean difference.
        self.assertGreater(abs(fit_direction(Xl, y, 'pca', 0) @ md), .9)
        t = gate_threshold(Xl, y, md)
        self.assertTrue(-1 <= t <= 1)

    def test_logit_baselines_orientation(self):
        s = score_logits([-0.01, -2.0], [0.1, 1.5])
        self.assertGreater(s['neg_logprob'][1], s['neg_logprob'][0])
        self.assertGreater(s['entropy'][1], s['entropy'][0])


if __name__ == '__main__':
    unittest.main()
