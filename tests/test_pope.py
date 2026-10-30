import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.evaluation import cluster_bootstrap_idx
from src.pope import POPE_SPLITS, image_disjoint_split, limit_split, load_pope, parse_yes_no, pope_metrics


class PopeParsingAndMetrics(unittest.TestCase):
    def test_official_parser(self):
        # Mirrors AoiDragon/POPE evaluate.py: first sentence, 'No'/'no'/'not' anywhere => no.
        cases = {'Yes': 'yes', 'No': 'no', 'No.': 'no', 'Yes, there is a dog.': 'yes',
                 'There is not a dog.': 'no', 'Yes. There is no cat.': 'yes', 'no, none': 'no',
                 'Nope': 'yes', 'NO': 'yes', '': 'yes', 'I do not see one': 'no'}
        for text, want in cases.items():
            self.assertEqual(parse_yes_no(text), want, text)

    def test_metrics_hand_computed(self):
        labels = ['yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'yes']
        preds = ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes']
        m = pope_metrics(labels, preds)
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['tn']), (3, 1, 1, 3))
        self.assertAlmostEqual(m['accuracy'], 6 / 8)
        self.assertAlmostEqual(m['precision'], 3 / 4)
        self.assertAlmostEqual(m['recall'], 3 / 4)
        self.assertAlmostEqual(m['f1'], 3 / 4)
        self.assertAlmostEqual(m['yes_ratio'], 4 / 8)
        self.assertAlmostEqual(m['false_yes_rate'], 1 / 4)
        self.assertEqual(pope_metrics(['no'], ['no'])['precision'], 0.0)

    def test_vendored_files_load_and_match_pins(self):
        import hashlib
        sources = json.loads(Path('dataset/pope/sources.json').read_text())
        for name, meta in sources['files'].items():
            self.assertEqual(hashlib.sha256(Path('dataset/pope', name).read_bytes()).hexdigest(), meta['sha256'])
        items = load_pope('dataset/pope')
        self.assertEqual(len(items), 9000)
        for split in POPE_SPLITS:
            rows = [i for i in items if i['pope_split'] == split]
            self.assertEqual((len(rows), len({i['image'] for i in rows}), sum(i['label'] == 'yes' for i in rows)), (3000, 500, 1500))

    def test_committed_split_manifest_is_reproducible_and_disjoint(self):
        manifest = json.loads(Path('dataset/pope/splits_seed20261030.json').read_text())
        items = load_pope('dataset/pope')
        split = image_disjoint_split([i['image'] for i in items], manifest['seed'], manifest['fractions'])
        for part in ('calibration', 'validation', 'test'):
            self.assertEqual(split[part], manifest[part])
        sets = [set(manifest[p]) for p in ('calibration', 'validation', 'test')]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(len(set.union(*sets)), 500)
        self.assertEqual(manifest['counts'], dict(calibration=100, validation=50, test=350))

    def test_split_depends_only_on_image_ids(self):
        images = [f'im{i}' for i in range(40)]
        fr = dict(calibration=.25, validation=.25, test=.5)
        a = image_disjoint_split(images, 1, fr)
        self.assertEqual(a, image_disjoint_split(list(reversed(images)) * 2, 1, fr))
        self.assertNotEqual(a, image_disjoint_split(images, 2, fr))
        quick = limit_split(a, dict(calibration=3, validation=2, test=4))
        self.assertEqual(quick['calibration'], a['calibration'][:3])
        with self.assertRaises(ValueError):
            image_disjoint_split(images, 1, dict(calibration=.5, validation=.5, test=.5))

    def test_missing_file_names_source(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(FileNotFoundError, 'sources.json'):
            load_pope(d, ['random'])


class ClusterBootstrap(unittest.TestCase):
    def test_resamples_whole_images_and_pairs_arms(self):
        groups = np.repeat(np.arange(10), 3)
        x = np.repeat(np.arange(10.), 3)
        r = cluster_bootstrap_idx(groups, lambda i: x[i].mean(), 0, 200)
        self.assertEqual((r['image_clusters'], r['n'], r['valid_bootstrap']), (10, 30, 200))
        self.assertLess(r['ci95'][0], 4.5); self.assertGreater(r['ci95'][1], 4.5)
        same = cluster_bootstrap_idx(groups, lambda i: x[i].mean() - x[i].mean(), 0, 50)
        self.assertEqual(same['ci95'], [0.0, 0.0])
        none = cluster_bootstrap_idx(groups, lambda i: None, 0, 10)
        self.assertIsNone(none['ci95'])


if __name__ == '__main__':
    unittest.main()
