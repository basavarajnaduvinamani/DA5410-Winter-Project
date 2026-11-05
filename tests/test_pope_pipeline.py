import os
os.environ.setdefault('USE_TF', '0')
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from src.adapters import LlavaAdapter
from src.pope import load_pope
from src.pope_pipeline import Pipeline, read_jsonl
from stubs import StubProcessor, synthetic_config, tiny_llava, write_synthetic_pope


class PipelineEndToEnd(unittest.TestCase):
    """Whole GPU pipeline on CPU with a random-weight tiny LLaVA; checks mechanics, not science."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = Path(tempfile.mkdtemp())
        pope_dir, cls.images = write_synthetic_pope(cls.tmp / 'pope', n_images=20)
        cls.items = load_pope(pope_dir)
        cls.config = synthetic_config()
        cls.out = cls.tmp / 'run'
        cls.adapter = LlavaAdapter(tiny_llava(), StubProcessor())
        cls.pipe = cls.make()
        cls.pipe.baseline(); cls.pipe.calibrate(); cls.pipe.sweep(); cls.pipe.test()
        (cls.out / 'identity.json').write_text(json.dumps(dict(run_id='synthetic', config=cls.config)))
        (cls.out / 'splits.json').write_text(json.dumps(dict(used_counts={k: len(v) for k, v in cls.pipe.split.items()})))

    @classmethod
    def make(cls):
        return Pipeline(cls.adapter, cls.items, lambda n: Image.open(cls.images / n).convert('RGB'), cls.config, cls.out, log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_raw_rows_complete_and_partitions_image_disjoint(self):
        rows = read_jsonl(self.out / 'predictions.jsonl')
        ids = [r['prediction_id'] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        base = [r for r in rows if r['method'] == 'baseline']
        self.assertEqual(len(base), 120)
        parts = {}
        for r in base:
            parts.setdefault(r['image'], set()).add(r['partition'])
        self.assertTrue(all(len(p) == 1 for p in parts.values()))
        self.assertTrue(all(r['pred'] in {'yes', 'no'} and isinstance(r['correct'], bool) for r in rows))
        test_n = sum(r['partition'] == 'test' for r in base)
        for method in ('slerp_pca', 'add_mean_difference', 'random_direction', 'shuffled_labels'):
            got = [r for r in rows if r['method'] == method]
            self.assertEqual(len(got), test_n, method)
            self.assertTrue(all(r['partition'] == 'test' and r['intervention']['layer'] in (0, 1) for r in got))
        self.assertFalse(any(r['partition'] == 'calibration' and r['method'] != 'baseline' for r in rows))
        sweep_rows = [r for r in rows if '@' in r['method']]
        self.assertTrue(sweep_rows and all(r['partition'] == 'validation' for r in sweep_rows))

    def test_resume_does_not_duplicate(self):
        before = (self.out / 'predictions.jsonl').read_text()
        pipe = self.make()
        self.assertEqual(pipe.baseline()[0], 0)
        pipe.sweep(); pipe.test()
        self.assertEqual((self.out / 'predictions.jsonl').read_text(), before)

    def test_selection_never_reads_test_labels(self):
        selection = json.loads((self.out / 'selection.json').read_text())
        plan = json.loads((self.out / 'steering_plan.json').read_text())
        sweep = json.loads((self.out / 'validation_sweep.json').read_text())
        self.assertEqual(selection['n_calibration'], len(self.pipe.part('calibration')))
        pipe = self.make()
        for key, row in pipe.rows.items():
            if row['partition'] == 'test':  # corrupt every test outcome
                row['label'] = 'yes' if row['label'] == 'no' else 'no'; row['correct'] = not row['correct']
        backup = {p.name: p.read_bytes() for p in self.out.iterdir() if p.is_file()}
        try:
            sel2, plan2 = pipe.calibrate()
            self.assertEqual(json.loads(json.dumps(sel2))['sweep'], selection['sweep'])
            self.assertEqual(plan2, plan)
            self.assertEqual(json.loads(json.dumps(pipe.sweep())), sweep)
        finally:
            for name, data in backup.items():
                (self.out / name).write_bytes(data)

    def test_detection_files_cover_held_out_only(self):
        rows = read_jsonl(self.out / 'detection_scores.jsonl')
        self.assertEqual({r['partition'] for r in rows}, {'validation', 'test'})
        self.assertEqual(set(rows[0]['scores']), {'probe', 'mean_difference', 'pca', 'neg_logprob', 'entropy'})
        d = np.load(self.out / 'layer_scores_test.npz')
        self.assertEqual(d['scores'].shape, (sum(r['partition'] == 'test' for r in rows), 2))

    def test_report_from_saved_files(self):
        from scripts.make_report import README_END, README_START, build_report
        readme = self.tmp / 'README.md'
        readme.write_text(f'# x\n{README_START}\nold\n{README_END}\ntail\n')
        metrics = build_report(self.out, self.tmp / 'results' / 'REPORT.md', self.tmp / 'results' / 'figures', readme)
        text = (self.tmp / 'results' / 'REPORT.md').read_text(encoding='utf8')
        for needle in ('POPE baseline', 'Linear probe', 'SLERP', 'Control: random direction', 'Δ accuracy', 'Retention'):
            self.assertIn(needle, text)
        self.assertTrue((self.tmp / 'results' / 'figures' / 'layer_sweep.png').exists())
        self.assertTrue((self.tmp / 'results' / 'figures' / 'steering_tradeoff_validation.png').exists())
        self.assertIn('slerp_pca', metrics['steering'])
        self.assertIsNotNone(metrics['steering']['slerp_pca']['delta_accuracy']['ci95'])
        r = readme.read_text(encoding='utf8')
        self.assertNotIn('old', r); self.assertIn('tail', r); self.assertIn('Auto-filled from run', r)


class CommandLine(unittest.TestCase):
    def test_write_splits_and_prepare_only_without_gpu(self):
        from scripts.run_pope import main
        with tempfile.TemporaryDirectory() as d:
            main(['--write-splits', f'{d}/s.json'])
            self.assertEqual(json.loads(Path(d, 's.json').read_text())['counts'], dict(calibration=100, validation=50, test=350))
            main(['--output', f'{d}/run', '--profile', 'quick', '--prepare-only'])
            used = json.loads(Path(d, 'run', 'splits.json').read_text())['used_counts']
            self.assertEqual(used, dict(calibration=40, validation=20, test=100))
            with self.assertRaisesRegex(ValueError, 'different config'):
                main(['--output', f'{d}/run', '--profile', 'full', '--prepare-only'])

    def test_placeholder_report(self):
        from scripts.make_report import build_report
        with tempfile.TemporaryDirectory() as d:
            build_report(None, f'{d}/REPORT.md', f'{d}/fig', placeholder=True)
            self.assertIn('no GPU run yet', Path(d, 'REPORT.md').read_text())

    def test_notebook_cells_parse(self):
        import ast
        nb = json.loads(Path('notebooks/CounterPrior_Mechanistic_Steering.ipynb').read_text())
        code = [c for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertTrue(code)
        for c in code:
            src = ''.join(c['source'])
            ast.parse('\n'.join(l for l in src.splitlines() if not l.lstrip().startswith(('!', '%'))))
        self.assertIn('scripts/colab_run.sh', ''.join(''.join(c['source']) for c in code))


if __name__ == '__main__':
    unittest.main()
