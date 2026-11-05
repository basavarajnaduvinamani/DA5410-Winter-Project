"""Synthetic fixtures exercise analysis plumbing; these are not research results."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.analyze_run import analyze


class AnalysisTests(unittest.TestCase):
    def test_complete_reviews_and_paired_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'identity.json').write_text(json.dumps(dict(run_id='synthetic-unit-test',commit='fixture',dataset_sha256='fixture',config=dict(seed=20261030,bootstrap_iterations=10))))
            split=dict(a='calibration',b='validation',c='validation',d='test',e='test')
            (p/'splits.json').write_text(json.dumps(split))
            rows=[];reviews=[]
            for i,(key,s) in enumerate(split.items()):
                for method in (['baseline'] if s=='calibration' else ['baseline','steered']):
                    pid=key+':'+method
                    r=dict(id=key,prediction_id=pid,run_id='synthetic-unit-test',split=s,method=method,image_group=key,category='counter_existence',probe_probability=.9 if i%2 else .1)
                    rows.append(r)
                    reviews.append(dict(prediction_id=pid,label='hallucinated' if i%2 and method=='baseline' else 'correct',reviewer='unit-test-fixture',reason='synthetic fixture'))
            (p/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            reviewfile=p/'reviews.jsonl'
            reviewfile.write_text(''.join(json.dumps(r)+'\n' for r in reviews[:-1]))
            with self.assertRaisesRegex(ValueError,'manual review'):
                analyze(p,reviewfile)
            reviewfile.write_text(''.join(json.dumps(r)+'\n' for r in reviews))
            result=analyze(p,reviewfile)
            self.assertEqual(result['metrics']['test']['probe:auroc']['estimate'],1.)
            self.assertIsNone(result['metrics']['test']['baseline:counter_attribute']['accuracy']['estimate'])
            self.assertEqual(result['metrics']['test']['steered:counter_existence:hallucination_delta']['estimate'],-.5)
            self.assertEqual(result['metrics']['test']['steered:overall:hallucination_delta']['estimate'],-.5)


if __name__=='__main__':unittest.main()
