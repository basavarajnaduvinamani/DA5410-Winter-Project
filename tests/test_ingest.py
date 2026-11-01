import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from scripts import ingest_run
from scripts.export_site_examples import build_examples

class SavedRunSafety(unittest.TestCase):
    def test_rejects_traversal_and_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bad.zip'
            with zipfile.ZipFile(bundle,'w') as z:z.writestr('run/../escape.json','{}')
            with patch.object(ingest_run,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'Unsafe'):
                    ingest_run.extract_bundle(bundle,root/'incoming/run')
                with zipfile.ZipFile(bundle,'w') as z:z.writestr('run/identity.json','first')
                ingest_run.extract_bundle(bundle,root/'incoming/run')
                with zipfile.ZipFile(bundle,'w') as z:z.writestr('run/identity.json','different')
                with self.assertRaisesRegex(ValueError,'overwrite'):
                    ingest_run.extract_bundle(bundle,root/'incoming/run')
                self.assertEqual((root/'incoming/run/identity.json').read_text(),'first')

    def test_rejects_incomplete_and_forged_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            with self.assertRaisesRegex(ValueError,'missing identity'):
                ingest_run.validate_run(run)
            for name in ['identity.json','splits.json','predictions.jsonl','detection_scores.jsonl','selection.json','status.json','environment.json']:
                (run/name).write_text('{}')
            (run/'identity.json').write_text(json.dumps(dict(config={},run_id='forged')))
            with self.assertRaisesRegex(ValueError,'identity hash'):
                ingest_run.validate_run(run)

    def test_never_extracts_credentials_or_unrelated_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle.zip'
            with zipfile.ZipFile(bundle,'w') as z:
                z.writestr('run/identity.json','{}');z.writestr('run/.env.local','secret fixture');z.writestr('README.md','unrelated')
            with patch.object(ingest_run,'ROOT',root):
                dest=ingest_run.extract_bundle(bundle,root/'incoming/run')
            self.assertEqual([p.name for p in dest.iterdir()],['identity.json'])

    def test_example_limit_is_enforced_before_reading_any_run(self):
        with self.assertRaisesRegex(ValueError,'limit'):
            build_examples('missing','missing','missing',limit=25)

if __name__=='__main__':unittest.main()
