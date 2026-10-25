import base64
import hashlib
import json
import tarfile
import zipfile
import unittest

from scripts.prepare_controller_e_v2a_transport import RELEASE, dataset_variant


class TransportTests(unittest.TestCase):
    def test_only_the_loader_changes_and_the_exact_archive_is_preserved(self):
        old = json.loads((RELEASE / 'controller_e_v2a_development_v1.ipynb').read_text())
        new = dataset_variant(old)
        before = ''.join(old['cells'][1]['source']); after = ''.join(new['cells'][1]['source'])
        payload = next(line for line in before.splitlines() if line.startswith('raw=base64.b64decode('))
        embedded = payload.split("'", 2)[1]
        archive = RELEASE / 'controller_e_v2a_development_v1.tar.gz'
        self.assertEqual(base64.b64decode(embedded, validate=True), archive.read_bytes())
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertIn(digest, after)
        self.assertIn('controller_e_v2a_development_v1.tar.gz.bin', after)
        self.assertEqual(before[before.index('assert hashlib.sha256'):], after[after.index('assert hashlib.sha256'):])
        self.assertEqual(old['metadata'], new['metadata'])
        self.assertEqual(old['cells'][0], new['cells'][0])
        self.assertLess(len(json.dumps(new).encode()), 1_000_000)

    def test_missing_or_duplicate_loader_is_rejected(self):
        old = json.loads((RELEASE / 'controller_e_v2a_development_v1.ipynb').read_text())
        for code in ('print("wrong notebook")', 'raw=base64.b64decode("x")\nraw=base64.b64decode("y")'):
            old['cells'][1]['source'] = [code]
            with self.assertRaises(ValueError): dataset_variant(old)

    def test_actual_sealed_archive_is_runtime_only_and_complete(self):
        with tarfile.open(RELEASE / 'controller_e_v2a_development_v1.tar.gz') as t:
            prefix = 'controller_e_v2a_development_v1/'
            manifest = json.load(t.extractfile(prefix + 'execution_manifest.json'))
            self.assertEqual(manifest['role'], 'development')
            self.assertFalse(manifest['contains_real_gold'])
            self.assertFalse(manifest['contains_selection_or_final_inputs'])
            plan = [json.loads(l) for l in t.extractfile(prefix + 'inputs/development_plan.jsonl')]
            self.assertEqual(len(plan), 800)
            for r in plan:
                self.assertEqual(set(r['input']), {'sample_id', 'dataset', 'query', 'evidence', 'initial_answer'})
                self.assertIn(r['input']['sample_id'], manifest['sample_ids'][r['input']['dataset']])
            for rel, digest in manifest['source_files'].items():
                self.assertEqual(hashlib.sha256(t.extractfile(prefix + rel).read()).hexdigest(), digest)

    def test_real_gate_failure_keeps_zero_development_predictions_and_raw_evidence(self):
        folder = RELEASE.parent / 'kaggle_v1'
        audit = json.loads((folder / 'audit.json').read_text())
        archive = folder / 'outputs.zip'
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), audit['output_zip_sha256'])
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read('manifest.json'))
            self.assertEqual(manifest['n_predictions'], 0)
            self.assertFalse(manifest['engineering_passed'])
            self.assertNotIn('predictions.jsonl', z.namelist())
            case = json.loads(z.read('engineering/v2a-fiction-nq-unsupported_replace.json'))
            self.assertFalse(case['passed'])
            model = json.loads(case['record']['module_trace']['raw_output'])
            self.assertEqual(model['original_verification'], 'unsupported')
            self.assertEqual(model['original_evidence_ids'], [])
            self.assertEqual(model['candidate_answer'], case['record']['initial_answer'])


if __name__ == '__main__': unittest.main()
