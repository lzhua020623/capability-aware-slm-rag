import hashlib
import json
from pathlib import Path
import tarfile
import unittest

from scripts.build_controller_fever_c1_bundle import build, BUNDLE_NAME
from tests import test_controller_fever_c1 as contract_fixtures


class RecordedC1SubmissionTests(unittest.TestCase):
    def test_submission_is_bound_to_frozen_private_input_and_does_not_claim_completion(self):
        from urllib.parse import urlparse
        root=Path(__file__).resolve().parents[1]
        folder=root/'artifacts/controller/fever_c1_v1'
        status=json.loads((folder/'cloud_status.json').read_text())
        receipt=json.loads((folder/status.get('bundle_receipt_path','bundle_receipt.json')).read_text())
        submitted=json.loads((folder/status.get('submission_receipt_path','submission_v1.json')).read_text())
        digest=hashlib.sha256((folder/status.get('archive_path','controller_fever_c1_gpu_v1.tar.gz')).read_bytes()).hexdigest()
        self.assertEqual(status['bundle_sha256'],digest)
        self.assertEqual(submitted['archive_sha256'],digest)
        self.assertEqual(receipt['archive_sha256'],digest)
        self.assertEqual(status['checksum_index_sha256'],receipt['checksum_index_sha256'])
        self.assertEqual(status['dataset_visibility'],'private')
        self.assertEqual(sum(status['planned_counts'].values()),900)
        self.assertEqual(status['version'],submitted['version'])
        self.assertTrue(status['notebook_imported'] and status['bundle_uploaded'] and status['gpu_started'])
        self.assertTrue(status['internet'])
        for key in ('notebook_url','dataset_url'):
            url=urlparse(status[key])
            self.assertEqual(url.hostname,'www.kaggle.com')
            self.assertEqual(url.query,'')
        if status['state']=='running':
            self.assertFalse(status['generation_completed'])
        if status['generation_completed']:
            self.assertTrue(status['qwen_generation_confirmed'])


class C1BundleTests(unittest.TestCase):
    def setUp(self):
        fixture=contract_fixtures.CollectionTests();fixture.setUp();self.addCleanup(fixture.temp.cleanup)
        self.fixture=fixture;self.root=fixture.root

    def test_package_has_exact_checksum_index_runtime_only_rows_and_compiled_notebook(self):
        archive,notebook=self.root/'gpu.tar.gz',self.root/'notebook.ipynb'
        receipt=build(self.fixture.collection,archive,notebook,root=self.root)
        self.assertEqual(receipt['n_tasks'],900)
        self.assertFalse(receipt['gold_included'])
        with tarfile.open(archive) as handle:
            index=handle.extractfile(BUNDLE_NAME+'/bundle_checksums.json').read()
            self.assertEqual(hashlib.sha256(index).hexdigest(),receipt['checksum_index_sha256'])
            for path,digest in json.loads(index).items():
                self.assertEqual(hashlib.sha256(handle.extractfile(BUNDLE_NAME+'/'+path).read()).hexdigest(),digest)
                self.assertNotIn('offline/',path)
                self.assertNotIn('gold.jsonl',path)
        book=json.loads(notebook.read_text())
        for c in book['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'<test-notebook>','exec')
        self.assertIn(receipt['archive_sha256'],notebook.read_text())
        with self.assertRaises(FileExistsError):build(self.fixture.collection,archive,notebook,root=self.root)


if __name__=='__main__':
    unittest.main()
