import hashlib
import json
from pathlib import Path
import tarfile
import unittest

from scripts.build_controller_fever_c1_bundle import build, BUNDLE_NAME
from tests import test_controller_fever_c1 as contract_fixtures


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
