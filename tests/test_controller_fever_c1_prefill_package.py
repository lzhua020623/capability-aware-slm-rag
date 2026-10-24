import copy
import json
from pathlib import Path
import unittest

from scripts.prepare_controller_fever_c1_prefill_r2 import RUNNER, prepare
from scripts.run_controller_fever_c1 import PREFILL_RUNTIME
from tests import test_controller_fever_c1 as fixtures
from src.retrieval.offset_corpus import file_sha


class PrefillPackageTests(unittest.TestCase):
    def test_registration_preserves_inputs_and_declares_uniform_new_computation(self):
        f=fixtures.CollectionTests();f.setUp();self.addCleanup(f.temp.cleanup)
        (f.root/RUNNER).write_text('registered chunked prefill runner')
        output=f.root/'prefill-r2'
        revision=prepare(f.collection,output,root=f.root)
        revised=json.loads((output/'manifest.json').read_text())
        expected=copy.deepcopy(f.manifest);expected['source_files'][RUNNER]=file_sha(f.root/RUNNER)
        self.assertEqual(revised.pop('execution_revision'),revision)
        self.assertEqual(revised.pop('generation_runtime'),PREFILL_RUNTIME)
        self.assertEqual(revised,expected)
        self.assertTrue(revision['uniform_full_collection_run'])
        self.assertFalse(revision['reuse_previous_answers'])
        for role,info in f.manifest['files'].items():
            self.assertEqual((f.collection/info['path']).read_bytes(),(output/info['path']).read_bytes())
        with self.assertRaises(FileExistsError):prepare(f.collection,output,root=f.root)
        (f.root/'src/generation/prompts.py').write_text('changed semantic prompt')
        with self.assertRaisesRegex(ValueError,'only the C1 runner'):
            prepare(f.collection,f.root/'bad',root=f.root)

    def test_recorded_prefill_collection_has_exact_900_original_inputs(self):
        root=Path(__file__).resolve().parents[1]/'artifacts/controller/fever_c1_v1'
        parent=json.loads((root/'transport_r1/collection/manifest.json').read_text())
        revised=json.loads((root/'prefill_r2/collection/manifest.json').read_text())
        self.assertEqual(revised.pop('generation_runtime'),PREFILL_RUNTIME)
        revision=revised.pop('execution_revision')
        self.assertEqual(set(revision['source_delta']),{RUNNER})
        revised['source_files'][RUNNER]=parent['source_files'][RUNNER]
        self.assertEqual(revised,parent)
        for role,info in parent['files'].items():
            self.assertEqual(file_sha(root/'prefill_r2/collection'/info['path']),info['sha256'])


if __name__=='__main__':unittest.main()
