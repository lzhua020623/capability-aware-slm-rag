"""A single declared import-only revision must preserve the original experiment."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.prepare_controller_fever_c1_transport_r1 import IMPORT_FIX, prepare
from src.retrieval.offset_corpus import file_sha
from tests import test_controller_fever_c1 as fixtures


class TransportTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.CollectionTests(); fixture.setUp(); self.addCleanup(fixture.temp.cleanup)
        self.fixture = fixture
        self.source = fixture.root/IMPORT_FIX
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self.source.write_text('old import')
        fixture.manifest['source_files'][IMPORT_FIX] = file_sha(self.source)
        fixture.save_manifest()
        self.source.write_text('lazy import')

    def test_registration_preserves_all_original_fields_and_bytes_except_declared_source(self):
        f = self.fixture
        before = (f.collection/'manifest.json').read_bytes()
        output = f.root/'retry'
        delta = prepare(f.collection, output, root=f.root)
        revised = json.loads((output/'manifest.json').read_text())
        self.assertEqual(set(delta['source_delta']), {IMPORT_FIX})
        expected = copy.deepcopy(f.manifest)
        expected['source_files'][IMPORT_FIX] = file_sha(self.source)
        self.assertEqual({k:v for k,v in revised.items() if k != 'transport_revision'}, expected)
        self.assertEqual((f.collection/'manifest.json').read_bytes(), before)
        for role in f.manifest['files']:
            relative = f.manifest['files'][role]['path']
            self.assertEqual((f.collection/relative).read_bytes(), (output/relative).read_bytes())
        with self.assertRaises(FileExistsError): prepare(f.collection, output, root=f.root)

    def test_prompt_or_task_changes_cannot_be_registered_as_import_only_fix(self):
        f = self.fixture
        (f.root/'src/generation/prompts.py').write_text('different prompt')
        with self.assertRaisesRegex(ValueError, 'only the retrieval'):
            prepare(f.collection, f.root/'retry', root=f.root)
        (f.root/'src/generation/prompts.py').write_text('fixture')
        (f.collection/'development/retrieval.jsonl').write_text('changed question')
        with self.assertRaisesRegex(ValueError, 'retrieval changed'):
            prepare(f.collection, f.root/'retry', root=f.root)


class RecordedTransportTests(unittest.TestCase):
    def test_recorded_revision_has_one_source_delta_and_original_input_fingerprints(self):
        root = Path(__file__).resolve().parents[1]
        folder = root/'artifacts/controller/fever_c1_v1'
        parent = json.loads((folder/'collection/manifest.json').read_text())
        child = json.loads((folder/'transport_r1/collection/manifest.json').read_text())
        revision = child.pop('transport_revision')
        self.assertEqual(revision['parent_manifest_sha256'], file_sha(folder/'collection/manifest.json'))
        changed = {p for p,digest in child['source_files'].items() if parent['source_files'][p] != digest}
        self.assertEqual(changed, {IMPORT_FIX})
        child['source_files'][IMPORT_FIX] = parent['source_files'][IMPORT_FIX]
        self.assertEqual(child, parent)
        failure = json.loads((folder/'attempt1_failure/failure_receipt.json').read_text())
        self.assertEqual(failure['n_completed'], 0)
        self.assertFalse(failure['qwen_called'])
        self.assertEqual(failure['log_sha256'], file_sha(folder/'attempt1_failure/kaggle_version1.log'))

    def test_real_retry_preflight_works_without_faiss_torch_transformers_or_encoders(self):
        root = Path(__file__).resolve().parents[1]
        collection = root/'artifacts/controller/fever_c1_v1/transport_r1/collection'
        code = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'faiss','torch','sentence_transformers','transformers'}:
            raise ImportError('forbidden optional dependency: '+fullname)
sys.meta_path.insert(0, Block())
sys.argv=['scripts/run_controller_fever_c1.py','--collection',sys.argv[1],
          '--output-dir',sys.argv[2],'--preflight']
runpy.run_path('scripts/run_controller_fever_c1.py',run_name='__main__')
'''
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, '-c', code, str(collection), temp+'/out'],
                                    cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(json.loads(result.stdout)['counts'],
                         {'development':200,'selection':200,'final_candidate':500})


if __name__ == '__main__':
    unittest.main()
