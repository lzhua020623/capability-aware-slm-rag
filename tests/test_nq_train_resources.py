"""Provenance gates, lazy backend wiring, and unchanged query/search behavior."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_controller_integration import main, smoke_inputs
from src.compensation.retrieval_enhancement.adapters import BGE_QUERY_INSTRUCTION, NQFaissBackend
from src.controller.nq_train_resources import load_backend, validate_resources
from src.retrieval.offset_corpus import file_sha, validate_complete_mapping


class NQTrainResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {k: self.root / v for k, v in dict(corpus='corpus', ids='ids', offsets='offsets.npy',
            index='index', recovery_metadata='meta', mapping_validation='audit').items()}
        self.paths['corpus'].write_text(''.join(json.dumps({'passage_id': f'p{i}', 'text': f'Passage {i}',
                                                       'gold': 'PRIVATE_SENTINEL'})+'\n' for i in range(3)))
        raw = self.paths['corpus'].read_bytes()
        np.save(self.paths['offsets'], np.array([0]+[i+1 for i, c in enumerate(raw[:-1]) if c == 10], dtype=np.int64))
        self.paths['ids'].write_text(''.join(json.dumps({'passage_id': f'p{i}'})+'\n' for i in range(3)))
        self.paths['index'].write_bytes(b'fake known index')
        self.revision = 'a' * 40
        meta = {'schema_version': 'nq_train_index_recovery_v1', 'corpus_size': 3, 'index_type': 'IndexFlatIP',
                'original_model_revision': None, 'embedding_alignment': {'performed': True, 'passed': True,
                    'model': {'revision': self.revision}}}
        for name in ('corpus', 'ids', 'offsets', 'index'):
            meta[f'{name}_fingerprint'] = self.fingerprint(self.paths[name])
        self.paths['recovery_metadata'].write_text(json.dumps(meta))
        audit = validate_complete_mapping(*(self.paths[k] for k in ('corpus', 'ids', 'offsets', 'recovery_metadata')),
                                          index_path=self.paths['index'])
        self.paths['mapping_validation'].write_text(json.dumps(audit))
        self.registry = {'schema_version': 'controller_nq_train_resources_v1', 'dataset': 'nq_train',
            'model_name': 'BAAI/bge-base-en-v1.5', 'verified_model_revision': self.revision,
            'query_instruction': BGE_QUERY_INSTRUCTION, 'passage_prefix': '', 'embedding_dimension': 768,
            'corpus_size': 3, 'files': {k: self.fingerprint(p) for k,p in self.paths.items()}}
        self.manifest = self.root / 'registry.json'; self.save_registry()

    def fingerprint(self, path):
        return {'path': path.name, 'bytes': path.stat().st_size, 'sha256': file_sha(path)}

    def save_registry(self):
        self.manifest.write_text(json.dumps(self.registry))

    def test_registry_verifies_all_bindings_without_model_imports(self):
        _, paths, receipt = validate_resources(self.manifest, root=self.root)
        self.assertEqual(paths, self.paths)
        self.assertIsNone(receipt['historical_model_revision'])
        self.assertEqual(receipt['mapping_records_checked'], 3)
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(receipt))

    def test_changed_resource_and_rehashed_but_unbound_resource_are_rejected(self):
        self.paths['index'].write_bytes(b'same-looking wrong index')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            validate_resources(self.manifest, root=self.root)
        self.registry['files']['index'] = self.fingerprint(self.paths['index']); self.save_registry()
        with self.assertRaisesRegex(ValueError, 'Audit/resource'):
            validate_resources(self.manifest, root=self.root)

    def test_incomplete_audit_and_unverified_encoder_revision_are_rejected(self):
        self.registry['verified_model_revision'] = 'b' * 40; self.save_registry()
        with self.assertRaisesRegex(ValueError, 'BGE'):
            validate_resources(self.manifest, root=self.root)
        self.registry['verified_model_revision'] = self.revision
        audit = json.loads(self.paths['mapping_validation'].read_text()); audit['records_checked'] = 2
        self.paths['mapping_validation'].write_text(json.dumps(audit))
        self.registry['files']['mapping_validation'] = self.fingerprint(self.paths['mapping_validation']); self.save_registry()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_resources(self.manifest, root=self.root)

    def test_wrong_dataset_or_query_prefix_is_rejected(self):
        for key, value in [('dataset', 'nq_dev'), ('query_instruction', 'query: '), ('passage_prefix', 'passage: ')]:
            old = self.registry[key]; self.registry[key] = value; self.save_registry()
            with self.assertRaisesRegex(ValueError, 'registry'):
                validate_resources(self.manifest, root=self.root)
            self.registry[key] = old

    def fake_modules(self):
        self.encode_calls, self.factory_calls, self.search_calls = [], [], []
        outer = self
        class Encoder:
            def __init__(self, *args, **kwargs):
                outer.factory_calls.append((args, kwargs))
            def encode(self, texts, **kwargs):
                outer.encode_calls.append((texts, kwargs))
                result = np.zeros((1, 768)); result[0, 0] = 1
                return result
        class IndexFlatIP:
            d, ntotal, metric_type, is_trained = 768, 3, 0, True
            def search(self, vectors, k):
                outer.search_calls.append(k)
                return np.array([[.9, .8, .7]]), np.array([[2, 0, 1]])
        self.index = IndexFlatIP()
        fake_faiss = SimpleNamespace(IO_FLAG_MMAP_IFC=512, IO_FLAG_READ_ONLY=2,
            read_index=lambda *args: self.index,
            normalize_L2=lambda x: x.__itruediv__(np.linalg.norm(x, axis=1, keepdims=True)))
        return {'faiss': fake_faiss, 'sentence_transformers': SimpleNamespace(SentenceTransformer=Encoder)}

    def test_backend_uses_pinned_encoder_once_prefixed_query_and_only_evidence_fields(self):
        with patch.dict(sys.modules, self.fake_modules()):
            backend, receipt = load_backend(self.manifest, root=self.root)
        self.addCleanup(backend.retriever.close)
        self.assertIsInstance(backend, NQFaissBackend)
        result = backend.retrieve('Raw question?', dataset='nq', top_k=20)
        self.assertEqual([r['passage_id'] for r in result], ['p2', 'p0', 'p1'])
        self.assertEqual(set(result[0]), {'passage_id', 'text', 'score'})
        self.assertEqual(self.factory_calls[0][1], {'revision': self.revision, 'device': 'cpu'})
        self.assertEqual(self.encode_calls[0][0], [BGE_QUERY_INSTRUCTION + 'Raw question?'])
        self.assertEqual(self.encode_calls[0][1]['batch_size'], 16)
        self.assertEqual(self.search_calls, [20])
        self.assertNotIn('SENTINEL', json.dumps(result))

    def test_loaded_wrong_index_is_rejected_before_search(self):
        modules = self.fake_modules(); self.index.ntotal = 4
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, 'structure'):
            load_backend(self.manifest, root=self.root)
        self.assertFalse(self.search_calls)

    def test_cli_preflight_checks_registry_and_preserves_legacy_resource_exclusivity(self):
        inp, out = self.root / 'input.jsonl', self.root / 'run'
        inp.write_text(json.dumps(asdict(smoke_inputs()[0]))+'\n')
        with patch('scripts.run_controller_integration.ROOT', self.root), \
             patch('scripts.run_controller_integration.code_manifest', return_value={}):
            # Use explicit fixture config/classifier because ROOT is patched only
            # to exercise portable relative resource paths.
            repo = Path(__file__).resolve().parents[1]
            args = ['--input', str(inp), '--output-dir', str(out), '--preflight',
                '--nq-resources', str(self.manifest), '--config', str(repo/'configs/controller/integration_v1.json'),
                '--classifier', str(repo/'models/diagnosis/diagnosis_v2_expanded_hierarchical.json')]
            with patch('scripts.run_controller_integration.subprocess.check_output', return_value='fixture\n'):
                self.assertEqual(main(args), 0)
        manifest = json.loads((out/'manifest.json').read_text())
        self.assertEqual(manifest['nq_train_resources']['mapping_records_checked'], 3)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--input', str(inp), '--output-dir', str(self.root/'bad'), '--preflight',
                  '--nq-resources', str(self.manifest), '--nq-index', str(self.paths['index'])])


if __name__ == '__main__':
    unittest.main()
