"""No unsealed/unmatched CUDA receipt may start the real development run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_controller_t4 import accept_engineering, engineering_environment, verify_manifest
from src.controller.engineering import inspect_engineering_calls


class T3GateTests(unittest.TestCase):
    def receipt(self):
        manifest = {'source_files': {'src/controller/execution.py': 'frozen'}, 'policy_sha256': 'policy'}
        environment = {'model_revision': 'fixed', 'packages': {'transformers': '5.16.1'},
                       'controller_generation_runtime': {'prefill_chunk_size': 256}}
        receipt = {'passed': True, 'minimum_input_tokens': 14000, **manifest,
                   'engineering_environment': engineering_environment(environment),
                   'checks': [{'dataset': ds, 'path': mode, 'passed': True, 'minimum_actual_input_tokens': 14000,
                               'record': {'calls': [{'kind': 'llm', 'attempted': True, 'status': 'completed',
                                                     'output': {'input_tokens': 14000}}]}}
                              for ds in ('nq', 'fever')
                              for mode in ('probe', 'always_retrieve', 'always_rerank', 'always_repair')]}
        return receipt, manifest, environment

    def test_accepts_only_complete_matching_real_receipt(self):
        receipt, manifest, environment = self.receipt()
        accept_engineering(receipt, manifest, environment)
        for mutation in ('source', 'environment', 'length', 'coverage', 'passed'):
            bad = deepcopy(receipt)
            if mutation == 'source': bad['source_files'] = {}
            elif mutation == 'environment': bad['engineering_environment']['model_revision'] = 'different'
            elif mutation == 'length': bad['minimum_input_tokens'] = 13000
            elif mutation == 'coverage': bad['checks'].pop()
            else: bad['passed'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                accept_engineering(bad, manifest, environment)

    def test_short_truncated_failed_and_zero_call_paths_fail_engineering_gate(self):
        good = {'kind': 'llm', 'attempted': True, 'status': 'completed', 'output': {'input_tokens': 14000}}
        self.assertEqual(inspect_engineering_calls([good], 14000)['minimum_actual_input_tokens'], 14000)
        for bad in ([], [{**good, 'status': 'failed'}], [{**good, 'output': {'input_tokens': 13999}}],
                    [{**good, 'attempted': False}]):
            with self.assertRaises(RuntimeError):
                inspect_engineering_calls(bad, 14000)

    def test_selection_or_gold_packaged_scope_is_rejected_before_loading_models(self):
        for flag in ('contains_gold', 'contains_selection_or_final_inputs'):
            m = {'schema_version': 'controller_t3_execution_manifest_v1', 'role': 'development',
                 'contains_gold': False, 'contains_selection_or_final_inputs': False,
                 'datasets': {'nq': {}, 'fever': {}}}
            m[flag] = True
            with self.assertRaises(ValueError):
                verify_manifest(Path('/not-read'), m)

    def test_mutated_code_is_rejected_before_input_or_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'code.py').write_text('changed')
            m = {'schema_version': 'controller_t3_execution_manifest_v1', 'role': 'development',
                 'contains_gold': False, 'contains_selection_or_final_inputs': False,
                 'datasets': {'nq': {}, 'fever': {}}, 'source_files': {'code.py': hashlib.sha256(b'old').hexdigest()}}
            with self.assertRaises(ValueError):
                verify_manifest(root, m)


if __name__ == '__main__':
    unittest.main()
