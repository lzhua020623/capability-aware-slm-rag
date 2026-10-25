from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_controller_e_v2a import frozen_plan, notebook
from scripts.prepare_controller_t5_review import page, prepare as prepare_review
from scripts.run_controller_e_v2a import fictional_cases, validate_plan, execute_one, check_expected, gate, long_case
from src.controller.execution import ExecutionConfig
from tests.test_controller_integration import pipeline
from tests.test_evidence_v2a import Backend, output


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.ids, cls.routed = frozen_plan()

    def test_every_actual_full_entering_state_is_verified_before_packaging(self):
        self.assertEqual(len(self.plan), 800)
        self.assertEqual({k: len(v) for k, v in self.routed.items()}, {'nq': 16, 'fever': 0})
        indexed = {(r['input']['dataset'], r['input']['sample_id'], r['arm']): r for r in self.plan}
        self.assertTrue(any(r['input_sha256'] != indexed[(r['input']['dataset'], r['input']['sample_id'], 'always_repair')]['input_sha256']
                            for r in self.plan if r['arm'] == 'full_E'))
        self.assertTrue(all(set(r['input']) == {'sample_id', 'dataset', 'query', 'evidence', 'initial_answer'} for r in self.plan))

    def test_plan_rejects_missing_duplicate_and_changed_states(self):
        manifest = {'role': 'development', 'contains_real_gold': False, 'sample_ids': self.ids}
        validate_plan(self.plan, manifest)
        for broken in (self.plan[:-1], [*self.plan[:-1], self.plan[0]]):
            with self.assertRaises(ValueError): validate_plan(broken, manifest)
        changed = deepcopy(self.plan); changed[0]['input']['initial_answer'] = 'changed'
        with self.assertRaises(ValueError): validate_plan(changed, manifest)
        with self.assertRaises(ValueError): validate_plan(self.plan, {**manifest, 'role': 'final'})

    def test_fresh_expected_behaviors_and_runtime_accounting(self):
        cases = fictional_cases(); self.assertEqual(len(cases), 6)
        for c in cases:
            exp = c['expected']
            p = pipeline(config=ExecutionConfig(nq_repair_strategy='evidence_id_v2a', fever_repair_strategy='evidence_id_v2a'))
            candidate = exp['final_answer'] if exp['final_action'] == 'replace' else ''
            p.repair_generators[c['input']['dataset']] = Backend(output(exp['original_verification'], candidate,
                original_ids=[] if exp['original_verification'] == 'insufficient' else ['R1S1']))
            r = execute_one(p, {'input': c['input'], 'arm': 'engineering_unit_test'})
            self.assertTrue(check_expected(r, c), c['case_id'])
            self.assertEqual(r['cost']['llm_calls'], 1)

    def test_failed_fictional_gate_stops_before_long_or_benchmark(self):
        case = fictional_cases()[0]
        bad = {'module_trace': {'support_check': {'contract_passed': False}},
               'cost': {'failed_calls': 1, 'budget_blocks': 0}}
        with tempfile.TemporaryDirectory() as td, patch('scripts.run_controller_e_v2a.execute_one', return_value=bad) as execute:
            with self.assertRaises(RuntimeError): gate(object(), [case], Path(td))
            self.assertEqual(execute.call_count, 1)
            self.assertFalse(json.loads(next(Path(td).iterdir()).read_text())['passed'])

    def test_long_fixture_preserves_real_fact_and_bounds_prompt(self):
        class Tokenizer:
            def encode(self, text): return text.split()
            def apply_chat_template(self, messages, **kw): return messages[0]['content']
        case, n = long_case(Tokenizer(), fictional_cases()[1])
        self.assertGreaterEqual(n, 14000)
        self.assertLessEqual(n + 384, 32768)
        self.assertEqual(case['input']['evidence'][0], fictional_cases()[1]['input']['evidence'][0])

    def test_notebook_exports_failures_and_contains_no_hidden_retry(self):
        nb = notebook(b'fake_archive_for_code_compilation_only', 'a' * 64)
        code = ''.join(nb['cells'][1]['source'])
        compile(code, '<unit-notebook>', 'exec')
        self.assertIn('finally:', code)
        self.assertIn('strict_timestamps=False', code)
        self.assertIn('requirements-controller-e-v2a.txt', code)
        self.assertNotIn('while True', code)

    def test_review_generation_preserves_pending_humans_and_records_all_searches(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'review'
            prepare_review(out)
            manifest = json.loads((out / 'review_manifest.json').read_text())
            self.assertEqual((manifest['n'], manifest['human_confirmed']), (15, 0))
            trials = json.loads((out / 'routing_search_development.json').read_text())
            self.assertEqual(trials['attempts'], 3)
            self.assertFalse(trials['candidate_frozen'])
            self.assertEqual(json.loads((out / 'replay_cache_audit.json').read_text())['checked'], 800)
            self.assertEqual(json.loads((out / 'correct_keep_regression_ids.json').read_text())['n'], 73)
            with self.assertRaises(FileExistsError): prepare_review(out)

    def test_review_page_escapes_evidence_and_answers(self):
        case = {'review_id': 'R01', 'sample_id': '1', 'query': '<img src=x onerror=bad()>',
                'initial_answer': '</script>', 'adaptive_answer': '<svg>', 'reference': ['x'],
                'original_evidence': [], 'variants': []}
        text = page([case])
        self.assertNotIn('<img src=x', text)
        self.assertIn('&lt;img', text)
        self.assertIn('localStorage', text)


if __name__ == '__main__': unittest.main()
