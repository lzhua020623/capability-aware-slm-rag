"""Preserved first r2 GPU result: intact export is not engineering acceptance."""
import json
from pathlib import Path
import unittest

from scripts.audit_controller_e_v2a_outputs import audit
from scripts.run_controller_e_v2a import execute_one
from src.controller.execution import ExecutionConfig
from src.controller.v2a_policy import choose_r2_modes
from tests.test_controller_integration import pipeline
from tests.test_evidence_v2a import Backend

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/controller/e_v2a_interface_r2'


class R2GPUResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit(FOLDER / 'kaggle_v1/outputs.zip', FOLDER / 'gpu_release_v2')

    def record(self, suffix):
        return json.loads((FOLDER / ('kaggle_v1/raw/engineering/v2a-r2-fiction-' + suffix + '.json')).read_text())['record']

    def test_all_twenty_real_checks_are_auditable_but_failed_without_development(self):
        saved = json.loads((FOLDER / 'kaggle_v1/audit.json').read_text())
        self.assertEqual(self.result, saved)
        self.assertTrue(saved['audit_passed'])
        self.assertEqual(saved['checksums_verified'], 26)
        self.assertEqual(saved['gate_checks_passed'], 10)
        self.assertEqual(saved['failure_counts'], {'execution': 0, 'structure': 1, 'semantic': 9})
        self.assertEqual((saved['short_checks_completed'], saved['long_checks_completed']), (18, 2))
        self.assertEqual(saved['state'], 'stopped_gate_failed')
        self.assertEqual(saved['n_development_predictions'], 0)
        self.assertFalse(saved['engineering_passed'])
        self.assertFalse(saved['quality_scored'])

    def test_correct_candidate_with_wrong_sentence_is_rejected_and_original_kept(self):
        record = self.record('nq-contradicted_repair-1')
        raw = json.loads(record['module_trace']['raw_output'])
        self.assertEqual(raw['replacement_answer'], 'Ilsa Marrow')
        self.assertEqual(raw['replacement_evidence_ids'], ['R2S2'])
        check = record['module_trace']['support_check']
        self.assertEqual(check['candidate_check'], 'invalid_span')
        self.assertFalse(check['contract_passed'])
        self.assertTrue(check['unresolved'])
        self.assertEqual(record['final_answer'], record['initial_answer'])

    def test_long_paths_completed_with_grammar_and_chunking_but_not_behavior_pass(self):
        checks = [c for c in self.result['gate_checks'] if c['long']]
        self.assertEqual(len(checks), 2)
        for c in checks:
            record = self.record(c['case_id'].removeprefix('v2a-r2-fiction-'))
            event, = [e for e in record['calls'] if e['kind'] == 'llm' and e['attempted']]
            self.assertEqual(event['status'], 'completed')
            self.assertGreaterEqual(event['output']['input_tokens'], 14000)
            self.assertEqual(event['structured_decoder']['decoder'], 'xgrammar')
            self.assertEqual(event['structured_decoder']['prefill_chunk_size'], 256)
            self.assertFalse(event['structured_decoder']['hit_token_budget'])
            self.assertTrue(c['contract_passed'])
            self.assertEqual(c['failure_class'], 'semantic')

    def test_failed_gate_freezes_shared_unavailable_without_fabricated_compliance(self):
        decision = json.loads((FOLDER / 'deployment_decision.json').read_text())
        self.assertEqual(decision['modes'], choose_r2_modes(self.result['engineering_outcome'], None))
        self.assertEqual({v['mode'] for v in decision['modes'].values()}, {'unavailable'})
        self.assertEqual(decision['real_development_contract_rates'], {'nq': None, 'fever': None})
        self.assertEqual(decision['evidence_zip_sha256'], self.result['output_zip_sha256'])
        self.assertFalse(decision['proposal_E_completion_accepted'])
        self.assertFalse(decision['r3_in_main_line'])
        self.assertFalse(decision['selection_or_final_started'])

    def test_unavailable_runtime_never_calls_backend_or_fabricates_verdict_for_either_dataset(self):
        config = ExecutionConfig(nq_repair_strategy='evidence_id_v2a', fever_repair_strategy='evidence_id_v2a',
                                 nq_v2a_mode='unavailable', fever_v2a_mode='unavailable')
        for ds in ('nq', 'fever'):
            case = self.record(ds + '-supported_keep-1')['input']
            p = pipeline(config=config)
            backend = Backend(AssertionError('Unavailable E must never generate'))
            p.repair_generators[ds] = backend
            record = execute_one(p, {'input': case, 'arm': 'always_repair'})
            self.assertEqual(backend.calls, [])
            self.assertEqual(record['final_answer'], case['initial_answer'])
            check = record['module_trace']['support_check']
            self.assertEqual(check['original_verification'], 'incomplete')
            self.assertTrue(check['unresolved'])
            self.assertFalse(check['contract_passed'])


if __name__ == '__main__':
    unittest.main()
