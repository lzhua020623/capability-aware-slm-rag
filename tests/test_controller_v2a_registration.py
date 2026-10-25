"""Post-T4 amendment boundaries, not a rewrite of the original protocol."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'configs/controller/e_v2a_amendment_v1.json'


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads(POLICY.read_text())

    def test_parent_snapshots_remain_unchanged(self):
        for rel, digest in self.p['historical_files_unchanged'].items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), digest, rel)
        self.assertIn('after_T4', self.p['registration_stage'])

    def test_engineering_cannot_silently_expand_model_or_budget(self):
        e, gate = self.p['E'], self.p['engineering_gate']
        self.assertEqual((e['calls'], e['max_new_tokens'], e['nq_max_answer_words']), (1, 384, 20))
        self.assertEqual((e['xgrammar_version'], e['prefill_chunk_size']), ('0.2.8', 256))
        self.assertEqual(gate['real_development_min_contract_rate_per_dataset'], .9)
        self.assertGreaterEqual(gate['long_input_tokens_per_dataset'], 14000)
        self.assertTrue(gate['all_expected_cases_must_pass'])
        self.assertTrue(gate['invalid_and_failed_calls_remain_in_denominator'])

    def test_mode_and_search_are_fixed_before_new_predictions(self):
        mode, search = self.p['deployment_mode'], self.p['routing_search']
        self.assertEqual(mode['reference_n'], {'nq': 16, 'fever': 0})
        self.assertEqual(mode['compliant_but_negative_or_unestimable'], 'verify_only')
        self.assertTrue(mode['same_mode_in_all_E_arms'])
        self.assertEqual(search['grid'], [.2, .3, .4])
        self.assertEqual((search['dataset'], search['eligible_action']), ('nq', 'no_action'))
        self.assertEqual(search['selection_candidates'], 1)
        self.assertEqual(search['evaluation_count'], 200)
        self.assertTrue(search['preserve_diagnosis'])

    def test_selection_and_final_do_not_become_development(self):
        self.assertEqual(self.p['selection']['gpu_modes'],
                         ['always_retrieve', 'always_rerank', 'always_repair', 'adaptive'])
        self.assertTrue(self.p['selection']['no_new_thresholds_after_selection'])
        self.assertEqual(self.p['final']['counts_per_dataset'], 500)
        self.assertEqual(len(self.p['final']['modes']), 6)
        self.assertEqual(self.p['final']['runs'], 1)


if __name__ == '__main__':
    unittest.main()
