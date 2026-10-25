from copy import deepcopy
import unittest
from src.controller.v2a_policy import route_override, choose_deployment_modes, choose_r2_modes


def row(dataset='nq', probability=.3, action='no_action', probe=None):
    return {'dataset': dataset, 'routing': {'decision': {'action': action, 'diagnosis_label': 'correct'},
            'diagnosis': {'stage1_failure_probability': probability}, 'probe': probe}}


def stats():
    return {ds: {'n': 200, 'contract_passed': 190, 'unresolved_or_insufficient': 30,
                 'routed': [{'initial_correct': False, 'final_correct': False} for _ in range(16)] if ds == 'nq' else []}
            for ds in ('nq', 'fever')}


class PolicyTests(unittest.TestCase):
    def test_nq_only_keep_only_and_supported_probe_veto(self):
        for r in (row('fever'), row(action='generation_repair'), row(probe={'answer_supported': 'supported'})):
            self.assertFalse(route_override(r, .2)['overridden'])
        self.assertTrue(route_override(row(probe=None), .2)['overridden'])
        self.assertFalse(route_override(row(), .4)['overridden'])

    def test_no_new_probe_no_mutation_and_no_id_rule(self):
        r = row(); before = deepcopy(r)
        a = route_override(r, .2)
        r['sample_id'] = 'different_id'; r['gold'] = 'offline_sentinel'
        self.assertEqual(route_override(r, .2), a)
        self.assertEqual({k: r[k] for k in before}, before)
        self.assertEqual(a['original_decision']['diagnosis_label'], 'correct')

    def test_only_registered_thresholds_and_valid_probabilities(self):
        for t in (True, .25, .5):
            with self.assertRaises(ValueError): route_override(row(), t)
        for p in (float('nan'), -1, True):
            with self.assertRaises(ValueError): route_override(row(probability=p), .2)

    def test_nonnegative_and_unestimable_modes_are_distinct(self):
        result = choose_deployment_modes(stats())
        self.assertEqual(result['nq']['mode'], 'allow_repair')
        self.assertEqual(result['fever']['mode'], 'verify_only')
        self.assertEqual(result['fever']['reason'], 'routed_net_unestimable')

    def test_negative_net_and_review_line_disable_replacement(self):
        s = stats(); s['nq']['routed'][0] = {'initial_correct': True, 'final_correct': False}
        self.assertEqual(choose_deployment_modes(s)['nq']['mode'], 'verify_only')
        s = stats(); s['nq']['unresolved_or_insufficient'] = 181
        self.assertEqual(choose_deployment_modes(s)['nq']['reason'], 'review_line_exceeded')

    def test_contract_failures_cannot_be_removed_from_denominator(self):
        s = stats(); s['nq']['contract_passed'] = 179
        self.assertEqual({v['mode'] for v in choose_deployment_modes(s).values()}, {'unavailable'})
        s = stats(); s['nq']['n'] = 199
        with self.assertRaises(ValueError): choose_deployment_modes(s)


    def test_r2_gate_outcome_fixes_mode_before_any_development_statistic(self):
        self.assertEqual({v['mode'] for v in choose_r2_modes('failed', None).values()}, {'unavailable'})
        semantic = choose_r2_modes('semantic_only', None)
        self.assertEqual({v['mode'] for v in semantic.values()}, {'verify_only'})
        self.assertIn('experimental', semantic['nq']['reason'])
        with self.assertRaises(ValueError): choose_r2_modes('passed_with_warnings', stats())

    def test_r2_misrepair_cap_counts_pending_as_harm_and_needs_explicit_counts(self):
        s = stats()
        with self.assertRaises(ValueError): choose_r2_modes('passed', s)
        s['nq']['misrepair'] = {'frozen_harms': 0, 'semantic_harms': 0, 'semantic_pending': 0}
        s['fever']['misrepair'] = {'frozen_harms': 0}
        result = choose_r2_modes('passed', s)
        self.assertEqual({v['mode'] for v in result.values()}, {'allow_repair'})
        self.assertIsNone(result['fever']['fever_routed_net_correct'])
        s['nq']['misrepair']['semantic_pending'] = 1
        result = choose_r2_modes('passed', s)
        self.assertEqual({v['mode'] for v in result.values()}, {'verify_only'})
        self.assertTrue(result['nq']['provisional'])
        s = stats(); s['nq']['contract_passed'] = 179
        self.assertEqual({v['mode'] for v in choose_r2_modes('passed', s).values()}, {'unavailable'})

    def test_r2_either_dataset_harm_or_review_line_protects_both_datasets(self):
        for ds, key in (('nq', 'frozen_harms'), ('nq', 'semantic_harms'), ('fever', 'frozen_harms')):
            s = stats()
            s['nq']['misrepair'] = {'frozen_harms': 0, 'semantic_harms': 0, 'semantic_pending': 0}
            s['fever']['misrepair'] = {'frozen_harms': 0}
            s[ds]['misrepair'][key] = 1
            self.assertEqual({v['mode'] for v in choose_r2_modes('passed', s).values()}, {'verify_only'})
        s = stats(); s['fever']['unresolved_or_insufficient'] = 181
        self.assertEqual({v['reason'] for v in choose_r2_modes('passed', s).values()}, {'review_line_exceeded'})

    def test_r2_semantic_only_still_requires_actual_development_compliance(self):
        s = stats(); s['fever']['contract_passed'] = 179
        result = choose_r2_modes('semantic_only', s)
        self.assertEqual({v['mode'] for v in result.values()}, {'unavailable'})
        s = stats(); s['nq']['n'] = 199
        with self.assertRaises(ValueError): choose_r2_modes('semantic_only', s)
        with self.assertRaises(ValueError): choose_r2_modes('passed', None)

    def test_r2_net_counts_fixes_and_harms_but_does_not_replace_the_harm_guard(self):
        s = stats()
        s['nq']['misrepair'] = {'frozen_harms': 1, 'semantic_harms': 1, 'semantic_pending': 0}
        s['fever']['misrepair'] = {'frozen_harms': 0}
        s['nq']['routed'][0] = {'initial_correct': True, 'final_correct': False}
        s['nq']['routed'][1] = {'initial_correct': False, 'final_correct': True}
        result = choose_r2_modes('passed', s)
        self.assertEqual(result['nq']['nq_routed_net_correct'], 0)
        self.assertEqual({v['mode'] for v in result.values()}, {'verify_only'})


if __name__ == '__main__': unittest.main()
