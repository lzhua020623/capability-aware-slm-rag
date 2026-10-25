from copy import deepcopy
import unittest
from src.controller.v2a_policy import route_override, choose_deployment_modes


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


if __name__ == '__main__': unittest.main()
