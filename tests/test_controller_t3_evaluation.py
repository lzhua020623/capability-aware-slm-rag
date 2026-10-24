"""Complete frozen coverage, paired uncertainty, observed Oracle and misrepair."""
from copy import deepcopy
from dataclasses import asdict, replace
import unittest

from scripts.run_controller_integration import smoke_inputs
from src.controller.execution import DEFAULT_MODES, ExecutionConfig, fingerprint
from src.controller.evaluation import analyze, holm, paired_noninferiority, paired_statistics, transition
from tests.test_controller_integration import pipeline


class T3EvaluationTests(unittest.TestCase):
    def fixture(self):
        inputs = smoke_inputs()
        records = [pipeline(config=ExecutionConfig(max_scoring_calls=2),
                            repair=('RANK: 1\nQUOTE: The capital of Luma is Amber.\nANSWER: Amber'
                                    if i.dataset.value == 'nq' else 'bad contract')).run(i, mode=m)
                   for i in inputs for m in DEFAULT_MODES]
        registration = {'role': 'development', 'required_modes': list(DEFAULT_MODES),
                        'datasets': {i.dataset.value: {'sample_ids': [i.sample_id],
                                     'effective_input_hashes': {i.sample_id: fingerprint(asdict(i))}}
                                     for i in inputs}}
        references = [{'dataset': 'nq', 'sample_id': inputs[0].sample_id, 'gold_answer_or_label': ['Amber']},
                      {'dataset': 'fever', 'sample_id': inputs[1].sample_id, 'gold_answer_or_label': 'REFUTES'}]
        return records, references, registration

    def test_dispatched_but_kept_answer_is_not_misrepair(self):
        self.assertEqual(transition([1, 1, 0, 0], [1, 0, 1, 0]),
                         {'n': 4, 'initial_correct': 2, 'initial_wrong': 2, 'fixed': 1,
                          'regressed': 1, 'net_correct': 0, 'repair_rate': .5, 'regression_rate': .5})

    def test_paired_interval_and_mcnemar_use_discordant_pairs(self):
        stats = paired_statistics([1] * 10, [0] * 10)
        self.assertEqual(stats['difference'], 1.)
        self.assertEqual(stats['bootstrap_two_sided_95'], [1., 1.])
        self.assertAlmostEqual(stats['mcnemar_two_sided_p'], 2 / 1024)
        with self.assertRaises(ValueError):
            paired_statistics([0, 2], [0, 1])

    def test_holm_is_monotone_and_handles_ties(self):
        self.assertEqual(holm({'a': .01, 'b': .02, 'c': .02, 'd': .4}),
                         {'a': .04, 'b': .06, 'c': .06, 'd': .4})

    def test_noninferiority_identical_pairs_not_treated_as_zero_pvalue(self):
        small = paired_noninferiority([1] * 10, [1] * 10)
        large = paired_noninferiority([1] * 500, [1] * 500)
        self.assertGreater(small['pvalue'], .05)
        self.assertGreater(large['pvalue'], 0)
        self.assertLess(large['pvalue'], .05)
        self.assertLess(small['one_sided_95_lower'], -.02)
        self.assertGreater(large['one_sided_95_lower'], -.02)

    def test_complete_fixture_reports_oracle_and_four_test_family(self):
        predictions, references, registration = self.fixture()
        report = analyze(predictions, references, registration, replicates=100)
        self.assertTrue(report['acceptance']['complete'])
        self.assertEqual(len(report['holm_adjusted_pvalues']), 4)
        nq = report['datasets']['nq']
        self.assertEqual(nq['routing_opportunities']['missed_repair_opportunities'], 1)
        self.assertEqual(nq['routing_opportunities']['fixed_action_residual_errors'], 0)
        self.assertEqual(nq['transitions']['adaptive']['fixed'], 0)
        self.assertEqual(nq['transitions']['always_repair']['fixed'], 1)
        self.assertIn('not_certified', nq['cost_comparison']['full_pipeline_AC5'])

    def test_whole_missing_arm_partial_cohort_and_extra_rows_rejected(self):
        predictions, references, registration = self.fixture()
        for bad in ([r for r in predictions if r['mode'] != 'always_repair'],
                    predictions[:-1], predictions + predictions[:1]):
            with self.assertRaises(ValueError):
                analyze(bad, references, registration, replicates=10)

    def test_forged_input_call_accounting_and_prompt_hash_rejected(self):
        predictions, references, registration = self.fixture()
        bad = deepcopy(predictions); bad[0]['input']['query'] = 'changed'
        with self.assertRaises(ValueError):
            analyze(bad, references, registration, replicates=10)
        bad = deepcopy(predictions); bad[0]['cost']['llm_calls'] += 1
        with self.assertRaises(ValueError):
            analyze(bad, references, registration, replicates=10)
        bad = deepcopy(predictions)
        row = next(r for r in bad if any(e['kind'] == 'llm' for e in r['calls']))
        next(e for e in row['calls'] if e['kind'] == 'llm')['arguments']['prompt'] += 'forged'
        with self.assertRaises(ValueError):
            analyze(bad, references, registration, replicates=10)

    def test_missing_tokens_cannot_be_zero_cost_or_complete_savings(self):
        predictions, references, registration = self.fixture()
        row = next(r for r in predictions if r['dataset'] == 'nq' and r['mode'] == 'adaptive')
        row['cost']['input_tokens'] = None
        report = analyze(predictions, references, registration, replicates=10)
        self.assertIsNone(report['datasets']['nq']['cost_comparison']['incremental_token_reduction_estimate'])


if __name__ == '__main__':
    unittest.main()
