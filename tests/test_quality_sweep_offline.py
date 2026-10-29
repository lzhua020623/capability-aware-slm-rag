"""Offline quality sweep: recalibrated thresholds hit their coverage, answers come from the right module, the rule picks one."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts import quality_sweep_offline as offline
from scripts.precompute_quality_sweep import load_registration

LABELS = ('evidence_insufficiency', 'evidence_accessibility', 'evidence_utilisation')


def record(i, label='evidence_insufficiency', p_fail=0.6, p_insuff=0.2, rec=0.1, p_corr=0.3, stage1='failure'):
    actions = {'retrieval_enhancement': {'final_answer': f'p3-{i}'}}
    if label == 'evidence_accessibility':
        actions['evidence_refinement'] = {'final_answer': f'ref-{i}'}
    if label == 'evidence_utilisation':
        actions['generation_repair'] = {'final_answer': f'c1-{i}'}
    return {'sample_id': f's{i}', 'query': f'q{i}', 'initial_answer': f'c1-{i}',
            'diagnosis': {'final_label': label, 'stage1_label': stage1, 'stage1_failure_probability': p_fail},
            'frozen_selection': {'p_fail': p_fail, 'p_insuff': p_insuff, 'rec': rec, 'p_corr_3c': p_corr},
            'actions': actions}


def pool(n=40):
    return [record(i, LABELS[i % 3], p_fail=0.5 + i / 100, p_insuff=(i * 7 % 40) / 40, rec=(i * 3 % 40) / 40,
                   p_corr=(i * 11 % 40) / 40) for i in range(n)]


class CalibrationTests(unittest.TestCase):
    def test_thresholds_hit_coverage_and_lane_share(self):
        items = pool()
        for coverage, share in ((0.1, 0.25), (0.35, 0.5), (0.75, 0.25), (1.0, 0.5)):
            cal = offline.calibrate(items, coverage, share, 0.4, 0.5)
            self.assertEqual(len(cal['selected']), round(coverage * len(items)))
            by_insuff = {r['sample_id'] for r in items if r['frozen_selection']['p_insuff'] >= cal['t_insuff']}
            self.assertEqual(len(by_insuff), round(share * round(coverage * len(items))))
            # The production rule reproduces the same set from the stored thresholds.
            rule = {r['sample_id'] for r in items if r['frozen_selection']['p_insuff'] >= cal['t_insuff']
                    or offline.combo(r, cal['z_stats'], 0.4, 0.5) >= cal['t_combo']}
            self.assertEqual(rule, cal['selected'])

    def test_z_stats_come_from_the_pool(self):
        z = offline.z_stats(pool())
        self.assertAlmostEqual(z['p_fail']['mean'], sum(0.5 + i / 100 for i in range(40)) / 40)
        self.assertEqual(set(z), {'p_fail', 'rec', 'p_corr'})


class AnswerTests(unittest.TestCase):
    def test_final_answer_follows_selection_map_and_gate(self):
        frozen = {'evidence_insufficiency': 'retrieval_enhancement', 'evidence_accessibility': 'evidence_refinement',
                  'evidence_utilisation': 'generation_repair'}
        e_off = {**frozen, 'evidence_accessibility': 'retrieval_enhancement', 'evidence_utilisation': 'no_action'}
        access, util = record(1, 'evidence_accessibility'), record(2, 'evidence_utilisation')
        self.assertEqual(offline.final_answer(access, {'s1'}, frozen, 'closed', 0.39), 'ref-1')
        self.assertEqual(offline.final_answer(access, {'s1'}, e_off, 'closed', 0.39), 'p3-1')
        self.assertEqual(offline.final_answer(access, set(), frozen, 'closed', 0.39), 'c1-1')
        self.assertEqual(offline.final_answer(util, {'s2'}, e_off, 'closed', 0.39), 'c1-2')
        correct = record(3, 'correct', p_fail=0.45, stage1='correct')
        self.assertEqual(offline.final_answer(correct, {'s3'}, frozen, 'closed', 0.39), 'c1-3')
        self.assertEqual(offline.final_answer(correct, set(), frozen, 'p_fail_at_least_0.39_to_p3', 0.39), 'p3-3')
        low = record(4, 'correct', p_fail=0.2, stage1='correct')
        self.assertEqual(offline.final_answer(low, set(), frozen, 'p_fail_at_least_0.39_to_p3', 0.39), 'c1-4')

    def test_judge_rows_cover_every_changed_answer_once_and_flag_normalized_matches(self):
        items = [record(1, 'evidence_accessibility'), record(2, 'evidence_utilisation')]
        gold = {'s1': ['p3-1'], 's2': ['x']}
        rows = offline.judge_rows(items, gold)
        self.assertEqual([(r['source_sample_id'], r['prediction']) for r in rows],
                         [('s1', 'c1-1'), ('s1', 'p3-1'), ('s1', 'ref-1'), ('s2', 'c1-2'), ('s2', 'p3-2')])
        self.assertEqual([r['normalized_match'] for r in rows], [False, True, False, False, False])
        again = offline.judge_rows(items, gold, exclude={rows[0]['sample_id']})
        self.assertEqual(len(again), 4)

    def test_evaluate_counts_judge_and_normalized_fixes(self):
        item = record(1, 'evidence_insufficiency')
        gold = {'s1': ['p3-1']}
        judged = {offline.judge_id('s1', 'c1-1'): False, offline.judge_id('s1', 'p3-1'): True}
        frozen = {label: 'retrieval_enhancement' for label in LABELS}
        out = offline.evaluate([item], gold, judged, {'s1'}, frozen, 'closed', 0.39)
        self.assertEqual((out['changed'], out['fixes'], out['harms'], out['net'], out['nm_net']), (1, 1, 0, 1, 1))
        out = offline.evaluate([item], gold, {}, {'s1'}, frozen, 'closed', 0.39)
        self.assertEqual(out['unjudged'], 1)


class ChoiceTests(unittest.TestCase):
    def test_rule_prefers_lower_coverage_within_three_items(self):
        registration, _ = load_registration()
        rows = [{'coverage': 0.5, 'insuff_share': 0.25, 'action_map': 'frozen', 'correct_gate': 'closed', 'net': 20},
                {'coverage': 0.35, 'insuff_share': 0.25, 'action_map': 'all_p3', 'correct_gate': 'closed', 'net': 18},
                {'coverage': 0.2, 'insuff_share': 0.25, 'action_map': 'frozen', 'correct_gate': 'closed', 'net': 10}]
        self.assertEqual(offline.choose(rows, registration)['coverage'], 0.35)
        rows.append({'coverage': 0.35, 'insuff_share': 0.5, 'action_map': 'frozen', 'correct_gate': 'closed', 'net': 17})
        self.assertEqual(offline.choose(rows, registration)['action_map'], 'frozen')

    def test_grid_is_the_registered_product(self):
        registration, _ = load_registration()
        g = registration['grid']
        configs = list(offline.grid(registration))
        self.assertEqual(len(configs), len(g['coverage_of_failure_pool']) * len(g['insuff_lane_share'])
                         * len(g['action_maps']) * len(g['correct_gate']))


class ReferenceTests(unittest.TestCase):
    def test_reference_export_needs_every_registered_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            delivery = Path(tmp)
            row = {'sample_id': 'a', 'query': 'q', 'annotation_reference': {'gold_answer': 'x'}}
            (delivery / 'nq_train_annotation_000.jsonl').write_text(json.dumps(row) + '\n')
            offline.export_references(delivery, ['a'], delivery / 'refs.jsonl')
            self.assertEqual(offline.read_jsonl(delivery / 'refs.jsonl'), [{'sample_id': 'a', 'gold': ['x']}])
            with self.assertRaises(ValueError):
                offline.export_references(delivery, ['a', 'b'], delivery / 'refs2.jsonl')


if __name__ == '__main__':
    unittest.main()
