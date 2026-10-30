"""L2 offline selection, generation reuse, scoring and the choice rule."""
import unittest

from scripts import l2_generate as gen
from scripts import l2_offline as off


def cand_row(sid='s', n_bge=100, bm25=(), ce=None):
    bge = [[f'b{i}', 1.0 - i / 1000] for i in range(n_bge)]
    ids = [p for p, _ in bge] + [p for p, _ in bm25 if p not in {q for q, _ in bge}]
    return {'sample_id': sid, 'bge': bge, 'bm25': [list(x) for x in bm25], 'ce': ce or {p: 0.0 for p in ids}}


def trace(ids, answer, reason=None):
    return {'native': {'module_trace': {'evidence': [{'passage_id': p} for p in ids], 'generated_answer': answer,
                                        'reject_reason': reason}}}


class L2OfflineTests(unittest.TestCase):
    def test_grid_has_eighteen_settings_and_the_control(self):
        settings = off.grid(off.load_registration())
        self.assertEqual(len(settings), 18)
        self.assertIn('bge20_bm250_rrf', [off.name(c) for c in settings])

    def test_selection_uses_only_the_first_k_of_each_list(self):
        ce = {f'b{i}': 0.0 for i in range(100)}
        ce.update({'b30': 9.0, 'x': 8.0})
        row = cand_row(bm25=[('x', 3.0)], ce=ce)
        control = off.select_ids(row, {'k_bge': 20, 'k_bm25': 0, 'fusion': 'ce_only'})
        self.assertNotIn('b30', control)
        self.assertNotIn('x', control)
        deep = off.select_ids(row, {'k_bge': 50, 'k_bm25': 20, 'fusion': 'ce_only'})
        self.assertEqual(deep[:2], ('b30', 'x'))

    def test_final_answer_follows_p3_rules(self):
        gens = {('s', ('a',)): (' Paris ', None), ('s', ('b',)): ('I do not know', 'refusal_or_empty')}
        self.assertEqual(off.final_answer('c1', ('z',), ('z',), gens, 's'), 'c1')
        self.assertEqual(off.final_answer('c1', ('a',), ('z',), gens, 's'), 'Paris')
        self.assertEqual(off.final_answer('c1', ('b',), ('z',), gens, 's'), 'c1')

    def test_stored_generations_come_from_route_1_and_new_coverage_rows(self):
        records = [{'sample_id': 's', 'actions': {off.P3: trace(['a'], 'x')}}, {'sample_id': 't', 'actions': {}}]
        coverage = [{'sample_id': 't', 'source': 'new', 'action': trace(['b'], 'y', 'not_grounded_in_selected_evidence')},
                    {'sample_id': 's', 'source': 'repro', 'action': trace(['c'], 'z')}]
        self.assertEqual(off.stored_generations(records, coverage),
                         {('s', ('a',)): ('x', None), ('t', ('b',)): ('y', 'not_grounded_in_selected_evidence')})

    def test_needed_contexts_skip_no_change_stored_and_repeats(self):
        row = cand_row()
        control_ids = off.select_ids(row, {'k_bge': 20, 'k_bm25': 0, 'fusion': 'rrf'})
        records = [{'sample_id': 's', 'query': 'q', 'evidence': [{'passage_id': p} for p in control_ids], 'actions': {}}]
        self.assertEqual(off.needed_contexts(records, {'s': row}, {}, off.load_registration()), [])

    def test_evaluate_counts_routed_changes_against_c1(self):
        records = [{'sample_id': 'a', 'initial_answer': 'x'}, {'sample_id': 'b', 'initial_answer': 'y'},
                   {'sample_id': 'c', 'initial_answer': 'z'}]
        gold = {'a': ['right'], 'b': ['y'], 'c': ['q']}
        finals = {'a': 'right', 'b': 'wrong', 'c': 'other'}
        judged = {off.judge_id('c', 'z'): False, off.judge_id('c', 'other'): False, off.judge_id('a', 'x'): False,
                  off.judge_id('b', 'wrong'): False}
        self.assertEqual(off.evaluate(records, finals, gold, judged, {'a', 'b', 'c'}),
                         {'changed': 3, 'fixes': 1, 'harms': 1, 'net': 0, 'unjudged': 0})
        self.assertEqual(off.evaluate(records, finals, gold, judged, {'a'})['net'], 1)

    def test_choice_prefers_fewer_candidates_then_rrf_within_three(self):
        rows = [{'k_bge': 100, 'k_bm25': 50, 'fusion': 'ce_only', 'net': 40},
                {'k_bge': 50, 'k_bm25': 20, 'fusion': 'ce_only', 'net': 38},
                {'k_bge': 50, 'k_bm25': 20, 'fusion': 'rrf', 'net': 37},
                {'k_bge': 20, 'k_bm25': 0, 'fusion': 'rrf', 'net': 30}]
        self.assertEqual((off.choose(rows)['k_bge'], off.choose(rows)['fusion']), (50, 'rrf'))

    def test_generation_outcome_uses_the_p3_grounding_check(self):
        passages = [{'passage_id': 'p', 'text': 'Paris is the capital of France.'}]
        self.assertEqual(gen.outcome('q', passages, lambda prompt: 'Paris'), ('Paris', None))
        self.assertEqual(gen.outcome('q', passages, lambda prompt: 'Berlin')[1], 'not_grounded_in_selected_evidence')
        self.assertEqual(gen.key({'sample_id': 's', 'passage_ids': ['a', 'b']}), ('s', ('a', 'b')))


if __name__ == '__main__':
    unittest.main()
