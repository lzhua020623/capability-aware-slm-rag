"""NQ recoverability oracle: registration and the L3 / L4 contexts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import recoverability_oracle as ro

ROOT = Path(__file__).resolve().parents[1]
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def record(sid, top5, query='q'):
    return {'sample_id': sid, 'query': query, 'evidence': [{'passage_id': p} for p in top5]}


class RegistrationTests(unittest.TestCase):
    def test_registration_binds_the_sweep_ids_only(self):
        reg = ro.load_registration()
        self.assertEqual(reg['set']['ids_sha256'], sha(reg['set']['ids']))
        self.assertEqual(len(ro.sweep_ids()), 6000)
        self.assertEqual(set(reg['levels']), {'L1', 'L2', 'L2_any', 'L3', 'L4'})
        self.assertIn('never a product component', reg['role'].lower())
        sweep = json.loads((ROOT / 'configs/controller/quality_sweep_v1.json').read_text(encoding='utf-8'))
        confirm = json.loads((ROOT / sweep['split']['confirm']['path']).read_text(encoding='utf-8'))
        self.assertFalse(set(confirm) & set(ro.sweep_ids()))


class ContextTests(unittest.TestCase):
    def test_levels_put_the_gold_passage_first_or_alone(self):
        c1 = ('a', 'b', 'g', 'c', 'd')
        self.assertEqual(ro.oracle_contexts(c1, 'g'), {'L3': ('g', 'a', 'b', 'c', 'd'), 'L4': ('g',)})
        self.assertEqual(ro.oracle_contexts(c1, 'z'), {'L3': ('z', 'a', 'b', 'g', 'c'), 'L4': ('z',)})
        self.assertEqual(ro.oracle_contexts(('g', 'a', 'b', 'c', 'd'), 'g')['L3'], ('g', 'a', 'b', 'c', 'd'))

    def test_rows_skip_c1_contexts_and_missing_gold_and_mark_nothing(self):
        records = [record('s1', ('a', 'b', 'g', 'c', 'd')), record('s2', ('g', 'a', 'b', 'c', 'd')),
                   record('s3', ('a', 'b', 'c', 'd', 'e'))]
        rows, missing = ro.context_rows(records, {'s1': 'g', 's2': 'g', 's3': 'x'}, {'a', 'b', 'c', 'd', 'e', 'g'})
        self.assertEqual([(r['sample_id'], r['passage_ids']) for r in rows],
                         [('s1', ['g', 'a', 'b', 'c', 'd']), ('s1', ['g']), ('s2', ['g'])])
        self.assertEqual(missing, ['s3'])
        self.assertTrue(all(set(r) == {'sample_id', 'query', 'passage_ids'} for r in rows))

    def test_gold_passages_read_the_delivery_reference_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{'sample_id': 's1', 'annotation_reference': {'gold_passage_id': 'g1', 'gold_answer': ['x']}},
                    {'sample_id': 's9', 'annotation_reference': {'gold_passage_id': 'g9'}}]
            (Path(tmp) / 'nq_train_annotation_000.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.assertEqual(ro.gold_passages(tmp, ['s1']), {'s1': 'g1'})
            with self.assertRaises(ValueError):
                ro.gold_passages(tmp, ['s1', 's2'])


def ladder_item(sid, c1='Oslo', route_1='Oslo', b_bm25='Oslo', ctx=('Oslo',), gold='g', rank=None, share=1.0):
    return {'sample_id': sid, 'query': 'q', 'c1': c1, 'c1_top5': ['a', 'b', 'c', 'd', 'e'], 'gold_passage_id': gold,
            'gold_bge_rank': rank, 'route_1': route_1, 'b_bm25': b_bm25, 'routed': True, 'p_fail': 0.5,
            'contexts': {f'k{i}': a for i, a in enumerate(ctx)}, 'agreement': {'groups': 2, 'share': share}}


class ScoreTests(unittest.TestCase):
    def test_oracle_answers_take_the_generation_or_the_p3_rule(self):
        items = [ladder_item('s1'), ladder_item('s2', gold='a')]
        gens = {('s1', ('g', 'a', 'b', 'c', 'd')): (' Rome ', None), ('s1', ('g',)): ('Paris', 'not_grounded'),
                ('s2', ('a',)): ('Bergen', None)}
        out = ro.oracle_answers(items, gens)
        self.assertEqual(out['s1'], {'L3': ('Rome', 'Rome'), 'L4': ('Paris', 'Oslo')})
        self.assertEqual(out['s2'], {'L3': ('Oslo', 'Oslo'), 'L4': ('Bergen', 'Bergen')})   # gold first in C1: L3 is C1

    def test_first_fix_positions_and_report(self):
        gold = {s: ['Rome'] for s in ('s1', 's2', 's3', 's4', 's5')}
        items = [ladder_item('s1', route_1='Rome'), ladder_item('s2', ctx=('Oslo', 'Rome'), rank=12),
                 ladder_item('s3', gold='a'), ladder_item('s4', share=0.3), ladder_item('s5', c1='Rome', route_1='Rome',
                                                                                         b_bm25='Rome', ctx=('Rome',))]
        answers = {'s1': {'L3': ('x', 'x'), 'L4': ('x', 'x')}, 's2': {'L3': ('x', 'x'), 'L4': ('x', 'x')},
                   's3': {'L3': ('Rome', 'Oslo'), 'L4': ('x', 'x')}, 's4': {'L3': ('x', 'x'), 'L4': ('x', 'x')},
                   's5': {'L3': ('x', 'x'), 'L4': ('Rome', 'Rome')}}
        label = ro.labeller({}, {(s, 'x'): False for s in gold} | {(s, 'Oslo'): False for s in gold}, gold)
        report = ro.ladder_report(items, answers, label)
        p = report['primary']
        self.assertEqual((report['complete'], p['c1_failures']), (True, 4))
        self.assertEqual(p['first_fix'], {'L1': 1, 'L2': 0, 'L2_any': 1, 'L3': 1, 'L4': 0, 'none': 1})
        self.assertEqual(p['by_gold_position']['bge_6_20'], {'L2_any': 1})
        self.assertEqual(p['by_gold_position']['c1_top1'], {'L3': 1})
        self.assertEqual(p['by_agreement']['below'], {'none': 1})
        self.assertEqual(p['c1_correct'], {'c1_correct': 1, 'L3_breaks': 1, 'L4_breaks': 0})
        self.assertEqual(report['p3_rule']['first_fix']['none'], 2)   # s3's L3 answer is rejected under the P3 rule
        self.assertEqual(p['boundary_auc']['disagreement'], 1.0)

    def test_unjudged_answers_make_the_report_incomplete(self):
        items = [ladder_item('s1')]
        label = ro.labeller({}, {}, {'s1': ['Rome']})
        report = ro.ladder_report(items, {'s1': {'L3': ('x', 'x'), 'L4': ('x', 'x')}}, label)
        self.assertEqual((report['complete'], report['unjudged_items']), (False, 1))
        self.assertEqual(ro.needed_strings(items[0], {'L3': ('x', 'Oslo'), 'L4': ('y', 'y')}), {'Oslo', 'x', 'y'})
        self.assertEqual((ro.auc([2, 3], [1]), ro.auc([1], [1]), ro.auc([], [1])), (1.0, 0.5, None))


if __name__ == '__main__':
    unittest.main()
