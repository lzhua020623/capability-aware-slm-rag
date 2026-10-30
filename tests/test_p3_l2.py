"""L2 candidate union and selection; the control setting must equal P3-20."""
import json
from pathlib import Path
import random
import unittest

from src.compensation.retrieval_enhancement.p3 import select_p3_20
from src.compensation.retrieval_enhancement.p3_l2 import same_context, select_l2, union_candidates

ROOT = Path(__file__).resolve().parents[1]


def hits(ids, start_score=0.9):
    return [{'passage_id': pid, 'text': 'text ' + pid, 'score': start_score - i / 100} for i, pid in enumerate(ids)]


class P3L2Tests(unittest.TestCase):
    def test_union_keeps_bge_order_then_bm25_only_passages_with_both_ranks(self):
        pool = union_candidates(hits(['a', 'b', 'c']), hits(['c', 'd', 'a']))
        self.assertEqual([p['passage_id'] for p in pool], ['a', 'b', 'c', 'd'])
        self.assertEqual([p['ranks'] for p in pool], [{'bge': 1, 'bm25': 3}, {'bge': 2}, {'bge': 3, 'bm25': 1}, {'bm25': 2}])

    def test_union_rejects_conflicting_texts_and_repeats(self):
        with self.assertRaises(ValueError):
            union_candidates(hits(['a']), [{'passage_id': 'a', 'text': 'other'}])
        with self.assertRaises(ValueError):
            union_candidates(hits(['a', 'a']), [])

    def test_control_setting_equals_p3_20_including_ties(self):
        rng = random.Random(7)
        for trial in range(300):
            bge = hits([f'p{i}' for i in range(20)])
            scores = [rng.choice([0.0, 0.5, 1.0, rng.uniform(-5, 5)]) for _ in range(20)]
            expected = [r['passage_id'] for r in select_p3_20(bge, scores)['evidence']]
            got = [r['passage_id'] for r in select_l2(union_candidates(bge, []), scores, 'rrf')]
            self.assertEqual(got, expected, trial)

    def test_ce_only_takes_the_five_best_scores(self):
        pool = union_candidates(hits([f'p{i}' for i in range(8)]), [])
        scores = [0, 7, 1, 6, 2, 5, 3, 4]
        self.assertEqual([r['passage_id'] for r in select_l2(pool, scores, 'ce_only')], ['p1', 'p3', 'p5', 'p7', 'p6'])

    def test_a_passage_in_both_lists_gains_in_rrf(self):
        pool = union_candidates(hits([f'p{i}' for i in range(6)]), hits(['p5']))
        flat = [0.0] * 6
        self.assertEqual(select_l2(pool, flat, 'rrf')[0]['passage_id'], 'p5')
        self.assertEqual(select_l2(union_candidates(hits([f'p{i}' for i in range(6)]), []), flat, 'rrf')[0]['passage_id'], 'p0')

    def test_same_context_needs_ids_texts_and_order(self):
        rows = hits(['a', 'b'])
        self.assertTrue(same_context(rows, [dict(r) for r in rows]))
        self.assertFalse(same_context(rows, rows[::-1]))

    def test_registered_grid_holds_the_p3_20_control(self):
        step = json.loads((ROOT / 'configs/controller/l2_round_v1.json').read_text(encoding='utf-8'))['step_2_l2']
        self.assertIn(20, step['grid']['k_bge'])
        self.assertIn(0, step['grid']['k_bm25'])
        self.assertIn('rrf', step['grid']['fusion'])
        self.assertEqual(step['bm25']['library'], 'bm25s==0.2.14')


if __name__ == '__main__':
    unittest.main()
