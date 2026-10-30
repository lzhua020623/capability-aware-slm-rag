"""Adaptive multi-context selection: registration binding and the GPU context list."""
import hashlib
import json
from pathlib import Path
import unittest

from scripts import adaptive_selection as ad
from scripts import l2_offline
from tests.test_l2_confirm import L2_IDS, cand_row, result

ROOT = Path(__file__).resolve().parents[1]
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class AdaptiveSelectionTests(unittest.TestCase):
    def test_registration_binds_the_product_the_contexts_and_the_test_ids(self):
        reg = ad.load_registration()
        self.assertEqual(reg['base']['product_sha256'], sha(reg['base']['product']))
        self.assertEqual(reg['test']['unrouted_ids_sha256'], sha(reg['test']['unrouted_ids']))
        ids = json.loads((ROOT / reg['test']['unrouted_ids']).read_text(encoding='utf-8'))
        routed = json.loads((ROOT / 'configs/controller/bm25_gate_v1_test_ids.json').read_text(encoding='utf-8'))
        self.assertEqual((len(ids), len(set(ids) & set(routed))), (reg['test']['n_unrouted'], 0))
        configs = ad.settings(reg)
        self.assertEqual(len(configs), 8)
        grid = l2_offline.grid(l2_offline.load_registration())
        self.assertTrue(all(c in grid for c in configs))
        self.assertEqual([l2_offline.name(c) for c in configs[:2]], ['bge20_bm250_rrf', 'bge20_bm2520_ce_only'])
        self.assertEqual(reg['contexts']['budget_stages'], [0, 3, 5, 8])

    def test_contexts_skip_c1_top5_stored_answers_and_duplicates(self):
        configs = [{'k_bge': 20, 'k_bm25': 0, 'fusion': 'rrf'}, {'k_bge': 20, 'k_bm25': 20, 'fusion': 'ce_only'},
                   {'k_bge': 20, 'k_bm25': 50, 'fusion': 'ce_only'}]
        results = [result('a', 'Bonn', 'Bonn', 'no_action'),
                   result('b', 'Lyon', 'Nice', ad.P3, ('b5', 'b0', 'b1', 'b2', 'b3'), 'Nice')]
        candidates = {'a': cand_row('a', 'x'), 'b': cand_row('b', 'x')}
        stored = ad.stored_generations(results, [])
        self.assertEqual(list(stored), [('b', ('b5', 'b0', 'b1', 'b2', 'b3'))])
        rows = ad.context_rows(results, candidates, stored, configs)
        # rrf over flat scores gives C1's Top-5; both ce_only settings pick (x, b0..b3) once per item.
        self.assertEqual([(r['sample_id'], tuple(r['passage_ids'])) for r in rows], [('a', L2_IDS), ('b', L2_IDS)])
        rows = ad.context_rows(results, candidates, {**stored, ('b', L2_IDS): ('Lille', None)}, configs)
        self.assertEqual([r['sample_id'] for r in rows], ['a'])


if __name__ == '__main__':
    unittest.main()
