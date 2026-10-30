"""BM25-only replacement: registration binding, the rule, the confirm replay and the Test row and pairing logic."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import bm25_gate as gate
from scripts import l2_candidates, l2_confirm
from scripts.quality_sweep_offline import judge_id
from tests.test_l2_confirm import L2_IDS, cand_row, result

ROOT = Path(__file__).resolve().parents[1]
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
BGE_ONLY = ('b7', 'b0', 'b1', 'b2', 'b3')


class Bm25GateTests(unittest.TestCase):
    def test_registration_binds_route_1_the_l2_round_and_the_test_ids(self):
        registration = gate.load_registration()
        for key in ('base', 'l2_round'):
            self.assertEqual(registration[key]['sha256'], sha(registration[key]['registration']))
        test = registration['test']
        self.assertEqual(sha(test['ids']), test['ids_sha256'])
        ids = json.loads((ROOT / test['ids']).read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            predictions, ours, team = gate.route_1_test(ROOT / gate.ROUTE_1_TEST, tmp)
            self.assertEqual((len(ours), sorted(team)), (4, sorted(gate.TEAM)))
        self.assertEqual(ids, gate.routed_ids(predictions))
        self.assertEqual(len(ids), test['n_ids'])
        self.assertIn('p < 0.05', test['adoption'])

    def test_rule_needs_a_passage_from_bm25(self):
        gens = {('s', L2_IDS): (' Paris ', None), ('s', BGE_ONLY): ('Rome', None)}
        self.assertTrue(gate.brings_bm25(cand_row('s', 'x'), L2_IDS))
        self.assertFalse(gate.brings_bm25(cand_row('s', 'b7'), BGE_ONLY))
        self.assertEqual(gate.bm25_answer('Oslo', cand_row('s', 'x'), (), gens, 's'), 'Paris')
        self.assertIsNone(gate.bm25_answer('Oslo', cand_row('s', 'b7'), (), gens, 's'))
        rejected = {('s', L2_IDS): ('Paris', 'not_grounded_in_selected_evidence')}
        self.assertEqual(gate.bm25_answer('Oslo', cand_row('s', 'x'), (), rejected, 's'), 'Oslo')
        self.assertEqual(gate.gate({'a': 'c', 'b': 'c'}, {'a': 'r', 'b': 'r'}, {'a': 'n', 'b': None}), {'a': 'n', 'b': 'r'})

    def test_confirm_replay_counts_against_route_1(self):
        """a: BM25 passage, L2 fixes. b: BM25 passage, L2 harms. c: BGE re-order only, route 1 stays. d: not routed."""
        results = [result('a', 'Bonn', 'Bern', l2_confirm.P3, ('b5', 'b0', 'b1', 'b2', 'b3'), 'Bern'),
                   result('b', 'Lyon', 'Nice', l2_confirm.P3, ('b5', 'b0', 'b1', 'b2', 'b3'), 'Nice'),
                   result('c', 'Oslo', 'Rome', l2_confirm.P3, ('b5', 'b0', 'b1', 'b2', 'b3'), 'Rome'),
                   result('d', 'Lima', 'Lima', 'no_action')]
        candidates = [cand_row('a', 'x'), cand_row('b', 'x'), cand_row('c', 'b7'), cand_row('d', 'x')]
        gens = [{'sample_id': s, 'passage_ids': list(ids), 'generated_answer': a, 'reject_reason': None}
                for s, ids, a in (('a', L2_IDS, 'Paris'), ('b', L2_IDS, 'Lille'), ('c', BGE_ONLY, 'Oslo'))]
        gold = {'a': ['Paris'], 'b': ['Nice'], 'c': ['Rome'], 'd': ['Lima']}
        judged = [{'sample_id': judge_id(s, x), 'judge_status': 'llm_ok', 'auto_correct': False, 'judge_label': 'INCORRECT'}
                  for s, x in (('a', 'Bern'), ('a', 'Bonn'), ('b', 'Lille'), ('b', 'Lyon'))]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def lines(name, items):
                (tmp / name).write_text(''.join(json.dumps(i) + '\n' for i in items), encoding='utf-8')
                return str(tmp / name)
            with mock.patch.object(l2_confirm, 'role_ids', return_value=['a', 'b', 'c', 'd']), mock.patch('sys.stdout'):
                gate.main(['replay', '--role', 'confirm', '--results', lines('r.jsonl', results),
                           '--candidates', lines('c.jsonl', candidates), '--generations', lines('g.jsonl', gens),
                           '--references', lines('ref.jsonl', [{'sample_id': s, 'gold': g} for s, g in gold.items()]),
                           '--judged', lines('j.jsonl', judged), '--output', str(tmp / 'out.json')])
            out = json.loads((tmp / 'out.json').read_text(encoding='utf-8'))
        self.assertEqual((out['routed'], out['bm25_items']), (3, 2))
        self.assertEqual({k: out['against_route_1'][k] for k in ('changed', 'fixes', 'harms', 'net')},
                         {'changed': 2, 'fixes': 1, 'harms': 1, 'net': 0})

    def test_test_rows_contexts_and_pairing(self):
        predictions = [{'sample_id': s, 'query': f'q {s}', 'prediction': p, 'initial_answer': c, 'gold': [g],
                        'controller_action': act}
                       for s, p, c, g, act in (('a', 'Bern', 'Bonn', 'Paris', gate.P3), ('b', 'Nice', 'Lyon', 'Nice', gate.P3),
                                               ('c', 'Rome', 'Oslo', 'Rome', gate.P3), ('d', 'Lima', 'Lima', 'Lima', 'no_action'))]
        candidates = {'a': cand_row('a', 'x'), 'b': cand_row('b', 'x'), 'c': cand_row('c', 'b7')}
        ids = ['a', 'b', 'c']
        self.assertEqual([r['sample_id'] for r in gate.test_contexts(predictions, candidates, ids)], ['a', 'b'])
        gens = {('a', L2_IDS): ('Paris', None), ('b', L2_IDS): ('Lille', None)}
        rows = gate.test_rows(predictions, candidates, gens, ids)
        self.assertEqual([r['prediction'] for r in rows], ['Paris', 'Lille', 'Rome', 'Lima'])
        self.assertEqual((rows[0]['normalized_match'], rows[1]['normalized_match']), (1, 0))
        self.assertIs(rows[2], predictions[2])
        labels = {('a', 'Bern'): False, ('a', 'Paris'): True, ('b', 'Nice'): True, ('b', 'Lille'): False}
        self.assertEqual({k: v for k, v in gate.paired_rows(predictions, rows, labels).items() if k != 'mcnemar_p'},
                         {'changed': 2, 'fixes': 1, 'harms': 1, 'net': 0})
        self.assertEqual(gate.routed_ids(predictions), ['a', 'b', 'c'])

    def test_candidates_test_role_needs_the_registered_ids(self):
        with self.assertRaises(ValueError):
            l2_candidates.test_ids(None)
        self.assertEqual(l2_candidates.test_ids(ROOT / gate.load_registration()['test']['ids'])[:1],
                         json.loads((ROOT / 'configs/controller/bm25_gate_v1_test_ids.json').read_text(encoding='utf-8'))[:1])


if __name__ == '__main__':
    unittest.main()
