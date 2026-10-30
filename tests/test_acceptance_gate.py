"""Acceptance gate: registration binding, the keep-C1 rule, its scope and the confirm direction check."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import acceptance_gate as gate
from scripts import l2_confirm
from scripts.quality_sweep_offline import judge_id
from tests.test_l2_confirm import L2_IDS, cand_row, result

ROOT = Path(__file__).resolve().parents[1]
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
OTHER = ('b5', 'b0', 'b1', 'b2', 'b3')


def fixture():
    """a: route 1 changed C1, voter returns to C1 (case and article differ) -> C1. b: voter agrees with route 1.
    c: voter gives a third answer. d: voter rejected (counts as C1) -> C1. e: route 1 kept C1. f: not routed."""
    results = [result('a', 'Bonn', 'Berlin', l2_confirm.P3, OTHER, 'Berlin'),
               result('b', 'Lyon', 'Nice', l2_confirm.P3, OTHER, 'Nice'),
               result('c', 'Oslo', 'Rome', l2_confirm.P3, OTHER, 'Rome'),
               result('d', 'Bern', 'Basel', l2_confirm.P3, OTHER, 'Basel'),
               result('e', 'Kyiv', 'Kyiv', l2_confirm.P3, OTHER, 'I do not know', 'refusal_or_empty'),
               result('f', 'Lima', 'Lima', 'no_action')]
    candidates = {r['sample_id']: cand_row(r['sample_id'], 'x') for r in results}
    generations = {('a', L2_IDS): ('the bonn', None), ('b', L2_IDS): ('Nice', None), ('c', L2_IDS): ('Paris', None),
                   ('d', L2_IDS): ('Zurich', 'not_grounded_in_selected_evidence')}
    gold = {'a': ['Bonn'], 'b': ['Nice'], 'c': ['Paris'], 'd': ['Basel'], 'e': ['Kyiv'], 'f': ['Lima']}
    return results, candidates, generations, gold


class AcceptanceGateTests(unittest.TestCase):
    def test_registration_binds_route_1_and_the_l2_round(self):
        registration = gate.load_registration()
        self.assertEqual(registration['base']['sha256'], sha(registration['base']['registration']))
        self.assertEqual(registration['l2_round']['sha256'], sha(registration['l2_round']['registration']))
        self.assertIn('bge20_bm2520_ce_only', registration['gate']['voter'])
        self.assertIn('p < 0.05', registration['test']['adoption'])

    def test_rule_keeps_c1_only_when_the_voter_returns_to_it(self):
        self.assertTrue(gate.keep_c1('Bonn', 'Berlin', 'the bonn'))
        self.assertFalse(gate.keep_c1('Lyon', 'Nice', 'nice'))
        self.assertFalse(gate.keep_c1('Oslo', 'Rome', 'Paris'))
        self.assertFalse(gate.keep_c1('Paris', 'paris.', 'Paris'))

    def test_confirm_answers_and_gate_scope(self):
        results, candidates, generations, _ = fixture()
        c1, route_1, voter = gate.confirm_answers(results, candidates, generations)
        self.assertEqual(set(voter), {'a', 'b', 'c', 'd'})
        self.assertEqual(voter['d'], 'Bern')
        after = gate.gated(c1, route_1, voter)
        self.assertEqual(after, {'a': 'Bonn', 'b': 'Nice', 'c': 'Rome', 'd': 'Bern', 'e': 'Kyiv', 'f': 'Lima'})

    def test_confirm_score_end_to_end(self):
        results, candidates, generations, gold = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def lines(name, items):
                (tmp / name).write_text(''.join(json.dumps(i) + '\n' for i in items), encoding='utf-8')
                return str(tmp / name)
            gens = [{'sample_id': s, 'passage_ids': list(ids), 'generated_answer': a, 'reject_reason': r}
                    for (s, ids), (a, r) in generations.items()]
            judged = [{'sample_id': judge_id(s, a), 'judge_status': 'llm_ok', 'auto_correct': False, 'judge_label': 'INCORRECT'}
                      for s, a in (('a', 'Berlin'), ('b', 'Lyon'), ('c', 'Oslo'), ('c', 'Rome'), ('d', 'Bern'))]
            with mock.patch.object(l2_confirm, 'role_ids', return_value=[r['sample_id'] for r in results]), \
                 mock.patch('sys.stdout'):
                gate.main(['score', '--role', 'confirm', '--results', lines('r.jsonl', results),
                           '--candidates', lines('c.jsonl', candidates.values()), '--generations', lines('g.jsonl', gens),
                           '--references', lines('ref.jsonl', [{'sample_id': s, 'gold': g} for s, g in gold.items()]),
                           '--judged', lines('j.jsonl', judged), '--output', str(tmp / 'out.json')])
            out = json.loads((tmp / 'out.json').read_text(encoding='utf-8'))
        self.assertEqual((out['in_scope'], out['kept_c1'], out['complete']), (4, 2, True))
        self.assertEqual({k: out['against_route_1'][k] for k in ('changed', 'fixes', 'harms', 'net')},
                         {'changed': 2, 'fixes': 1, 'harms': 1, 'net': 0})
        self.assertEqual(out['against_c1']['route_1']['net'], 1)
        self.assertFalse(out['proceed_to_test'])


if __name__ == '__main__':
    unittest.main()
