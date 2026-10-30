"""L2 confirm: choice loading, answers under route 1, generation reuse and the paired adoption test."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import l2_confirm as conf
from scripts.quality_sweep_offline import judge_id

BASE = ('b0', 'b1', 'b2', 'b3', 'b4')
CHOSEN = {'k_bge': 20, 'k_bm25': 20, 'fusion': 'ce_only'}


def cand_row(sid, lift=None):
    """BGE b0..b99 in order; BM25 hit 'x'. CE is flat unless `lift` raises one passage to the top."""
    bge = [[f'b{i}', 1.0 - i / 1000] for i in range(100)]
    ce = {p: 0.0 for p, _ in bge} | {'x': 0.0}
    if lift:
        ce[lift] = 5.0
    return {'sample_id': sid, 'bge': bge, 'bm25': [['x', 2.0]], 'ce': ce}


def result(sid, c1, final, action, trace_ids=None, gen=None, reason=None):
    native = {'module_trace': {'evidence': [{'passage_id': p} for p in trace_ids], 'generated_answer': gen,
                               'reject_reason': reason}} if trace_ids else {}
    return {'sample_id': sid, 'query': f'q {sid}', 'initial_answer': c1, 'final_answer': final,
            'initial_evidence': [{'passage_id': p} for p in BASE], 'controller_decision': {'action': action},
            'enhancement_result': {'metadata': {'native': native} if native else {}}}


L2_IDS = ('x', 'b0', 'b1', 'b2', 'b3')


def fixture():
    """s1 routed, route 1 kept C1 (no_change), L2 fixes. s2 not routed, L2 would fix. s3 routed, route 1
    fixed with a b5 Top-5, L2 harms. s4 routed, L2 rejected (keeps C1, as route 1 did)."""
    results = [result('s1', 'London', 'London', conf.P3, BASE),
               result('s2', 'Oslo', 'Oslo', 'no_action'),
               result('s3', 'Lyon', 'Nice', conf.P3, ('b5', 'b0', 'b1', 'b2', 'b3'), ' Nice '),
               result('s4', 'Bern', 'Bern', conf.P3, BASE)]
    candidates = {'s1': cand_row('s1', 'x'), 's2': cand_row('s2', 'x'), 's3': cand_row('s3', 'x'), 's4': cand_row('s4', 'x')}
    generations = {('s1', L2_IDS): ('Paris', None), ('s2', L2_IDS): ('Rome', None), ('s3', L2_IDS): ('Lille', None),
                   ('s4', L2_IDS): ('I do not know', 'refusal_or_empty'),
                   ('s3', ('b5', 'b0', 'b1', 'b2', 'b3')): (' Nice ', None)}
    gold = {'s1': ['Paris'], 's2': ['Rome'], 's3': ['Nice'], 's4': ['Bern']}
    return results, candidates, generations, gold


def report(results, chosen='bge20_bm2520_ce_only', complete=True):
    return {'complete': complete, 'chosen': {**CHOSEN, 'name': chosen} if complete else None, 'results': results}


def setting(k_bge, k_bm25, fusion, net):
    return {'k_bge': k_bge, 'k_bm25': k_bm25, 'fusion': fusion, 'name': f'bge{k_bge}_bm25{k_bm25}_{fusion}', 'net': net}


class L2ConfirmTests(unittest.TestCase):
    def write(self, path, obj):
        path.write_text(json.dumps(obj), encoding='utf-8')
        return path

    def test_choice_must_be_complete_and_follow_the_rule(self):
        rows = [setting(20, 20, 'ce_only', 139), setting(20, 50, 'ce_only', 132), setting(20, 0, 'rrf', 99)]
        with tempfile.TemporaryDirectory() as tmp:
            good = self.write(Path(tmp) / 'good.json', report(rows))
            self.assertEqual(conf.load_choice(good), CHOSEN)
            with self.assertRaises(ValueError):
                conf.load_choice(self.write(Path(tmp) / 'open.json', report(rows, complete=False)))
            with self.assertRaises(ValueError):
                conf.load_choice(self.write(Path(tmp) / 'wrong.json', report(rows, chosen='bge20_bm2550_ce_only')))

    def test_answers_replace_p3_only_on_route_1_items(self):
        results, candidates, generations, _ = fixture()
        finals = conf.answers(results, candidates, generations, CHOSEN)
        self.assertEqual(finals['route_1'], {'s1': 'London', 's2': 'Oslo', 's3': 'Nice', 's4': 'Bern'})
        self.assertEqual(finals['route_1_l2'], {'s1': 'Paris', 's2': 'Oslo', 's3': 'Lille', 's4': 'Bern'})
        self.assertEqual(finals['l2'], {'s1': 'Paris', 's2': 'Rome', 's3': 'Lille', 's4': 'Bern'})

    def test_contexts_skip_c1_top5_and_stored_route_1_generations(self):
        results, candidates, _, _ = fixture()
        stored = {('s3', L2_IDS): ('Lille', None)}
        rows = conf.needed_contexts(results, candidates, stored, CHOSEN)
        self.assertEqual([r['sample_id'] for r in rows], ['s1', 's2', 's4'])
        self.assertEqual(rows[0]['passage_ids'], list(L2_IDS))
        self.assertEqual(conf.needed_contexts(results, candidates, stored, conf.CONTROL), [])

    def test_control_check_compares_with_route_1_top5(self):
        results, candidates, _, _ = fixture()
        self.assertEqual(conf.control_check(results, candidates), {'routed': 3, 'same_top5': 2})

    def test_paired_counts_only_changed_answers(self):
        results, candidates, generations, gold = fixture()
        finals = conf.answers(results, candidates, generations, CHOSEN)
        judged = {judge_id('s1', 'London'): False, judge_id('s3', 'Lyon'): False, judge_id('s3', 'Lille'): False}
        out = conf.paired(finals['route_1'], finals['route_1_l2'], gold, judged)
        self.assertEqual({k: out[k] for k in ('changed', 'fixes', 'harms', 'net', 'unjudged')},
                         {'changed': 2, 'fixes': 1, 'harms': 1, 'net': 0, 'unjudged': 0})
        self.assertEqual(out['mcnemar_p'], 1.0)
        self.assertEqual(conf.paired(finals['c1'], finals['l2'], gold, {})['unjudged'], 3)

    def test_score_end_to_end(self):
        results, candidates, generations, gold = fixture()
        rows = [setting(20, 20, 'ce_only', 139), setting(20, 0, 'rrf', 99)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def lines(path, items):
                path.write_text(''.join(json.dumps(i) + '\n' for i in items), encoding='utf-8')
                return path
            gens = [{'sample_id': s, 'passage_ids': list(ids), 'generated_answer': a, 'reject_reason': r}
                    for (s, ids), (a, r) in generations.items() if ids == L2_IDS]
            judged = [{'sample_id': judge_id(s, a), 'judge_status': 'llm_ok', 'auto_correct': False, 'judge_label': 'INCORRECT'}
                      for s, a in (('s1', 'London'), ('s2', 'Oslo'), ('s3', 'Lyon'), ('s3', 'Lille'))]
            args = ['--results', str(lines(tmp / 'r.jsonl', results)),
                    '--candidates', str(lines(tmp / 'c.jsonl', candidates.values())),
                    '--choice', str(self.write(tmp / 'choice.json', report(rows))),
                    '--generations', str(lines(tmp / 'g.jsonl', gens)),
                    '--references', str(lines(tmp / 'ref.jsonl', [{'sample_id': s, 'gold': g} for s, g in gold.items()])),
                    '--judged', str(lines(tmp / 'j.jsonl', judged))]
            with mock.patch.object(conf, 'role_ids', return_value=['s1', 's2', 's3', 's4']), \
                 mock.patch('sys.stdout'):
                conf.main(['score', *args, '--output', str(tmp / 'out.json')])
                out = json.loads((tmp / 'out.json').read_text(encoding='utf-8'))
                self.assertTrue(out['complete'])
                self.assertEqual((out['adoption']['net'], out['adoption']['passed']), (0, False))
                self.assertEqual(out['against_c1']['route_1']['net'], 1)
                self.assertEqual(out['against_c1']['l2']['fixes'], 2)
                self.assertEqual(out['control_check'], {'routed': 3, 'same_top5': 2})
                conf.main(['judge-inputs', *args, '--output-dir', str(tmp / 'ji'), '--shards', '2'])
                pending = [json.loads(l) for p in sorted((tmp / 'ji').glob('*.jsonl')) for l in p.read_text().splitlines()]
                self.assertEqual(sorted(r['sample_id'] for r in pending),
                                 sorted(judge_id(s, a) for s, a in (('s1', 'Paris'), ('s2', 'Rome'), ('s3', 'Nice'))))
            with mock.patch.object(conf, 'role_ids', return_value=['s2', 's1', 's3', 's4']):
                with self.assertRaises(ValueError):
                    conf.main(['score', *args, '--output', str(tmp / 'out2.json')])


if __name__ == '__main__':
    unittest.main()
