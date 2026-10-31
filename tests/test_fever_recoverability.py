"""FEVER recoverability research set: registration, evidence resolution and the eligibility-ordered sample."""
import hashlib
import json
from pathlib import Path
import unittest

from scripts import fever_recoverability as fr

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'artifacts/controller/fever_recoverability_v1/sample'


def train_row(i, label, pairs, claim=None):
    return {'id': i, 'label': label, 'claim': claim or f'claim number {i}',
            'evidence': [[[0, 0, p, s] for p, s in group] for group in pairs]}


class RegistrationTests(unittest.TestCase):
    def test_registration_fixes_the_set_and_conditions(self):
        reg = fr.load_registration()
        spec = reg['sample']
        self.assertEqual((spec['per_label'], spec['development_per_label'], spec['held_out_per_label']), (1000, 800, 200))
        self.assertEqual(set(reg['conditions']) - {'truncation'}, {'C0', 'C1', 'L1', 'L3', 'L4'})
        self.assertEqual(reg['team_test']['n'], 13332)
        self.assertIn('not a fever product evaluation', reg['role'].lower())


class EvidenceTests(unittest.TestCase):
    def test_union_of_groups_in_first_seen_order(self):
        evidence = [[[1, 2, 'A', 3], [1, 2, 'B', 0]], [[1, 3, 'A', 3]], [[1, 4, None, None]]]
        self.assertEqual(fr.evidence_sentences(evidence), [('A', 3), ('B', 0)])

    def test_lines_skip_empty_sentences(self):
        self.assertEqual(fr.parse_lines('0\tFirst .\tlink\n1\t\n2\tThird .'), {0: 'First .', 2: 'Third .'})

    def test_claim_normalization_and_jaccard(self):
        self.assertEqual(fr.norm_claim('The YOG -- a Sport event!'), 'the yog a sport event')
        self.assertEqual(fr.jaccard('a b c d', 'a b c e'), 0.6)


class SampleTests(unittest.TestCase):
    def test_each_eligibility_rule(self):
        pages = {'P': (True, {0: 'p zero .', 1: 'p one .'}), 'E': (False, {0: 'e .'})}
        test_norm, by_page = {'test claim here'}, {'P': ['a near test claim here now']}
        check = lambda pairs, claim, seen=(): fr.ineligible(pairs, claim, pages, test_norm, by_page, set(seen), 0.8)
        self.assertEqual(check([], 'x'), 'no_evidence')
        self.assertEqual(check([('P', 5)], 'x'), 'unresolved_sentence')
        self.assertEqual(check([('Z', 0)], 'x'), 'unresolved_sentence')
        self.assertEqual(check([('E', 0)], 'x'), 'page_without_text')
        self.assertEqual(check([('P', 0)], 'test claim here'), 'equals_test_claim')
        self.assertEqual(check([('P', 0)], 'near test claim here now'), 'near_test_claim')
        self.assertIsNone(check([('E', 0)][:0] + [('P', 1)], 'near test claim elsewhere'))
        self.assertEqual(check([('P', 0)], 'seen claim', ['seen claim']), 'repeats_sampled_claim')
        self.assertIsNone(check([('P', 0), ('P', 1)], 'fresh claim'))

    def test_order_roles_and_evidence(self):
        reg = {'sample': {'seed': 's', 'per_label': 2, 'development_per_label': 1, 'near_duplicate_jaccard': 0.8, 'batch': 3}}
        pages = {'P': (True, {0: 'p zero .', 1: 'p one .'}), 'Q': (True, {0: 'q zero .'})}
        train = [train_row(i, 'SUPPORTS' if i % 2 else 'REFUTES', [[('P', 0)], [('P', 1)]] if i == 4 else [[('Q', 0)]])
                 for i in range(1, 9)] + [train_row(9, 'NOT ENOUGH INFO', [[('P', 0)]])]
        items, excluded = fr.sample(train, [], lambda ids: {k: pages[k] for k in ids if k in pages}, reg)
        order = sorted((str(r['id']) for r in train if r['label'] != 'NOT ENOUGH INFO'), key=lambda s: fr.order_key('s', s))
        expected = [s for s in order if int(s) % 2][:2] + [s for s in order if not int(s) % 2][:2]
        self.assertEqual(sorted(r['sample_id'] for r in items), sorted(expected))
        self.assertEqual([r['sample_id'] for r in items], [s for s in order if s in expected])
        for label in fr.LABELS:
            roles = [r['role'] for r in items if r['label'] == label]
            self.assertEqual(sorted(roles), ['development', 'held_out'])
        four = [r for r in items if r['sample_id'] == '4']
        if four:
            self.assertEqual((four[0]['gold_evidence_text'], four[0]['gold_page_ids'], four[0]['evidence_groups']),
                             (['p zero .', 'p one .'], ['P'], 2))
        self.assertEqual(sum(excluded.values()), 0)


class ArchivedSampleTests(unittest.TestCase):
    def test_archived_sample_counts_and_binding(self):
        manifest = json.loads((SAMPLE / 'manifest.json').read_text(encoding='utf-8'))
        rows = fr.read_jsonl(SAMPLE / 'items.jsonl')
        self.assertEqual(manifest['items_sha256'], hashlib.sha256((SAMPLE / 'items.jsonl').read_bytes()).hexdigest())
        self.assertEqual(manifest['registration_sha256'], hashlib.sha256((ROOT / fr.REGISTRATION).read_bytes()).hexdigest())
        self.assertEqual(manifest['counts'], {'development': {'SUPPORTS': 800, 'REFUTES': 800}, 'held_out': {'SUPPORTS': 200, 'REFUTES': 200}})
        self.assertEqual(manifest['inputs']['team_test_sha256'], '09072a8314e57b5f2557366ae7a51e760e3a0a40220709e998aac82fcf28c8f2')
        self.assertEqual(len({r['sample_id'] for r in rows}), 2000)
        self.assertEqual(len({fr.norm_claim(r['claim']) for r in rows}), 2000)
        self.assertTrue(all(r['gold_evidence_text'] and r['gold_page_ids'] for r in rows))
        self.assertEqual([r['sample_id'] for r in rows], sorted((r['sample_id'] for r in rows),
                                                               key=lambda s: fr.order_key(fr.load_registration()['sample']['seed'], s)))


class RunAndScoreTests(unittest.TestCase):
    row = {'sample_id': '7', 'claim': 'X is Y.', 'gold_evidence_text': ['X is Y .', 'Y is Z .'], 'gold_page_ids': ['X', 'Y']}
    hits = [{'rank': i + 1, 'passage_id': p, 'text': f'{p} page text', 'score': 1.0 - i / 10} for i, p in enumerate('AXBCD')]

    def test_oracle_contexts_put_the_gold_block_first_or_alone(self):
        l3, l4 = fr.oracle_passages(self.row['gold_evidence_text'], self.hits, self.row['gold_page_ids'])
        self.assertEqual([p['passage_id'] for p in l3], ['gold_evidence', 'A', 'B', 'C', 'D'])
        self.assertEqual(l3[0]['text'], 'X is Y . Y is Z .')
        self.assertEqual(l4, [l3[0]])

    def test_run_item_uses_one_prompt_per_condition(self):
        from src.compensation.answer_repair import Generation
        from src.generation.prompts import build_fever_rag_prompt
        prompts = []

        def generate(prompt):
            prompts.append(prompt)
            return Generation('SUPPORTS' if 'Evidence' in prompt else 'refutes.', 0.5, input_tokens=len(prompt))
        p3 = lambda row, hits, c1: {'raw': 'REFUTES', 'label': 'REFUTES', 'passage_ids': ['X', 'Y', 'A', 'B', 'C'], 'seconds': 1.0}
        out = fr.run_item(self.row, generate, lambda claim, k: self.hits[:k], p3)
        self.assertEqual({c: out[c]['label'] for c in fr.CONDITIONS},
                         {'C0': 'REFUTES', 'C1': 'SUPPORTS', 'L1': 'REFUTES', 'L3': 'SUPPORTS', 'L4': 'SUPPORTS'})
        self.assertEqual(prompts[0], fr.c0_prompt('X is Y.'))
        self.assertTrue(prompts[0].startswith('Determine whether the claim is supported or refuted.'))
        self.assertEqual(prompts[1], build_fever_rag_prompt('X is Y.', self.hits))
        self.assertEqual(out['C1']['passage_ids'], list('AXBCD'))
        self.assertEqual(len(prompts), 4)
        self.assertEqual(fr.gpu_rows([{**self.row, 'label': 'SUPPORTS', 'role': 'development'}]), [self.row])

    def test_score_counts_rfr_flips_and_coverage(self):
        items = [{'sample_id': str(i), 'label': 'SUPPORTS' if i < 2 else 'REFUTES', 'gold_page_ids': ['X'],
                  'role': 'development' if i < 3 else 'held_out'} for i in range(4)]
        res = lambda c0, c1, l1, l3, l4, l1_ids=('X',): {
            'C0': {'label': c0, 'passage_ids': None, 'seconds': 1}, 'C1': {'label': c1, 'passage_ids': ['A'], 'seconds': 1},
            'L1': {'label': l1, 'passage_ids': list(l1_ids), 'seconds': 2}, 'L3': {'label': l3, 'passage_ids': [], 'seconds': 1},
            'L4': {'label': l4, 'passage_ids': [], 'seconds': 1}}
        results = [{'sample_id': '0', **res('SUPPORTS', 'REFUTES', 'SUPPORTS', 'SUPPORTS', 'SUPPORTS')},
                   {'sample_id': '1', **res('SUPPORTS', 'SUPPORTS', 'REFUTES', 'SUPPORTS', 'UNKNOWN', ('B',))},
                   {'sample_id': '2', **res('SUPPORTS', 'SUPPORTS', 'REFUTES', 'REFUTES', 'REFUTES')},
                   {'sample_id': '3', **res('REFUTES', 'REFUTES', 'REFUTES', 'REFUTES', 'REFUTES')}]
        report = fr.score(items, results)
        dev = report['development']
        self.assertEqual((dev['n'], dev['c1_wrong']), (3, 2))
        self.assertEqual(dev['rfr'], {'C0': 0.5, 'L1': 1.0, 'L3': 1.0, 'L4': 1.0})
        self.assertEqual(dev['flips_vs_c1']['L4'], {'wrong_to_correct': 2, 'correct_to_wrong': 1})
        self.assertEqual(dev['unparsed']['L4'], 1)
        self.assertEqual(dev['gold_pages_covered'], {'C1_top5': 0, 'L1_top5': 2})
        self.assertEqual(report['all']['n'], 4)
        self.assertEqual(report['held_out']['accuracy']['C1'], 1.0)
        with self.assertRaises(ValueError):
            fr.score(items, results[:3])


if __name__ == '__main__':
    unittest.main()
