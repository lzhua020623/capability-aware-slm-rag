"""Adaptive multi-context selection: registration binding and the GPU context list."""
import gzip
import hashlib
import json
from pathlib import Path
import unittest

from scripts import adaptive_selection as ad
from scripts import l2_offline
from tests.test_l2_confirm import L2_IDS, cand_row, result

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'artifacts/controller/adaptive_selection_v1'
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def item(contexts, c1='Bonn', base='Bonn', routed=True, p_fail=0.5):
    """contexts: name -> (top5, answer); passages p0..p9 text 'p<i> about <word>'."""
    order = ad.load_registration()['contexts']['order']
    full = {n: {'top5': list(contexts.get(n, (['p0', 'p1', 'p2', 'p3', 'p4'], c1))[0]),
                'answer': contexts.get(n, (None, c1))[1]} for n in order}
    return {'c1': c1, 'base': base, 'routed': routed, 'p_fail': p_fail, 'bm25_brings': False,
            'c1_top5': ['p0', 'p1', 'p2', 'p3', 'p4'], 'contexts': full,
            'ce': {f'p{i}': float(i) for i in range(10)}, 'bge_rank': {f'p{i}': i + 1 for i in range(10)}, 'bm25_rank': {}}


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


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads((POLICY / 'policy.json').read_text(encoding='utf-8'))
        cls.cv = json.loads((POLICY / 'sweep_cv.json').read_text(encoding='utf-8'))

    def test_artifact_is_bound_to_the_registration(self):
        checks = json.loads((POLICY / 'checksums.json').read_text(encoding='utf-8'))
        self.assertEqual(set(checks), {p.name for p in POLICY.iterdir() if p.name != 'checksums.json'})
        for name, digest in checks.items():
            self.assertEqual(hashlib.sha256((POLICY / name).read_bytes()).hexdigest(), digest, name)
        reg = ad.load_registration()
        self.assertEqual(self.policy['registration_sha256'], sha(ad.REGISTRATION))
        self.assertEqual((self.policy['order'], self.policy['stages'], self.policy['features']),
                         (reg['contexts']['order'], [3, 5, 8], list(ad.FEATURES)))

    def test_chosen_policy_follows_the_registered_rule(self):
        table, cap = self.cv['table'], round(self.cv['fixed5_gens'], 2)
        eligible = [t for t in table if t['gens_per_item'] <= cap]
        best = max(t['net'] for t in eligible)
        chosen = min((t for t in eligible if t['net'] >= best - 3), key=lambda t: (t['gens'], -t['net']))
        self.assertEqual(self.policy['chosen'], {k: chosen[k] for k in ('theta', 'tau', 'decision')})
        self.assertEqual(self.policy['chosen'], {'theta': 0.0, 'tau': 0.8, 'decision': 'selector'})

    def test_selectors_refit_from_the_archived_training_rows(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        training = json.loads(gzip.decompress((POLICY / 'sweep_training.json.gz').read_bytes()))
        for k, model in self.policy['models'].items():
            X = np.array([r['x'] for r in training[k]]); y = np.array([r['label'] for r in training[k]])
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
            self.assertEqual(model['n_train'], len(y))
            np.testing.assert_allclose(pipe[-1].coef_[0], model['coef'], atol=1e-6)
            np.testing.assert_allclose(pipe.predict_proba(X[:50])[:, 1], [ad.probability(model, x) for x in X[:50]], atol=1e-9)


class SelectionTests(unittest.TestCase):
    def test_groups_votes_and_support(self):
        texts = {f'p{i}': f'p{i} about Rome' if i >= 5 else f'p{i} about Bonn' for i in range(10)}
        it = item({'bge20_bm250_rrf': (['p5', 'p6', 'p0', 'p1', 'p2'], 'Rome'), 'bge20_bm2520_ce_only': (['p5', 'p7', 'p0', 'p1', 'p2'], 'rome')})
        gs = {g['answer']: g['x'] for g in ad.groups(it, ad.load_registration()['contexts']['order'][:3], texts)}
        self.assertEqual(set(gs), {'Bonn', 'Rome'})
        self.assertEqual((gs['Rome'][0], gs['Bonn'][0], gs['Bonn'][3]), (2, 2, 1.0))
        self.assertEqual((gs['Rome'][4], gs['Rome'][5], gs['Bonn'][8]), (3, 7.0, 1.0))
        names = ad.load_registration()['contexts']['order'][:3]
        self.assertEqual(ad.decide(it, names, texts, 'vote'), 'Bonn')   # a tie keeps the B+BM25 answer
        model = {'mean': [0.0] * 15, 'scale': [1.0] * 15, 'coef': [0.0] * 5 + [1.0] + [0.0] * 9, 'intercept': 0.0}
        self.assertEqual(ad.decide(it, names, texts, 'selector', model), 'Rome')  # higher supporting CE score

    def test_policy_gate_stages_and_cost(self):
        texts = {f'p{i}': f'p{i} about Bonn' for i in range(10)}
        model = {'mean': [0.0] * 15, 'scale': [1.0] * 15, 'coef': [0.0] * 15, 'intercept': 0.0}
        models = {str(k): model for k in ad.STAGES}
        order = ad.load_registration()['contexts']['order']
        agree = item({'bge50_bm250_rrf': (['p5', 'p6', 'p7', 'p8', 'p9'], 'Bonn')}, routed=False, p_fail=0.1)
        self.assertEqual(ad.apply_policy(agree, {'theta': 0.5, 'tau': 0.8, 'decision': 'selector'}, models, texts, order), ('Bonn', 0, 0))
        self.assertEqual(ad.apply_policy(agree, {'theta': 0.0, 'tau': 0.8, 'decision': 'selector'}, models, texts, order), ('Bonn', 3, 1))
        split = item({n: (['p5', 'p6', 'p7', 'p8', f'p{9 - i % 2}'], f'A{i}') for i, n in enumerate(order)})
        answer, stage, cost = ad.apply_policy(split, {'theta': 0.0, 'tau': 0.8, 'decision': 'vote'}, models, texts, order)
        self.assertEqual((answer, stage), ('Bonn', 8))
        self.assertEqual(cost, ad.extra_generations(split, order))


class ItemsAndScoreTests(unittest.TestCase):
    def test_items_policies_and_summary(self):
        import tempfile
        results = [result('a', 'Bonn', 'Bonn', 'no_action'), result('b', 'Lyon', 'Nice', ad.P3, ('b5', 'b0', 'b1', 'b2', 'b3'), 'Nice')]
        for r, p in zip(results, (0.2, 0.9)):
            r['diagnosis'] = {'stage1_failure_probability': p}
        candidates = {'a': cand_row('a', 'x'), 'b': cand_row('b', 'x')}
        stored = {**ad.stored_generations(results, []), ('a', L2_IDS): ('Paris', None), ('b', L2_IDS): ('Paris', None)}
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / 'corpus.jsonl'
            corpus.write_text(''.join(json.dumps({'passage_id': p, 'text': f'{p} about Paris'}) + '\n'
                                      for p in [f'b{i}' for i in range(100)] + ['x']), encoding='utf-8')
            rows = ad.build_items(results, {'a': 'Bonn', 'b': 'Paris'}, candidates, stored, corpus,
                                  ad.load_registration()['contexts']['order'])
        a, b = rows
        self.assertEqual((a['routed'], b['routed'], a['query']), (False, True, 'q a'))
        self.assertEqual((a['stages']['8']['cost'], b['stages']['8']['cost']), (1, 0))
        self.assertEqual({g['answer'] for g in a['stages']['3']['groups']}, {'Bonn', 'Paris'})
        policy = json.loads((POLICY / 'policy.json').read_text(encoding='utf-8'))
        names = set(ad.policies(policy))
        self.assertEqual(len(names), 1 + len(policy['reported_points']) + 6)
        self.assertIn('fixed5_vote', names)
        outcomes = ad.answers_of(rows, policy)
        self.assertEqual(outcomes['fixed3_vote']['a'], ('Bonn', 3, 1))
        pairs = ad.needed_pairs(rows, outcomes)
        self.assertTrue(all(sid in ('a', 'b') for sid, _ in pairs))
        labels = {('a', 'Bonn'): False, ('a', 'Paris'): True, ('b', 'Paris'): True, ('b', 'Lyon'): False}
        report = ad.summarize(rows, {'x': {'a': ('Paris', 3, 1), 'b': ('Paris', 3, 0)}}, labels, {'a': ['Rome'], 'b': ['Paris']})
        self.assertEqual({k: report['x']['against_b_bm25'][k] for k in ('fixes', 'harms', 'net', 'unjudged')},
                         {'fixes': 1, 'harms': 0, 'net': 1, 'unjudged': 0})
        self.assertEqual(report['x']['against_c1']['net'], 2)
        self.assertEqual(report['x']['extra_generations_per_item'], 0.5)


if __name__ == '__main__':
    unittest.main()
