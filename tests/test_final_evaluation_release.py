"""Replay the final NQ evaluation release: archive integrity, routing rows, the judge batch, latency sample and report."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts import adaptive_selection as ad
from scripts import final_evaluation as fe
from scripts.quality_sweep_offline import judge_id
from scripts.run_quality_eval import expected_ids

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / fe.RELEASE
RAW = {name: Path(os.environ.get(env, ROOT / default)) for name, env, default in (
    ('test_results', 'ROUTE_1_TEST_RESULTS', 'data/processed/quality_sweep_v1/test/results.jsonl.gz'),
    ('test_candidates', 'BM25_TEST_CANDIDATES', 'data/processed/quality_sweep_v1/bm25_test/candidates.jsonl'),
    ('test_unrouted_candidates', 'ADAPTIVE_TEST_CANDIDATES', 'data/processed/quality_sweep_v1/adaptive_test/candidates_unrouted.jsonl'))}
CONFIRM_PRODUCT = Path(os.environ.get('ADAPTIVE_PRODUCT_CONFIRM_RESULTS',
                                      ROOT / 'data/processed/quality_sweep_v1/adaptive_product_confirm/results.jsonl.gz'))
sha = lambda data: hashlib.sha256(data).hexdigest()


class FinalEvaluationReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members = fe.untar(RELEASE / 'results.tar.gz')
        cls.provenance = json.loads((RELEASE / 'provenance.json').read_text(encoding='utf-8'))
        cls.report = json.loads((RELEASE / 'report.json').read_text(encoding='utf-8'))
        cls.rows = fe.jsonl(cls.members['test_p3_rows.jsonl'])
        cls.items = fe.jsonl(fe.untar(ROOT / fe.AD_RELEASE)['test/items.jsonl'])
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_archived_byte_is_bound_without_secrets(self):
        checks = json.loads((RELEASE / 'checksums.json').read_text(encoding='utf-8'))
        actual = {p.relative_to(RELEASE).as_posix() for p in RELEASE.rglob('*') if p.is_file() and p.name != 'checksums.json'}
        self.assertEqual(set(checks), actual)
        for name, expected in checks.items():
            self.assertEqual(sha((RELEASE / name).read_bytes()), expected, name)
        archive = self.provenance['curated_archive']
        self.assertEqual((sha((RELEASE / 'results.tar.gz').read_bytes()), len(self.members)), (archive['sha256'], archive['files']))
        for name, payload in self.members.items():
            self.assertFalse(Path(name).is_absolute() or '..' in Path(name).parts)
            for secret in (b'AZURE_API_KEY=', b'AZURE_OPENAI_API_KEY=', b'Authorization: Bearer ', b'password='):
                self.assertNotIn(secret, payload, name)

    def test_routing_rows_bind_the_test_items_and_route_1(self):
        from scripts import bm25_gate as gate
        preds, _, _ = gate.route_1_test(ROOT / gate.ROUTE_1_TEST, self.tmp.name)
        route_1 = {r['sample_id']: r['prediction'] for r in preds}
        self.assertEqual([r['sample_id'] for r in self.rows], [it['sample_id'] for it in self.items])
        for row, it in zip(self.rows, self.items):
            self.assertEqual((row['c1'], row['b_bm25'], row['routed'], row['p_fail']), (it['c1'], it['base'], it['routed'], it['p_fail']))
            if row['routed']:
                self.assertEqual((row['p3'], row['bm25_all']), (route_1[row['sample_id']], row['b_bm25']))
            else:
                self.assertEqual(route_1[row['sample_id']], row['c1'])

    def test_judge_batch_is_exactly_the_unlabelled_routing_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            preds, _, labels, _ = fe.load_test(ROOT, tmp)
        gold = {r['sample_id']: r['gold'] for r in preds}
        wanted = sorted(judge_id(sid, answer) for sid, answer in fe.unlabelled_pairs(self.rows, labels, gold))
        inputs = [r for i in range(4) for r in fe.jsonl(self.members[f'judge/inputs/shard_{i}.jsonl'])]
        self.assertEqual(sorted(r['sample_id'] for r in inputs), wanted)
        self.assertEqual(len(wanted), 370)
        outputs = [r for i in range(4) for r in fe.jsonl(self.members[f'judge/outputs/shard_{i}/judge_results.jsonl'])]
        self.assertEqual(sorted(r['sample_id'] for r in outputs), wanted)
        self.assertEqual({r['judge_status'] for r in outputs}, {'llm_ok'})

    def test_latency_sample_and_rows(self):
        ids = json.loads(self.members['latency/ids.json'])
        self.assertEqual(ids, fe.latency_ids(expected_ids('confirm')))
        rows = fe.jsonl(self.members['latency/latency.jsonl'])
        self.assertEqual([r['sample_id'] for r in rows], ids['timed'])
        manifest = json.loads(self.members['latency/manifest.json'])
        self.assertEqual((manifest['state'], manifest['n'], manifest['warmup']), ('completed', 300, 10))
        self.assertEqual(manifest['registration'], 'configs/controller/quality_adaptive_v1.json')
        for r in rows:
            parts = r['c1_s'] + r.get('diagnosis_classifier_s', 0) + r.get('diagnosis_features_s', 0) + r.get('controller_s', 0) + r['answer_selection_s']
            self.assertLessEqual(parts, r['total_s'] + 1e-6)
            self.assertGreater(parts, 0.95 * r['total_s'])

    @unittest.skipUnless(CONFIRM_PRODUCT.exists(), 'the confirm product run is kept locally only')
    def test_latency_run_answers_equal_the_confirm_product_run(self):
        rows = {r['sample_id']: r for r in fe.jsonl(self.members['latency/latency.jsonl'])}
        seen = 0
        with gzip.open(CONFIRM_PRODUCT, 'rt', encoding='utf-8') as handle:
            for line in handle:
                r = json.loads(line)
                if r['sample_id'] in rows:
                    t = rows[r['sample_id']]
                    self.assertEqual((t['c1'], t['controller_answer'], t['final_answer']),
                                     (r['initial_answer'], r['enhancement_result']['final_answer'], r['final_answer']))
                    seen += 1
        self.assertEqual(seen, 300)

    @unittest.skipUnless(all(p.exists() for p in RAW.values()), 'raw Test candidates and route-1 results are kept locally only')
    def test_routing_rows_rebuild_from_the_raw_inputs(self):
        from scripts import score_quality_eval as scorer
        results = list(ad.read_results(RAW['test_results']))
        candidates = {r['sample_id']: r for p in (RAW['test_candidates'], RAW['test_unrouted_candidates']) for r in scorer.read_jsonl(p)}
        bm25, rel = fe.untar(ROOT / fe.BM25_RELEASE), fe.untar(ROOT / fe.AD_RELEASE)
        files = [fe.write(self.tmp.name, 'bm25_gen', bm25['generations/generations.jsonl'])]
        files += [fe.write(self.tmp.name, f'ad_{p}', rel[f'test/generations/{p}/generations.jsonl']) for p in ('routed', 'unrouted')]
        items = {it['sample_id']: it for it in self.items}
        self.assertEqual(fe.test_p3_rows(results, candidates, ad.stored_generations(results, files), items), self.rows)

    def test_report_replays_and_its_headline_numbers(self):
        report = fe.report(ROOT)
        self.assertEqual(report, self.report)
        t = report['test']
        self.assertEqual({k: round(t['metrics'][k]['judge'], 2) for k in ('rag', 'c1', 'b', 'b_bm25', 'product')},
                         {'rag': 57.5, 'c1': 57.59, 'b': 59.67, 'b_bm25': 60.62, 'product': 62.76})
        self.assertEqual(round(t['metrics']['product']['em'], 2), 34.07)
        main = t['paired']['product-b_bm25']
        self.assertEqual((main['judge']['fixes'], main['judge']['harms'], main['em']['pp']), (521, 283, 1.9435))
        self.assertGreater(main['em']['ci95'][0], 0)
        for k in (3, 5, 8):
            self.assertGreater(t['policies']['paired'][f'fixed{k}_selector-fixed{k}_vote']['judge']['ci95'][0], 0)
        sweep, test = report['routing']['sweep'], report['routing']['test']
        self.assertEqual((sweep['net_vs_c1'], test['net_vs_c1']), ({'P60_routed': 99, 'all_to_P3': 84, 'unrouted_if_sent': -15},
                                                                   {'P60_routed': 232, 'all_to_P3': 224, 'unrouted_if_sent': -8}))
        self.assertEqual((sweep['random_same_size']['share_at_or_above_P60'], test['random_same_size']['share_at_or_above_P60']), (0.0, 0.0))
        self.assertLess(test['auc_p_fail']['p3_fixes_among_c1_wrong'], 0.55)
        self.assertEqual(report['latency']['n'], 300)


class AucTests(unittest.TestCase):
    def test_matches_pairwise_definition_with_ties(self):
        from scripts.recoverability_oracle import auc as pairwise
        scores, positive = [0.1, 0.4, 0.4, 0.8, 0.2, 0.4], [False, True, False, True, False, True]
        self.assertAlmostEqual(fe.auc(scores, positive),
                               pairwise([s for s, p in zip(scores, positive) if p], [s for s, p in zip(scores, positive) if not p]))


if __name__ == '__main__':
    unittest.main()
