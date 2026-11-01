"""The current product's Test outputs in the team baseline format: format, headline numbers, answers, labels, evidence rule."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from baselines._lib.judge_io import summarize_results
from baselines.run_product_evaluation import aggregate_product_metrics
from scripts import adaptive_selection as ad
from scripts import export_product_v2_outputs as ex
from scripts import final_evaluation as fe

ROOT = Path(__file__).resolve().parents[1]
PRED, JUDGE = ROOT / ex.PRED_DIR, ROOT / ex.JUDGE_DIR
TEAM = ROOT / 'baselines/outputs/adaptive_product_test/predictions.jsonl'
TEAM_JUDGE = ROOT / 'baselines/outputs/judge/adaptive_product_full/judge_results.jsonl'
CONFIRM_PRODUCT = ROOT / 'data/processed/quality_sweep_v1/adaptive_product_confirm/results.jsonl.gz'
lines = lambda path: [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
load = lambda path: json.loads(path.read_text(encoding='utf-8'))


class ProductV2OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.judge = lines(PRED / 'predictions.jsonl'), lines(JUDGE / 'judge_results.jsonl')
        cls.metrics, cls.summary = load(PRED / 'metrics.json'), load(JUDGE / 'judge_summary.json')

    def test_rows_follow_the_team_format_and_test_items(self):
        team = lines(TEAM)
        self.assertEqual(len(self.rows), 11114)
        self.assertEqual([list(r) for r in self.rows[:1]], [list(team[0])])
        self.assertEqual({tuple(r) for r in self.rows}, {ex.ROW_KEYS})
        self.assertEqual([(r['sample_id'], r['query'], r['gold']) for r in self.rows],
                         [(r['sample_id'], r['query'], r['gold']) for r in team])
        self.assertEqual({tuple(r) for r in self.judge}, {tuple(lines(TEAM_JUDGE)[0])})
        self.assertEqual([(j['sample_id'], j['prediction'], j['em'], j['f1'], j['normalized_match']) for j in self.judge],
                         [(r['sample_id'], r['prediction'], r['em'], r['f1'], r['normalized_match']) for r in self.rows])

    def test_metrics_and_summary_recompute_from_the_rows(self):
        expected = aggregate_product_metrics(self.rows) | {'eval_split': 'test', 'rag_prompt_style': 'context_fallback',
                                                            'smoke': False}
        self.assertEqual(self.metrics, expected)
        self.assertEqual({k: self.summary[k] for k in summarize_results(self.judge)}, summarize_results(self.judge))
        config = load(PRED / 'run_config.json')
        self.assertEqual((config['status'], config['config_path'], config['eval_n']),
                         ('REPLAYED_FROM_ARCHIVED_TEST_RUN', ex.CONFIG, 11114))
        for path, sha in config['replay']['archives_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), sha, path)

    def test_headline_numbers_equal_the_final_evaluation(self):
        product = load(ROOT / fe.RELEASE / 'report.json')['test']['metrics']['product']
        for key in ('em', 'f1', 'normalized_accuracy'):
            self.assertAlmostEqual(100 * self.metrics[key], product[key], places=9)
        self.assertAlmostEqual(100 * self.summary['judge_assisted_accuracy'], product['judge'], places=9)
        self.assertEqual((round(100 * self.metrics['em'], 2), round(100 * self.metrics['f1'], 2),
                          round(100 * self.summary['judge_assisted_accuracy'], 2)), (34.07, 49.55, 62.76))

    def test_predictions_are_the_archived_product_answers_with_their_labels(self):
        policy = load(ROOT / ad.POLICY)
        chosen = ad.answers_of(fe.jsonl(fe.untar(ROOT / fe.AD_RELEASE)['test/items.jsonl']), policy)['chosen']
        self.assertEqual({r['sample_id']: r['prediction'] for r in self.rows}, {s: v[0] for s, v in chosen.items()})
        members = fe.untar(ROOT / fe.RELEASE / 'results.tar.gz')
        with tempfile.TemporaryDirectory() as tmp:
            preds, _, labels, _ = fe.load_test(ROOT, tmp, fe.batch_files(tmp, 'p60', members, 'judge/outputs'))
        gold = {r['sample_id']: r['gold'] for r in preds}
        for j in self.judge:
            correct = bool(j['auto_correct']) or j['judge_label'] == 'CORRECT'
            self.assertEqual(correct, bool(ad.label_of(labels, gold, j['sample_id'], j['prediction'])), j['sample_id'])
            self.assertEqual(j['auto_correct'], j['judge_status'] == 'auto_correct')

    @unittest.skipUnless(CONFIRM_PRODUCT.exists(), 'the confirm production run is kept locally only')
    def test_evidence_rule_equals_the_confirm_production_run(self):
        order, seen = load(ROOT / ad.POLICY)['order'], 0
        with gzip.open(CONFIRM_PRODUCT, 'rt', encoding='utf-8') as handle:
            for line in handle:
                r = json.loads(line)
                trace, enh = r['metadata']['answer_selection'], r['enhancement_result']
                c1, base, final = r['initial_answer'], enh['final_answer'], r['final_answer']
                c1_top5 = [e['passage_id'] for e in r['initial_evidence']]
                contexts = {n: {'top5': tuple(c['top5']), 'answer': c['answer']} for n, c in trace['contexts'].items()}
                routed = r['controller_decision']['action'] == 'retrieval_enhancement'
                mt = ad.trace_of(r)
                applied = bool((mt.get('bm25') or {}).get('applied'))
                ids = lambda row: [e['passage_id'] for e in (row or {}).get('evidence') or ()]
                base_top5 = ex.base_evidence_ids(c1, base, c1_top5, routed, applied,
                                                 ids(mt.get('p3_20') if applied else mt), ids(mt) if applied else None)
                self.assertEqual(base_top5, [e['passage_id'] for e in enh['evidence']], r['sample_id'])
                self.assertEqual(ex.final_evidence_ids(c1, base, final, c1_top5, base_top5, contexts, order, trace['stage']),
                                 [e['passage_id'] for e in r['final_evidence']], r['sample_id'])
                seen += 1
        self.assertEqual(seen, 2440)

    @unittest.skipUnless(all(p.exists() for p in ex.raw_paths().values()), 'raw Test candidates and corpus are local only')
    def test_outputs_rebuild_from_the_raw_inputs(self):
        rows, judge, sources = ex.build()
        with tempfile.TemporaryDirectory() as tmp:
            ex.write_outputs(tmp, rows, judge, sources)
            for d in (ex.PRED_DIR, ex.JUDGE_DIR):
                for path in sorted((Path(tmp) / d).iterdir()):
                    committed = (ROOT / d / path.name).read_bytes().replace(b'\r\n', b'\n')
                    self.assertEqual(path.read_bytes(), committed, path.name)


if __name__ == '__main__':
    unittest.main()
