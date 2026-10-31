"""Replay the adaptive selection confirm and Test release: archive integrity, item bindings, judge inputs and scores."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts import adaptive_selection as ad
from scripts import bm25_gate as gate
from scripts import l2_offline as off
from scripts import score_quality_eval as scorer
from scripts.quality_sweep_offline import judge_id

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/controller/adaptive_selection_eval_v1'
CONFIRM = ROOT / 'artifacts/controller/quality_confirm_v1/results.tar.gz'
L2_CONFIRM = ROOT / 'artifacts/controller/l2_confirm_v1/results.tar.gz'
EQUIVALENCE = ROOT / 'artifacts/controller/bm25_product_equivalence_v1/results.tar.gz'
BM25_TEST = ROOT / 'artifacts/controller/bm25_gate_test_v1/results.tar.gz'
RAW = {name: Path(os.environ.get(env, ROOT / default)) for name, env, default in (
    ('corpus', 'NQ_TRAIN_CORPUS', 'data/processed/retrieval/nq_train_corpus.jsonl'),
    ('confirm_candidates', 'L2_CONFIRM_CANDIDATES', 'data/processed/quality_sweep_v1/l2_confirm/candidates.jsonl'),
    ('test_results', 'ROUTE_1_TEST_RESULTS', 'data/processed/quality_sweep_v1/test/results.jsonl.gz'),
    ('test_candidates', 'BM25_TEST_CANDIDATES', 'data/processed/quality_sweep_v1/bm25_test/candidates.jsonl'),
    ('test_unrouted_candidates', 'ADAPTIVE_TEST_CANDIDATES', 'data/processed/quality_sweep_v1/adaptive_test/candidates_unrouted.jsonl'))}
sha = lambda data: hashlib.sha256(data).hexdigest()


def untar(path):
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def jsonl(data):
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


def write(tmp, name, data):
    path = Path(tmp) / name.replace('/', '__')
    path.write_bytes(data)
    return path


class AdaptiveSelectionEvalReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members = untar(RELEASE / 'results.tar.gz')
        cls.provenance = json.loads((RELEASE / 'provenance.json').read_text(encoding='utf-8'))
        cls.score = json.loads((RELEASE / 'score.json').read_text(encoding='utf-8'))
        cls.policy = json.loads((ROOT / ad.POLICY).read_text(encoding='utf-8'))
        cls.registration = ad.load_registration()
        cls.rows = {role: jsonl(cls.members[f'{role}/items.jsonl']) for role in ('confirm', 'test')}
        cls.reports = {role: json.loads(cls.members[f'{role}/report.json']) for role in ('confirm', 'test')}
        cls.confirm = untar(CONFIRM)
        cls.l2_confirm = untar(L2_CONFIRM)
        cls.bm25_test = untar(BM25_TEST)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.predictions, cls.ours, cls.team = gate.route_1_test(ROOT / gate.ROUTE_1_TEST, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def batch(self, archive, prefix):
        return [write(self.tmp.name, f'{prefix}/shard_{i}', archive[f'{prefix}/shard_{i}/judge_results.jsonl']) for i in range(4)]

    def label_files(self, role):
        """(prior files, this release's files) in the registered order; the caller appends nothing after RAG."""
        new = self.batch(self.members, f'{role}/judge/outputs')
        if role == 'confirm':
            round2 = [write(self.tmp.name, f'round2_{i}', self.members[f'confirm/prior_labels/round2_paired/shard_{i}.jsonl']) for i in range(4)]
            prior = self.batch(self.confirm, 'judge/outputs') + round2 + self.batch(self.l2_confirm, 'judge/l2_confirm/outputs')
            return prior, new, []
        bm25 = self.batch(self.bm25_test, 'judge/outputs')
        return [self.team['qwen_only'], self.team['product'], *self.ours, *bm25], new, [self.team['rag']]

    def gold(self, role):
        rows = jsonl(self.confirm['scored/predictions.jsonl']) if role == 'confirm' else self.predictions
        return {r['sample_id']: r['gold'] for r in rows}

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
        self.assertEqual(self.provenance['registration_sha256'], sha((ROOT / ad.REGISTRATION).read_bytes()))
        self.assertEqual((self.provenance['policy_sha256'], self.score['policy_sha256']), (sha((ROOT / ad.POLICY).read_bytes()),) * 2)
        for role in ('confirm', 'test'):
            self.assertEqual(self.reports[role]['policy_sha256'], self.provenance['policy_sha256'])

    def test_confirm_items_bind_route_1_and_the_product(self):
        results = jsonl(self.confirm['run/results.jsonl'])
        product = {r['sample_id']: r['final_answer'] for r in jsonl(untar(EQUIVALENCE)['product.jsonl'])}
        rows = self.rows['confirm']
        self.assertEqual([r['sample_id'] for r in rows], [r['sample_id'] for r in results])
        for row, r in zip(rows, results):
            self.assertEqual((row['c1'], row['routed'], row['p_fail'], row['query']),
                             (r['initial_answer'], r['controller_decision']['action'] == ad.P3,
                              r['diagnosis']['stage1_failure_probability'], r['query']))
            self.assertEqual(row['base'], product[row['sample_id']])

    def test_test_items_bind_route_1_and_b_bm25(self):
        generations = {(g['sample_id'], tuple(g['passage_ids'])): (g['generated_answer'], g['reject_reason'])
                       for g in jsonl(self.bm25_test['generations/generations.jsonl'])}
        selected = {s['sample_id']: tuple(s['top5']) for s in jsonl(self.bm25_test['selections.jsonl']) if s['bm25_only']}
        rows = {r['sample_id']: r for r in self.rows['test']}
        self.assertEqual(set(rows), {r['sample_id'] for r in self.predictions})
        self.assertEqual(len(rows), len(self.rows['test']))
        changed = 0
        for r in self.predictions:
            row, sid = rows[r['sample_id']], r['sample_id']
            base = off.final_answer(r['initial_answer'], selected[sid], (), generations, sid) if sid in selected else r['prediction']
            changed += base != r['prediction']
            self.assertEqual((row['c1'], row['routed'], row['base'], row['query']),
                             (r['initial_answer'], r['controller_action'] == ad.P3, base, r['query']))
        self.assertEqual(changed, 764)
        unrouted = json.loads((ROOT / self.registration['test']['unrouted_ids']).read_text(encoding='utf-8'))
        self.assertEqual(sorted(unrouted), sorted(sid for sid, row in rows.items() if not row['routed']))
        manifest = json.loads(self.members['test/candidates_unrouted/manifest.json'])
        self.assertEqual((manifest['state'], manifest['role'], manifest['ids_sha256'], manifest['engineering_run'], manifest['resumed_with']),
                         ('completed', 'test', self.registration['test']['unrouted_ids_sha256'], False, 0))

    def test_generation_runs_completed_once_without_gold(self):
        runs = {'confirm/generations': 'validation', 'test/generations/routed': 'test', 'test/generations/unrouted': 'test'}
        for prefix, role in runs.items():
            manifest = json.loads(self.members[f'{prefix}/manifest.json'])
            self.assertEqual((manifest['state'], manifest['data_role'], manifest['resumed_with']), ('completed', role, 0), prefix)
            self.assertEqual(manifest['contexts_sha256'], sha(self.members[f'{prefix}/contexts.jsonl']), prefix)
            contexts = jsonl(self.members[f'{prefix}/contexts.jsonl'])
            generated = jsonl(self.members[f'{prefix}/generations.jsonl'])
            self.assertTrue(all(set(c) == {'sample_id', 'query', 'passage_ids'} for c in contexts), prefix)
            self.assertEqual(sorted((c['sample_id'], tuple(c['passage_ids'])) for c in contexts),
                             sorted((g['sample_id'], tuple(g['passage_ids'])) for g in generated), prefix)
            self.assertEqual(manifest['n'], len(contexts))

    def test_judge_inputs_are_exactly_the_missing_pairs(self):
        for role in ('confirm', 'test'):
            prior, new, last = self.label_files(role)
            labels = scorer.prior_labels(prior + last)
            outcomes = ad.answers_of(self.rows[role], self.policy)
            wanted = sorted(judge_id(sid, answer) for sid, answer in ad.needed_pairs(self.rows[role], outcomes)
                            if (sid, answer) not in labels)
            inputs = sorted(r['sample_id'] for i in range(4) for r in jsonl(self.members[f'{role}/judge/inputs/shard_{i}.jsonl']))
            outputs = [r for p in new for r in scorer.read_jsonl(p)]
            self.assertEqual(inputs, wanted, role)
            self.assertEqual(sorted(r['sample_id'] for r in outputs), inputs, role)
            self.assertNotIn('llm_failed', {r['judge_status'] for r in outputs}, role)

    def test_scores_replay(self):
        for role in ('confirm', 'test'):
            prior, new, last = self.label_files(role)
            labels = scorer.prior_labels(prior + new + last)
            outcomes = ad.answers_of(self.rows[role], self.policy)
            report = ad.summarize(self.rows[role], outcomes, labels, self.gold(role))
            self.assertEqual(report, self.reports[role]['policies'], role)
            self.assertTrue(self.reports[role]['complete'], role)
            self.assertEqual(self.score[role]['policies'], {name: {k: v[k] for k in ('against_b_bm25', 'against_c1', 'extra_generations_per_item', 'stages')}
                                                            for name, v in report.items()})
        chosen = [{'sample_id': r['sample_id'], 'prediction': outcomes['chosen'][r['sample_id']][0]} for r in self.rows['test']]
        team = scorer.compare(chosen, labels, {'rag': self.team['rag'], 'product': self.team['product']})
        self.assertEqual(team, self.reports['test']['chosen_against_team'])
        self.assertEqual(team, self.score['test']['chosen_against_team'])
        runs = {k: {str(r['sample_id']): r['prediction'] for r in scorer.read_jsonl(self.team[k])} for k in ('qwen_only', 'rag', 'product')}
        runs['c1'] = {r['sample_id']: r['initial_answer'] for r in self.predictions}
        runs['route_1'] = {r['sample_id']: r['prediction'] for r in self.predictions}
        runs['b_bm25'] = {r['sample_id']: r['base'] for r in self.rows['test']}
        runs.update({name: {sid: v[0] for sid, v in result.items()} for name, result in outcomes.items()})
        table = {name: sum(labels[(sid, a)] for sid, a in answers.items()) for name, answers in runs.items()}
        self.assertEqual(table, {name: v['correct'] for name, v in self.score['test']['one_label_set'].items()})
        self.assertEqual({k: table[k] for k in ('qwen_only', 'rag', 'product', 'c1', 'route_1', 'b_bm25', 'chosen')},
                         {'qwen_only': 3281, 'rag': 6391, 'product': 6446, 'c1': 6400, 'route_1': 6632, 'b_bm25': 6737, 'chosen': 6975})

    def test_registered_checks_pass(self):
        confirm, test = (self.reports[role]['policies']['chosen']['against_b_bm25'] for role in ('confirm', 'test'))
        self.assertEqual((confirm['fixes'], confirm['harms'], confirm['net']), (97, 49, 48))
        self.assertEqual((test['fixes'], test['harms'], test['net']), (521, 283, 238))
        self.assertLess(test['mcnemar_p'], 0.05)
        self.assertEqual((self.reports['confirm']['rule'], self.reports['test']['rule']),
                         (self.registration['confirm']['rule'], self.registration['test']['adoption']))
        self.assertTrue(self.reports['confirm']['passed'] and self.reports['test']['passed'])
        self.assertEqual((self.score['confirm']['passed'], self.score['test']['passed']), (True, True))
        self.assertEqual(self.policy['chosen'], {'theta': 0.0, 'tau': 0.8, 'decision': 'selector'})

    @unittest.skipUnless(all(p.exists() for p in RAW.values()), 'raw candidates, Test results and the corpus are kept locally only')
    def test_items_rebuild_from_the_raw_inputs(self):
        self.assertEqual(sha(RAW['test_unrouted_candidates'].read_bytes()),
                         self.provenance['raw_local']['test_unrouted_candidates']['sha256'])
        order = self.registration['contexts']['order']
        candidates = {r['sample_id']: r for r in scorer.read_jsonl(RAW['confirm_candidates'])}
        results = jsonl(self.confirm['run/results.jsonl'])
        stored = ad.stored_generations(results, [write(self.tmp.name, 'l2_gen', self.l2_confirm['generations/generations.jsonl']),
                                                 write(self.tmp.name, 'ad_gen', self.members['confirm/generations/generations.jsonl'])])
        base = {r['sample_id']: r['base'] for r in self.rows['confirm']}
        self.assertEqual(ad.build_items(results, base, candidates, stored, RAW['corpus'], order), self.rows['confirm'])
        candidates = {r['sample_id']: r for p in (RAW['test_candidates'], RAW['test_unrouted_candidates']) for r in scorer.read_jsonl(p)}
        results = list(ad.read_results(RAW['test_results']))
        files = [write(self.tmp.name, 'bm25_gen', self.bm25_test['generations/generations.jsonl'])]
        files += [write(self.tmp.name, f'ad_{p}', self.members[f'test/generations/{p}/generations.jsonl']) for p in ('routed', 'unrouted')]
        base = {r['sample_id']: r['base'] for r in self.rows['test']}
        self.assertEqual(ad.build_items(results, base, candidates, ad.stored_generations(results, files), RAW['corpus'], order), self.rows['test'])


if __name__ == '__main__':
    unittest.main()
