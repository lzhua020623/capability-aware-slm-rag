"""Replay the BM25 rule's Test release: archive integrity, the registered IDs, and the paired, C1, RAG and product scores."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from baselines._lib.answer_scoring import score_answer
from scripts import bm25_gate as gate
from scripts import l2_offline as off
from scripts import score_quality_eval as scorer

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/controller/bm25_gate_test_v1'
RAW = Path(os.environ.get('BM25_TEST_CANDIDATES', ROOT / 'data/processed/quality_sweep_v1/bm25_test/candidates.jsonl'))
sha = lambda data: hashlib.sha256(data).hexdigest()


def jsonl(data):
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


class Bm25GateTestReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tarfile.open(RELEASE / 'results.tar.gz') as archive:
            cls.members = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
        cls.provenance = json.loads((RELEASE / 'provenance.json').read_text(encoding='utf-8'))
        cls.score = json.loads((RELEASE / 'score.json').read_text(encoding='utf-8'))
        cls.report = json.loads(cls.members['report.json'])
        cls.selections = jsonl(cls.members['selections.jsonl'])
        cls.registration = gate.load_registration()

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

    def test_runs_cover_the_registered_test_ids(self):
        self.assertEqual(self.provenance['registration_sha256'], sha((ROOT / gate.REGISTRATION).read_bytes()))
        self.assertEqual(self.provenance['route_1_test_archive']['sha256'], sha((ROOT / gate.ROUTE_1_TEST).read_bytes()))
        ids = json.loads((ROOT / self.registration['test']['ids']).read_text(encoding='utf-8'))
        self.assertEqual([s['sample_id'] for s in self.selections], ids)
        manifest = json.loads(self.members['candidates/manifest.json'])
        self.assertEqual((manifest['state'], manifest['role'], manifest['ids_sha256'], manifest['engineering_run'], manifest['resumed_with']),
                         ('completed', 'test', self.registration['test']['ids_sha256'], False, 0))
        generation = json.loads(self.members['generations/manifest.json'])
        self.assertEqual((generation['state'], generation['data_role'], generation['resumed_with']), ('completed', 'test', 0))
        self.assertEqual(generation['contexts_sha256'], sha(self.members['generations/contexts.jsonl']))
        contexts = jsonl(self.members['generations/contexts.jsonl'])
        self.assertEqual([(c['sample_id'], c['passage_ids']) for c in contexts],
                         [(s['sample_id'], s['top5']) for s in self.selections if s['bm25_only']])
        self.assertTrue(all(set(c) == {'sample_id', 'query', 'passage_ids'} for c in contexts))

    @unittest.skipUnless(RAW.exists(), 'raw Test candidates are kept locally only')
    def test_selections_replay_from_the_raw_candidates(self):
        self.assertEqual(sha(RAW.read_bytes()), self.provenance['raw_local']['candidates']['sha256'])
        rows = {json.loads(line)['sample_id']: json.loads(line) for line in RAW.read_text(encoding='utf-8').splitlines() if line.strip()}
        for s in self.selections:
            top5 = off.select_ids(rows[s['sample_id']], gate.SETTING)
            bge = {p for p, _ in rows[s['sample_id']]['bge'][:gate.SETTING['k_bge']]}
            self.assertEqual((list(top5), [p for p in top5 if p not in bge]), (s['top5'], s['bm25_only']))

    def test_scores_replay_and_pass_the_registered_rule(self):
        generations = {(g['sample_id'], tuple(g['passage_ids'])): (g['generated_answer'], g['reject_reason'])
                       for g in jsonl(self.members['generations/generations.jsonl'])}
        selected = {s['sample_id']: tuple(s['top5']) for s in self.selections if s['bm25_only']}
        with tempfile.TemporaryDirectory() as tmp:
            predictions, ours, team = gate.route_1_test(ROOT / gate.ROUTE_1_TEST, tmp)
            new = []
            for i in range(4):
                path = Path(tmp) / f'new_{i}.jsonl'
                path.write_bytes(self.members[f'judge/outputs/shard_{i}/judge_results.jsonl'])
                new.append(path)
            self.assertNotIn('llm_failed', {r['judge_status'] for p in new for r in scorer.read_jsonl(p)})
            rows = []
            for r in predictions:
                if r['sample_id'] in selected:
                    answer = off.final_answer(r['initial_answer'], selected[r['sample_id']], (), generations, r['sample_id'])
                    r = {**r, 'prediction': answer, **score_answer(answer, r['gold'])}
                rows.append(r)
            labels = scorer.prior_labels([team['qwen_only'], team['product'], *ours, *new, team['rag']])
            expected = scorer.judge_rows(rows, scorer.prior_labels([team['qwen_only'], team['product'], *ours, team['rag']]))
            archived = [r for i in range(4) for r in jsonl(self.members[f'judge/inputs/shard_{i}.jsonl'])]
            self.assertEqual(sorted(archived, key=lambda r: r['sample_id']), sorted(expected, key=lambda r: r['sample_id']))
            against = gate.paired_rows(predictions, rows, labels)
            self.assertEqual(against, self.report['against_route_1'])
            self.assertEqual(scorer.score(rows, labels), self.report['score'])
            self.assertEqual(scorer.compare(rows, labels, team), self.report['comparisons'])
        self.assertEqual((against['net'], against['fixes'], against['harms']), (105, 226, 121))
        self.assertTrue(self.report['adoption']['passed'])
        self.assertEqual(self.report['adoption']['rule'], self.registration['test']['adoption'])
        self.assertEqual(self.score['route_1_judge_assisted_correct'], 6632)

    def test_judge_inputs_are_exactly_the_missing_strings(self):
        archived = sorted((r['sample_id'] for i in range(4) for r in jsonl(self.members[f'judge/inputs/shard_{i}.jsonl'])))
        outputs = sorted((r['sample_id'] for i in range(4) for r in jsonl(self.members[f'judge/outputs/shard_{i}/judge_results.jsonl'])))
        self.assertEqual(archived, outputs)
        self.assertEqual(len(archived), len(set(archived)))


if __name__ == '__main__':
    unittest.main()
