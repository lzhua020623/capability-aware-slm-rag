"""Replay the quality round-1 Test run (exposed-Test re-evaluation): archive integrity, binding to the frozen
registration, the CPU-thread disclosure and the judge-assisted results against the team's Test runs."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts import run_quality_eval as runner
from scripts import score_quality_eval as scorer

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/controller/quality_test_v1'
BACKUP = Path(os.environ.get('QUALITY_TEST_RESULTS', ROOT / 'data/processed/quality_sweep_v1/test/results.jsonl.gz'))
TEAM = ('qwen_only', 'product', 'rag')
sha = lambda data: hashlib.sha256(data).hexdigest()


class QualityTestReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tarfile.open(RELEASE / 'results.tar.gz') as archive:
            cls.members = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
        cls.provenance = json.loads((RELEASE / 'provenance.json').read_text(encoding='utf-8'))
        cls.score = json.loads((RELEASE / 'score.json').read_text(encoding='utf-8'))
        cls.predictions = cls.jsonl('scored/predictions.jsonl')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = {}
        for name, payload in cls.members.items():
            if name.startswith(('judge/outputs/', 'team/')) and name.endswith('judge_results.jsonl'):
                path = Path(cls.tmp.name) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                cls.paths[name] = path
        cls.ours = [cls.paths[f'judge/outputs/shard_{i}/judge_results.jsonl'] for i in range(4)]
        cls.team = {k: cls.paths[f'team/{k}/judge_results.jsonl'] for k in TEAM}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def jsonl(cls, name):
        return [json.loads(line) for line in cls.members[name].decode('utf-8').splitlines() if line.strip()]

    def test_every_archived_byte_is_bound_without_weights_or_secrets(self):
        checks = json.loads((RELEASE / 'checksums.json').read_text(encoding='utf-8'))
        actual = {p.relative_to(RELEASE).as_posix() for p in RELEASE.rglob('*') if p.is_file() and p.name != 'checksums.json'}
        self.assertEqual(set(checks), actual)
        for name, expected in checks.items():
            self.assertEqual(sha((RELEASE / name).read_bytes()), expected, name)
        archive = self.provenance['curated_archive']
        self.assertEqual((sha((RELEASE / 'results.tar.gz').read_bytes()), len(self.members)), (archive['sha256'], archive['files']))
        for name, payload in self.members.items():
            self.assertFalse(Path(name).is_absolute() or '..' in Path(name).parts)
            self.assertFalse(name.endswith('.safetensors'))
            for secret in (b'AZURE_API_KEY=', b'AZURE_OPENAI_API_KEY=', b'Authorization: Bearer ', b'password='):
                self.assertNotIn(secret, payload, name)

    def test_run_is_the_single_full_test_run_of_the_frozen_registration(self):
        manifest = json.loads(self.members['run/manifest.json'])
        self.assertEqual(manifest['registration'], runner.REGISTRATION)
        self.assertEqual(manifest['registration_sha256'], sha((ROOT / runner.REGISTRATION).read_bytes()))
        self.assertEqual((manifest['state'], manifest['role'], manifest['data_role']), ('completed', 'test', 'test'))
        self.assertFalse(manifest['engineering_run'])
        self.assertEqual((manifest['resumed_with'], manifest['shard'], manifest['shards']), (0, 0, 1))
        ids = runner.expected_ids('test')
        self.assertEqual((manifest['n_ids'], manifest['results']), (len(ids), len(ids)))
        self.assertEqual([r['sample_id'] for r in self.predictions], ids)
        self.assertEqual(self.provenance['test_ids_sha256'], sha(json.dumps(ids).encode()))
        for key in ('registration_sha256', 'query_sha256', 'results_sha256'):
            self.assertEqual(self.provenance[key], manifest[key])
        sweep = json.loads((ROOT / 'configs/controller/quality_sweep_v1.json').read_text(encoding='utf-8'))
        self.assertIn('exposed-Test re-evaluation', sweep['roles']['test'])
        self.assertTrue(self.score['role'].startswith('exposed Test re-evaluation'))

    def test_cpu_thread_change_is_disclosed_with_the_confirm_replay(self):
        env = self.members['run/env_threads.txt'].decode().split()
        self.assertEqual(sorted(env), ['MKL_NUM_THREADS=16', 'OMP_NUM_THREADS=16'])
        threads = self.provenance['cpu_threads']
        self.assertEqual(sorted(threads['env']), sorted(env))
        replay = threads['equivalence_replay']
        self.assertEqual(replay['same']['final_answer'], replay['n'])
        self.assertEqual(replay['same']['c1_answer'], replay['n'])
        confirm = json.loads((ROOT / 'artifacts/controller/quality_confirm_v1/provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(replay['original_results_sha256'], confirm['results_sha256'])

    def test_predictions_replay_from_the_full_results(self):
        if not BACKUP.exists():
            self.skipTest(f'full results kept outside git; set QUALITY_TEST_RESULTS ({BACKUP} missing)')
        self.assertEqual(sha(BACKUP.read_bytes()), self.provenance['full_results']['gzip_sha256'])
        raw = gzip.decompress(BACKUP.read_bytes())
        self.assertEqual(sha(raw), self.provenance['results_sha256'])
        gold = {r['sample_id']: r['gold'] for r in self.predictions}
        rebuilt = []
        for line in raw.decode('utf-8').splitlines():
            if line.strip():
                result = json.loads(line)
                rebuilt.append(scorer.prediction_row(result, gold[str(result['sample_id'])]))
        self.assertEqual(rebuilt, self.predictions)

    def test_judge_inputs_are_the_strings_no_team_file_had_judged(self):
        archived = [row for i in range(4) for row in self.jsonl(f'judge/inputs/shard_{i}.jsonl')]
        expected = scorer.judge_rows(self.predictions, scorer.prior_labels([self.team[k] for k in TEAM]))
        self.assertEqual(sorted(archived, key=lambda r: r['sample_id']), sorted(expected, key=lambda r: r['sample_id']))
        judged = [row['sample_id'] for path in self.ours for row in scorer.read_jsonl(path)]
        self.assertEqual(sorted(judged), sorted(r['sample_id'] for r in archived))
        for k, spec in self.provenance['judge']['team_files'].items():
            self.assertEqual(sha(self.members[f'team/{k}/judge_results.jsonl']), spec['sha256'], k)

    def test_score_and_comparisons_replay_with_the_rag_label_winning(self):
        judged = [row for path in self.ours for row in scorer.read_jsonl(path)]
        self.assertNotIn('llm_failed', {row['judge_status'] for row in judged})
        labels = scorer.prior_labels([self.team['qwen_only'], self.team['product'], *self.ours, self.team['rag']])
        report = scorer.score(self.predictions, labels)
        self.assertEqual(report, {k: self.score[k] for k in report})
        c1 = sum(labels[(r['sample_id'], r['initial_answer'])] for r in self.predictions)
        self.assertEqual(c1, self.score['c1_judge_assisted_correct'])
        self.assertEqual(report['judge_assisted_correct'] - c1, report['net_vs_c1'])
        self.assertEqual(scorer.compare(self.predictions, labels, self.team), self.score['comparisons'])
        self.assertTrue(self.score['label_order'][-1].startswith('team rag'))
        alt = scorer.prior_labels([self.team['rag'], self.team['qwen_only'], self.team['product'], *self.ours])
        sensitivity = self.score['sensitivity_this_run_labels_last']
        self.assertEqual(scorer.score(self.predictions, alt)['net_vs_c1'], sensitivity['net_vs_c1'])
        self.assertEqual({k: v['net'] for k, v in scorer.compare(self.predictions, alt, self.team).items()},
                         sensitivity['net_vs'])

    def test_team_judge_self_disagreement_is_reported(self):
        rag, product = (scorer.prior_labels([self.team[k]]) for k in ('rag', 'product'))
        common = set(rag) & set(product)
        spec = self.provenance['judge']['team_self_disagreement']
        self.assertEqual((len(common), sum(rag[p] != product[p] for p in common)),
                         (spec['identical_strings_rag_and_product'], spec['different_labels']))


if __name__ == '__main__':
    unittest.main()
