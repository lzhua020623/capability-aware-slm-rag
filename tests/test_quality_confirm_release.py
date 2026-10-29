"""Replay the quality round-1 confirm run: archive integrity, binding to the frozen registration and the judge-assisted result."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts import run_quality_eval as runner
from scripts import score_quality_eval as scorer

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/controller/quality_confirm_v1'
sha = lambda data: hashlib.sha256(data).hexdigest()


class QualityConfirmReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tarfile.open(RELEASE / 'results.tar.gz') as archive:
            cls.members = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
        cls.provenance = json.loads((RELEASE / 'provenance.json').read_text(encoding='utf-8'))
        cls.score = json.loads((RELEASE / 'score.json').read_text(encoding='utf-8'))
        cls.predictions = cls.jsonl('scored/predictions.jsonl')

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

    def test_run_is_the_single_full_confirm_run_of_the_frozen_registration(self):
        manifest = json.loads(self.members['run/manifest.json'])
        registration = ROOT / runner.REGISTRATION
        self.assertEqual(manifest['registration'], runner.REGISTRATION)
        self.assertEqual(manifest['registration_sha256'], sha(registration.read_bytes()))
        self.assertEqual((manifest['state'], manifest['role'], manifest['data_role']), ('completed', 'confirm', 'validation'))
        self.assertFalse(manifest['engineering_run'])
        self.assertEqual(manifest['resumed_with'], 0)
        self.assertEqual(manifest['results_sha256'], sha(self.members['run/results.jsonl']))
        results = self.jsonl('run/results.jsonl')
        self.assertEqual([r['sample_id'] for r in results], runner.expected_ids('confirm'))
        self.assertEqual(self.provenance['registration_sha256'], manifest['registration_sha256'])

    def test_predictions_and_judge_inputs_replay_from_the_results(self):
        gold = {r['sample_id']: r['gold'] for r in self.predictions}
        rebuilt = [scorer.prediction_row(r, gold[r['sample_id']]) for r in self.jsonl('run/results.jsonl')]
        self.assertEqual(rebuilt, self.predictions)
        archived = [row for i in range(4) for row in self.jsonl(f'judge/inputs/shard_{i}.jsonl')]
        expected = scorer.judge_rows(self.predictions, {})
        self.assertEqual(sorted(archived, key=lambda r: r['sample_id']), sorted(expected, key=lambda r: r['sample_id']))

    def test_judge_assisted_score_replays_and_passes_the_registered_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(4):
                path = Path(tmp) / f'judge_{i}.jsonl'
                path.write_bytes(self.members[f'judge/outputs/shard_{i}/judge_results.jsonl'])
                paths.append(path)
            judged = [row for path in paths for row in scorer.read_jsonl(path)]
            self.assertNotIn('llm_failed', {row['judge_status'] for row in judged})
            report = scorer.score(self.predictions, scorer.prior_labels(paths))
        self.assertEqual(report, {k: self.score[k] for k in report})
        self.assertEqual((report['n'], report['changed'], report['fixes'], report['harms'], report['net_vs_c1']),
                         (2440, 421, 102, 57, 45))
        sweep = json.loads((ROOT / 'configs/controller/quality_sweep_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(self.score['rule'], sweep['choice']['confirm_success'])
        self.assertTrue(self.score['passed'])
        self.assertLess(self.score['mcnemar_p'], 0.001)


if __name__ == '__main__':
    unittest.main()
