"""Replay the FEVER recoverability release: archive integrity, GPU inputs, contexts, label parsing and the score."""
import hashlib
import json
from pathlib import Path
import tarfile
import unittest

from scripts import fever_recoverability as fr
from src.evaluation.fever import parse_fever_prediction

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/controller/fever_recoverability_v1'
RELEASE = BASE / 'results'
sha = lambda data: hashlib.sha256(data).hexdigest()


def jsonl(data):
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


class FeverRecoverabilityReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tarfile.open(RELEASE / 'results.tar.gz') as archive:
            cls.members = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
        cls.provenance = json.loads((RELEASE / 'provenance.json').read_text(encoding='utf-8'))
        cls.items = fr.read_jsonl(BASE / 'sample/items.jsonl')
        cls.results = jsonl(cls.members['run/results.jsonl'])
        cls.manifest = json.loads(cls.members['run/manifest.json'])

    def test_every_archived_byte_is_bound(self):
        checks = json.loads((RELEASE / 'checksums.json').read_text(encoding='utf-8'))
        actual = {p.relative_to(RELEASE).as_posix() for p in RELEASE.rglob('*') if p.is_file() and p.name != 'checksums.json'}
        self.assertEqual(set(checks), actual)
        for name, expected in checks.items():
            self.assertEqual(sha((RELEASE / name).read_bytes()), expected, name)
        archive = self.provenance['curated_archive']
        self.assertEqual((sha((RELEASE / 'results.tar.gz').read_bytes()), len(self.members)), (archive['sha256'], archive['files']))
        self.assertEqual(self.provenance['registration_sha256'], sha((ROOT / fr.REGISTRATION).read_bytes()))
        for name, payload in self.members.items():
            self.assertFalse(Path(name).is_absolute() or '..' in Path(name).parts)
            for secret in (b'AZURE_API_KEY=', b'AZURE_OPENAI_API_KEY=', b'Authorization: Bearer ', b'password='):
                self.assertNotIn(secret, payload, name)

    def test_gpu_inputs_follow_the_sample_without_labels(self):
        sample = json.loads((BASE / 'sample/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(sha((BASE / 'sample/items.jsonl').read_bytes()), sample['items_sha256'])
        inputs = self.members['gpu_inputs.jsonl']
        self.assertEqual(jsonl(inputs), fr.gpu_rows(self.items))
        self.assertTrue(all('label' not in row for row in jsonl(inputs)))
        retrieval = json.loads(self.members['retrieval/manifest.json'])
        self.assertEqual((self.manifest['inputs_sha256'], retrieval['items_sha256']), (sha(inputs),) * 2)
        self.assertEqual((self.manifest['state'], self.manifest['n'], self.manifest['resumed_with']), ('completed', 2000, 0))
        self.assertEqual(self.manifest['results_sha256'], sha(self.members['run/results.jsonl']))
        self.assertEqual(self.manifest['registration_sha256'], self.provenance['registration_sha256'])
        replay = self.manifest['environment']['fever_retrieval']
        self.assertEqual(replay['hits_sha256'], sha(self.members['retrieval/hits.jsonl']))
        self.assertEqual(replay['manifest'], retrieval)

    def test_contexts_are_the_registered_conditions(self):
        hits = {r['sample_id']: r for r in jsonl(self.members['retrieval/hits.jsonl'])}
        items = {r['sample_id']: r for r in self.items}
        self.assertEqual([r['sample_id'] for r in self.results], [r['sample_id'] for r in self.items])
        for r in self.results:
            item = items[r['sample_id']]
            top5 = [{'passage_id': h['passage_id']} for h in hits[r['sample_id']]['top5']]
            l3, l4 = fr.oracle_passages(item['gold_evidence_text'], top5, item['gold_page_ids'])
            self.assertIsNone(r['C0']['passage_ids'])
            self.assertEqual(r['C1']['passage_ids'], [h['passage_id'] for h in top5])
            self.assertEqual(r['L3']['passage_ids'], [p['passage_id'] for p in l3])
            self.assertEqual(r['L4']['passage_ids'], [p['passage_id'] for p in l4])
            for c in fr.CONDITIONS:
                self.assertEqual(r[c]['label'], parse_fever_prediction(r[c]['raw']), (r['sample_id'], c))
            self.assertIsNone(r['L1']['error'])

    def test_score_replays_and_its_headline_numbers(self):
        report = fr.score(self.items, self.results)
        self.assertEqual(report, json.loads((RELEASE / 'score.json').read_text(encoding='utf-8')))
        d, h, a = report['development'], report['held_out'], report['all']
        self.assertEqual((d['n'], d['c1_wrong'], h['n'], h['c1_wrong'], a['c1_wrong']), (1600, 247, 400, 59, 306))
        self.assertEqual({c: v for c, v in d['flips_vs_c1'].items()},
                         {'C0': {'wrong_to_correct': 77, 'correct_to_wrong': 271}, 'L1': {'wrong_to_correct': 36, 'correct_to_wrong': 47},
                          'L3': {'wrong_to_correct': 116, 'correct_to_wrong': 88}, 'L4': {'wrong_to_correct': 179, 'correct_to_wrong': 50}})
        self.assertEqual(a['flips_vs_c1']['L1'], {'wrong_to_correct': 44, 'correct_to_wrong': 53})
        self.assertEqual(set(a['unparsed'].values()), {0})
        self.assertEqual(a['gold_pages_covered'], {'C1_top5': 1304, 'L1_top5': 1358})


if __name__ == '__main__':
    unittest.main()
