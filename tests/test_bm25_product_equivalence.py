"""Replay the P3-BM25 production equivalence check on confirm against the archived route-1 and L2 confirm results."""
import hashlib
import json
from pathlib import Path
import tarfile
import unittest

from scripts.l2_candidates import role_ids

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/controller/bm25_product_equivalence_v1'
L2_CONFIRM = ROOT / 'artifacts/controller/l2_confirm_v1/results.tar.gz'
sha = lambda data: hashlib.sha256(data).hexdigest()


def untar(path):
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def jsonl(data):
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


class Bm25ProductEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members = untar(RELEASE / 'results.tar.gz')
        cls.product = jsonl(cls.members['product.jsonl'])
        cls.expected = jsonl(cls.members['expected.jsonl'])
        cls.score = json.loads((RELEASE / 'score.json').read_text(encoding='utf-8'))
        cls.provenance = json.loads((RELEASE / 'provenance.json').read_text(encoding='utf-8'))

    def test_every_archived_byte_is_bound_without_secrets(self):
        checks = json.loads((RELEASE / 'checksums.json').read_text(encoding='utf-8'))
        actual = {p.relative_to(RELEASE).as_posix() for p in RELEASE.rglob('*') if p.is_file() and p.name != 'checksums.json'}
        self.assertEqual(set(checks), actual)
        for name, expected in checks.items():
            self.assertEqual(sha((RELEASE / name).read_bytes()), expected, name)
        self.assertEqual(sha((RELEASE / 'results.tar.gz').read_bytes()), self.provenance['curated_archive']['sha256'])
        for name, payload in self.members.items():
            for secret in (b'AZURE_API_KEY=', b'AZURE_OPENAI_API_KEY=', b'Authorization: Bearer ', b'password='):
                self.assertNotIn(secret, payload, name)

    def test_run_is_one_full_confirm_run_of_the_registered_product(self):
        manifest = json.loads(self.members['run/manifest.json'])
        self.assertEqual(manifest['registration'], 'configs/controller/quality_bm25_v1.json')
        self.assertEqual(manifest['registration_sha256'], sha((ROOT / manifest['registration']).read_bytes()))
        self.assertEqual((manifest['state'], manifest['role'], manifest['data_role'], manifest['engineering_run'], manifest['resumed_with']),
                         ('completed', 'confirm', 'validation', False, 0))
        self.assertEqual([p['sample_id'] for p in self.product], role_ids('confirm'))
        self.assertEqual([e['sample_id'] for e in self.expected], role_ids('confirm'))

    def test_expectations_follow_the_archived_confirm_results(self):
        items = {i['sample_id']: i for i in jsonl(untar(L2_CONFIRM)['items.jsonl'])}
        for e in self.expected:
            item = items[e['sample_id']]
            self.assertEqual((e['route_1_initial'], e['route_1_final'], e['offline_l2_top5']), (item['c1'], item['route_1'], item['l2_top5']))
            self.assertEqual(e['expected_final'], item['l2'] if e['offline_bm25'] else item['route_1'])
            self.assertTrue(not e['offline_bm25'] or item['routed'])

    def test_product_reproduces_every_final_answer(self):
        self.assertEqual([p['final_answer'] for p in self.product], [e['expected_final'] for e in self.expected])
        self.assertEqual([p['initial_answer'] for p in self.product], [e['route_1_initial'] for e in self.expected])
        routed = [(p, e) for p, e in zip(self.product, self.expected) if e['route_1_action'] == 'retrieval_enhancement']
        self.assertEqual(sum(bool((p['bm25'] or {}).get('bm25_only')) == e['offline_bm25'] for p, e in routed), len(routed))
        self.assertEqual((self.score['same_final'], self.score['n'], self.score['bm25_errors']), (2440, 2440, 0))
        self.assertEqual(self.score['same_l2_top5'], sum(tuple((p['bm25'] or {}).get('selected') or ()) == tuple(e['offline_l2_top5'])
                                                          for p, e in zip(self.product, self.expected) if e['offline_bm25']))


if __name__ == '__main__':
    unittest.main()
