"""Replay the adaptive selection production equivalence check on confirm against the archived offline evaluation."""
import hashlib
import json
from pathlib import Path
import tarfile
import unittest

from scripts import adaptive_selection as ad
from scripts.l2_candidates import role_ids

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/controller/adaptive_product_equivalence_v1'
EVAL = ROOT / 'artifacts/controller/adaptive_selection_eval_v1/results.tar.gz'
BM25_PRODUCT = ROOT / 'artifacts/controller/bm25_product_equivalence_v1/results.tar.gz'
sha = lambda data: hashlib.sha256(data).hexdigest()


def untar(path):
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def jsonl(data):
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


class AdaptiveProductEquivalenceTests(unittest.TestCase):
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
        self.assertEqual(manifest['registration'], 'configs/controller/quality_adaptive_v1.json')
        self.assertEqual(manifest['registration_sha256'], sha((ROOT / manifest['registration']).read_bytes()))
        self.assertEqual((manifest['state'], manifest['role'], manifest['data_role'], manifest['engineering_run'], manifest['resumed_with']),
                         ('completed', 'confirm', 'validation', False, 0))
        spec = json.loads((ROOT / manifest['registration']).read_text(encoding='utf-8'))['answer_selection']
        self.assertEqual(manifest['environment']['answer_selection'], spec)
        self.assertEqual([p['sample_id'] for p in self.product], role_ids('confirm'))

    def test_expectations_follow_the_archived_evaluation(self):
        items = jsonl(untar(EVAL)['confirm/items.jsonl'])
        policy = json.loads((ROOT / ad.POLICY).read_text(encoding='utf-8'))
        b_bm25 = {r['sample_id']: r['final_answer'] for r in jsonl(untar(BM25_PRODUCT)['product.jsonl'])}
        self.assertEqual([e['sample_id'] for e in self.expected], [r['sample_id'] for r in items])
        for e, row in zip(self.expected, items):
            answer, stage, cost = ad.apply_stages(row, policy['chosen'], policy['models'])
            self.assertEqual((e['expected_final'], e['expected_stage'], e['expected_generations']), (answer, stage, cost))
            self.assertEqual(e['expected_groups'], [g['x'] for g in row['stages'][str(stage)]['groups']] if stage else [])
            self.assertEqual((e['c1'], e['b_bm25']), (row['c1'], b_bm25[row['sample_id']]))

    def test_product_reproduces_every_selection(self):
        pairs = list(zip(self.product, self.expected))
        self.assertEqual([p['initial_answer'] for p, _ in pairs], [e['c1'] for _, e in pairs])
        self.assertEqual([p['controller_answer'] for p, _ in pairs], [e['b_bm25'] for _, e in pairs])
        self.assertEqual([p['selection']['stage'] for p, _ in pairs], [e['expected_stage'] for _, e in pairs])
        # Two items regenerate one Top-5 the item had already answered in another order (GPU vs CPU rerank ties);
        # the offline cost counted it as paid. Same answers, one more generation each.
        extra = {p['sample_id']: p['selection']['generations'] - e['expected_generations'] for p, e in pairs
                 if p['selection']['generations'] != e['expected_generations']}
        self.assertEqual(extra, {'1226764714413157858': 1, '-1504254136609936585': 1})
        self.assertEqual(self.score['same_generations'], len(pairs) - 2)
        self.assertEqual([p['selection']['groups'] for p, _ in pairs], [e['expected_groups'] for _, e in pairs])
        self.assertEqual([p['final_answer'] for p, _ in pairs], [e['expected_final'] for _, e in pairs])
        self.assertTrue(all(p['selection']['error'] is None and p['selection']['entered'] for p, _ in pairs))
        self.assertTrue(all(p['selection']['cost']['llm_calls'] == p['selection']['generations'] for p, _ in pairs))
        self.assertEqual((self.score['same_final'], self.score['n'], self.score['errors']), (2440, 2440, 0))


if __name__ == '__main__':
    unittest.main()
