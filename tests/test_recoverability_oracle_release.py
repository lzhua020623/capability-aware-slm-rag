"""Replay the NQ recoverability oracle release: archive integrity, contexts, judge inputs and the ladder report."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts import recoverability_oracle as ro
from scripts.quality_sweep_offline import judge_id
from scripts.score_quality_eval import prior_labels

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/controller/recoverability_oracle_v1'
L2_SWEEP = ROOT / 'artifacts/controller/l2_sweep_v1/results.tar.gz'
sha = lambda data: hashlib.sha256(data).hexdigest()


def untar(path):
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def jsonl(data):
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


class RecoverabilityOracleReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members = untar(RELEASE / 'results.tar.gz')
        cls.provenance = json.loads((RELEASE / 'provenance.json').read_text(encoding='utf-8'))
        cls.items = jsonl(cls.members['ladder_inputs.jsonl'])
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / 'generations.jsonl'
        path.write_bytes(cls.members['generations/generations.jsonl'])
        cls.answers = ro.oracle_answers(cls.items, ro.read_generations([path]))
        cls.gold = {r['sample_id']: r['gold'] for r in jsonl(cls.members['references.jsonl'])}
        cls.prior = {(r['sample_id'], r['answer']): r['correct'] for r in jsonl(cls.members['prior_labels.jsonl'])}
        outputs = []
        for i in range(4):
            out = Path(cls.tmp.name) / f'out_{i}.jsonl'
            out.write_bytes(cls.members[f'judge/outputs/shard_{i}/judge_results.jsonl'])
            outputs.append(out)
        cls.judged = prior_labels(outputs)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_archived_byte_is_bound_without_secrets(self):
        checks = json.loads((RELEASE / 'checksums.json').read_text(encoding='utf-8'))
        actual = {p.relative_to(RELEASE).as_posix() for p in RELEASE.rglob('*') if p.is_file() and p.name != 'checksums.json'}
        self.assertEqual(set(checks), actual)
        for name, expected in checks.items():
            self.assertEqual(sha((RELEASE / name).read_bytes()), expected, name)
        self.assertEqual(sha((RELEASE / 'results.tar.gz').read_bytes()), self.provenance['curated_archive']['sha256'])
        self.assertEqual(self.provenance['registration_sha256'], sha((ROOT / ro.REGISTRATION).read_bytes()))
        for name, payload in self.members.items():
            for secret in (b'AZURE_API_KEY=', b'AZURE_OPENAI_API_KEY=', b'Authorization: Bearer ', b'password='):
                self.assertNotIn(secret, payload, name)

    def test_ladder_inputs_follow_the_sweep_and_the_l2_archive(self):
        self.assertEqual([it['sample_id'] for it in self.items], ro.sweep_ids())
        l2 = {r['sample_id']: r for r in jsonl(untar(L2_SWEEP)['items.jsonl'])}
        for it in self.items:
            ref = l2[it['sample_id']]
            self.assertEqual((it['c1'], it['route_1']), (ref['c1'], ref['route_1']))
            self.assertEqual(it['contexts'], {n: ref['changed'].get(n, it['c1']) for n in it['contexts']})

    def test_contexts_are_the_registered_oracle_contexts_without_markers(self):
        manifest = json.loads(self.members['generations/manifest.json'])
        self.assertEqual((manifest['state'], manifest['data_role'], manifest['resumed_with']), ('completed', 'validation', 0))
        self.assertEqual(manifest['contexts_sha256'], sha(self.members['generations/contexts.jsonl']))
        contexts = jsonl(self.members['generations/contexts.jsonl'])
        self.assertTrue(all(set(c) == {'sample_id', 'query', 'passage_ids'} for c in contexts))
        expected = [(it['sample_id'], list(ids)) for it in self.items
                    for ids in ro.oracle_contexts(tuple(it['c1_top5']), it['gold_passage_id']).values() if ids != tuple(it['c1_top5'])]
        self.assertEqual([(c['sample_id'], c['passage_ids']) for c in contexts], expected)
        self.assertEqual(len(contexts), 10069)

    def test_judge_inputs_are_exactly_the_unlabelled_strings(self):
        label = ro.labeller(self.prior, {}, self.gold)
        wanted = sorted(judge_id(it['sample_id'], a) for it in self.items
                        for a in ro.needed_strings(it, self.answers[it['sample_id']]) if label(it['sample_id'], a) is None)
        inputs = sorted(r['sample_id'] for i in range(4) for r in jsonl(self.members[f'judge/inputs/shard_{i}.jsonl']))
        self.assertEqual(inputs, wanted)
        outputs = [r for i in range(4) for r in jsonl(self.members[f'judge/outputs/shard_{i}/judge_results.jsonl'])]
        self.assertEqual(sorted(r['sample_id'] for r in outputs), inputs)
        self.assertNotIn('llm_failed', {r['judge_status'] for r in outputs})

    def test_report_replays_and_its_headline_numbers(self):
        report = ro.ladder_report(self.items, self.answers, ro.labeller(self.prior, self.judged, self.gold))
        self.assertEqual(report, json.loads(self.members['report.json']))
        p, q = report['primary'], report['p3_rule']
        self.assertTrue(report['complete'])
        self.assertEqual((p['c1_failures'], p['standalone']['L4'], p['oracle_persistent'], p['unrecovered_under_all_tested']),
                         (2096, 1522, 574, 431))
        self.assertEqual((q['oracle_persistent'], q['unrecovered_under_all_tested']), (791, 591))
        self.assertEqual(p['first_fix'], {'L1': 239, 'L2': 79, 'L2_any': 628, 'L3': 223, 'L4': 496, 'none': 431})
        score = json.loads((RELEASE / 'score.json').read_text(encoding='utf-8'))
        self.assertEqual((score['primary'], score['p3_rule']), (p, q))


if __name__ == '__main__':
    unittest.main()
