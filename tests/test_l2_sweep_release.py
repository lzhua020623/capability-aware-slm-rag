"""Replay the L2 sweep release: archive integrity, both score reports and the registered choice from the archived items."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts import l2_offline as off
from scripts.quality_sweep_offline import judge_id, load_judged

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/controller/l2_sweep_v1'
sha = lambda data: hashlib.sha256(data).hexdigest()


class L2SweepReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tarfile.open(RELEASE / 'results.tar.gz') as archive:
            cls.members = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
        cls.provenance = json.loads((RELEASE / 'provenance.json').read_text(encoding='utf-8'))
        cls.score = json.loads((RELEASE / 'score.json').read_text(encoding='utf-8'))
        cls.items = cls.jsonl('items.jsonl')
        cls.labels = {(r['sample_id'], r['answer']): r for r in cls.jsonl('labels.jsonl')}
        cls.reports = {k: json.loads(cls.members[f'reports/{k}.json']) for k in ('routed', 'all')}

    @classmethod
    def jsonl(cls, name):
        return [json.loads(line) for line in cls.members[name].decode('utf-8').splitlines() if line.strip()]

    def replay(self, name, scope):
        out = {'changed': 0, 'fixes': 0, 'harms': 0}
        for item in self.items:
            after = item['changed'].get(name, item['c1']) if (item['routed'] or scope == 'all') else item['c1']
            if after == item['c1']:
                continue
            before, now = (self.labels[(item['sample_id'], a)]['correct'] for a in (item['c1'], after))
            out['changed'] += 1
            out['fixes'] += now and not before
            out['harms'] += before and not now
        return {**out, 'net': out['fixes'] - out['harms']}

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

    def test_items_cover_the_sweep_and_the_registered_grid(self):
        from scripts.precompute_quality_sweep import load_registration as load_sweep
        self.assertEqual([i['sample_id'] for i in self.items], load_sweep()[1])
        names = {off.name(c) for c in off.grid(off.load_registration())}
        self.assertEqual(set(self.score['settings']), names)
        self.assertTrue(all(set(i['changed']) <= names for i in self.items))
        self.assertEqual(self.provenance['registration_sha256'], sha((ROOT / off.REGISTRATION).read_bytes()))

    def test_control_is_route_1_on_routed_items(self):
        routed = [i for i in self.items if i['routed']]
        self.assertEqual(len(routed), self.score['routed'])
        self.assertTrue(all(i['changed'].get('bge20_bm250_rrf', i['c1']) == i['route_1'] for i in routed))
        self.assertEqual(self.reports['routed']['control_equals_route_1'], len(routed))

    def test_reports_replay_from_items_and_labels(self):
        for scope, report in self.reports.items():
            self.assertTrue(report['complete'])
            for row in report['results']:
                self.assertEqual(self.replay(row['name'], 'routed'), {k: row[k] for k in ('changed', 'fixes', 'harms', 'net')})
                if scope == 'all':
                    self.assertEqual(self.replay(row['name'], 'all'), {k: row['all_items'][k] for k in ('changed', 'fixes', 'harms', 'net')})
                    self.assertEqual(self.score['settings'][row['name']]['all_items']['net'], row['all_items']['net'])

    def test_choice_follows_the_registered_rule(self):
        for report in self.reports.values():
            self.assertEqual(off.choose(report['results'])['name'], report['chosen']['name'])
        self.assertEqual(self.score['chosen'], self.reports['routed']['chosen']['name'])
        self.assertEqual(self.reports['all']['chosen']['name'], self.score['chosen'])
        self.assertEqual(self.score['rule'], off.load_registration()['step_2_l2']['choice'])

    def test_l2_judge_labels_match_the_archived_judge_outputs(self):
        for batch in ('l2_routed', 'l2_all'):
            rows = [r for i in range(4) for r in self.jsonl(f'judge/{batch}/outputs/shard_{i}/judge_results.jsonl')]
            self.assertNotIn('llm_failed', {r['judge_status'] for r in rows})
            inputs = {r['sample_id'] for i in range(4) for r in self.jsonl(f'judge/{batch}/inputs/shard_{i}.jsonl')}
            self.assertEqual({r['sample_id'] for r in rows}, inputs)
        labelled = {}
        with tempfile.TemporaryDirectory() as tmp:
            for batch in ('l2_routed', 'l2_all'):
                for i in range(4):
                    path = Path(tmp) / f'{batch}_{i}.jsonl'
                    path.write_bytes(self.members[f'judge/{batch}/outputs/shard_{i}/judge_results.jsonl'])
                    labelled.update({jid: (batch, v) for jid, v in load_judged([path]).items()})
        for (sid, answer), label in self.labels.items():
            if label['source'] in ('l2_routed', 'l2_all'):
                self.assertEqual(labelled[judge_id(sid, answer)], (label['source'], label['correct']))


if __name__ == '__main__':
    unittest.main()
