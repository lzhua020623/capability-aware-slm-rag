"""Replay the L2 confirm release: archive integrity, L2 answers from route-1 traces and L2 generations, the L2
adoption test and the acceptance-gate direction check."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts import acceptance_gate as gate
from scripts import l2_confirm as conf
from scripts import l2_offline as off
from scripts.l2_candidates import role_ids
from scripts.quality_sweep_offline import judge_id, load_judged

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'artifacts/controller/l2_confirm_v1'
ROUTE_1 = ROOT / 'artifacts/controller/quality_confirm_v1/results.tar.gz'
sha = lambda data: hashlib.sha256(data).hexdigest()


def untar(path):
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def jsonl(data):
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


class L2ConfirmReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members = untar(RELEASE / 'results.tar.gz')
        cls.provenance = json.loads((RELEASE / 'provenance.json').read_text(encoding='utf-8'))
        cls.score = json.loads((RELEASE / 'score.json').read_text(encoding='utf-8'))
        cls.items = {i['sample_id']: i for i in jsonl(cls.members['items.jsonl'])}
        cls.labels = {(r['sample_id'], r['answer']): r for r in jsonl(cls.members['labels.jsonl'])}
        cls.reports = {k: json.loads(cls.members[f'reports/{k}.json']) for k in ('l2_adoption', 'gate_confirm', 'gate_sweep')}

    def paired(self, before, after):
        out = {'changed': 0, 'fixes': 0, 'harms': 0}
        for sid in before:
            if after[sid] == before[sid]:
                continue
            was, now = (self.labels[(sid, a)]['correct'] for a in (before[sid], after[sid]))
            out['changed'] += 1
            out['fixes'] += now and not was
            out['harms'] += was and not now
        return {**out, 'net': out['fixes'] - out['harms']}

    def column(self, key):
        return {sid: i[key] for sid, i in self.items.items()}

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

    def test_items_follow_the_confirm_ids_and_the_registrations(self):
        self.assertEqual(list(self.items), role_ids('confirm'))
        for path, expected in self.provenance['registrations'].items():
            self.assertEqual(sha((ROOT / path).read_bytes()), expected, path)
        self.assertEqual(self.score['l2']['setting'], off.name(gate.VOTER))

    def test_l2_answers_replay_from_route_1_traces_and_l2_generations(self):
        route_1 = jsonl(untar(ROUTE_1)['run/results.jsonl'])
        self.assertEqual(sha(untar(ROUTE_1)['run/results.jsonl']), self.provenance['route_1_confirm']['results_sha256'])
        generations = {}
        for r in route_1:
            t = conf.trace_of(r)
            if r['controller_decision']['action'] == conf.P3 and t.get('generated_answer') is not None:
                generations[(r['sample_id'], tuple(e['passage_id'] for e in t['evidence']))] = (t['generated_answer'], t['reject_reason'])
        for g in jsonl(self.members['generations/generations.jsonl']):
            generations[(g['sample_id'], tuple(g['passage_ids']))] = (g['generated_answer'], g['reject_reason'])
        for r in route_1:
            item = self.items[r['sample_id']]
            self.assertEqual((item['c1'], item['route_1'], item['c1_top5']),
                             (r['initial_answer'], r['final_answer'], list(conf.base_ids(r))))
            self.assertEqual(item['routed'], r['controller_decision']['action'] == conf.P3)
            self.assertEqual(off.final_answer(item['c1'], tuple(item['l2_top5']), tuple(item['c1_top5']), generations,
                                              r['sample_id']), item['l2'])

    def test_l2_adoption_replays_and_fails(self):
        c1, route_1, l2 = self.column('c1'), self.column('route_1'), self.column('l2')
        route_1_l2 = {s: l2[s] if self.items[s]['routed'] else c1[s] for s in c1}
        report = self.reports['l2_adoption']
        self.assertEqual(self.paired(route_1, route_1_l2), {k: report['adoption'][k] for k in ('changed', 'fixes', 'harms', 'net')})
        for key, after in (('route_1', route_1), ('route_1_l2', route_1_l2), ('l2', l2)):
            self.assertEqual(self.paired(c1, after), {k: report['against_c1'][key][k] for k in ('changed', 'fixes', 'harms', 'net')})
        self.assertEqual((report['adoption']['net'], report['adoption']['passed']), (-1, False))

    def test_gate_direction_check_replays_and_fails(self):
        c1, route_1, l2 = self.column('c1'), self.column('route_1'), self.column('l2')
        voter = {s: l2[s] for s in c1 if route_1[s] != c1[s]}
        after = gate.gated(c1, route_1, voter)
        report = self.reports['gate_confirm']
        self.assertEqual((len(voter), sum(after[s] != route_1[s] for s in c1)), (report['in_scope'], report['kept_c1']))
        self.assertEqual(self.paired(route_1, after), {k: report['against_route_1'][k] for k in ('changed', 'fixes', 'harms', 'net')})
        self.assertEqual(self.paired(c1, after), {k: report['against_c1']['gated'][k] for k in ('changed', 'fixes', 'harms', 'net')})
        self.assertEqual((report['against_route_1']['net'], report['proceed_to_test']), (-1, False))
        self.assertEqual(report['rule'], gate.load_registration()['confirm']['rule'])
        self.assertEqual(self.reports['gate_sweep']['against_route_1']['net'], 26)

    def test_new_judge_labels_match_the_archived_judge_outputs(self):
        rows = [r for i in range(4) for r in jsonl(self.members[f'judge/l2_confirm/outputs/shard_{i}/judge_results.jsonl'])]
        self.assertNotIn('llm_failed', {r['judge_status'] for r in rows})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'judged.jsonl'
            path.write_bytes(b''.join(self.members[f'judge/l2_confirm/outputs/shard_{i}/judge_results.jsonl'] for i in range(4)))
            judged = load_judged([path])
        for (sid, answer), label in self.labels.items():
            if label['source'] == 'l2_confirm':
                self.assertEqual(judged[judge_id(sid, answer)], label['correct'])


if __name__ == '__main__':
    unittest.main()
