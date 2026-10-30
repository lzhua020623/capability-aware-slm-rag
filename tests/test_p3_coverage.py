"""L2 round step 1: P3-20 on every sweep item, rebuilt from stored route-1 records."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import precompute_p3_coverage as coverage

ROOT = Path(__file__).resolve().parents[1]


def record(sid, p3=True, needs=False, label='correct', p_fail=0.2):
    evidence = [{'rank': i, 'passage_id': f'nq_p_{i}', 'text': f'passage {i}', 'score': 0.9 - i / 10} for i in range(1, 6)]
    actions = {coverage.P3: {'final_answer': 'b', 'status': 'success', 'final_evidence': evidence[:1],
                             'native': {'module_trace': {'status': 'answer_replaced'}}}} if p3 else {}
    return {'sample_id': sid, 'query': 'q ' + sid, 'initial_answer': 'a', 'evidence': evidence,
            'diagnosis': {'final_label': label, 'stage1_label': label, 'stage1_failure_probability': p_fail,
                          'confidence': 0.7},
            'frozen_selection': {'needs_enhancement': needs, 'failure_type': label, 'selection_lane': 'no_action',
                                 'selection_score': 0.0, 'p_fail': p_fail, 'p_insuff': 0.0, 'd_combo': None,
                                 'rec': None, 'p_corr_3c': None},
            'actions': actions}


class P3CoverageTests(unittest.TestCase):
    def test_registration_binds_route_1(self):
        reg = json.loads((ROOT / coverage.REGISTRATION).read_text(encoding='utf-8'))
        sha = hashlib.sha256((ROOT / reg['base']['registration']).read_bytes()).hexdigest()
        self.assertEqual(sha, reg['base']['sha256'])
        self.assertEqual(reg['base']['tag'], 'quality_round1_B_final')
        self.assertIn('C1 is not regenerated', reg['step_1_p3_coverage']['input'])

    def test_items_and_selections_are_rebuilt_from_stored_records(self):
        r = record('s1')
        item = coverage.item_from_record(r)
        self.assertEqual((item.sample_id, item.dataset.value, item.query, item.initial_answer), ('s1', 'nq', 'q s1', 'a'))
        self.assertEqual([e.passage_id for e in item.evidence], [f'nq_p_{i}' for i in range(1, 6)])
        s = coverage.selection_from_record(r)
        self.assertEqual((s.needs_enhancement, s.failure_type.value, s.confidence, s.p_fail), (False, 'correct', 0.7, 0.2))

    def test_plan_runs_missing_items_then_the_first_stored_ones(self):
        records = [record('a'), record('b', p3=False), record('c'), record('d', p3=False), record('e')]
        self.assertEqual(coverage.plan(records, repro_n=2), [('b', 'new'), ('d', 'new'), ('a', 'repro'), ('c', 'repro')])

    def test_agreement_counts_each_field(self):
        stored = {'a': record('a')['actions'][coverage.P3], 'b': record('b')['actions'][coverage.P3]}
        changed = dict(stored['b'], final_answer='x')
        report = coverage.agreement(stored, {'a': stored['a'], 'b': changed})
        self.assertEqual(report, {'n': 2, 'final_answer': 1, 'status': 2, 'final_evidence': 2, 'native_status': 2})

    def test_done_keys_drop_a_cut_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'p3_records.jsonl'
            path.write_text(json.dumps({'sample_id': 'a', 'source': 'new'}) + '\n{"sample_id": "b"')
            self.assertEqual(coverage.done_keys(path), {('a', 'new')})
            self.assertTrue(path.read_text().endswith('\n'))


if __name__ == '__main__':
    unittest.main()
