"""NQ recoverability oracle: registration and the L3 / L4 contexts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import recoverability_oracle as ro

ROOT = Path(__file__).resolve().parents[1]
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def record(sid, top5, query='q'):
    return {'sample_id': sid, 'query': query, 'evidence': [{'passage_id': p} for p in top5]}


class RegistrationTests(unittest.TestCase):
    def test_registration_binds_the_sweep_ids_only(self):
        reg = ro.load_registration()
        self.assertEqual(reg['set']['ids_sha256'], sha(reg['set']['ids']))
        self.assertEqual(len(ro.sweep_ids()), 6000)
        self.assertEqual(set(reg['levels']), {'L1', 'L2', 'L2_any', 'L3', 'L4'})
        self.assertIn('never a product component', reg['role'].lower())
        sweep = json.loads((ROOT / 'configs/controller/quality_sweep_v1.json').read_text(encoding='utf-8'))
        confirm = json.loads((ROOT / sweep['split']['confirm']['path']).read_text(encoding='utf-8'))
        self.assertFalse(set(confirm) & set(ro.sweep_ids()))


class ContextTests(unittest.TestCase):
    def test_levels_put_the_gold_passage_first_or_alone(self):
        c1 = ('a', 'b', 'g', 'c', 'd')
        self.assertEqual(ro.oracle_contexts(c1, 'g'), {'L3': ('g', 'a', 'b', 'c', 'd'), 'L4': ('g',)})
        self.assertEqual(ro.oracle_contexts(c1, 'z'), {'L3': ('z', 'a', 'b', 'g', 'c'), 'L4': ('z',)})
        self.assertEqual(ro.oracle_contexts(('g', 'a', 'b', 'c', 'd'), 'g')['L3'], ('g', 'a', 'b', 'c', 'd'))

    def test_rows_skip_c1_contexts_and_missing_gold_and_mark_nothing(self):
        records = [record('s1', ('a', 'b', 'g', 'c', 'd')), record('s2', ('g', 'a', 'b', 'c', 'd')),
                   record('s3', ('a', 'b', 'c', 'd', 'e'))]
        rows, missing = ro.context_rows(records, {'s1': 'g', 's2': 'g', 's3': 'x'}, {'a', 'b', 'c', 'd', 'e', 'g'})
        self.assertEqual([(r['sample_id'], r['passage_ids']) for r in rows],
                         [('s1', ['g', 'a', 'b', 'c', 'd']), ('s1', ['g']), ('s2', ['g'])])
        self.assertEqual(missing, ['s3'])
        self.assertTrue(all(set(r) == {'sample_id', 'query', 'passage_ids'} for r in rows))

    def test_gold_passages_read_the_delivery_reference_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{'sample_id': 's1', 'annotation_reference': {'gold_passage_id': 'g1', 'gold_answer': ['x']}},
                    {'sample_id': 's9', 'annotation_reference': {'gold_passage_id': 'g9'}}]
            (Path(tmp) / 'nq_train_annotation_000.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.assertEqual(ro.gold_passages(tmp, ['s1']), {'s1': 'g1'})
            with self.assertRaises(ValueError):
                ro.gold_passages(tmp, ['s1', 's2'])


if __name__ == '__main__':
    unittest.main()
