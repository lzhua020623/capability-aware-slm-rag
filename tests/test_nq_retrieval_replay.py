import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.replay_nq_controller_retrieval import development_queries, PreencodedQueryModel
from src.compensation.retrieval_enhancement.adapters import BGE_QUERY_INSTRUCTION


class ReplayInputTests(unittest.TestCase):
    def test_reader_stops_at_requested_development_rows_and_strips_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'dev'
            row = {'dataset': 'nq', 'sample_id': 'one', 'query': 'Question?',
                   'initial_answer': 'SENTINEL', 'gold_answer': 'GOLD_SENTINEL',
                   'evidence': [{'passage_id': 'p1'}]}
            path.write_text(json.dumps(row)+'\ninvalid unused tail\n')
            rows = development_queries(path, 1)
            self.assertEqual(rows, [{'sample_id': 'one', 'query': 'Question?', 'reference_top5': ['p1']}])
            self.assertNotIn('SENTINEL', json.dumps(rows))
            with self.assertRaises(ValueError):
                development_queries(path, 0)

    def test_replay_fails_for_missing_or_duplicate_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'dev'
            row = {'dataset': 'nq', 'sample_id': 'one', 'query': 'Q', 'evidence': []}
            path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'Insufficient'):
                development_queries(path, 2)
            path.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                development_queries(path, 2)

    def test_preencoded_vectors_require_exactly_the_once_prefixed_query(self):
        model = PreencodedQueryModel(['Q'], np.array([[1., 0.]]))
        self.assertEqual(model.encode([BGE_QUERY_INSTRUCTION+'Q']).tolist(), [[1., 0.]])
        for text in ['Q', BGE_QUERY_INSTRUCTION*2+'Q']:
            with self.assertRaises(KeyError):
                model.encode([text])
        with self.assertRaises(ValueError):
            PreencodedQueryModel(['Q'], [])


if __name__ == '__main__':
    unittest.main()
