import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.retrieval.isolated_bge import encode_queries
from src.compensation.retrieval_enhancement.adapters import BGE_QUERY_INSTRUCTION


class IsolatedBGETests(unittest.TestCase):
    def fake_worker(self, command, **kwargs):
        self.assertTrue(kwargs['check'])
        self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '1')
        self.assertEqual(command[-1], 'a'*40)
        source, result, info = map(Path, command[-4:-1])
        self.assertEqual(json.loads(source.read_text()), ['Raw question?'])
        vectors = np.zeros((1, 768), dtype=np.float32); vectors[0, 0] = 1
        np.save(result, vectors)
        info.write_text(json.dumps({'revision': 'a'*40, 'query_instruction': BGE_QUERY_INSTRUCTION}))

    def test_pinned_revision_and_single_prefix_owned_by_worker(self):
        with patch('src.retrieval.isolated_bge.subprocess.run', side_effect=self.fake_worker):
            vectors, receipt = encode_queries(['Raw question?'], 'a'*40)
        self.assertEqual(vectors.shape, (1, 768))
        self.assertEqual(receipt['revision'], 'a'*40)

    def test_bad_worker_provenance_or_nonfinite_vectors_fail_closed(self):
        def bad(command, **kwargs):
            self.fake_worker(command, **kwargs)
            Path(command[-2]).write_text(json.dumps({'revision': 'b'*40}))
        with patch('src.retrieval.isolated_bge.subprocess.run', side_effect=bad), self.assertRaises(ValueError):
            encode_queries(['Raw question?'], 'a'*40)
        for queries, revision in [([], 'a'*40), ([''], 'a'*40), (['Q'], 'main')]:
            with self.assertRaises(ValueError):
                encode_queries(queries, revision)


if __name__ == '__main__':
    unittest.main()
