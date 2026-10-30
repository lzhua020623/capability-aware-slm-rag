"""L2 candidate stage: BM25 index, deeper BGE backend, GPU check and row format."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts import l2_candidates as l2
from src.compensation.retrieval_enhancement.adapters import NQFaissBackend
from tests.test_retrieval_enhancement_adapters import FakeLoadedRetriever

HAS_BM25S = importlib.util.find_spec('bm25s') is not None


class L2CandidateTests(unittest.TestCase):
    @unittest.skipUnless(HAS_BM25S, 'bm25s is not installed')
    def test_bm25_index_keeps_corpus_order_and_drops_zero_scores(self):
        rows = [{'passage_id': 'p0', 'document_title': 'France', 'text': 'Paris is the capital.'},
                {'passage_id': 'p1', 'document_title': 'Germany', 'text': 'Berlin is the capital.'},
                {'passage_id': 'p2', 'document_title': 'Cats', 'text': 'The cat sat on the mat.'}]
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / 'corpus.jsonl'
            corpus.write_text(''.join(json.dumps(r) + '\n' for r in rows) + '\n')
            receipt = l2.build_bm25(corpus, Path(tmp) / 'bm25', l2.load_registration())
            index = l2.Bm25Index(Path(tmp) / 'bm25')
            self.assertEqual((receipt['n'], index.passage_ids), (3, ['p0', 'p1', 'p2']))
            hits = index.search('capital of germany', 3)
            self.assertEqual([index.passage_ids[p] for p, _ in hits], ['p1', 'p0'])
            self.assertEqual(index.search('zzzz', 3), [])

    def test_deep_backend_raises_only_the_top_k_cap(self):
        backend = NQFaissBackend(FakeLoadedRetriever(), max_top_k=20)
        deep = l2.deep_backend(backend)
        self.assertIs(deep.retriever, backend.retriever)
        self.assertEqual(deep.retrieve('q', dataset='nq', top_k=100), backend.retrieve('q', dataset='nq', top_k=20))
        with self.assertRaises(ValueError):
            backend.retrieve('q', dataset='nq', top_k=21)

    def test_compare_trace_reports_gap_and_top5_agreement(self):
        candidates = [{'passage_id': f'p{i}', 'text': f't{i}', 'score': 0.9 - i / 100} for i in range(20)]
        stored = [float(20 - i) for i in range(20)]
        trace = {'candidates': candidates, 'scores': [{'cross_encoder_score': s} for s in stored],
                 'evidence': [{'passage_id': f'p{i}'} for i in range(5)]}
        self.assertEqual(l2.compare_trace(trace, stored), {'max_gap': 0.0, 'same_top5': True})
        flipped = list(reversed(stored))
        self.assertFalse(l2.compare_trace(trace, flipped)['same_top5'])

    def test_candidate_rows_hold_ids_and_scores_only(self):
        row = l2.candidate_row('s', [{'passage_id': 'a', 'text': 'x', 'score': 0.5}], [(1, 2.5)], ['z', 'b'], {'a': 1.0, 'b': 0.2})
        self.assertEqual(row, {'sample_id': 's', 'bge': [['a', 0.5]], 'bm25': [['b', 2.5]], 'ce': {'a': 1.0, 'b': 0.2}})

    def test_roles_are_the_registered_sweep_and_confirm_ids(self):
        sweep, confirm = l2.role_ids('sweep'), l2.role_ids('confirm')
        self.assertEqual((len(sweep), len(confirm)), (6000, 2440))
        self.assertFalse(set(sweep) & set(confirm))

    def test_depths_cover_the_registered_grid(self):
        grid = l2.load_registration()['step_2_l2']['grid']
        self.assertEqual((max(grid['k_bge']), max(grid['k_bm25'])), (l2.BGE_DEPTH, l2.BM25_DEPTH))


if __name__ == '__main__':
    unittest.main()
