"""Tests for the deeper candidate pool. No index and no model are loaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.compensation.candidates import IndexCandidateProvider, load_corpus, load_ids
from src.compensation.rerank import build_candidate_pool
from src.diagnosis.schema import DatasetName, DiagnosisInput, EvidenceItem

CORPUS = {f"p{index}": f"passage {index}" for index in range(6)}


class FakeIndex:
    """Returns a scripted row order, like FAISS does."""

    def __init__(self, rows, scores=None, ntotal=6):
        self.rows = rows
        self.scores = scores or [[1.0 - 0.1 * position for position in range(len(rows[0]))]]
        self.ntotal = ntotal
        self.searched = []

    def search(self, embedding, top_k):
        self.searched.append(top_k)
        return np.array(self.scores), np.array(self.rows)


def provider(rows, **kwargs) -> IndexCandidateProvider:
    ids = [f"p{index}" for index in range(6)]
    return IndexCandidateProvider(lambda query: np.zeros((1, 4)), FakeIndex(rows), ids, dict(CORPUS), **kwargs)


def item(*passage_ids: str) -> DiagnosisInput:
    return DiagnosisInput(
        sample_id="s1",
        dataset=DatasetName.NQ,
        query="who wrote hamlet",
        evidence=[
            EvidenceItem(rank=rank, passage_id=passage_id, text=CORPUS[passage_id], score=0.9)
            for rank, passage_id in enumerate(passage_ids, start=1)
        ],
        initial_answer="Marlowe",
    )


class LoadersTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "corpus.jsonl"

    def test_corpus_accepts_passage_id_or_id(self):
        rows = [{"passage_id": "p1", "text": "one"}, {"id": "p2", "text": "two"}]
        self.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.assertEqual(load_corpus(self.path), {"p1": "one", "p2": "two"})

    def test_corpus_can_be_filtered(self):
        rows = [{"passage_id": "p1", "text": "one"}, {"passage_id": "p2", "text": "two"}]
        self.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.assertEqual(load_corpus(self.path, wanted={"p2"}), {"p2": "two"})

    def test_ids_preserve_row_order(self):
        rows = [{"passage_id": "p9"}, {"passage_id": "p3"}]
        self.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.assertEqual(load_ids(self.path), ["p9", "p3"])

    def test_blank_lines_are_skipped(self):
        self.path.write_text('{"passage_id": "p1", "text": "one"}\n\n', encoding="utf-8")
        self.assertEqual(load_corpus(self.path), {"p1": "one"})


class ProviderTest(unittest.TestCase):
    def test_it_returns_ranked_evidence_items(self):
        candidates = list(provider([[3, 1, 0]], top_n=3)(item("p0")))
        self.assertEqual([candidate.passage_id for candidate in candidates], ["p3", "p1", "p0"])
        self.assertEqual([candidate.rank for candidate in candidates], [1, 2, 3])
        self.assertEqual(candidates[0].text, "passage 3")

    def test_it_asks_the_index_for_top_n(self):
        pool = provider([[3, 1]], top_n=2)
        list(pool(item("p0")))
        self.assertEqual(pool.index.searched, [2])

    def test_faiss_padding_is_dropped(self):
        candidates = list(provider([[3, -1, -1]], top_n=3)(item("p0")))
        self.assertEqual([candidate.passage_id for candidate in candidates], ["p3"])

    def test_an_id_missing_from_the_corpus_is_skipped(self):
        pool = provider([[3, 1]], top_n=2)
        pool.corpus.pop("p3")
        self.assertEqual([candidate.passage_id for candidate in pool(item("p0"))], ["p1"])

    def test_it_validates_its_arguments(self):
        ids = [f"p{index}" for index in range(6)]
        with self.assertRaises(TypeError):
            IndexCandidateProvider("not callable", FakeIndex([[0]]), ids, dict(CORPUS))
        with self.assertRaises(ValueError):
            IndexCandidateProvider(lambda q: None, FakeIndex([[0]]), ids, dict(CORPUS), top_n=0)
        with self.assertRaises(ValueError):
            IndexCandidateProvider(lambda q: None, FakeIndex([[0]], ntotal=99), ids, dict(CORPUS))

    def test_missing_artifacts_raise_a_helpful_error(self):
        with self.assertRaises(FileNotFoundError) as caught:
            IndexCandidateProvider.from_frozen_artifacts(corpus_path=Path("nowhere/corpus.jsonl"))
        self.assertIn("gitignored", str(caught.exception))


class PoolMergeTest(unittest.TestCase):
    def test_the_shown_evidence_stays_first_and_duplicates_collapse(self):
        request = item("p0", "p1")
        pool = build_candidate_pool(request, provider([[1, 4, 5]], top_n=3))
        self.assertEqual([candidate.passage_id for candidate in pool], ["p0", "p1", "p4", "p5"])
        self.assertEqual([candidate.rank for candidate in pool], [1, 2, 3, 4])

    def test_the_original_text_wins_for_a_duplicate(self):
        request = item("p0")
        pool = build_candidate_pool(request, provider([[0]], top_n=1))
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0].text, CORPUS["p0"])

    def test_a_deeper_pool_can_add_passages_the_top_k_never_had(self):
        request = item("p0", "p1")
        pool = build_candidate_pool(request, provider([[2, 3, 4, 5]], top_n=4))
        added = {candidate.passage_id for candidate in pool} - {"p0", "p1"}
        self.assertEqual(added, {"p2", "p3", "p4", "p5"})


if __name__ == "__main__":
    unittest.main()
