"""NQ-Train matched BGE FAISS loader (formal Reference RAG stack).

Ported from feature/baseline-evaluation ``baselines/_lib/matched_retrieval.py``.
Refuses the legacy Dev ``nq_bge_base`` index.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.retrieval.e5_faiss import E5FaissRetriever

class CorpusLookup:
    """Random-access passage rows by FAISS row id (byte offset into corpus JSONL)."""

    def __init__(self, corpus_path: Path, offsets: np.ndarray):
        self.offsets = offsets
        self.handle = Path(corpus_path).open("r", encoding="utf-8")

    def get(self, faiss_id: int) -> dict:
        self.handle.seek(int(self.offsets[faiss_id]))
        return json.loads(self.handle.readline())

    def close(self) -> None:
        self.handle.close()


def assert_not_dev_index(index_path: Path | str) -> None:
    """Hard refuse invalidated Dev-index path for formal / matched RAG."""
    text = str(index_path).replace("\\", "/")
    name = Path(index_path).name
    if name == "nq_bge_base.index" or text.endswith("/nq_bge_base.index"):
        raise ValueError(
            "Legacy Dev index nq_bge_base is invalidated for formal Reference RAG; "
            "use nq_train_bge_base matched stack"
        )
    if "nq_bge_base" in text and "nq_train_bge_base" not in text:
        raise ValueError(
            f"Refusing non-matched BGE index path: {index_path}"
        )


def load_passage_ids(ids_path: Path) -> list[str]:
    ids: list[str] = []
    with Path(ids_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line)["passage_id"])
    return ids


def load_matched_retriever(
    *,
    embedding_model: str,
    index_path: Path,
    ids_path: Path,
    offsets_path: Path,
    corpus_path: Path,
    passage_prefix: str = "",
    query_prefix: str = "",
    expected_ntotal: int | None = None,
    device: str | None = None,
) -> tuple[E5FaissRetriever, CorpusLookup, list[str]]:
    """Return (retriever with FAISS loaded, corpus lookup, passage_ids)."""
    index_path = Path(index_path)
    ids_path = Path(ids_path)
    offsets_path = Path(offsets_path)
    corpus_path = Path(corpus_path)
    assert_not_dev_index(index_path)
    for path in (index_path, ids_path, offsets_path, corpus_path):
        if not path.is_file():
            raise FileNotFoundError(f"Matched retrieval asset missing: {path}")

    passage_ids = load_passage_ids(ids_path)
    offsets = np.load(offsets_path)
    if len(passage_ids) != len(offsets):
        raise RuntimeError(
            f"ids/offsets length mismatch: {len(passage_ids)} vs {len(offsets)}"
        )

    index = faiss.read_index(str(index_path))
    if expected_ntotal is not None and int(index.ntotal) != int(expected_ntotal):
        raise RuntimeError(
            f"FAISS ntotal {index.ntotal} != expected_ntotal {expected_ntotal}"
        )
    if int(index.ntotal) != len(passage_ids):
        raise RuntimeError(
            f"FAISS ntotal {index.ntotal} != ids {len(passage_ids)}"
        )

    retriever = E5FaissRetriever(
        embedding_model,
        device=device,
        passage_prefix=passage_prefix,
        query_prefix=query_prefix,
    )
    retriever.index = index
    retriever.passage_ids = passage_ids
    lookup = CorpusLookup(corpus_path, offsets)
    return retriever, lookup, passage_ids


def retrieve_top_k_matched(
    retriever: E5FaissRetriever,
    lookup: CorpusLookup,
    passage_ids: list[str],
    questions: list[str],
    top_k: int,
    *,
    batch_size: int = 16,
) -> list[list[dict]]:
    """Top-k hits with passage text resolved via corpus offsets."""
    query_embeddings = retriever.encode_queries(questions, batch_size=batch_size)
    scores, indices = retriever.search(query_embeddings, top_k=top_k)
    all_hits: list[list[dict]] = []
    for row_scores, row_indices in zip(scores, indices):
        hits: list[dict] = []
        for rank, (score, idx) in enumerate(zip(row_scores, row_indices), start=1):
            faiss_id = int(idx)
            if faiss_id < 0:
                continue
            row = lookup.get(faiss_id)
            pid = row["passage_id"]
            if pid != passage_ids[faiss_id]:
                raise RuntimeError(
                    f"passage id misalignment at faiss_id={faiss_id}: "
                    f"{pid!r} vs {passage_ids[faiss_id]!r}"
                )
            text = row.get("text") or ""
            if not str(text).strip():
                raise RuntimeError(f"empty passage text at faiss_id={faiss_id} pid={pid}")
            hits.append(
                {
                    "passage_id": pid,
                    "text": text,
                    "score": float(score),
                    "rank": rank,
                }
            )
        if len(hits) != top_k:
            raise RuntimeError(f"Expected {top_k} hits, got {len(hits)}")
        all_hits.append(hits)
    return all_hits
