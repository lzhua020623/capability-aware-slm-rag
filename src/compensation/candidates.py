"""Widen the candidate pool before reranking: retrieve Top-N, rerank, keep Top-k.

The experiments so far reordered the frozen Top-5 and found that ordering it
better changes no answers. That tests only the last step of the usual reranking
recipe. The full recipe retrieves a deeper pool first -- N of 20 to 50 -- and
lets the cross-encoder choose which 5 passages the generator sees, which is a
different mechanism: it can bring in a passage the Top-5 never contained.

There is room for it. On the NQ Train corpus the frozen bi-encoder puts the
gold passage in the Top-5 for only 62.9% of queries
(`results/compensation/nq15_train_label_stats.json`'s source run), and 9.4% of
the held-out accessibility samples have no passage containing the answer string
at all; no reordering of five passages can fix either.

`IndexCandidateProvider` plugs into `EvidenceRefiner(candidate_provider=...)`:
the refiner merges what it returns with the evidence already shown, keeping the
original passages first and deduplicating by passage id, then reranks the union
and keeps `keep_k`.

It needs the frozen retrieval artifacts, which are gitignored and built by
`scripts/build_nq_train_bge_index.py`:

* `data/processed/retrieval/nq_train_corpus.jsonl` -- passage id and text
* `data/processed/indexes/nq_train_bge_base.index` -- FAISS IndexFlatIP
* `data/processed/indexes/nq_train_bge_base_ids.jsonl` -- row -> passage id
* `data/processed/indexes/nq_train_bge_metadata.json` -- fingerprints

The embedding model and the query instruction are read from the frozen config,
so a widened pool is still the project's own retriever, only deeper.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from src.diagnosis.schema import DiagnosisInput, EvidenceItem

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "processed" / "retrieval" / "nq_train_corpus.jsonl"
DEFAULT_INDEX = ROOT / "data" / "processed" / "indexes" / "nq_train_bge_base.index"
DEFAULT_IDS = ROOT / "data" / "processed" / "indexes" / "nq_train_bge_base_ids.jsonl"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEFAULT_TOP_N = 50


def _native_dependencies():
    """Import heavy/native dependencies only after resource validation."""
    import faiss
    from sentence_transformers import SentenceTransformer
    return faiss, SentenceTransformer


def load_corpus(path: Path, wanted: set[str] | None = None) -> dict[str, str]:
    """passage_id -> text. `wanted` keeps only those ids, which saves memory."""
    corpus: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            passage_id = str(row.get("passage_id") or row.get("id"))
            if wanted is not None and passage_id not in wanted:
                continue
            corpus[passage_id] = row["text"]
    return corpus


def load_ids(path: Path) -> list[str]:
    """Row order of the FAISS index -> passage id."""
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(str(row.get("passage_id") or row.get("id")))
    return ids


class IndexCandidateProvider:
    """Deeper Top-N from the frozen index, as `EvidenceItem`s the refiner can rerank.

    Scores carry the retriever's similarity, so a candidate that also appears in
    the shown evidence keeps a comparable score; the refiner deduplicates by
    passage id and keeps the originals' own ranks.
    """

    def __init__(self, encode_query, index, ids: list[str], corpus: dict[str, str], top_n: int = DEFAULT_TOP_N):
        if not callable(encode_query):
            raise TypeError("encode_query must be callable")
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError(f"top_n must be a positive int, got {top_n!r}")
        if len(ids) != getattr(index, "ntotal", len(ids)):
            raise ValueError(f"index holds {index.ntotal} rows but {len(ids)} ids were given")
        self._encode_query = encode_query
        self.index = index
        self.ids = ids
        self.corpus = corpus
        self.top_n = top_n

    @classmethod
    def from_frozen_artifacts(
        cls,
        corpus_path: Path = DEFAULT_CORPUS,
        index_path: Path = DEFAULT_INDEX,
        ids_path: Path = DEFAULT_IDS,
        top_n: int = DEFAULT_TOP_N,
    ) -> IndexCandidateProvider:
        """Load the team's NQ Train BGE index. Raises a clear error when it is absent."""
        if type(top_n) is not int or top_n < 1:
            raise ValueError('top_n must be a positive integer')
        for path in (corpus_path, index_path, ids_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing. The retrieval artifacts are gitignored; build them with "
                    "scripts/build_nq_train_bge_index.py or copy them from whoever ran it."
                )
        from src.controller.nq_train_resources import validate_resources
        registry, paths, receipt = validate_resources(
            ROOT / 'configs/controller/nq_train_resources_v1.json', root=ROOT)
        for name, path in (('corpus', corpus_path), ('index', index_path), ('ids', ids_path)):
            if Path(path).resolve() != paths[name].resolve():
                raise ValueError('Custom candidate artifacts need a separate registered resource manifest')
        # Complete resource preflight before importing native/model libraries.
        faiss, SentenceTransformer = _native_dependencies()
        encoder = SentenceTransformer(registry['model_name'],
            revision=registry['verified_model_revision'], device='cpu')

        def encode_query(question: str):
            return encoder.encode(
                [registry['query_instruction'] + question],
                normalize_embeddings=True, convert_to_numpy=True,
            )

        ids = load_ids(ids_path)
        index = faiss.read_index(str(index_path))
        if (type(index).__name__ != 'IndexFlatIP' or index.d != registry['embedding_dimension']
                or index.ntotal != registry['corpus_size']):
            raise ValueError('Candidate index structure differs from registered resources')
        provider = cls(encode_query, index, ids, load_corpus(corpus_path), top_n)
        provider.resource_receipt = receipt
        return provider

    def __call__(self, item: DiagnosisInput) -> Iterable[EvidenceItem]:
        scores, rows = self.index.search(self._encode_query(item.query), self.top_n)
        candidates = []
        for rank, (row, score) in enumerate(zip(rows[0], scores[0]), start=1):
            if row < 0:
                continue  # FAISS pads a short result with -1
            passage_id = self.ids[row]
            text = self.corpus.get(passage_id)
            if text is None:
                continue  # the corpus and the index disagree; skip rather than invent a passage
            candidates.append(EvidenceItem(rank=rank, passage_id=passage_id, text=text, score=float(score)))
        return candidates
