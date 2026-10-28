"""Online P26 semantic runtime features.

Exact formulas recovered from the deleted Temp generator
``elec5623_semantic_runtime_features_p26`` (agent transcript Write + StrReplace
history) and validated against ``semantic_arrays_v3.npz`` /
``semantic_meta_v3.json``.

Models:
  - BAAI/bge-base-en-v1.5  (max_seq_length=128, L2-normalized encode)
  - BAAI/bge-reranker-base (CrossEncoder; 2 pairs / query)

No historical sample_id lookup. Fail-fast on missing resources / bad shapes.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

BGE_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
BGE_MAX_SEQ_LENGTH = 128
PASSAGE_TRUNCATE = 512
S5_TRUNCATE = 512
DEDUP_KEY_MAX = 600

Q_TYPE_NAMES: tuple[str, ...] = (
    "WHO",
    "WHEN",
    "WHERE",
    "HOW_MANY",
    "WHAT",
    "WHICH",
    "HOW",
    "OTHER",
)
A_TYPE_NAMES: tuple[str, ...] = (
    "PERSON",
    "DATE",
    "NUMBER",
    "LOCATION",
    "SHORT_TEXT",
    "LONG_TEXT",
)

MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

# Scalar + derived names as assembled into P60 (P53 build_p26_matrix order).
P26_SCALAR_NAMES: tuple[str, ...] = (
    "qa_support_max",
    "qa_support_mean_top3",
    "qa_support_first_rank",
    "qa_support_margin",
    "qa_support_concentration",
    "support_peak_dispersion",
    "top_support_margin",
    "evidence_agreement_mean",
    "evidence_agreement_min",
    "answer_evidence_max_similarity",
    "answer_evidence_top1_similarity",
    "answer_evidence_mean_topk",
    "qa_type_match",
    "s5_rerank_best_chunk",
    "s5_rerank_top1_passage",
)

P26_FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"p26_{n}" for n in P26_SCALAR_NAMES]
    + ["p26_chunk_max", "p26_chunk_mean", "p26_chunk_std"]
    + ["p26_pass_max", "p26_pass_mean", "p26_pass_std"]
    + [f"p26_qtype_{j}" for j in range(8)]
    + [f"p26_atype_{j}" for j in range(6)]
)


def normalize_query(q: str) -> str:
    return " ".join(q.lower().split())


def question_type(query: str) -> str:
    toks = normalize_query(query).split()
    if not toks:
        return "OTHER"
    if len(toks) >= 2 and toks[0] == "how" and toks[1] in {"many", "much"}:
        return "HOW_MANY"
    mapping = {
        "who": "WHO",
        "whom": "WHO",
        "whose": "WHO",
        "when": "WHEN",
        "where": "WHERE",
        "which": "WHICH",
        "what": "WHAT",
        "how": "HOW",
    }
    return mapping.get(toks[0], "OTHER")


def answer_type(answer: str) -> str:
    a = " ".join((answer or "").split())
    if not a:
        return "SHORT_TEXT"
    al = a.lower()
    if re.fullmatch(r"[\d,.\-%]+", a.replace(" ", "")) or re.search(
        r"^\d+(\.\d+)?\s*(%|percent|million|billion|thousand|years?|days?|km|miles?)?$", al
    ):
        return "NUMBER"
    if re.search(rf"\b({MONTHS})\b", al) or re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", al) or re.search(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", al
    ):
        return "DATE"
    tokens = a.split()
    if (
        1 <= len(tokens) <= 4
        and all(t[:1].isupper() for t in tokens if t.isalpha())
        and not any(ch.isdigit() for ch in a)
        and not re.search(
            r"\b(city|county|state|province|country|river|ocean|sea|island|university|company)\b", al
        )
    ):
        return "PERSON"
    if re.search(
        r"\b(city|county|state|province|country|river|ocean|sea|island|united states|uk|usa)\b", al
    ) or (1 <= len(tokens) <= 5 and all(t[:1].isupper() for t in tokens if t.isalpha())):
        return "LOCATION"
    return "SHORT_TEXT" if len(tokens) <= 6 else "LONG_TEXT"


def qa_compatible(qtype: str, atype: str) -> float:
    mapping = {
        "WHO": {"PERSON"},
        "WHEN": {"DATE", "NUMBER"},
        "WHERE": {"LOCATION", "SHORT_TEXT"},
        "HOW_MANY": {"NUMBER"},
        "WHAT": {"SHORT_TEXT", "LONG_TEXT", "NUMBER", "DATE", "PERSON", "LOCATION"},
        "WHICH": {"SHORT_TEXT", "LONG_TEXT", "PERSON", "LOCATION", "NUMBER", "DATE"},
        "HOW": {"SHORT_TEXT", "LONG_TEXT", "NUMBER"},
        "OTHER": {"SHORT_TEXT", "LONG_TEXT", "PERSON", "LOCATION", "NUMBER", "DATE"},
    }
    return 1.0 if atype in mapping.get(qtype, set()) else 0.0


def _encode_texts(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Dedup encode identical to P26 (key = text if len<=600 else text[:600])."""
    if not texts:
        dim = int(model.get_sentence_embedding_dimension())
        return np.zeros((0, dim), dtype=np.float32)
    key_to_idx: dict[str, int] = {}
    unique: list[str] = []
    remap = np.empty(len(texts), dtype=np.int32)
    for i, t in enumerate(texts):
        k = t if len(t) <= DEDUP_KEY_MAX else t[:DEDUP_KEY_MAX]
        if k not in key_to_idx:
            key_to_idx[k] = len(unique)
            unique.append(k)
        remap[i] = key_to_idx[k]
    emb_u = model.encode(
        unique,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb_u, dtype=np.float32)[remap]


def _require_five_passages(passages: Sequence[str]) -> list[str]:
    if len(passages) != 5:
        raise ValueError(f"P26 requires exactly 5 evidence passages, got {len(passages)}")
    out = []
    for t in passages:
        s = str(t or "").strip() or "[empty]"
        out.append(s[:PASSAGE_TRUNCATE])
    return out


@dataclass
class P26SemanticResult:
    """Raw arrays for one sample (shapes match semantic_arrays_v3 row)."""

    qa_support_max: float
    qa_support_mean_top3: float
    qa_support_first_rank: float
    qa_support_margin: float
    qa_support_concentration: float
    chunk_support: np.ndarray  # (5,)
    pass_support: np.ndarray  # (5,)
    support_peak_dispersion: float
    top_support_margin: float
    evidence_agreement_mean: float
    evidence_agreement_min: float
    answer_evidence_max_similarity: float
    answer_evidence_top1_similarity: float
    answer_evidence_mean_topk: float
    qtypes: np.ndarray  # (8,)
    atypes: np.ndarray  # (6,)
    qa_type_match: float
    s5_rerank_best_chunk: float
    s5_rerank_top1_passage: float

    def p60_vector(self) -> np.ndarray:
        """35-dim vector in P53 ``build_p26_matrix`` / P60 order."""
        chunk = np.asarray(self.chunk_support, dtype=np.float64)
        pass_ = np.asarray(self.pass_support, dtype=np.float64)
        scalars = np.array(
            [
                self.qa_support_max,
                self.qa_support_mean_top3,
                self.qa_support_first_rank,
                self.qa_support_margin,
                self.qa_support_concentration,
                self.support_peak_dispersion,
                self.top_support_margin,
                self.evidence_agreement_mean,
                self.evidence_agreement_min,
                self.answer_evidence_max_similarity,
                self.answer_evidence_top1_similarity,
                self.answer_evidence_mean_topk,
                self.qa_type_match,
                self.s5_rerank_best_chunk,
                self.s5_rerank_top1_passage,
            ],
            dtype=np.float64,
        )
        derived = np.array(
            [
                float(chunk.max()),
                float(chunk.mean()),
                float(chunk.std()),
                float(pass_.max()),
                float(pass_.mean()),
                float(pass_.std()),
            ],
            dtype=np.float64,
        )
        return np.concatenate(
            [scalars, derived, self.qtypes.astype(np.float64), self.atypes.astype(np.float64)]
        )


class P26SemanticRuntime:
    """Lazy-loaded BGE + CrossEncoder runtime for online P26 features."""

    def __init__(self, *, device: str | None = None):
        self.device = device
        self._bge = None
        self._ce = None
        self._last_calls: list[dict] = []

    def runtime_calls(self) -> list[dict]:
        return list(self._last_calls)

    def _ensure_models(self) -> None:
        if self._bge is not None and self._ce is not None:
            return
        try:
            import torch
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "P26 semantic runtime requires sentence-transformers + torch "
                f"(BGE={BGE_MODEL}, reranker={RERANKER_MODEL}): {exc}"
            ) from exc
        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        try:
            self._bge = SentenceTransformer(BGE_MODEL, device=device)
            self._bge.max_seq_length = BGE_MAX_SEQ_LENGTH
        except Exception as exc:
            raise RuntimeError(f"Failed to load BGE model {BGE_MODEL}: {exc}") from exc
        try:
            self._ce = CrossEncoder(RERANKER_MODEL, device=device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load reranker {RERANKER_MODEL}: {exc}") from exc

    def compute(
        self,
        query: str,
        initial_answer: str,
        passage_texts: Sequence[str],
    ) -> P26SemanticResult:
        self._ensure_models()
        self._last_calls = []
        passages = _require_five_passages(passage_texts)
        q = str(query)
        a = str(initial_answer or "")
        qa_text = f"Question: {q}\nCandidate answer: {a}"
        ans_text = a if a.strip() else "[empty]"
        qa_ce = f"Question: {q} Candidate answer: {a}"

        t0 = time.perf_counter()
        try:
            qa_emb = _encode_texts(self._bge, [qa_text], batch_size=1)[0]
            ans_emb = _encode_texts(self._bge, [ans_text], batch_size=1)[0]
            pass_emb = _encode_texts(self._bge, passages, batch_size=5)
        except Exception as exc:
            self._last_calls.append(
                {
                    "kind": "bge_encode",
                    "attempted": True,
                    "seconds": time.perf_counter() - t0,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            raise RuntimeError(f"BGE encode failed: {exc}") from exc
        self._last_calls.append(
            {
                "kind": "bge_encode",
                "attempted": True,
                "seconds": time.perf_counter() - t0,
                "status": "completed",
                "n_texts": 2 + len(passages),
                "model": BGE_MODEL,
            }
        )
        if pass_emb.shape != (5, qa_emb.shape[0]):
            raise RuntimeError(f"Unexpected BGE embedding shapes: pass={pass_emb.shape}")

        sims_p = pass_emb @ qa_emb
        if not np.isfinite(sims_p).all():
            raise RuntimeError("Non-finite chunk_support from BGE")
        order = np.argsort(-sims_p)
        top = sims_p[order]
        qa_support_max = float(top[0])
        qa_support_mean_top3 = float(top[: min(3, len(top))].mean())
        qa_support_margin = float(top[0] - top[1]) if len(top) > 1 else float(top[0])
        pos = np.clip(sims_p, 0, None)
        qa_support_concentration = (
            float(pos.max() / pos.sum()) if float(pos.sum()) > 1e-8 else 0.0
        )
        qa_support_first_rank = float(order[0] + 1)

        sims_ae = pass_emb @ ans_emb
        answer_evidence_max = float(sims_ae.max())
        answer_evidence_top1 = float(sims_ae[0])
        answer_evidence_mean = float(sims_ae.mean())

        grams = [float(pass_emb[i] @ pass_emb[j]) for i in range(5) for j in range(i + 1, 5)]
        evidence_agreement_mean = float(np.mean(grams))
        evidence_agreement_min = float(np.min(grams))

        ps_sorted = np.sort(sims_p)[::-1]
        top_support_margin = float(ps_sorted[0] - ps_sorted[1])
        thr_rel = float(ps_sorted[0] - 0.05)
        peaks = np.where(sims_p >= thr_rel)[0]
        if len(peaks) <= 1:
            support_peak_dispersion = 0.0
        else:
            pe_sel = pass_emb[peaks]
            dists = [
                1.0 - float(pe_sel[i] @ pe_sel[j])
                for i in range(len(peaks))
                for j in range(i + 1, len(peaks))
            ]
            support_peak_dispersion = float(np.mean(dists))

        bi = int(order[0])
        best_chunk = passages[bi][:S5_TRUNCATE]
        top1_chunk = passages[0][:S5_TRUNCATE]
        pairs_best = [(qa_ce, best_chunk)]
        pairs_top1 = [(qa_ce, top1_chunk)]

        t1 = time.perf_counter()
        try:
            s5_best = float(
                np.asarray(
                    self._ce.predict(pairs_best, batch_size=1, show_progress_bar=False),
                    dtype=np.float32,
                ).reshape(-1)[0]
            )
            s5_top1 = float(
                np.asarray(
                    self._ce.predict(pairs_top1, batch_size=1, show_progress_bar=False),
                    dtype=np.float32,
                ).reshape(-1)[0]
            )
        except Exception as exc:
            self._last_calls.append(
                {
                    "kind": "bge_reranker",
                    "attempted": True,
                    "seconds": time.perf_counter() - t1,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            raise RuntimeError(f"CrossEncoder rerank failed: {exc}") from exc
        self._last_calls.append(
            {
                "kind": "bge_reranker",
                "attempted": True,
                "seconds": time.perf_counter() - t1,
                "status": "completed",
                "n_pairs": 2,
                "model": RERANKER_MODEL,
            }
        )
        if not np.isfinite([s5_best, s5_top1]).all():
            raise RuntimeError("Non-finite S5 reranker scores")

        qt = question_type(q)
        at = answer_type(a)
        qtypes = np.zeros(8, dtype=np.float32)
        atypes = np.zeros(6, dtype=np.float32)
        qtypes[Q_TYPE_NAMES.index(qt)] = 1.0
        atypes[A_TYPE_NAMES.index(at)] = 1.0
        qa_match = float(qa_compatible(qt, at))

        chunk = np.asarray(sims_p, dtype=np.float32)
        return P26SemanticResult(
            qa_support_max=qa_support_max,
            qa_support_mean_top3=qa_support_mean_top3,
            qa_support_first_rank=qa_support_first_rank,
            qa_support_margin=qa_support_margin,
            qa_support_concentration=qa_support_concentration,
            chunk_support=chunk,
            pass_support=chunk.copy(),
            support_peak_dispersion=support_peak_dispersion,
            top_support_margin=top_support_margin,
            evidence_agreement_mean=evidence_agreement_mean,
            evidence_agreement_min=evidence_agreement_min,
            answer_evidence_max_similarity=answer_evidence_max,
            answer_evidence_top1_similarity=answer_evidence_top1,
            answer_evidence_mean_topk=answer_evidence_mean,
            qtypes=qtypes,
            atypes=atypes,
            qa_type_match=qa_match,
            s5_rerank_best_chunk=s5_best,
            s5_rerank_top1_passage=s5_top1,
        )
