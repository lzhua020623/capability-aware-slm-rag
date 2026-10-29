"""Formal matched Reference RAG → DiagnosisInput.

Reuses the NQ-Train BGE FAISS stack and FREEZE_CONTEXT_FALLBACK prompt validated
on feature/baseline-evaluation (baselines/configs/rag_matched.json).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.diagnosis.schema import DiagnosisInput, EvidenceItem, parse_dataset
from src.generation.prompts import RAG_PROMPT_STYLES, build_baseline_rag_prompt
from src.retrieval.matched_nq_train import (
    CorpusLookup,
    assert_not_dev_index,
    load_matched_retriever,
    retrieve_top_k_matched,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/controller/reference_rag_v1.json"
FORMAL_TOP_K = 5
FORMAL_PROMPT_STYLE = "context_fallback"
FORMAL_EMBEDDING = "BAAI/bge-base-en-v1.5"
FORMAL_GENERATOR = "Qwen/Qwen2.5-7B-Instruct"


GenerateFn = Callable[[Any, Any, str, int], tuple[str, float]]


def _default_generate(tokenizer, model, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    from src.generation.qwen import generate_greedy

    return generate_greedy(tokenizer, model, prompt, max_new_tokens)


def full_prefill_generate(tokenizer, model, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    """C1 with full prefill, as in the frozen RAG baseline.

    The shared Qwen carries prefill_chunk_size=256 for the enhancement modules;
    it is cleared only for this call and restored afterwards, even on error.
    """
    config = model.generation_config
    original = config.prefill_chunk_size
    config.prefill_chunk_size = None
    try:
        return _default_generate(tokenizer, model, prompt, max_new_tokens)
    finally:
        config.prefill_chunk_size = original


def load_reference_rag_config(path: Path | str | None = None, *, root: Path | str = ROOT) -> dict:
    root = Path(root)
    cfg_path = Path(path) if path is not None else root / DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Reference RAG config missing: {cfg_path}")
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    validate_reference_rag_config(payload, root=root)
    payload = dict(payload)
    payload["_config_path"] = str(cfg_path)
    return payload


def validate_reference_rag_config(cfg: dict, *, root: Path | str = ROOT) -> None:
    """Refuse legacy Dev index and silent drift from the frozen matched contract."""
    root = Path(root)
    retrieval = cfg.get("retrieval") or {}
    prompt = cfg.get("rag_prompt") or {}
    generator = cfg.get("generator") or {}

    index_rel = retrieval.get("index_path")
    if not index_rel:
        raise ValueError("reference_rag config missing retrieval.index_path")
    assert_not_dev_index(root / index_rel)
    assert_not_dev_index(index_rel)

    if int(retrieval.get("top_k", -1)) != FORMAL_TOP_K:
        raise ValueError(f"Formal Reference RAG requires top_k={FORMAL_TOP_K}")
    if retrieval.get("embedding_model") != FORMAL_EMBEDDING:
        raise ValueError(f"Formal Reference RAG requires embedding_model={FORMAL_EMBEDDING!r}")
    if retrieval.get("backend") != "corpus_offsets":
        raise ValueError("Formal Reference RAG requires backend=corpus_offsets")
    style = prompt.get("default_style")
    if style != FORMAL_PROMPT_STYLE:
        raise ValueError(f"Formal Reference RAG requires prompt style {FORMAL_PROMPT_STYLE!r}")
    if style not in RAG_PROMPT_STYLES:
        raise ValueError(f"Unknown rag prompt style {style!r}")
    if generator.get("model_name") != FORMAL_GENERATOR:
        raise ValueError(f"Formal Reference RAG requires generator {FORMAL_GENERATOR!r}")
    if int(generator.get("max_new_tokens", -1)) != 64:
        raise ValueError("Formal Reference RAG requires max_new_tokens=64")

    for key in ("index_path", "ids_path", "offsets_path", "corpus_path"):
        rel = retrieval[key]
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"Matched retrieval asset missing: {path}")
        if "nq_bge_base" in str(rel).replace("\\", "/") and "nq_train_bge_base" not in str(rel):
            raise ValueError(f"Refusing legacy Dev retrieval asset: {rel}")


def evidence_from_hits(hits: list[dict]) -> list[EvidenceItem]:
    if len(hits) != FORMAL_TOP_K:
        raise ValueError(f"Expected Top-{FORMAL_TOP_K} evidence, got {len(hits)}")
    items: list[EvidenceItem] = []
    for expected_rank, hit in enumerate(hits, start=1):
        rank = int(hit.get("rank", expected_rank))
        if rank != expected_rank:
            raise ValueError(f"evidence ranks must be 1..{FORMAL_TOP_K} contiguous, got {rank} at {expected_rank}")
        text = hit.get("text") or ""
        if not str(text).strip():
            raise ValueError(f"evidence[{rank}] text must be non-empty")
        score = float(hit["score"])
        items.append(
            EvidenceItem(
                rank=rank,
                passage_id=str(hit["passage_id"]),
                text=str(text),
                score=score,
            )
        )
    return items


class MatchedReferenceRAG:
    """BGE Top-5 + Qwen2.5-7B context_fallback → DiagnosisInput."""

    def __init__(
        self,
        *,
        retriever: Any,
        lookup: CorpusLookup,
        passage_ids: list[str],
        tokenizer: Any,
        model: Any,
        top_k: int = FORMAL_TOP_K,
        max_new_tokens: int = 64,
        prompt_style: str = FORMAL_PROMPT_STYLE,
        embedding_model: str = FORMAL_EMBEDDING,
        generate_fn: GenerateFn | None = None,
        config: dict | None = None,
    ) -> None:
        if top_k != FORMAL_TOP_K:
            raise ValueError(f"Formal Reference RAG requires top_k={FORMAL_TOP_K}")
        if prompt_style != FORMAL_PROMPT_STYLE:
            raise ValueError(f"Formal Reference RAG requires prompt style {FORMAL_PROMPT_STYLE!r}")
        self.retriever = retriever
        self.lookup = lookup
        self.passage_ids = passage_ids
        self.tokenizer = tokenizer
        self.model = model
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.prompt_style = prompt_style
        self.embedding_model = embedding_model
        self._generate = generate_fn or _default_generate
        self.config = config or {}

    @classmethod
    def from_config(
        cls,
        config_path: Path | str | None = None,
        *,
        root: Path | str = ROOT,
        tokenizer: Any,
        model: Any,
        device: str | None = None,
        generate_fn: GenerateFn | None = None,
    ) -> MatchedReferenceRAG:
        root = Path(root)
        cfg = load_reference_rag_config(config_path, root=root)
        ret = cfg["retrieval"]
        retriever, lookup, passage_ids = load_matched_retriever(
            embedding_model=ret["embedding_model"],
            index_path=root / ret["index_path"],
            ids_path=root / ret["ids_path"],
            offsets_path=root / ret["offsets_path"],
            corpus_path=root / ret["corpus_path"],
            passage_prefix=ret.get("passage_prefix", ""),
            query_prefix=ret.get("query_prefix", ""),
            expected_ntotal=ret.get("expected_ntotal"),
            device=device,
        )
        return cls(
            retriever=retriever,
            lookup=lookup,
            passage_ids=passage_ids,
            tokenizer=tokenizer,
            model=model,
            top_k=int(ret["top_k"]),
            max_new_tokens=int(cfg["generator"]["max_new_tokens"]),
            prompt_style=str(cfg["rag_prompt"]["default_style"]),
            embedding_model=str(ret["embedding_model"]),
            generate_fn=generate_fn,
            config=cfg,
        )

    @classmethod
    def from_shared_generator(
        cls,
        generator: Any,
        *,
        root: Path | str = ROOT,
        config_path: Path | str | None = None,
        device: str | None = None,
        generate_fn: GenerateFn | None = None,
    ) -> MatchedReferenceRAG:
        tokenizer = getattr(generator, "tokenizer", None)
        model = getattr(generator, "model", None)
        if tokenizer is None or model is None:
            raise TypeError("generator must expose tokenizer and model for Reference RAG")
        return cls.from_config(
            config_path,
            root=root,
            tokenizer=tokenizer,
            model=model,
            device=device,
            generate_fn=generate_fn,
        )

    def close(self) -> None:
        close = getattr(self.lookup, "close", None)
        if callable(close):
            close()
        unload = getattr(self.retriever, "unload", None)
        if callable(unload):
            unload()

    def run(
        self,
        query: str,
        *,
        sample_id: str | None = None,
        dataset: str = "nq",
    ) -> DiagnosisInput:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        parse_dataset(dataset)
        hits = retrieve_top_k_matched(
            self.retriever,
            self.lookup,
            self.passage_ids,
            [query],
            self.top_k,
        )[0]
        evidence = evidence_from_hits(hits)
        prompt = build_baseline_rag_prompt(
            query,
            [{"text": item.text} for item in evidence],
            prompt_style=self.prompt_style,
        )
        answer, _latency = self._generate(
            self.tokenizer, self.model, prompt, self.max_new_tokens
        )
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Reference RAG generator returned an empty answer")
        sid = sample_id if sample_id is not None else f"runtime-{uuid.uuid4().hex}"
        return DiagnosisInput(
            sample_id=sid,
            dataset=dataset,
            query=query,
            evidence=evidence,
            initial_answer=answer.strip(),
        )
