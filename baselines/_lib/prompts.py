"""Closed-book + RAG prompt variants for baselines.

Fairness: reuse the same answer-format instruction as main's PROMPT_INSTRUCTION;
RAG context_only keeps the main clause \"using only the provided context\".
context_fallback is identical except it allows parametric knowledge when context
is insufficient. Context block formatting matches src.generation.qwen.build_rag_prompt.
"""

from __future__ import annotations

from src.generation.qwen import (
    FEVER_PROMPT_INSTRUCTION,
    PROMPT_INSTRUCTION,
    build_fever_rag_prompt,
    build_rag_prompt,
)

# Strip the evidence-only clause; keep shortest-span / no-explanation wording identical.
CLOSED_BOOK_INSTRUCTION = PROMPT_INSTRUCTION.replace(
    " using only the provided context",
    "",
)

# Same shortest-span / no-explanation constraints; only the context policy differs.
CONTEXT_FALLBACK_INSTRUCTION = (
    "Answer the question. Prefer the provided context when it contains the answer. "
    "If the context is insufficient, you may use your own knowledge. "
    "Return only the shortest answer span. "
    "Do not provide explanations or full sentences."
)

RAG_PROMPT_STYLES = ("context_only", "context_fallback")

# Closed-book FEVER: binary label only; no evidence block (Qwen-only baseline).
FEVER_CLOSED_BOOK_INSTRUCTION = (
    "Determine whether the claim is supported or refuted. "
    "Return only SUPPORTS or REFUTES. "
    "Do not provide explanations."
)


def build_closed_book_prompt(question: str) -> str:
    """Query-only prompt (no retrieved evidence). Same Question: footer as RAG."""
    return f"{CLOSED_BOOK_INSTRUCTION}\n\nQuestion: {question}"


def build_fever_closed_book_prompt(claim: str) -> str:
    """Claim-only FEVER prompt (no retrieved evidence)."""
    return f"{FEVER_CLOSED_BOOK_INSTRUCTION}\n\nClaim: {claim}"


def build_fever_baseline_rag_prompt(claim: str, passages: list[dict]) -> str:
    """FEVER RAG prompt: Claim + Top-k evidence → SUPPORTS|REFUTES only."""
    # Delegate to main-tree builder so wording stays frozen with C1 / product.
    prompt = build_fever_rag_prompt(claim, passages)
    if FEVER_PROMPT_INSTRUCTION not in prompt:
        raise RuntimeError("FEVER RAG prompt missing FEVER_PROMPT_INSTRUCTION")
    if "Evidence:" not in prompt:
        raise RuntimeError("FEVER RAG prompt missing Evidence block")
    return prompt


def _format_rag_body(question: str, passages: list[dict], instruction: str) -> str:
    context_blocks = []
    for i, passage in enumerate(passages, start=1):
        context_blocks.append(f"[{i}] {passage['text']}")
    context = "\n\n".join(context_blocks)
    return f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}"


def build_baseline_rag_prompt(
    question: str,
    passages: list[dict],
    *,
    prompt_style: str = "context_only",
) -> str:
    """Build RAG user content. context_only delegates to main build_rag_prompt."""
    if prompt_style == "context_only":
        return build_rag_prompt(question, passages)
    if prompt_style == "context_fallback":
        return _format_rag_body(question, passages, CONTEXT_FALLBACK_INSTRUCTION)
    raise ValueError(
        f"Unknown rag prompt_style {prompt_style!r}; expected one of {RAG_PROMPT_STYLES}"
    )
