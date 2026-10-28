"""Shared prompt text and formatting; no model dependencies."""

PROMPT_INSTRUCTION = (
    "Answer the question using only the provided context. "
    "Return only the shortest answer span. "
    "Do not provide explanations or full sentences."
)
# Formal matched Reference RAG (FREEZE_CONTEXT_FALLBACK on feature/baseline-evaluation).
CONTEXT_FALLBACK_INSTRUCTION = (
    "Answer the question. Prefer the provided context when it contains the answer. "
    "If the context is insufficient, you may use your own knowledge. "
    "Return only the shortest answer span. "
    "Do not provide explanations or full sentences."
)
FEVER_PROMPT_INSTRUCTION = (
    "Based only on the provided evidence, determine whether the claim is "
    "supported or refuted. Return only SUPPORTS or REFUTES."
)
RAG_PROMPT_STYLES = ("context_only", "context_fallback")


def build_rag_prompt(question: str, passages: list[dict]) -> str:
    # Keep this body self-contained: P3 frozen AST tests extract only
    # PROMPT_INSTRUCTION + build_rag_prompt without helper helpers.
    context_blocks = []
    for i, passage in enumerate(passages, start=1):
        context_blocks.append(f"[{i}] {passage['text']}")
    context = "\n\n".join(context_blocks)
    return (
        f"{PROMPT_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def build_context_fallback_rag_prompt(question: str, passages: list[dict]) -> str:
    """Frozen matched-RAG prompt (baseline FREEZE_CONTEXT_FALLBACK)."""
    context_blocks = []
    for i, passage in enumerate(passages, start=1):
        context_blocks.append(f"[{i}] {passage['text']}")
    context = "\n\n".join(context_blocks)
    return (
        f"{CONTEXT_FALLBACK_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def build_baseline_rag_prompt(
    question: str,
    passages: list[dict],
    *,
    prompt_style: str = "context_only",
) -> str:
    if prompt_style == "context_only":
        return build_rag_prompt(question, passages)
    if prompt_style == "context_fallback":
        return build_context_fallback_rag_prompt(question, passages)
    raise ValueError(
        f"Unknown rag prompt_style {prompt_style!r}; expected one of {RAG_PROMPT_STYLES}"
    )


def build_fever_rag_prompt(claim: str, passages: list[dict]) -> str:
    context_blocks = []
    for i, passage in enumerate(passages, start=1):
        context_blocks.append(f"[{i}] {passage['text']}")
    context = "\n\n".join(context_blocks)
    return (
        f"{FEVER_PROMPT_INSTRUCTION}\n\n"
        f"Evidence:\n{context}\n\n"
        f"Claim: {claim}"
    )
