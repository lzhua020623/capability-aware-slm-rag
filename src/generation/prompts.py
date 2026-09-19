"""Shared prompt text and formatting; no model dependencies."""

PROMPT_INSTRUCTION = (
    "Answer the question using only the provided context. "
    "Return only the shortest answer span. "
    "Do not provide explanations or full sentences."
)
FEVER_PROMPT_INSTRUCTION = (
    "Based only on the provided evidence, determine whether the claim is "
    "supported or refuted. Return only SUPPORTS or REFUTES."
)


def build_rag_prompt(question: str, passages: list[dict]) -> str:
    context_blocks = []
    for i, passage in enumerate(passages, start=1):
        context_blocks.append(f"[{i}] {passage['text']}")
    context = "\n\n".join(context_blocks)
    return (
        f"{PROMPT_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
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
