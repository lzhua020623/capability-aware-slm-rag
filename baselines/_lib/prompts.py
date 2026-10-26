"""Closed-book prompt for qwen_only. RAG prompts come from src.generation.qwen.

Fairness: reuse the same answer-format instruction as main's PROMPT_INSTRUCTION;
the only intentional difference is the RAG clause \"using only the provided context\"
(and the Context block itself).
"""

from __future__ import annotations

from src.generation.qwen import PROMPT_INSTRUCTION

# Strip the evidence-only clause; keep shortest-span / no-explanation wording identical.
CLOSED_BOOK_INSTRUCTION = PROMPT_INSTRUCTION.replace(
    " using only the provided context",
    "",
)


def build_closed_book_prompt(question: str) -> str:
    """Query-only prompt (no retrieved evidence). Same Question: footer as RAG."""
    return f"{CLOSED_BOOK_INSTRUCTION}\n\nQuestion: {question}"
