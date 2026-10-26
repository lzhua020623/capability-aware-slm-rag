"""Closed-book prompt for qwen_only. RAG prompts come from src.generation.qwen."""

from __future__ import annotations

CLOSED_BOOK_INSTRUCTION = (
    "Answer the question. "
    "Return only the shortest answer span. "
    "Do not provide explanations or full sentences."
)


def build_closed_book_prompt(question: str) -> str:
    """Query-only prompt (no retrieved evidence)."""
    return f"{CLOSED_BOOK_INSTRUCTION}\n\nQuestion: {question}"
