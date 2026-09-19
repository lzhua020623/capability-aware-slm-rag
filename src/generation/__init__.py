"""Generation components."""

from src.generation.prompts import build_fever_rag_prompt, build_rag_prompt

__all__ = [
    "apply_qwen_chat_template",
    "build_fever_rag_prompt",
    "build_rag_prompt",
    "free_cuda_memory",
    "generate_greedy",
    "load_qwen_nf4",
]


def __getattr__(name):
    if name in __all__:
        from importlib import import_module

        return getattr(import_module("src.generation.qwen"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
