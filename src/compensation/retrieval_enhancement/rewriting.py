"""Injectable rewriting contract and intentionally narrow, LLM-free NQ rules."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol


class QueryRewriter(Protocol):
    """Providers must preserve meaning and declare their LLM cost up front.

    A future Qwen adapter can implement this interface, but this stage's
    orchestrator accepts only providers declaring llm_calls_per_rewrite=0.
    Lexical filtering is not a general proof of semantic equivalence.
    """

    llm_calls_per_rewrite: int

    def rewrite(self, query: str, dataset: str) -> Sequence[str]: ...


def _clean(query: str) -> str:
    return " ".join(unicodedata.normalize("NFC", query).split())


# Skip rules for constraints whose safe interpretation needs more than a template.
_CONSTRAINTS = re.compile(
    r"\b(?:not|no|never|without|except|excluding|only|before|after|between|"
    r"during|since|until|in|from|and|or|nor|that|which|whose|who|with|by|at|on|"
    r"for|if|when|where|because|than|under|over|first|last|most|least)\b|n['\u2019]t\b|[<>=]",
    re.IGNORECASE,
)
_NEGATIONS = re.compile(r"\b(?:not|no|never|without|nor)\b|\b\w+n['\u2019]t\b", re.IGNORECASE)
_NUMBERS = re.compile(r"\d+(?:[.,:/-]\d+)*")


def filter_rewrites(original_query: str, candidates: Sequence[str]) -> list[str]:
    """Remove empty/original/duplicate outputs, retaining at most two queries.

    NFC/whitespace/casefold are used for duplicate comparison. Changed numeric
    or negation tokens are rejected as an extra safeguard for injected providers.
    Entity/constraint fidelity beyond these checks is the provider's contract.
    Malformed provider outputs raise TypeError and are caught by the orchestrator.
    """
    if not isinstance(original_query, str) or not original_query.strip():
        raise ValueError("original_query must be a nonempty string")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("rewriter output must be a sequence of strings")
    original = _clean(original_query)
    seen = {original.casefold()}
    numbers = set(_NUMBERS.findall(original))
    negations = {word.casefold() for word in _NEGATIONS.findall(original)}
    result = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            raise TypeError("each rewritten query must be a string")
        query = _clean(candidate)
        if not query or query.casefold() in seen:
            continue
        if set(_NUMBERS.findall(query)) != numbers:
            continue
        if {word.casefold() for word in _NEGATIONS.findall(query)} != negations:
            continue
        seen.add(query.casefold())
        if len(result) < 2:
            result.append(query)
    return result


class RuleQueryRewriter:
    """Copy the subject verbatim into a search phrase, never supply an answer.

    Supports only four anchored simple NQ forms. Subjects with constraint words,
    extra question marks or sentence separators are skipped. FEVER/unknown tasks
    return no rewrites. This small rule set validates flow, not research efficacy.
    """

    llm_calls_per_rewrite = 0
    _RULES = (
        (r"Where was (?P<subject>.+?) born\??", "birthplace"),
        (r"When was (?P<subject>.+?) born\??", "date of birth"),
        (r"Who wrote (?P<subject>.+?)\??", "author"),
        (r"When was (?P<subject>.+?) founded\??", "founding date"),
    )

    def rewrite(self, query: str, dataset: str) -> list[str]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if dataset not in {"nq", "natural_questions"}:
            return []
        cleaned = _clean(query)
        for pattern, search_term in self._RULES:
            match = re.fullmatch(pattern, cleaned, flags=re.IGNORECASE)
            if match is None:
                continue
            subject = match.group("subject").strip()
            if _CONSTRAINTS.search(subject) or any(ch in subject for ch in "?;!\n"):
                return []
            # Every entity/number in the subject is copied, with no answer facts.
            return filter_rewrites(query, [f"{subject} {search_term}"])
        return []
