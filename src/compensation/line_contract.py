"""Framing for one labeled response; never salvage a block from a saved response."""
from __future__ import annotations

import re

LINE_CONTRACT_VERSION = "single_response_v3_fact_first_audit"
LEGACY_CHECK_LABELS = ("ENTITY", "RELATION", "TYPE", "COVERAGE", "SUPPORT", "CONFLICT_FREE",
                       "RANK", "QUOTE", "REASON")
CHECK_LABELS = ("FACT",) + LEGACY_CHECK_LABELS
CANDIDATE_LABELS = ("RANK", "QUOTE", "ANSWER")
CANDIDATE_SCHEMAS = ("labeled", "bare_rank_labeled_values", "three_positional_values")


class FormatError(ValueError):
    """Unparseable structure or field value."""


class GroundingError(ValueError):
    """A structured response fails its source-address or contiguous-slice guard."""


class ConstraintError(ValueError):
    """A structured response exceeds an explicit response length constraint."""


def parse_labeled_lines(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != len(labels):
        raise FormatError(f"Expected exactly {len(labels)} labeled lines")
    values = {}
    for line, label in zip(lines, labels):
        match = re.fullmatch(re.escape(label) + r":\s*(\S.*)", line)
        if match is None:
            raise FormatError(f"Expected {label} label and nonempty value")
        values[label] = match[1].strip()
    return values


def complete_labeled_output(text: str, labels: tuple[str, ...], *, insufficient: bool = False) -> bool:
    """Stop DURING generation after a newline closes the final nonempty value.

    Without a newline, the last field might still be a partial answer or reason.
    No text is cropped, and a saved repeated/conflicting response remains invalid.
    EOS can end a valid response without a newline; normal parsing then handles it.
    """
    if insufficient and text.strip() == "INSUFFICIENT":
        return True
    if not text.endswith(("\n", "\r")):
        return False
    try:
        if labels == CANDIDATE_LABELS:
            parse_candidate_lines(text)
        else:
            parse_labeled_lines(text, labels)
    except ValueError:
        return False
    return True


def parse_evidence_rank(text: str) -> int:
    """A single integer, optionally surrounded by ONE matching bracket pair."""
    match = re.fullmatch(r"(?:([0-9]+)|\[\s*([0-9]+)\s*\])", text.strip())
    if match is None:
        raise FormatError("RANK must be one evidence block number")
    value = match[1] if match[1] is not None else match[2]
    if len(value) > 1 and value.startswith("0"):
        raise FormatError("RANK must not have leading zeros")
    return int(value)


def parse_candidate_lines(text: str) -> tuple[dict[str, str], str]:
    """Three explicit grammars, followed by the same strict provenance checks.

    A missing label is not inferred from prose: the first bare line must be one
    integer, and exactly two values must follow in the fixed QUOTE/ANSWER order.
    Partial mixtures of labeled and positional values remain invalid.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 3:
        raise FormatError("Expected exactly three candidate lines")
    if lines[0].startswith("RANK:"):
        return parse_labeled_lines(text, CANDIDATE_LABELS), "labeled"
    parse_evidence_rank(lines[0])
    if lines[1].startswith("QUOTE:") and lines[2].startswith("ANSWER:"):
        values = parse_labeled_lines("RANK: " + "\n".join(lines), CANDIDATE_LABELS)
        return values, "bare_rank_labeled_values"
    # Keep r3 candidate grammars unchanged; FACT belongs only to audits.
    reserved = LEGACY_CHECK_LABELS + CANDIDATE_LABELS
    if any(line.startswith(label + ":") for line in lines[1:] for label in reserved):
        raise FormatError("Mixed candidate labels and positional values")
    return dict(zip(CANDIDATE_LABELS, lines)), "three_positional_values"
