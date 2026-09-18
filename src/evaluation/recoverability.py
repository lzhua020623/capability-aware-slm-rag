"""Frozen Preliminary Recoverability Experiment statistics."""

from __future__ import annotations


def recoverable_failure_rate(c1_wrong: set, c3_7b_correct: set) -> float:
    """number(C1 wrong AND C3-7B correct) / number(C1 wrong)."""
    if not c1_wrong:
        return 0.0
    return len(c1_wrong & c3_7b_correct) / len(c1_wrong)


def retrieval_induced_degradation(c0_correct: set, c1_wrong: set) -> set:
    """C0 correct AND C1 wrong."""
    return c0_correct & c1_wrong


def likely_small_model_capability_limitation(
    c1_wrong: set, c3_7b_wrong: set, c3_32b_correct: set
) -> set:
    """C1 wrong AND C3-7B wrong AND C3-32B correct."""
    return c1_wrong & c3_7b_wrong & c3_32b_correct


def unresolved_hard_cases(c3_32b_wrong: set) -> set:
    """32B still wrong."""
    return set(c3_32b_wrong)
