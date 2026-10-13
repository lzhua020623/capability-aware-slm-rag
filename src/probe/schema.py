"""Evidence Probe v1 schema and deterministic label mapping.

The Probe is a low-cost, selective re-check of the retrieved evidence by the
same Qwen2.5-7B-Instruct model used by the Reference RAG generator. It is not
answer repair, a second answer generator, a larger-model judge or a
replacement classifier. It sees only the task type, the query/claim, the
Top-5 evidence and the initial answer.

Judgment definitions
--------------------
answer_supported
    Whether the initial answer is supported by the given Top-5 evidence.
    NQ: the evidence must support the initial answer as an answer to the
    exact question, respecting every constraint in it (time, version, entity,
    location, quantity, ordering, event, work); semantic equivalence of the
    answer counts, topical relatedness does not. FEVER: computed by code as
    evidence_verdict == initial answer (see below).
evidence_sufficiency
    Whether the Top-5 evidence alone contains enough information to answer
    the exact question (NQ) or to decide SUPPORTS vs REFUTES (FEVER). A
    missing key fact or constraint means insufficient.
evidence_accessibility
    Meaningful only when the evidence is sufficient. clear: the key
    information is explicit, consistent and easy to locate and use.
    hard_to_access: the key information exists but is buried in irrelevant
    text, dispersed across passages, faced with competing or conflicting
    evidence, preceded by misleading higher-ranked passages, ambiguous, or
    surrounded by substantial retrieval noise.

FEVER verdict judgment
----------------------
For FEVER the model never sees the initial answer and never judges it. It
returns only evidence_verdict for the claim from the Top-5 evidence:
SUPPORTS (the evidence is enough to support the claim), REFUTES (enough to
refute it) or INSUFFICIENT (not enough to judge it reliably), plus
evidence_accessibility. fever_judgments then derives, deterministically:
evidence_sufficiency = sufficient for SUPPORTS/REFUTES, insufficient for
INSUFFICIENT; answer_supported = supported iff the verdict equals the initial
answer (SUPPORTS/REFUTES); evidence_accessibility = uncertain whenever the
evidence is insufficient.

The final probe_label is derived by map_probe_label, never chosen freely by
the model. supporting_ranks / conflicting_ranks refer to evidence ranks 1-5
that support / contradict the initial answer. reason is debug-only and never
used in any decision.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from src.diagnosis.labels import DiagnosisLabel

MIN_EVIDENCE_RANK = 1
MAX_EVIDENCE_RANK = 5


class AnswerSupport(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class EvidenceSufficiency(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    UNCERTAIN = "uncertain"


class EvidenceAccessibility(StrEnum):
    CLEAR = "clear"
    HARD_TO_ACCESS = "hard_to_access"
    UNCERTAIN = "uncertain"


class FeverVerdict(StrEnum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    INSUFFICIENT = "INSUFFICIENT"


class ProbeLabel(StrEnum):
    CORRECT = "correct"
    EVIDENCE_INSUFFICIENCY = "evidence_insufficiency"
    EVIDENCE_ACCESSIBILITY = "evidence_accessibility"
    EVIDENCE_UTILISATION = "evidence_utilisation"
    UNCERTAIN = "uncertain"

    def to_diagnosis_label(self) -> DiagnosisLabel | None:
        """Runtime label, or None for uncertain (the caller keeps the original diagnosis)."""
        return None if self is ProbeLabel.UNCERTAIN else DiagnosisLabel(self.value)


def _parse_enum(enum_type: type[StrEnum], name: str, value: object) -> StrEnum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            pass
    allowed = ", ".join(member.value for member in enum_type)
    raise ValueError(f"Invalid {name} {value!r}; expected one of: {allowed}")


def parse_ranks(name: str, values: Iterable[object], max_rank: int = MAX_EVIDENCE_RANK) -> tuple[int, ...]:
    """Validate evidence ranks: ints in [1, max_rank] (max_rank <= 5), no duplicates, sorted."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{name} must be a list of ints, got {values!r}")
    upper = min(max_rank, MAX_EVIDENCE_RANK)
    ranks = list(values)
    for rank in ranks:
        if isinstance(rank, bool) or not isinstance(rank, int) or not MIN_EVIDENCE_RANK <= rank <= upper:
            raise ValueError(f"{name} contains invalid rank {rank!r}; expected ints in [1, {upper}]")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{name} contains duplicate ranks: {ranks}")
    return tuple(sorted(ranks))


def map_probe_label(
    answer_supported: AnswerSupport,
    evidence_sufficiency: EvidenceSufficiency,
    evidence_accessibility: EvidenceAccessibility,
) -> ProbeLabel:
    """Deterministic mapping from the three judgments to the probe label."""
    if answer_supported is AnswerSupport.SUPPORTED:
        return ProbeLabel.CORRECT
    if evidence_sufficiency is EvidenceSufficiency.INSUFFICIENT:
        return ProbeLabel.EVIDENCE_INSUFFICIENCY
    if evidence_sufficiency is EvidenceSufficiency.SUFFICIENT:
        if evidence_accessibility is EvidenceAccessibility.HARD_TO_ACCESS:
            return ProbeLabel.EVIDENCE_ACCESSIBILITY
        if evidence_accessibility is EvidenceAccessibility.CLEAR:
            return ProbeLabel.EVIDENCE_UTILISATION
    return ProbeLabel.UNCERTAIN


def fever_judgments(
    evidence_verdict: FeverVerdict | str,
    initial_answer: str,
    evidence_accessibility: EvidenceAccessibility | str,
) -> tuple[AnswerSupport, EvidenceSufficiency, EvidenceAccessibility]:
    """Deterministic comparison of the model's claim verdict with the initial answer."""
    verdict = _parse_enum(FeverVerdict, "evidence_verdict", evidence_verdict)
    accessibility = _parse_enum(EvidenceAccessibility, "evidence_accessibility", evidence_accessibility)
    if verdict is FeverVerdict.INSUFFICIENT:
        return AnswerSupport.UNSUPPORTED, EvidenceSufficiency.INSUFFICIENT, EvidenceAccessibility.UNCERTAIN
    supported = isinstance(initial_answer, str) and initial_answer.strip().upper() == verdict.value
    support = AnswerSupport.SUPPORTED if supported else AnswerSupport.UNSUPPORTED
    return support, EvidenceSufficiency.SUFFICIENT, accessibility


@dataclass(frozen=True)
class ProbeResult:
    """Structured Probe output. raw_output and parse_error are for debugging only.

    evidence_verdict is set only for FEVER and records the model's raw claim verdict.
    """

    answer_supported: AnswerSupport
    evidence_sufficiency: EvidenceSufficiency
    evidence_accessibility: EvidenceAccessibility
    supporting_ranks: tuple[int, ...]
    conflicting_ranks: tuple[int, ...]
    probe_label: ProbeLabel
    reason: str | None = None
    raw_output: str | None = None
    parse_error: str | None = None
    evidence_verdict: FeverVerdict | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer_supported", _parse_enum(AnswerSupport, "answer_supported", self.answer_supported))
        object.__setattr__(
            self, "evidence_sufficiency", _parse_enum(EvidenceSufficiency, "evidence_sufficiency", self.evidence_sufficiency)
        )
        object.__setattr__(
            self,
            "evidence_accessibility",
            _parse_enum(EvidenceAccessibility, "evidence_accessibility", self.evidence_accessibility),
        )
        object.__setattr__(self, "supporting_ranks", parse_ranks("supporting_ranks", self.supporting_ranks))
        object.__setattr__(self, "conflicting_ranks", parse_ranks("conflicting_ranks", self.conflicting_ranks))
        label = _parse_enum(ProbeLabel, "probe_label", self.probe_label)
        expected = map_probe_label(self.answer_supported, self.evidence_sufficiency, self.evidence_accessibility)
        if label is not expected:
            raise ValueError(f"probe_label {label.value!r} does not follow the mapping (expected {expected.value!r})")
        object.__setattr__(self, "probe_label", label)
        if self.evidence_verdict is not None:
            object.__setattr__(
                self, "evidence_verdict", _parse_enum(FeverVerdict, "evidence_verdict", self.evidence_verdict)
            )

    @classmethod
    def from_judgments(
        cls,
        answer_supported: AnswerSupport | str,
        evidence_sufficiency: EvidenceSufficiency | str,
        evidence_accessibility: EvidenceAccessibility | str,
        supporting_ranks: Iterable[int] = (),
        conflicting_ranks: Iterable[int] = (),
        reason: str | None = None,
        raw_output: str | None = None,
        evidence_verdict: FeverVerdict | str | None = None,
    ) -> ProbeResult:
        support = _parse_enum(AnswerSupport, "answer_supported", answer_supported)
        sufficiency = _parse_enum(EvidenceSufficiency, "evidence_sufficiency", evidence_sufficiency)
        accessibility = _parse_enum(EvidenceAccessibility, "evidence_accessibility", evidence_accessibility)
        return cls(
            answer_supported=support,
            evidence_sufficiency=sufficiency,
            evidence_accessibility=accessibility,
            supporting_ranks=tuple(supporting_ranks),
            conflicting_ranks=tuple(conflicting_ranks),
            probe_label=map_probe_label(support, sufficiency, accessibility),
            reason=reason,
            raw_output=raw_output,
            evidence_verdict=evidence_verdict,
        )

    @classmethod
    def from_fever_verdict(
        cls,
        evidence_verdict: FeverVerdict | str,
        initial_answer: str,
        evidence_accessibility: EvidenceAccessibility | str,
        decisive_ranks: Iterable[int] = (),
        opposing_ranks: Iterable[int] = (),
        reason: str | None = None,
        raw_output: str | None = None,
    ) -> ProbeResult:
        """decisive_ranks back the model's verdict, opposing_ranks back the opposite verdict;
        they are re-expressed relative to the initial answer."""
        verdict = _parse_enum(FeverVerdict, "evidence_verdict", evidence_verdict)
        support, sufficiency, accessibility = fever_judgments(verdict, initial_answer, evidence_accessibility)
        if verdict is FeverVerdict.INSUFFICIENT:
            supporting, conflicting = (), ()
        elif support is AnswerSupport.SUPPORTED:
            supporting, conflicting = decisive_ranks, opposing_ranks
        else:
            supporting, conflicting = opposing_ranks, decisive_ranks
        return cls.from_judgments(
            support, sufficiency, accessibility, supporting, conflicting,
            reason=reason, raw_output=raw_output, evidence_verdict=verdict,
        )

    @classmethod
    def unparsable(cls, raw_output: str | None, error: str) -> ProbeResult:
        """All-uncertain result used when the model output cannot be parsed or validated."""
        return cls(
            answer_supported=AnswerSupport.UNCERTAIN,
            evidence_sufficiency=EvidenceSufficiency.UNCERTAIN,
            evidence_accessibility=EvidenceAccessibility.UNCERTAIN,
            supporting_ranks=(),
            conflicting_ranks=(),
            probe_label=ProbeLabel.UNCERTAIN,
            raw_output=raw_output,
            parse_error=error,
        )

    def to_dict(self) -> dict:
        return {
            "answer_supported": self.answer_supported.value,
            "evidence_sufficiency": self.evidence_sufficiency.value,
            "evidence_accessibility": self.evidence_accessibility.value,
            "supporting_ranks": list(self.supporting_ranks),
            "conflicting_ranks": list(self.conflicting_ranks),
            "probe_label": self.probe_label.value,
            "evidence_verdict": None if self.evidence_verdict is None else self.evidence_verdict.value,
            "reason": self.reason,
            "raw_output": self.raw_output,
            "parse_error": self.parse_error,
        }
