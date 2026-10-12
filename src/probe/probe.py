"""Evidence Probe v1: one greedy Qwen2.5-7B call per triggered sample.

The generation function is injected so that tests never load a model;
EvidenceProbe.from_qwen reuses the Reference RAG loader and greedy decoding
from src/generation/qwen.py (configs/base.yaml model, 4-bit NF4,
do_sample=False). Model output is parsed as JSON, validated, and mapped to a
probe label deterministically; any parse/validation failure yields an
all-uncertain result so the original Diagnosis is kept. For FEVER the model
returns only a claim verdict, which is compared with the initial answer here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import DatasetName, DiagnosisInput
from src.probe.prompts import build_probe_prompt
from src.probe.schema import MAX_EVIDENCE_RANK, ProbeLabel, ProbeResult, parse_ranks

PROBE_MAX_NEW_TOKENS = 160
REQUIRED_KEYS: tuple[str, ...] = (
    "answer_supported",
    "evidence_sufficiency",
    "evidence_accessibility",
    "supporting_ranks",
    "conflicting_ranks",
)
FEVER_REQUIRED_KEYS: tuple[str, ...] = (
    "evidence_verdict",
    "evidence_accessibility",
    "decisive_ranks",
    "opposing_ranks",
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ProbeParseError(ValueError):
    """Model output is not a valid Probe JSON object."""


def _load_json_object(text: str, required: tuple[str, ...]) -> tuple[dict, str | None]:
    if not isinstance(text, str):
        raise ProbeParseError(f"Probe output must be a string, got {type(text).__name__}")
    cleaned = _FENCE_RE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ProbeParseError("No JSON object found in probe output")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ProbeParseError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProbeParseError("Probe output JSON must be an object")
    missing = [key for key in required if key not in data]
    if missing:
        raise ProbeParseError(f"Probe output is missing keys: {missing}")
    reason = data.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ProbeParseError("reason must be a string")
    return data, reason


def parse_probe_output(text: str, max_rank: int = MAX_EVIDENCE_RANK) -> ProbeResult:
    """Parse and validate one NQ Probe JSON object; raise ProbeParseError on any violation."""
    data, reason = _load_json_object(text, REQUIRED_KEYS)
    try:
        supporting = parse_ranks("supporting_ranks", data["supporting_ranks"], max_rank)
        conflicting = parse_ranks("conflicting_ranks", data["conflicting_ranks"], max_rank)
        return ProbeResult.from_judgments(
            data["answer_supported"],
            data["evidence_sufficiency"],
            data["evidence_accessibility"],
            supporting,
            conflicting,
            reason=reason,
            raw_output=text,
        )
    except ValueError as exc:
        raise ProbeParseError(str(exc)) from exc


def parse_fever_probe_output(text: str, initial_answer: str, max_rank: int = MAX_EVIDENCE_RANK) -> ProbeResult:
    """Parse a FEVER claim verdict and compare it with initial_answer in code."""
    data, reason = _load_json_object(text, FEVER_REQUIRED_KEYS)
    try:
        return ProbeResult.from_fever_verdict(
            data["evidence_verdict"],
            initial_answer,
            data["evidence_accessibility"],
            parse_ranks("decisive_ranks", data["decisive_ranks"], max_rank),
            parse_ranks("opposing_ranks", data["opposing_ranks"], max_rank),
            reason=reason,
            raw_output=text,
        )
    except ValueError as exc:
        raise ProbeParseError(str(exc)) from exc


def refine_label(original: DiagnosisLabel, probe_result: ProbeResult | None) -> DiagnosisLabel:
    """Probe label when available; the original Diagnosis label when not probed or uncertain."""
    if probe_result is None or probe_result.probe_label is ProbeLabel.UNCERTAIN:
        return original
    return probe_result.probe_label.to_diagnosis_label()


class EvidenceProbe:
    def __init__(self, generate_fn: Callable[[str], str]):
        self._generate = generate_fn

    @classmethod
    def from_qwen(cls, max_new_tokens: int = PROBE_MAX_NEW_TOKENS) -> EvidenceProbe:
        """Load the Reference RAG Qwen2.5-7B (NF4) via src/generation/qwen.py."""
        from src.generation.qwen import generate_greedy, load_qwen_nf4

        tokenizer, model = load_qwen_nf4()

        def generate(prompt: str) -> str:
            text, _latency = generate_greedy(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
            return text

        return cls(generate)

    def run(self, item: DiagnosisInput) -> ProbeResult:
        """Exactly one generation call for one DiagnosisInput."""
        raw = self._generate(build_probe_prompt(item))
        max_rank = min(len(item.evidence), MAX_EVIDENCE_RANK)
        try:
            if item.dataset is DatasetName.FEVER:
                return parse_fever_probe_output(raw, item.initial_answer, max_rank)
            return parse_probe_output(raw, max_rank)
        except ProbeParseError as exc:
            return ProbeResult.unparsable(raw, str(exc))
