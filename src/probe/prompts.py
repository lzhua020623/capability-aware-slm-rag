"""Deterministic Evidence Probe prompts for NQ and FEVER.

build_probe_prompt takes only a DiagnosisInput. NQ prompts contain the
question, the initial answer and the Top-5 evidence texts (by rank). FEVER
prompts contain only the claim and the Top-5 evidence: the model judges the
claim, and the comparison with the initial answer is done in code. Sample
IDs, passage IDs, gold data and any Diagnosis prediction (probabilities,
labels, confidence) are never part of a prompt.
"""

from __future__ import annotations

from src.diagnosis.schema import DatasetName, DiagnosisInput
from src.probe.schema import MAX_EVIDENCE_RANK

ACCESSIBILITY_NOISE = (
    "buried in long or irrelevant text, spread across several passages, competing with or contradicted "
    "by other passages, preceded by misleading higher-ranked passages, ambiguous, or surrounded by "
    "heavy retrieval noise"
)

COMMON_RULES = (
    "Use only the numbered passages; do not use outside knowledge. "
    f"Ranks must be passage numbers 1-{MAX_EVIDENCE_RANK}. "
    "Return only one JSON object and nothing else, in this format:\n"
)

NQ_JSON_TEMPLATE = (
    '{"answer_supported": "supported|unsupported|uncertain", '
    '"evidence_sufficiency": "sufficient|insufficient|uncertain", '
    '"evidence_accessibility": "clear|hard_to_access|uncertain", '
    '"supporting_ranks": [], "conflicting_ranks": [], '
    '"reason": "<one short sentence>"}'
)

NQ_INSTRUCTION = (
    "You are checking the retrieved evidence of a question-answering system. "
    "Do not answer the question yourself.\n"
    "Semantic relevance alone is not enough. The evidence must support the proposed answer as an answer "
    "to the exact question. Check all temporal, entity, version, quantity and scope constraints in the "
    "question (time, year or decade; version, edition or adaptation; person or entity; location; number; "
    "before/after; first/last; specific event or work). If the passages support the proposed answer only "
    "for a different time, version, entity or scope (for example, the question asks about a 1980s version "
    "but the passages only support a 1963 version), the answer is unsupported, even if the topic matches.\n"
    '1. answer_supported: "supported" only if the passages state the proposed answer, or a semantically '
    "equivalent one (aliases, paraphrases and different formats count), as the answer to the exact "
    'question with all its constraints; "unsupported" if they do not, if a constraint does not match, or '
    'if they give a different answer; "uncertain" if you cannot tell.\n'
    '2. evidence_sufficiency: "sufficient" if the passages alone contain enough information to answer '
    'the exact question, including its constraints; "insufficient" if a key fact or constraint is '
    'missing, even when the passages are on the same topic; "uncertain" if you cannot tell.\n'
    '3. evidence_accessibility (only if sufficient, otherwise "uncertain"): "clear" if the information '
    'that answers the exact question is explicit, consistent and easy to locate; "hard_to_access" if it '
    f'is present but {ACCESSIBILITY_NOISE}; "uncertain" if you cannot tell.\n'
    "supporting_ranks: passages that support the proposed answer for the exact question. "
    "conflicting_ranks: passages that contradict it or give a different answer.\n"
    + COMMON_RULES
    + NQ_JSON_TEMPLATE
)

FEVER_JSON_TEMPLATE = (
    '{"evidence_verdict": "SUPPORTS|REFUTES|INSUFFICIENT", '
    '"evidence_accessibility": "clear|hard_to_access|uncertain", '
    '"decisive_ranks": [], "opposing_ranks": [], '
    '"reason": "<one short sentence>"}'
)

FEVER_INSTRUCTION = (
    "You are verifying a claim against retrieved evidence.\n"
    "1. evidence_verdict: decide what the passages say about the claim. "
    '"SUPPORTS" if the passages are enough to support the claim; '
    '"REFUTES" if the passages are enough to refute the claim (they state facts that contradict it); '
    '"INSUFFICIENT" if the passages are not enough to judge the claim reliably.\n'
    '2. evidence_accessibility (only for SUPPORTS or REFUTES, otherwise "uncertain"): "clear" if the '
    "decisive evidence for your verdict is clear, direct, consistent and easy to locate; "
    f'"hard_to_access" if it is enough but {ACCESSIBILITY_NOISE}; "uncertain" if you cannot tell.\n'
    "decisive_ranks: passages that back your verdict. "
    "opposing_ranks: passages that point to the opposite verdict.\n"
    + COMMON_RULES
    + FEVER_JSON_TEMPLATE
)


def format_passages(item: DiagnosisInput) -> str:
    evidence = sorted(item.evidence, key=lambda passage: passage.rank)[:MAX_EVIDENCE_RANK]
    return "\n\n".join(f"[{passage.rank}] {passage.text}" for passage in evidence)


def build_probe_prompt(item: DiagnosisInput) -> str:
    if not isinstance(item, DiagnosisInput):
        raise TypeError(f"Expected DiagnosisInput, got {type(item).__name__}")
    passages = format_passages(item)
    if item.dataset is DatasetName.NQ:
        return (
            f"{NQ_INSTRUCTION}\n\n"
            f"Question: {item.query}\n"
            f"Proposed answer: {item.initial_answer}\n\n"
            f"Passages:\n{passages}"
        )
    return f"{FEVER_INSTRUCTION}\n\nClaim: {item.query}\n\nPassages:\n{passages}"
