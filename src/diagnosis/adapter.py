"""Convert canonical C1 Reference RAG records into DiagnosisInput.

Only whitelisted runtime fields are read. Gold, correctness and evaluation
fields in the C1 record (gold_answer_or_label, correct, em, f1,
gold_evidence_in_top5, latency, ...) are never accessed. FEVER raw_output is
deliberately ignored in this version; prediction is the initial answer.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.diagnosis.schema import DatasetName, DiagnosisInput, EvidenceItem

QWEN_A_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
C1_CONDITION = "C1"

DATASET_ALIASES: dict[str, DatasetName] = {
    "natural_questions": DatasetName.NQ,
    "nq": DatasetName.NQ,
    "fever": DatasetName.FEVER,
}


def canonical_dataset(value: object) -> DatasetName:
    if isinstance(value, str) and value in DATASET_ALIASES:
        return DATASET_ALIASES[value]
    raise ValueError(f"Unsupported dataset {value!r}; expected one of: {', '.join(DATASET_ALIASES)}")


def _required(record: Mapping, key: str) -> object:
    if key not in record:
        raise ValueError(f"C1 record is missing required field {key!r}")
    return record[key]


def _evidence_items(evidence: object) -> list[EvidenceItem]:
    if not isinstance(evidence, list):
        raise ValueError("C1 evidence must be a list")
    items = []
    for index, passage in enumerate(evidence):
        if not isinstance(passage, Mapping):
            raise ValueError(f"C1 evidence[{index}] must be an object")
        items.append(
            EvidenceItem(
                rank=index + 1,
                passage_id=_required(passage, "passage_id"),
                text=_required(passage, "text"),
                score=_required(passage, "score"),
            )
        )
    return items


def c1_record_to_input(record: Mapping) -> DiagnosisInput:
    condition = _required(record, "condition")
    if condition != C1_CONDITION:
        raise ValueError(f"Expected condition {C1_CONDITION!r}, got {condition!r}")
    model_name = _required(record, "model_name")
    if model_name != QWEN_A_MODEL_NAME:
        raise ValueError(f"Expected model {QWEN_A_MODEL_NAME!r}, got {model_name!r}")

    evidence = _evidence_items(_required(record, "evidence"))
    retrieved_ids = _required(record, "retrieved_top5_ids")
    evidence_ids = [item.passage_id for item in evidence]
    if retrieved_ids != evidence_ids:
        raise ValueError(
            f"retrieved_top5_ids {retrieved_ids} do not match evidence passage IDs {evidence_ids}"
        )

    prediction = _required(record, "prediction")
    if not isinstance(prediction, str):
        raise ValueError("C1 prediction must be a string")

    return DiagnosisInput(
        sample_id=_required(record, "sample_id"),
        dataset=canonical_dataset(_required(record, "dataset")),
        query=_required(record, "question_or_claim"),
        evidence=evidence,
        initial_answer=prediction,
    )
