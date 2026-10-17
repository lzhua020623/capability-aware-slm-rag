"""Adapt frozen NQ Train delivery rows without exposing offline supervision to E."""
from __future__ import annotations

from dataclasses import asdict
import math

from src.compensation.answer_repair import MODEL_NAME
from src.diagnosis import DiagnosisInput, EvidenceItem
from src.evaluation.qa import best_em_f1


def validate_generation_source(stats: dict) -> None:
    """Validate documented settings; a model revision is not supplied by this delivery."""
    if (stats.get("generation_model") != MODEL_NAME or stats.get("quantization") != "nf4"
            or stats.get("decoding") != "greedy do_sample=False"
            or stats.get("max_new_tokens") != 64):
        raise ValueError("NQ Train source must document the frozen Qwen 7B NF4 C1 settings")


def runtime_request(row: dict) -> DiagnosisInput:
    """Explicit runtime allowlist; labels, references and any extra fields are ignored."""
    if row.get("dataset") != "nq":
        raise ValueError("Expected an NQ Train row")
    evidence = row.get("evidence")
    if (not isinstance(evidence, list) or len(evidence) != 5
            or any(not isinstance(e, dict) for e in evidence)
            or [e.get("rank") for e in evidence] != [1, 2, 3, 4, 5]):
        raise ValueError("Expected five evidence items in their frozen rank order")
    items = [EvidenceItem(rank=e["rank"], passage_id=e["passage_id"],
                          text=e["text"], score=e["score"]) for e in evidence]
    if len({e.passage_id for e in items}) != 5 or any(
            not e.text.strip() or not math.isfinite(e.score) for e in items):
        raise ValueError("Evidence must have distinct passage IDs, nonempty text and finite scores")
    request = DiagnosisInput(sample_id=row["sample_id"], dataset="nq", query=row["query"],
                             initial_answer=row["initial_answer"], evidence=items)
    if not request.query.strip() or not request.initial_answer.strip():
        raise ValueError("Query and initial answer must be nonempty")
    return request


def offline_reference(row: dict) -> tuple[list[str], bool]:
    ref = row.get("annotation_reference")
    if not isinstance(ref, dict):
        raise ValueError("Missing offline annotation_reference")
    gold = ref.get("gold_answer")
    answers = [gold] if isinstance(gold, str) else gold
    if (not isinstance(answers, list) or not answers
            or any(not isinstance(g, str) or not g.strip() for g in answers)):
        raise ValueError("Gold answers must be nonempty strings")
    evidence = row["evidence"]
    ids = [e["passage_id"] for e in evidence]
    pid = ref.get("gold_passage_id")
    if not isinstance(pid, str) or not pid:
        raise ValueError("Missing gold passage ID")
    for k in (1, 3, 5):
        if ref.get(f"gold_in_top{k}") is not (pid in ids[:k]):
            raise ValueError("Offline gold-hit flags disagree with the frozen evidence")
    found = next((e for e in evidence if e["passage_id"] == pid), None)
    if ref.get("gold_evidence_text") != (found["text"] if found else None):
        raise ValueError("Offline gold evidence text disagrees with the frozen evidence")
    return list(answers), pid in ids


def c1_record(row: dict, stats: dict, *, source_shard: str, source_sha256: str,
              data_role: str) -> dict:
    """Runner-compatible scored C1 record; references remain offline-only fields."""
    validate_generation_source(stats)
    request = runtime_request(row)
    answers, _ = offline_reference(row)
    em, f1 = best_em_f1(request.initial_answer, answers)
    return {
        "sample_id": request.sample_id, "dataset": "nq", "condition": "C1",
        "model_name": stats["generation_model"], "question_or_claim": request.query,
        "prediction": request.initial_answer,
        "evidence": [{"passage_id": e.passage_id, "text": e.text, "score": e.score}
                     for e in request.evidence],
        "gold_answer_or_label": answers, "correct": bool(em), "em": em, "f1": f1,
        "provenance": {"source": "nq_train_delivery", "source_shard": source_shard,
                       "source_sha256": source_sha256, "data_role": data_role,
                       "c1_model_revision": None,
                       "c1_revision_status": "not_exported_by_upstream",
                       "correctness_policy": "frozen_EM_only_not_semantic_label"},
    }


def runtime_record(row: dict) -> dict:
    return asdict(runtime_request(row))
