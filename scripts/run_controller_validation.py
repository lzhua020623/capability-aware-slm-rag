"""Offline Controller v1 dry-run on stored v2-expanded VALIDATION Diagnosis + Probe outputs.

No model is run and no test record is read. Only runtime-visible fields
(Diagnosis outputs and the raw Probe result) are passed to the Controller;
target labels and offline evaluation fields are never read. This checks
routing only and does not evaluate compensation. As a consistency check the
resolved labels must equal the offline Safe Binary rule S3 (p <= 0.70 and
Stage 2 max <= 0.50) from scripts/evaluate_probe_safe_override_v2.py.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_probe_safe_override_v2 import OverrideRule, safe_label
from scripts.run_probe_smoke import SAMPLES_PATH
from scripts.run_probe_v2_expanded_validation import EXPECTED_TRIGGERED, EXPECTED_VALIDATION
from scripts.run_probe_v2_expanded_validation import OUTPUT_PATH as PROBE_RESULTS
from scripts.train_diagnosis_classifier import write_json
from scripts.train_diagnosis_v2_expanded import OUTPUT_PATH as DIAGNOSIS_RESULTS
from src.controller.controller import SAFE_OVERRIDE_MAX_STAGE1, SAFE_OVERRIDE_MAX_STAGE2, decide_action
from src.controller.schema import ControllerAction
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.probe.schema import ProbeResult

OUTPUT_PATH = ROOT / "results" / "controller" / "controller_v1_validation.json"
PROBE_RESULT_FIELDS = (
    "answer_supported", "evidence_sufficiency", "evidence_accessibility", "supporting_ranks",
    "conflicting_ranks", "probe_label", "reason", "raw_output", "parse_error", "evidence_verdict",
)
DATASETS = ("nq", "fever")


def diagnosis_from_prediction(prediction: dict) -> HierarchicalDiagnosis:
    scores = prediction["stage2_class_scores"]
    return HierarchicalDiagnosis(
        stage1_failure_probability=prediction["stage1_failure_probability"],
        stage1_label=prediction["stage1_label"],
        stage2_class_scores=None if scores is None else {parse_label(k): v for k, v in scores.items()},
        final_label=parse_label(prediction["final_label"]),
        confidence=prediction["confidence"],
        class_scores=None,
    )


def probe_result_from_dict(data: dict | None) -> ProbeResult | None:
    if data is None:
        return None
    values = {name: data[name] for name in PROBE_RESULT_FIELDS}
    values["supporting_ranks"] = tuple(values["supporting_ranks"])
    values["conflicting_ranks"] = tuple(values["conflicting_ranks"])
    return ProbeResult(**values)


def probe_status(decision) -> str:
    if not decision.used_probe:
        return "probe_not_triggered"
    return "probe_override_to_correct" if decision.probe_overrode_diagnosis else "probe_triggered_no_override"


def distribution(decisions: list) -> dict:
    return {
        "n": len(decisions),
        "labels": {label.value: sum(d.diagnosis_label is label for d in decisions) for label in DiagnosisLabel},
        "actions": {action.value: sum(d.action is action for d in decisions) for action in ControllerAction},
        "probe_status": dict(Counter(probe_status(d) for d in decisions)),
        "needs_compensation": sum(d.needs_compensation for d in decisions),
    }


def report(rows: list[dict], decisions: list) -> dict:
    return {
        "all": distribution(decisions),
        **{d: distribution([x for r, x in zip(rows, decisions) if r["dataset"] == d]) for d in DATASETS},
    }


def main() -> int:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"{OUTPUT_PATH.relative_to(ROOT)} already exists; refusing to overwrite")
    rows = json.loads(PROBE_RESULTS.read_text(encoding="utf-8"))["samples"]
    predictions = {(p["dataset"], p["sample_id"]): p
                   for p in json.loads(DIAGNOSIS_RESULTS.read_text(encoding="utf-8"))["predictions"]}
    if len(rows) != EXPECTED_VALIDATION or sum(r["probe_triggered"] for r in rows) != EXPECTED_TRIGGERED:
        raise RuntimeError("Unexpected validation or trigger counts")

    decisions = []
    for row in rows:
        diagnosis = diagnosis_from_prediction(predictions[(row["dataset"], row["sample_id"])])
        if diagnosis.final_label.value != row["diagnosis_label_before_probe"] or not math.isclose(
                diagnosis.stage1_failure_probability, row["stage1_failure_probability"], abs_tol=1e-12):
            raise RuntimeError(f"Diagnosis mismatch for {row['sample_id']}")
        decisions.append(decide_action(diagnosis, row["probe_triggered"], probe_result_from_dict(row["probe_result"]),
                                       sample_id=row["sample_id"]))

    rule = OverrideRule("S3", t1=SAFE_OVERRIDE_MAX_STAGE1, t2=SAFE_OVERRIDE_MAX_STAGE2)
    mismatches = [r["sample_id"] for r, d in zip(rows, decisions) if safe_label(rule, r) is not d.diagnosis_label]
    if mismatches:
        raise RuntimeError(f"Controller labels differ from offline Safe Binary S3: {mismatches[:5]}")

    dev_ids = {(s["dataset"], s["sample_id"]) for s in json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]}
    unseen = [i for i, r in enumerate(rows) if (r["dataset"], r["sample_id"]) not in dev_ids]
    result = {
        "note": (
            "Controller v1 routing dry-run on stored v2-expanded validation outputs; no model run, no test record "
            "read, no target labels read. Routing only; compensation is not implemented or evaluated."
        ),
        "sources": [str(p.relative_to(ROOT).as_posix()) for p in (PROBE_RESULTS, DIAGNOSIS_RESULTS)],
        "safe_binary": {"max_stage1_failure_probability": SAFE_OVERRIDE_MAX_STAGE1,
                        "max_stage2_score": SAFE_OVERRIDE_MAX_STAGE2},
        "consistency_with_offline_safe_binary_s3": {"mismatches": len(mismatches)},
        "full_validation": report(rows, decisions),
        "unseen_validation": report([rows[i] for i in unseen], [decisions[i] for i in unseen]),
        "decisions": [d.to_dict() | {"dataset": r["dataset"]} for r, d in zip(rows, decisions)],
    }
    write_json(OUTPUT_PATH, result)
    for name in ("full_validation", "unseen_validation"):
        for subset, dist in result[name].items():
            print(f"[{name} {subset}] {json.dumps(dist)}")
    print(f"consistency mismatches: {len(mismatches)}; wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
