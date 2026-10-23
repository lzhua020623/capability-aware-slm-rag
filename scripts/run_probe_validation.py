"""Full VALIDATION evaluation of the frozen Hierarchical Diagnosis + Evidence Probe v1.

Runs the frozen Diagnosis on all validation samples, the Probe once on every
triggered sample, and compares Diagnosis only vs Diagnosis + Probe. The test
split is never loaded. Before any model call it checks that validation has
EXPECTED_VALIDATION samples, that the live trigger set equals the trigger set
derived from the stored hierarchical validation predictions, and that every
prompt-development sample is triggered; otherwise it stops.
Target labels are used only in the offline_evaluation section.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_probe_smoke import MODEL_PATH, SAMPLES_PATH, offline_category, outcome
from scripts.train_diagnosis_classifier import (
    DATASET_PATHS,
    SPLIT_PATH,
    evaluate_by_dataset,
    input_from_dict,
    write_json,
)
from src.diagnosis.dev_dataset import load_jsonl
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosisClassifier
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.probe.pipeline import apply_probe
from src.probe.probe import EvidenceProbe
from src.probe.schema import ProbeLabel
from src.probe.trigger import HIGH_THRESHOLD, LOW_THRESHOLD, STAGE2_THRESHOLD, should_probe

HIERARCHICAL_RESULTS = ROOT / "results" / "diagnosis" / "diagnosis_hierarchical_validation.json"
OUTPUT_PATH = ROOT / "results" / "probe" / "probe_validation_full.json"
EXPECTED_VALIDATION = 138
EXPECTED_TRIGGERED = 46
OUTCOMES = ("fixed", "kept_correct", "worsened", "still_wrong", "uncertain")


def load_validation() -> list[dict]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    records = []
    for dataset, path in DATASET_PATHS.items():
        by_id = {r["input"]["sample_id"]: r for r in load_jsonl(path)}
        records += [by_id[sample_id] for sample_id in split["splits"][dataset]["validation"]]
    return records


def stored_trigger_ids() -> set[tuple[str, str]]:
    stored = json.loads(HIERARCHICAL_RESULTS.read_text(encoding="utf-8"))["predictions"]
    return {
        (row["dataset"], row["sample_id"]) for row in stored
        if should_probe(row["stage1_failure_probability"], row["stage2_class_scores"])
    }


def error_counts(metrics: dict) -> dict:
    matrix = metrics["confusion_matrix"]["rows_true_cols_pred"]
    return {
        "correct_to_failure_false_positives": sum(matrix[0][1:]),
        "failure_to_correct_false_negatives": sum(row[0] for row in matrix[1:]),
        "failure_type_errors": sum(matrix[i][j] for i in range(1, 4) for j in range(1, 4) if i != j),
    }


def outcome_summary(rows: list[dict]) -> dict:
    counts = Counter(row["offline_evaluation"]["outcome"] for row in rows)
    before_correct = sum(row["offline_evaluation"]["correct_before_probe"] for row in rows)
    after_correct = sum(row["offline_evaluation"]["correct_after_probe"] for row in rows)
    before_wrong = len(rows) - before_correct
    return {
        "n": len(rows),
        "correct_before_probe": before_correct,
        "correct_after_probe": after_correct,
        "counts": {name: counts.get(name, 0) for name in OUTCOMES},
        "rates": {name: counts.get(name, 0) / len(rows) for name in OUTCOMES} if rows else {},
        "net_gain": counts.get("fixed", 0) - counts.get("worsened", 0),
        "correction_rate": counts.get("fixed", 0) / before_wrong if before_wrong else None,
        "harm_rate": counts.get("worsened", 0) / before_correct if before_correct else None,
    }


def label_distribution(rows: list[dict]) -> dict:
    counts = Counter(row["probe_result"]["probe_label"] for row in rows)
    return {label.value: counts.get(label.value, 0) for label in ProbeLabel}


def insufficiency_analysis(rows: list[dict]) -> dict:
    predicted = [r for r in rows if r["probe_result"]["probe_label"] == ProbeLabel.EVIDENCE_INSUFFICIENCY.value]
    true_counts = Counter(r["offline_evaluation"]["target_label"] for r in predicted)
    by_dataset = {}
    for dataset in DATASET_PATHS:
        subset = [r for r in rows if r["dataset"] == dataset]
        hits = sum(r["probe_result"]["probe_label"] == ProbeLabel.EVIDENCE_INSUFFICIENCY.value for r in subset)
        true_insufficient = sum(
            r["offline_evaluation"]["target_label"] == DiagnosisLabel.EVIDENCE_INSUFFICIENCY.value for r in subset
        )
        by_dataset[dataset] = {
            "n": len(subset),
            "predicted_insufficiency": hits,
            "prediction_rate": hits / len(subset) if subset else None,
            "true_insufficiency": true_insufficient,
            "true_rate": true_insufficient / len(subset) if subset else None,
        }
    return {
        "predicted_insufficiency": len(predicted),
        "true_label_of_predicted": {label.value: true_counts.get(label.value, 0) for label in DiagnosisLabel},
        "precision": true_counts.get(DiagnosisLabel.EVIDENCE_INSUFFICIENCY.value, 0) / len(predicted) if predicted else None,
        "true_insufficiency_in_subset": sum(
            r["offline_evaluation"]["target_label"] == DiagnosisLabel.EVIDENCE_INSUFFICIENCY.value for r in rows
        ),
        "by_dataset": by_dataset,
    }


def fever_verdicts(rows: list[dict]) -> dict:
    fever = [r for r in rows if r["dataset"] == "fever"]
    verdicts = Counter(r["probe_result"]["evidence_verdict"] for r in fever)
    matches = sum(r["probe_result"]["evidence_verdict"] == r["initial_answer"].strip().upper() for r in fever)
    insufficient = verdicts.get("INSUFFICIENT", 0)
    return {
        "n": len(fever),
        "verdicts": dict(verdicts),
        "verdict_equals_initial_answer": matches,
        "verdict_differs_from_initial_answer": len(fever) - matches - insufficient - verdicts.get(None, 0),
        "insufficient": insufficient,
        "unparsed": verdicts.get(None, 0),
    }


def main() -> int:
    records = load_validation()
    if len(records) != EXPECTED_VALIDATION:
        raise RuntimeError(f"Expected {EXPECTED_VALIDATION} validation samples, got {len(records)}")

    classifier = HierarchicalDiagnosisClassifier.load(MODEL_PATH)
    inputs = [input_from_dict(r["input"]) for r in records]
    diagnoses = classifier.diagnose_batch(inputs)
    triggered = [should_probe(d.stage1_failure_probability, d.stage2_class_scores) for d in diagnoses]
    live_ids = {(item.dataset.value, item.sample_id) for item, t in zip(inputs, triggered) if t}
    stored_ids = stored_trigger_ids()
    dev_ids = {(s["dataset"], s["sample_id"]) for s in json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]}
    checks = {
        "validation_count": len(records),
        "triggered_count": len(live_ids),
        "matches_stored_trigger_analysis": live_ids == stored_ids,
        "prompt_development_samples": len(dev_ids),
        "prompt_development_all_triggered": dev_ids <= live_ids,
    }
    print("pre-run checks:", checks)
    if len(live_ids) != EXPECTED_TRIGGERED or live_ids != stored_ids or not dev_ids <= live_ids:
        raise RuntimeError(f"Pre-run checks failed; not running the model: {checks}")

    probe = EvidenceProbe.from_qwen()
    rows = []
    for item, diagnosis, record, is_triggered in zip(inputs, diagnoses, records, triggered):
        start = time.perf_counter()
        result = apply_probe(item, diagnosis, probe)
        latency = time.perf_counter() - start if is_triggered else 0.0
        assert result.probe_triggered == is_triggered
        target = parse_label(record["target_label"])
        before, after = diagnosis.final_label, result.final_label
        row = {
            "sample_id": item.sample_id,
            "dataset": item.dataset.value,
            "initial_answer": item.initial_answer,
            "diagnosis_label_before_probe": before.value,
            "stage1_failure_probability": diagnosis.stage1_failure_probability,
            "stage2_class_scores": None if diagnosis.stage2_class_scores is None
            else {k.value: v for k, v in diagnosis.stage2_class_scores.items()},
            "probe_triggered": result.probe_triggered,
            "probe_result": None if result.probe_result is None else result.probe_result.to_dict(),
            "probe_latency_seconds": round(latency, 3) if is_triggered else None,
            "final_diagnosis_label": after.value,
            "offline_evaluation": {
                "target_label": target.value,
                "prompt_development_sample": (item.dataset.value, item.sample_id) in dev_ids,
                "category_before_probe": offline_category(target, before),
                "correct_before_probe": before is target,
                "correct_after_probe": after is target,
                "outcome": outcome(target, before, after, result.probe_result.probe_label) if is_triggered else None,
            },
        }
        rows.append(row)
        if is_triggered:
            pr = row["probe_result"]
            print(f"{item.sample_id:>22s} {item.dataset.value:5s} {before.value:24s} -> {after.value:24s} "
                  f"[{pr['answer_supported']}/{pr['evidence_sufficiency']}/{pr['evidence_accessibility']}"
                  f"{'/' + pr['evidence_verdict'] if pr['evidence_verdict'] else ''}] "
                  f"target={target.value:24s} {row['offline_evaluation']['outcome']} ({latency:.1f}s)")

    targets = [r["offline_evaluation"]["target_label"] for r in rows]
    before_labels = [parse_label(r["diagnosis_label_before_probe"]) for r in rows]
    after_labels = [parse_label(r["final_diagnosis_label"]) for r in rows]
    eval_records = [{"input": {"dataset": r["dataset"]}, "target_label": t} for r, t in zip(rows, targets)]
    before_metrics = evaluate_by_dataset(eval_records, before_labels)
    after_metrics = evaluate_by_dataset(eval_records, after_labels)
    comparison = {
        subset: {
            "diagnosis_only": before_metrics[subset] | {"errors": error_counts(before_metrics[subset])},
            "diagnosis_plus_probe": after_metrics[subset] | {"errors": error_counts(after_metrics[subset])},
        }
        for subset in before_metrics
    }

    probed = [r for r in rows if r["probe_triggered"]]
    unseen = [r for r in probed if not r["offline_evaluation"]["prompt_development_sample"]]
    latencies = [r["probe_latency_seconds"] for r in probed]
    fixed = sum(r["offline_evaluation"]["outcome"] == "fixed" for r in probed)
    report = {
        "triggered_46": outcome_summary(probed),
        "unseen_38": outcome_summary(unseen),
        "prompt_development_8": outcome_summary([r for r in probed if r["offline_evaluation"]["prompt_development_sample"]]),
        "triggered_by_dataset": {d: outcome_summary([r for r in probed if r["dataset"] == d]) for d in DATASET_PATHS},
        "unseen_by_dataset": {d: outcome_summary([r for r in unseen if r["dataset"] == d]) for d in DATASET_PATHS},
        "probe_label_distribution": {
            "triggered_46": label_distribution(probed),
            "unseen_38": label_distribution(unseen),
            "triggered_by_dataset": {d: label_distribution([r for r in probed if r["dataset"] == d]) for d in DATASET_PATHS},
        },
        "insufficiency_analysis": {"triggered_46": insufficiency_analysis(probed), "unseen_38": insufficiency_analysis(unseen)},
        "fever_verdicts": {"triggered": fever_verdicts(probed), "unseen": fever_verdicts(unseen)},
        "parse_failures": sum(r["probe_result"]["parse_error"] is not None for r in probed),
        "cost": {
            "validation_samples": len(rows),
            "probe_calls": len(probed),
            "probe_coverage": len(probed) / len(rows),
            "additional_calls_per_sample": len(probed) / len(rows),
            "max_calls_per_triggered_sample": 1,
            "mean_latency_seconds": sum(latencies) / len(latencies),
            "total_latency_seconds": sum(latencies),
            "calls_per_fixed_sample": len(probed) / fixed if fixed else None,
        },
    }

    write_json(OUTPUT_PATH, {
        "note": (
            "Frozen Hierarchical Diagnosis + Evidence Probe v1 on validation only; test split not loaded. "
            "Only offline_evaluation fields contain target labels; they never reach the Probe prompt."
        ),
        "model": "Qwen/Qwen2.5-7B-Instruct (4-bit NF4, do_sample=False), one call per triggered sample",
        "trigger": {"low": LOW_THRESHOLD, "high": HIGH_THRESHOLD, "stage2_threshold": STAGE2_THRESHOLD},
        "pre_run_checks": checks,
        "samples": rows,
        "offline_evaluation": {"validation_138": comparison, **report},
    })

    for subset in ("all", "nq", "fever"):
        b, a = comparison[subset]["diagnosis_only"], comparison[subset]["diagnosis_plus_probe"]
        print(f"[{subset}] acc {b['accuracy']:.3f}->{a['accuracy']:.3f} macro_f1 {b['macro_f1']:.3f}->{a['macro_f1']:.3f} "
              f"FDR {b['failure_detection_recall']:.3f}->{a['failure_detection_recall']:.3f} "
              f"errors {b['errors']} -> {a['errors']}")
    print(json.dumps({k: report[k] for k in ("triggered_46", "unseen_38", "parse_failures", "cost")}, indent=1))
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
