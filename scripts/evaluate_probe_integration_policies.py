"""Offline comparison of Diagnosis + Probe integration policies on VALIDATION.

Reads only existing validation result files; no model is loaded or run and
the test split is never read.

Policy 0 (diagnosis_only): the Hierarchical Diagnosis label.
Policy 1 (typed_probe): the Probe label when triggered and not uncertain.
Policy 2 (binary_verification_probe): when triggered, Probe "correct"
    overrides the Diagnosis label; any other Probe label keeps the Diagnosis
    label, so failure types always come from Diagnosis Stage 2.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_probe_smoke import SAMPLES_PATH
from scripts.run_probe_validation import HIERARCHICAL_RESULTS, error_counts
from scripts.train_diagnosis_classifier import evaluate_by_dataset, write_json
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.probe.schema import ProbeLabel

PROBE_RESULTS = ROOT / "results" / "probe" / "probe_validation_full.json"
OUTPUT_PATH = ROOT / "results" / "probe" / "probe_integration_policy_comparison.json"
EXPECTED_VALIDATION = 138
EXPECTED_TRIGGERED = 46
POLICIES = ("diagnosis_only", "typed_probe", "binary_verification_probe")


def policy_label(policy: str, row: dict) -> DiagnosisLabel:
    before = parse_label(row["diagnosis_label_before_probe"])
    if policy == "diagnosis_only" or not row["probe_triggered"]:
        return before
    probe_label = ProbeLabel(row["probe_result"]["probe_label"])
    if probe_label is ProbeLabel.UNCERTAIN:
        return before
    if policy == "typed_probe":
        return probe_label.to_diagnosis_label()
    if policy == "binary_verification_probe":
        return DiagnosisLabel.CORRECT if probe_label is ProbeLabel.CORRECT else before
    raise ValueError(f"Unknown policy {policy!r}")


def outcomes_vs_diagnosis(rows: list[dict], labels: dict[str, list[DiagnosisLabel]], policy: str) -> dict:
    counts = Counter()
    for i, row in enumerate(rows):
        target = parse_label(row["offline_evaluation"]["target_label"])
        before_ok = labels["diagnosis_only"][i] is target
        after_ok = labels[policy][i] is target
        counts[("kept_correct" if after_ok else "worsened") if before_ok else ("fixed" if after_ok else "still_wrong")] += 1
    names = ("fixed", "kept_correct", "worsened", "still_wrong")
    return {name: counts.get(name, 0) for name in names} | {"net_gain": counts["fixed"] - counts["worsened"]}


def subset_report(rows: list[dict], labels: dict[str, list[DiagnosisLabel]]) -> dict:
    targets = [parse_label(row["offline_evaluation"]["target_label"]) for row in rows]
    return {
        "n": len(rows),
        "final_correct": {p: sum(l is t for l, t in zip(labels[p], targets)) for p in POLICIES},
        "vs_diagnosis": {p: outcomes_vs_diagnosis(rows, labels, p) for p in POLICIES[1:]},
    }


def type_overrides(rows: list[dict], labels: dict[str, list[DiagnosisLabel]], policy: str) -> dict:
    """Failure-type changes made by a policy relative to the Diagnosis failure type."""
    changes = Counter()
    for i, _row in enumerate(rows):
        before, after = labels["diagnosis_only"][i], labels[policy][i]
        if before is not DiagnosisLabel.CORRECT and after is not DiagnosisLabel.CORRECT and before is not after:
            changes[f"{before.value}->{after.value}"] += 1
    return dict(changes)


def main() -> int:
    data = json.loads(PROBE_RESULTS.read_text(encoding="utf-8"))
    rows = data["samples"]
    stored = {
        (r["dataset"], r["sample_id"]): r["final_label"]
        for r in json.loads(HIERARCHICAL_RESULTS.read_text(encoding="utf-8"))["predictions"]
    }
    if len(rows) != EXPECTED_VALIDATION or sum(r["probe_triggered"] for r in rows) != EXPECTED_TRIGGERED:
        raise RuntimeError("Unexpected validation or trigger counts")
    mismatched = [r["sample_id"] for r in rows
                  if stored[(r["dataset"], r["sample_id"])] != r["diagnosis_label_before_probe"]]
    if mismatched:
        raise RuntimeError(f"Diagnosis labels differ from stored hierarchical results: {mismatched[:5]}")
    dev_ids = {(s["dataset"], s["sample_id"]) for s in json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]}

    labels = {p: [policy_label(p, r) for r in rows] for p in POLICIES}
    typed_check = [parse_label(r["final_diagnosis_label"]) for r in rows]
    if labels["typed_probe"] != typed_check:
        raise RuntimeError("Typed policy does not reproduce the stored Diagnosis + Probe labels")

    eval_records = [{"input": {"dataset": r["dataset"]}, "target_label": r["offline_evaluation"]["target_label"]}
                    for r in rows]
    metrics = {}
    for policy in POLICIES:
        report = evaluate_by_dataset(eval_records, labels[policy])
        metrics[policy] = {subset: m | {"errors": error_counts(m)} for subset, m in report.items()}

    triggered_idx = [i for i, r in enumerate(rows) if r["probe_triggered"]]
    unseen_idx = [i for i in triggered_idx if (rows[i]["dataset"], rows[i]["sample_id"]) not in dev_ids]

    def pick(indices: list[int]) -> tuple[list[dict], dict]:
        return [rows[i] for i in indices], {p: [labels[p][i] for i in indices] for p in POLICIES}

    triggered_rows, triggered_labels = pick(triggered_idx)
    unseen_rows, unseen_labels = pick(unseen_idx)
    comparison = {
        "note": (
            "Offline policy simulation on validation only; no model was run and the test split was not read. "
            "Minority classes are tiny; FEVER has no trusted evidence_accessibility samples (N/A, not 0)."
        ),
        "sources": [str(PROBE_RESULTS.relative_to(ROOT)), str(HIERARCHICAL_RESULTS.relative_to(ROOT))],
        "policies": {
            "diagnosis_only": "Policy 0: Hierarchical Diagnosis label only.",
            "typed_probe": "Policy 1: triggered and not uncertain -> Probe label; otherwise Diagnosis label.",
            "binary_verification_probe": (
                "Policy 2: triggered and Probe label correct -> correct; otherwise Diagnosis label "
                "(failure types always from Diagnosis Stage 2)."
            ),
        },
        "validation_138": metrics,
        "triggered_46": subset_report(triggered_rows, triggered_labels),
        "unseen_38": subset_report(unseen_rows, unseen_labels),
        "failure_type_overrides": {p: type_overrides(rows, labels, p) for p in POLICIES[1:]},
        "predicted_label_counts": {p: dict(Counter(l.value for l in labels[p])) for p in POLICIES},
    }
    write_json(OUTPUT_PATH, comparison)

    def fmt(value: float | None) -> str:
        return " N/A " if value is None else f"{value:.3f}"

    for subset in ("all", "nq", "fever"):
        print(f"\n== {subset} ==")
        for policy in POLICIES:
            m = metrics[policy][subset]
            print(f"{policy:26s} acc={fmt(m['accuracy'])} macro_f1={fmt(m['macro_f1'])} "
                  f"fdr={fmt(m['failure_detection_recall'])} {m['errors']}")
            for label, pc in m["per_class"].items():
                print(f"    {label:24s} P={fmt(pc['precision'])} R={fmt(pc['recall'])} F1={fmt(pc['f1'])} "
                      f"support={pc['support']} predicted={pc['predicted']}")
            if subset == "all":
                print(f"    confusion: {m['confusion_matrix']['rows_true_cols_pred']}")
    print("\ntriggered_46:", json.dumps(comparison["triggered_46"]))
    print("unseen_38:", json.dumps(comparison["unseen_38"]))
    print("failure_type_overrides:", json.dumps(comparison["failure_type_overrides"]))
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
