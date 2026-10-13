"""Offline comparison of integration policies for Diagnosis v2-expanded + Evidence Probe (VALIDATION only).

Reads results/probe/probe_v2_expanded_validation.json; no model is run and
no test record is read. PRIMARY metrics use the unseen validation subset that
excludes the Probe prompt-development samples; the full validation split is a
supplementary result. Policies are those of
scripts/evaluate_probe_integration_policies.py:
  Policy 0 diagnosis_only, Policy 1 typed_probe, Policy 2 binary_verification_probe.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_probe_integration_policies import POLICIES, outcomes_vs_diagnosis, policy_label
from scripts.run_probe_v2_expanded_validation import EXPECTED_TRIGGERED, EXPECTED_VALIDATION
from scripts.run_probe_v2_expanded_validation import OUTPUT_PATH as PROBE_RESULTS
from scripts.run_probe_validation import error_counts, fever_verdicts, label_distribution
from scripts.train_diagnosis_classifier import evaluate_by_dataset, write_json
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.probe.schema import ProbeLabel

OUTPUT_PATH = ROOT / "results" / "probe" / "probe_v2_expanded_policy_comparison.json"
EXPECTED_PROMPT_DEVELOPMENT = 4
DATASETS = ("nq", "fever")


def probe_label_analysis(rows: list[dict], label: ProbeLabel) -> dict:
    """Predicted vs true rate and precision/recall of one Probe label on triggered rows."""
    report = {}
    for subset in ("all",) + DATASETS:
        sub = [r for r in rows if subset == "all" or r["dataset"] == subset]
        predicted = [r["probe_result"]["probe_label"] == label.value for r in sub]
        true = [r["offline_evaluation"]["target_label"] == label.value for r in sub]
        hits = sum(p and t for p, t in zip(predicted, true))
        n_pred, n_true = sum(predicted), sum(true)
        report[subset] = {
            "n": len(sub),
            "predicted": n_pred,
            "predicted_rate": n_pred / len(sub) if sub else None,
            "true": n_true,
            "true_rate": n_true / len(sub) if sub else None,
            "true_positives": hits,
            "precision": hits / n_pred if n_pred else None,
            "recall": hits / n_true if n_true else None,
        }
    return report


def subset_metrics(rows: list[dict]) -> dict:
    labels = {p: [policy_label(p, r) for r in rows] for p in POLICIES}
    eval_records = [{"input": {"dataset": r["dataset"]}, "target_label": r["offline_evaluation"]["target_label"]}
                    for r in rows]
    metrics = {}
    for policy in POLICIES:
        report = evaluate_by_dataset(eval_records, labels[policy])
        metrics[policy] = {subset: m | {"errors": error_counts(m)} for subset, m in report.items()}

    triggered = [i for i, r in enumerate(rows) if r["probe_triggered"]]
    trig_rows = [rows[i] for i in triggered]
    trig_labels = {p: [labels[p][i] for i in triggered] for p in POLICIES}
    probe = {
        "n": len(rows),
        "triggered": len(trig_rows),
        "coverage": len(trig_rows) / len(rows),
        "triggered_by_dataset": {
            d: {"triggered": sum(r["dataset"] == d for r in trig_rows), "n": sum(r["dataset"] == d for r in rows)}
            for d in DATASETS
        },
        "uncertain": sum(r["probe_result"]["probe_label"] == ProbeLabel.UNCERTAIN.value for r in trig_rows),
        "parse_failures": sum(r["probe_result"]["parse_error"] is not None for r in trig_rows),
        "outcomes_vs_diagnosis": {p: outcomes_vs_diagnosis(trig_rows, trig_labels, p) for p in POLICIES[1:]},
        "outcomes_by_dataset": {
            d: {p: outcomes_vs_diagnosis(
                    [r for r in trig_rows if r["dataset"] == d],
                    {q: [l for l, r in zip(trig_labels[q], trig_rows) if r["dataset"] == d] for q in POLICIES}, p)
                for p in POLICIES[1:]}
            for d in DATASETS
        },
        "probe_label_distribution": {
            "all": label_distribution(trig_rows),
            **{d: label_distribution([r for r in trig_rows if r["dataset"] == d]) for d in DATASETS},
        },
        "probe_label_analysis": {
            label.value: probe_label_analysis(trig_rows, label)
            for label in ProbeLabel if label is not ProbeLabel.UNCERTAIN
        },
        "fever_verdicts": fever_verdicts(trig_rows),
    }
    return {
        "metrics": metrics,
        "probe": probe,
        "predicted_label_counts": {p: dict(Counter(l.value for l in labels[p])) for p in POLICIES},
    }


def main() -> int:
    data = json.loads(PROBE_RESULTS.read_text(encoding="utf-8"))
    rows = data["samples"]
    if len(rows) != EXPECTED_VALIDATION or sum(r["probe_triggered"] for r in rows) != EXPECTED_TRIGGERED:
        raise RuntimeError("Unexpected validation or trigger counts")
    if [policy_label("typed_probe", r) for r in rows] != [parse_label(r["final_diagnosis_label"]) for r in rows]:
        raise RuntimeError("Typed policy does not reproduce the stored Diagnosis + Probe labels")
    unseen = [r for r in rows if not r["offline_evaluation"]["prompt_development_sample"]]
    if len(rows) - len(unseen) != EXPECTED_PROMPT_DEVELOPMENT:
        raise RuntimeError(f"Expected {EXPECTED_PROMPT_DEVELOPMENT} prompt-development samples in validation")

    primary = subset_metrics(unseen)
    supplementary = subset_metrics(rows)
    write_json(OUTPUT_PATH, {
        "note": (
            "Offline policy comparison on v2-expanded validation; no model run, no test record read. PRIMARY = "
            "unseen validation (prompt-development samples excluded); full validation is supplementary. FEVER has "
            "no trusted evidence_accessibility samples (N/A, not 0)."
        ),
        "source": str(PROBE_RESULTS.relative_to(ROOT).as_posix()),
        "trigger": data["trigger"],
        "excluded_prompt_development": [
            {"dataset": r["dataset"], "sample_id": r["sample_id"], "probe_triggered": r["probe_triggered"]}
            for r in rows if r["offline_evaluation"]["prompt_development_sample"]
        ],
        "primary_unseen_validation": primary,
        "supplementary_full_validation": supplementary,
    })

    def fmt(value: float | None) -> str:
        return " N/A " if value is None else f"{value:.3f}"

    for name, block in (("PRIMARY unseen", primary), ("SUPPLEMENTARY full", supplementary)):
        print(f"\n######## {name} (n={block['probe']['n']}) ########")
        for subset in ("all",) + DATASETS:
            print(f"\n== {subset} ==")
            for policy in POLICIES:
                m = block["metrics"][policy][subset]
                print(f"{policy:26s} n={m['n']} acc={fmt(m['accuracy'])} macro_f1={fmt(m['macro_f1'])} "
                      f"fdr={fmt(m['failure_detection_recall'])} "
                      f"correct_R={fmt(m['per_class'][DiagnosisLabel.CORRECT.value]['recall'])} {m['errors']}")
                for label, pc in m["per_class"].items():
                    print(f"    {label:24s} P={fmt(pc['precision'])} R={fmt(pc['recall'])} F1={fmt(pc['f1'])} "
                          f"support={pc['support']} predicted={pc['predicted']}")
                print(f"    confusion: {m['confusion_matrix']['rows_true_cols_pred']}")
        p = block["probe"]
        print("\nprobe:", json.dumps({k: p[k] for k in ("triggered", "coverage", "triggered_by_dataset", "uncertain",
                                                        "parse_failures", "outcomes_vs_diagnosis",
                                                        "outcomes_by_dataset", "probe_label_distribution",
                                                        "fever_verdicts")}))
        for label, analysis in p["probe_label_analysis"].items():
            for subset, a in analysis.items():
                print(f"  {label:24s} [{subset:5s}] n={a['n']} pred={a['predicted']} ({fmt(a['predicted_rate'])}) "
                      f"true={a['true']} ({fmt(a['true_rate'])}) P={fmt(a['precision'])} R={fmt(a['recall'])}")
    print(f"\nWrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
