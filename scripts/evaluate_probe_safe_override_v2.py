"""Offline search of Safe Binary Override policies for Diagnosis v2-expanded + Evidence Probe (VALIDATION only).

Reads existing result files only; no model is run and no test record is read.
The Probe trigger (V2_EXPANDED_TRIGGER) and Probe outputs are unchanged; only
the rule deciding whether a Probe "correct" may override a Diagnosis failure
label varies:
  S0: always override (current binary_verification_probe);
  S1: override iff Stage 1 P(failure) <= t1;
  S2: override iff max Stage 2 score <= t2;
  S3: override iff P(failure) <= t1 and max Stage 2 score <= t2.
Stage 2 conditions require Stage 2 scores; without them no override happens.
Any other Probe label keeps the Diagnosis label. PRIMARY metrics exclude the
prompt-development validation samples.

Selection: failure detection recall >= MIN_RECALL; maximize macro-F1; within
TOLERANCE prefer higher failure detection recall, fewer correct->failure false
positives, then the simpler rule (S0 < S1 = S2 < S3), then looser thresholds.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_probe_integration_policies import policy_label
from scripts.evaluate_probe_v2_expanded_policies import EXPECTED_PROMPT_DEVELOPMENT
from scripts.run_probe_v2_expanded_validation import EXPECTED_TRIGGERED, EXPECTED_VALIDATION
from scripts.run_probe_v2_expanded_validation import OUTPUT_PATH as PROBE_RESULTS
from scripts.run_probe_validation import error_counts
from scripts.search_probe_trigger_v2_expanded import OUTPUT_PATH as TRIGGER_SEARCH, distribution
from scripts.train_diagnosis_classifier import evaluate_by_dataset, write_json
from scripts.train_diagnosis_v2_expanded import OUTPUT_PATH as DIAGNOSIS_RESULTS
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.probe.schema import ProbeLabel
from src.probe.trigger import V2_EXPANDED_TRIGGER

OUTPUT_PATH = ROOT / "results" / "probe" / "probe_v2_safe_override_comparison.json"
T1_CANDIDATES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
T2_CANDIDATES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
PRIMARY_MIN_RECALL = 0.65
STRICT_MIN_RECALL = 0.68
TOLERANCE = 0.01
DATASETS = ("nq", "fever")
COMPLEXITY = {"S0": 0, "S1": 1, "S2": 1, "S3": 2}


@dataclass(frozen=True)
class OverrideRule:
    family: str
    t1: float | None = None
    t2: float | None = None

    @property
    def name(self) -> str:
        parts = [self.family]
        if self.t1 is not None:
            parts.append(f"p<={self.t1:.2f}")
        if self.t2 is not None:
            parts.append(f"s2max<={self.t2:.2f}")
        return " ".join(parts)

    def allows(self, stage1_failure_probability: float, stage2_class_scores: dict | None) -> bool:
        if self.t1 is not None and stage1_failure_probability > self.t1:
            return False
        if self.t2 is not None:
            if not stage2_class_scores:
                return False
            if max(stage2_class_scores.values()) > self.t2:
                return False
        return True


def candidate_rules() -> list[OverrideRule]:
    rules = [OverrideRule("S0")]
    rules += [OverrideRule("S1", t1=t1) for t1 in T1_CANDIDATES]
    rules += [OverrideRule("S2", t2=t2) for t2 in T2_CANDIDATES]
    rules += [OverrideRule("S3", t1=t1, t2=t2) for t1 in T1_CANDIDATES for t2 in T2_CANDIDATES]
    return rules


def safe_label(rule: OverrideRule, row: dict) -> DiagnosisLabel:
    before = parse_label(row["diagnosis_label_before_probe"])
    if not row["probe_triggered"] or row["probe_result"]["probe_label"] != ProbeLabel.CORRECT.value:
        return before
    if before is DiagnosisLabel.CORRECT:
        return before
    return DiagnosisLabel.CORRECT if rule.allows(row["stage1_failure_probability"], row["stage2_class_scores"]) else before


def stage2_max(row: dict) -> float | None:
    scores = row["stage2_class_scores"]
    return max(scores.values()) if scores else None


def is_override_candidate(row: dict) -> bool:
    return (row["probe_triggered"]
            and row["probe_result"]["probe_label"] == ProbeLabel.CORRECT.value
            and row["diagnosis_label_before_probe"] != DiagnosisLabel.CORRECT.value)


def group_summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "by_dataset": {d: sum(r["dataset"] == d for r in rows) for d in DATASETS},
        "stage1_failure_probability": distribution(r["stage1_failure_probability"] for r in rows),
        "stage2_max": distribution(m for r in rows if (m := stage2_max(r)) is not None),
        "diagnosis_failure_type": dict(Counter(r["diagnosis_label_before_probe"] for r in rows)),
        "true_label": dict(Counter(r["offline_evaluation"]["target_label"] for r in rows)),
    }


def candidate_analysis(rows: list[dict]) -> dict:
    candidates = [r for r in rows if is_override_candidate(r)]
    fixable = [r for r in candidates if r["offline_evaluation"]["target_label"] == DiagnosisLabel.CORRECT.value]
    harmful = [r for r in candidates if r["offline_evaluation"]["target_label"] != DiagnosisLabel.CORRECT.value]
    is_harmful = [r["offline_evaluation"]["target_label"] != DiagnosisLabel.CORRECT.value for r in candidates]

    def auc(values: list[float]) -> float | None:
        return roc_auc_score(is_harmful, values) if 0 < sum(is_harmful) < len(is_harmful) else None

    by_dataset_type = Counter(
        (r["dataset"], r["diagnosis_label_before_probe"], "harmful" if h else "fixable")
        for r, h in zip(candidates, is_harmful)
    )
    return {
        "all_candidates": group_summary(candidates),
        "A_fixable_true_correct": group_summary(fixable),
        "B_harmful_true_failure": group_summary(harmful),
        "separability_auc_harmful_vs_fixable": {
            "stage1_failure_probability": auc([r["stage1_failure_probability"] for r in candidates]),
            "stage2_max": auc([stage2_max(r) or 0.0 for r in candidates]),
        },
        "by_dataset_and_diagnosis_type": {"|".join(k): v for k, v in sorted(by_dataset_type.items())},
    }


def metrics_for(rows: list[dict], labels: list[DiagnosisLabel]) -> dict:
    records = [{"input": {"dataset": r["dataset"]}, "target_label": r["offline_evaluation"]["target_label"]}
               for r in rows]
    return {subset: m | {"errors": error_counts(m)} for subset, m in evaluate_by_dataset(records, labels).items()}


def summary(name: str, metrics: dict, rule: OverrideRule | None = None) -> dict:
    m = metrics["all"]
    return {
        "policy": name,
        "family": None if rule is None else rule.family,
        "t1": None if rule is None else rule.t1,
        "t2": None if rule is None else rule.t2,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "failure_detection_recall": m["failure_detection_recall"],
        "correct_recall": m["per_class"][DiagnosisLabel.CORRECT.value]["recall"],
        **m["errors"],
    }


def select(results: list[dict], min_recall: float) -> dict | None:
    feasible = [r for r in results if r["failure_detection_recall"] >= min_recall]
    if not feasible:
        return None
    best = max(r["macro_f1"] for r in feasible)
    near = [r for r in feasible if best - r["macro_f1"] <= TOLERANCE]
    return min(near, key=lambda r: (
        -r["failure_detection_recall"],
        r["correct_to_failure_false_positives"],
        COMPLEXITY[r["family"]],
        -(r["t1"] if r["t1"] is not None else 1.0),
        -(r["t2"] if r["t2"] is not None else 1.0),
    ))


def override_outcomes(rows: list[dict], before: list[DiagnosisLabel], after: list[DiagnosisLabel]) -> dict:
    """Failure -> correct overrides relative to Diagnosis; other label changes are counted separately."""
    counts = Counter()
    for row, b, a in zip(rows, before, after):
        if a is b:
            continue
        if b is DiagnosisLabel.CORRECT or a is not DiagnosisLabel.CORRECT:
            counts["other_label_changes"] += 1
            continue
        target = parse_label(row["offline_evaluation"]["target_label"])
        if target is DiagnosisLabel.CORRECT:
            counts["fp_fixed"] += 1
        else:
            counts["fn_added"] += 1
            counts["fn_added_with_correct_failure_type"] += b is target
    fixed, added = counts["fp_fixed"], counts["fn_added"]
    return {
        "failure_to_correct_overrides": fixed + added,
        "fp_fixed": fixed,
        "fn_added": added,
        "fn_added_with_correct_failure_type": counts["fn_added_with_correct_failure_type"],
        "fn_added_per_fp_fixed": added / fixed if fixed else None,
        "other_label_changes": counts["other_label_changes"],
    }


def main() -> int:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"{OUTPUT_PATH.relative_to(ROOT)} already exists; refusing to overwrite")
    data = json.loads(PROBE_RESULTS.read_text(encoding="utf-8"))
    rows = data["samples"]
    if len(rows) != EXPECTED_VALIDATION or sum(r["probe_triggered"] for r in rows) != EXPECTED_TRIGGERED:
        raise RuntimeError("Unexpected validation or trigger counts")
    trigger = (V2_EXPANDED_TRIGGER.low, V2_EXPANDED_TRIGGER.high, V2_EXPANDED_TRIGGER.stage2_threshold)
    chosen = json.loads(TRIGGER_SEARCH.read_text(encoding="utf-8"))["step2_search"]["chosen"]
    if (data["trigger"]["low"], data["trigger"]["high"], data["trigger"]["stage2_threshold"]) != trigger or (
            chosen["low"], chosen["high"], chosen["stage2_threshold"]) != trigger:
        raise RuntimeError("Probe results were not produced with the frozen v2 trigger")
    stored = {(r["dataset"], r["sample_id"]): r for r in
              json.loads(DIAGNOSIS_RESULTS.read_text(encoding="utf-8"))["predictions"]}
    if any(stored[(r["dataset"], r["sample_id"])]["final_label"] != r["diagnosis_label_before_probe"] for r in rows):
        raise RuntimeError("Diagnosis labels differ from the stored v2 validation predictions")
    unseen = [r for r in rows if not r["offline_evaluation"]["prompt_development_sample"]]
    if len(rows) - len(unseen) != EXPECTED_PROMPT_DEVELOPMENT:
        raise RuntimeError("Unexpected number of prompt-development samples")

    def evaluate_set(subset_rows: list[dict]) -> dict:
        diagnosis = [policy_label("diagnosis_only", r) for r in subset_rows]
        typed = [policy_label("typed_probe", r) for r in subset_rows]
        binary = [policy_label("binary_verification_probe", r) for r in subset_rows]
        rule_labels = {rule: [safe_label(rule, r) for r in subset_rows] for rule in candidate_rules()}
        if rule_labels[OverrideRule("S0")] != binary:
            raise RuntimeError("S0 does not reproduce the current binary policy")
        results = [summary(rule.name, metrics_for(subset_rows, labels), rule) | {
            "override_outcomes": override_outcomes(subset_rows, diagnosis, labels)}
            for rule, labels in rule_labels.items()]
        best = {f"recall_ge_{m}": select(results, m) for m in (PRIMARY_MIN_RECALL, STRICT_MIN_RECALL)}
        by_name = {rule.name: rule for rule in rule_labels}
        compared = {"diagnosis_only": diagnosis, "typed_probe": typed, "current_binary": binary}
        for key, chosen_result in best.items():
            if chosen_result is not None:
                compared[f"safe_binary_{key}"] = rule_labels[by_name[chosen_result["policy"]]]
        return {
            "n": len(subset_rows),
            "search": results,
            "best": best,
            "comparison": {
                name: {
                    "metrics": metrics_for(subset_rows, labels),
                    "override_outcomes_vs_diagnosis": override_outcomes(subset_rows, diagnosis, labels),
                    "predicted_label_counts": dict(Counter(l.value for l in labels)),
                }
                for name, labels in compared.items()
            },
        }

    primary = evaluate_set(unseen)
    supplementary = evaluate_set(rows)
    report = {
        "note": (
            "Offline Safe Binary Override search on v2-expanded validation; no model run and no test record read. "
            "Probe trigger and Probe outputs are unchanged. PRIMARY = unseen validation (prompt-development samples "
            "excluded); full validation is supplementary. Thresholds are selected on validation only."
        ),
        "sources": [str(p.relative_to(ROOT).as_posix()) for p in (PROBE_RESULTS, DIAGNOSIS_RESULTS, TRIGGER_SEARCH)],
        "trigger": data["trigger"],
        "selection_rule": (
            f"failure_detection_recall >= {PRIMARY_MIN_RECALL} (strict reference: >= {STRICT_MIN_RECALL}); max "
            f"macro_f1; within {TOLERANCE} prefer higher failure_detection_recall, fewer correct->failure false "
            "positives, simpler rule (S0 < S1 = S2 < S3), looser thresholds"
        ),
        "primary_candidate_analysis": candidate_analysis(unseen),
        "primary_unseen_validation": primary,
        "supplementary_full_validation": {k: supplementary[k] for k in ("n", "best", "comparison")},
    }
    write_json(OUTPUT_PATH, report)

    def fmt(value: float | None) -> str:
        return " N/A " if value is None else f"{value:.3f}"

    print(json.dumps(report["primary_candidate_analysis"], indent=1))
    print("\n== PRIMARY search (sorted by macro_f1) ==")
    for r in sorted(primary["search"], key=lambda r: -r["macro_f1"])[:25]:
        o = r["override_outcomes"]
        print(f"{r['policy']:28s} acc={r['accuracy']:.3f} mf1={r['macro_f1']:.3f} fdr={r['failure_detection_recall']:.3f} "
              f"cR={r['correct_recall']:.3f} FP={r['correct_to_failure_false_positives']} "
              f"FN={r['failure_to_correct_false_negatives']} type={r['failure_type_errors']} "
              f"fixed={o['fp_fixed']} added={o['fn_added']}")
    for family in ("S1", "S2"):
        print(f"\n-- {family} sweep --")
        for r in primary["search"]:
            if r["family"] == family:
                print(f"{r['policy']:28s} mf1={r['macro_f1']:.3f} fdr={r['failure_detection_recall']:.3f} "
                      f"FP={r['correct_to_failure_false_positives']} FN={r['failure_to_correct_false_negatives']}")
    for key, best in primary["best"].items():
        print(f"BEST {key}: {best['policy'] if best else None}")
    for name, block in primary["comparison"].items():
        print(f"\n######## {name} ########")
        for subset in ("all",) + DATASETS:
            m = block["metrics"][subset]
            print(f"[{subset}] n={m['n']} acc={fmt(m['accuracy'])} macro_f1={fmt(m['macro_f1'])} "
                  f"fdr={fmt(m['failure_detection_recall'])} "
                  f"correct_R={fmt(m['per_class']['correct']['recall'])} {m['errors']}")
            for label, pc in m["per_class"].items():
                print(f"    {label:24s} P={fmt(pc['precision'])} R={fmt(pc['recall'])} F1={fmt(pc['f1'])} "
                      f"support={pc['support']}")
            print(f"    confusion: {m['confusion_matrix']['rows_true_cols_pred']}")
        print("override vs diagnosis:", block["override_outcomes_vs_diagnosis"])
    print("\nsupplementary best:", {k: v["policy"] if v else None for k, v in supplementary["best"].items()})
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
