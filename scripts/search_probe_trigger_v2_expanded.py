"""Offline Probe trigger analysis and search for Diagnosis v2-expanded (VALIDATION only).

Reads the stored v2-expanded validation predictions; no model is loaded and
no test record is read (the split file is read for sample IDs only).

Step 1 describes Stage 1 / Stage 2 probabilities by true outcome and by
Diagnosis error type:
  A = true correct predicted failure,
  B = true failure predicted correct,
  C = failure detected but Stage 2 failure type wrong.
Step 2 grid-searches TriggerPolicy(low, high, stage2_threshold). Selection:
  1. coverage within COVERAGE_RANGE;
  2. overall error capture rate within TOLERANCE of the best feasible;
  3. among those, A+C capture rate within TOLERANCE of the best;
  4. then fewest self-checks (triggered samples Diagnosis already got right),
     lower coverage, narrower band, lower stage2 threshold.
Step 3 locates the Probe prompt-development samples in the v2 split.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_diagnosis_v2_expanded_split import OUTPUT_PATH as SPLIT_PATH
from scripts.run_probe_smoke import SAMPLES_PATH
from scripts.train_diagnosis_classifier import write_json
from scripts.train_diagnosis_v2_expanded import OUTPUT_PATH as DIAGNOSIS_RESULTS
from src.probe.trigger import TriggerPolicy

V1_DIAGNOSIS_RESULTS = ROOT / "results" / "diagnosis" / "diagnosis_hierarchical_validation.json"
OUTPUT_PATH = ROOT / "results" / "probe" / "probe_trigger_v2_expanded_search.json"
LOW_CANDIDATES = tuple(round(0.25 + 0.05 * i, 2) for i in range(7))
HIGH_MAX = 0.80
STAGE2_CANDIDATES = tuple(round(0.35 + 0.05 * i, 2) for i in range(7))
COVERAGE_RANGE = (0.20, 0.35)
TOLERANCE = 0.02
CATEGORIES = ("diagnosis_correct", "A_correct_to_failure", "B_failure_to_correct", "C_failure_type_error")
ERRORS = CATEGORIES[1:]


def error_category(target_label: str, final_label: str) -> str:
    if target_label == final_label:
        return "diagnosis_correct"
    if target_label == "correct":
        return "A_correct_to_failure"
    if final_label == "correct":
        return "B_failure_to_correct"
    return "C_failure_type_error"


def distribution(values: Iterable[float]) -> dict:
    values = sorted(values)
    if not values:
        return {"n": 0}
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p25": quartiles[0],
        "p75": quartiles[2],
        "min": values[0],
        "max": values[-1],
    }


def stage2_max(row: dict) -> float | None:
    scores = row["stage2_class_scores"]
    return max(scores.values()) if scores else None


def stage1_analysis(rows: list[dict]) -> dict:
    report = {}
    for subset in ("all", "nq", "fever"):
        sub = [r for r in rows if subset == "all" or r["dataset"] == subset]
        failure = [r["target_label"] != "correct" for r in sub]
        p = [r["stage1_failure_probability"] for r in sub]
        report[subset] = {
            "roc_auc": roc_auc_score(failure, p) if 0 < sum(failure) < len(sub) else None,
            "true_correct": distribution(x for x, f in zip(p, failure) if not f),
            "true_failure": distribution(x for x, f in zip(p, failure) if f),
        }
        c, f = report[subset]["true_correct"], report[subset]["true_failure"]
        report[subset]["median_gap"] = f["median"] - c["median"]
    return report


def error_analysis(rows: list[dict]) -> dict:
    report = {}
    for category in CATEGORIES:
        sub = [r for r in rows if r["category"] == category]
        report[category] = {
            "count": len(sub),
            "by_dataset": dict(Counter(r["dataset"] for r in sub)),
            "stage1_failure_probability": distribution(r["stage1_failure_probability"] for r in sub),
            "stage2_max": distribution(m for r in sub if (m := stage2_max(r)) is not None),
        }
    entered = [r for r in rows if r["stage2_class_scores"]]
    true_failures = [r for r in entered if r["target_label"] != "correct"]
    report["stage2_max_entered_stage2"] = {
        "type_correct": distribution(stage2_max(r) for r in true_failures if r["category"] == "diagnosis_correct"),
        "type_wrong": distribution(stage2_max(r) for r in true_failures if r["category"] != "diagnosis_correct"),
        "true_correct_entered_stage2": distribution(stage2_max(r) for r in entered if r["target_label"] == "correct"),
    }
    return report


def candidate_policies() -> list[TriggerPolicy]:
    policies = []
    for low in LOW_CANDIDATES:
        high = round(low + 0.10, 2)
        while high <= HIGH_MAX + 1e-9:
            policies.extend(TriggerPolicy(low, high, s) for s in STAGE2_CANDIDATES)
            high = round(high + 0.05, 2)
    return policies


def evaluate_policy(policy: TriggerPolicy, rows: list[dict]) -> dict:
    triggered = [r for r in rows if policy.should_probe(r["stage1_failure_probability"], r["stage2_class_scores"])]
    totals = Counter(r["category"] for r in rows)
    hits = Counter(r["category"] for r in triggered)
    n_errors = sum(totals[c] for c in ERRORS)
    captured = sum(hits[c] for c in ERRORS)
    a_c_total = totals["A_correct_to_failure"] + totals["C_failure_type_error"]
    a_c = hits["A_correct_to_failure"] + hits["C_failure_type_error"]
    by_dataset = {}
    for dataset in ("nq", "fever"):
        n = sum(r["dataset"] == dataset for r in rows)
        t = sum(r["dataset"] == dataset for r in triggered)
        by_dataset[dataset] = {"triggered": t, "n": n, "coverage": t / n if n else None}
    return {
        "low": policy.low,
        "high": policy.high,
        "stage2_threshold": policy.stage2_threshold,
        "triggered": len(triggered),
        "coverage": len(triggered) / len(rows),
        "band_triggered": sum(policy.low <= r["stage1_failure_probability"] <= policy.high for r in triggered),
        "errors_captured": captured,
        "error_capture_rate": captured / n_errors if n_errors else None,
        "capture_by_category": {
            c: {"captured": hits[c], "total": totals[c], "rate": hits[c] / totals[c] if totals[c] else None}
            for c in ERRORS
        },
        "a_plus_c_captured": a_c,
        "a_plus_c_capture_rate": a_c / a_c_total if a_c_total else None,
        "self_checks": hits["diagnosis_correct"],
        "self_check_rate": hits["diagnosis_correct"] / len(triggered) if triggered else None,
        "by_dataset": by_dataset,
    }


def select_policy(results: list[dict]) -> dict:
    feasible = [r for r in results if COVERAGE_RANGE[0] <= r["coverage"] <= COVERAGE_RANGE[1]]
    if not feasible:
        raise RuntimeError("No trigger configuration has coverage within the target range")
    best = max(r["error_capture_rate"] for r in feasible)
    near = [r for r in feasible if best - r["error_capture_rate"] <= TOLERANCE]
    best_ac = max(r["a_plus_c_capture_rate"] for r in near)
    near = [r for r in near if best_ac - r["a_plus_c_capture_rate"] <= TOLERANCE]
    return min(near, key=lambda r: (r["self_checks"], r["coverage"], r["high"] - r["low"], r["stage2_threshold"]))


def load_predictions(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    for row in rows:
        row["category"] = error_category(row["target_label"], row["final_label"])
    return rows


def main() -> int:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["splits"]
    rows = load_predictions(DIAGNOSIS_RESULTS)
    validation_ids = {(d, i) for d, by in split.items() for i in by["validation"]}
    if {(r["dataset"], r["sample_id"]) for r in rows} != validation_ids or len(rows) != len(validation_ids):
        raise RuntimeError("Stored predictions do not match the v2 validation split")

    v1_rows = load_predictions(V1_DIAGNOSIS_RESULTS)
    step1 = {
        "v2_stage1": stage1_analysis(rows),
        "v1_stage1": stage1_analysis(v1_rows),
        "v2_errors": error_analysis(rows),
    }

    results = [evaluate_policy(policy, rows) for policy in candidate_policies()]
    chosen = select_policy(results)
    v1_reference = evaluate_policy(TriggerPolicy(0.40, 0.55, 0.50), rows)
    feasible = [r for r in results if COVERAGE_RANGE[0] <= r["coverage"] <= COVERAGE_RANGE[1]]

    location = {(d, i): s for d, by in split.items() for s, ids in by.items() for i in ids}
    dev = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]
    dev_locations = [{"dataset": s["dataset"], "sample_id": s["sample_id"],
                      "v2_split": location.get((s["dataset"], s["sample_id"]), "absent")} for s in dev]

    write_json(OUTPUT_PATH, {
        "note": (
            "Offline analysis of stored v2-expanded validation predictions; no model was run and no test record "
            "was read. Target labels are used only to score trigger candidates."
        ),
        "source": str(DIAGNOSIS_RESULTS.relative_to(ROOT).as_posix()),
        "step1_distributions": step1,
        "step2_search": {
            "grid": {"low": LOW_CANDIDATES, "high": f"low+0.10..{HIGH_MAX} step 0.05", "stage2": STAGE2_CANDIDATES},
            "selection_rule": (
                f"coverage in {COVERAGE_RANGE}; error capture rate within {TOLERANCE} of best feasible; "
                f"A+C capture rate within {TOLERANCE} of best among those; then fewest self-checks, lower "
                "coverage, narrower band, lower stage2 threshold"
            ),
            "n_candidates": len(results),
            "n_feasible": len(feasible),
            "chosen": chosen,
            "v1_thresholds_on_v2": v1_reference,
            "top_feasible_by_capture": sorted(
                feasible, key=lambda r: (-r["error_capture_rate"], r["self_checks"]))[:20],
            "all_results": results,
        },
        "step3_prompt_development_locations": dev_locations,
    })

    def fmt(d: dict) -> str:
        return " ".join(f"{k}={d[k]:.3f}" if isinstance(d[k], float) else f"{k}={d[k]}" for k in d)

    for version in ("v1_stage1", "v2_stage1"):
        for subset, m in step1[version].items():
            print(f"[{version} {subset}] auc={m['roc_auc']:.3f} gap={m['median_gap']:.3f}")
            print(f"    correct: {fmt(m['true_correct'])}")
            print(f"    failure: {fmt(m['true_failure'])}")
    for category, m in step1["v2_errors"].items():
        if category.startswith("stage2"):
            for k, v in m.items():
                print(f"[stage2 max] {k}: {fmt(v)}")
        else:
            print(f"[{category}] count={m['count']} {m['by_dataset']}")
            print(f"    p_failure: {fmt(m['stage1_failure_probability'])}")
            if m["stage2_max"]["n"]:
                print(f"    stage2_max: {fmt(m['stage2_max'])}")

    def show(name: str, r: dict) -> None:
        cap = " ".join(f"{c[:1]}={v['captured']}/{v['total']}" for c, v in r["capture_by_category"].items())
        print(f"{name}: low={r['low']} high={r['high']} s2={r['stage2_threshold']} n={r['triggered']} "
              f"cov={r['coverage']:.3f} capture={r['errors_captured']} ({r['error_capture_rate']:.3f}) {cap} "
              f"A+C={r['a_plus_c_capture_rate']:.3f} self={r['self_checks']} ({r['self_check_rate']:.3f}) "
              f"nq_cov={r['by_dataset']['nq']['coverage']:.3f} fever_cov={r['by_dataset']['fever']['coverage']:.3f}")

    print(f"\n{len(results)} candidates, {len(feasible)} feasible")
    for r in sorted(feasible, key=lambda r: (-r["error_capture_rate"], r["self_checks"]))[:12]:
        show("  top", r)
    show("v1 thresholds", v1_reference)
    show("CHOSEN", chosen)
    print("prompt-development locations:", Counter(d["v2_split"] for d in dev_locations), dev_locations)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
