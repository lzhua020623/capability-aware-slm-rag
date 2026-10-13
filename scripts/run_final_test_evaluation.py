"""Final frozen TEST evaluation: Diagnosis v2-expanded -> v2 Probe trigger -> Evidence Probe -> Safe Binary -> Controller v1.

Three separate stages (run in order):
  preflight  Split / schema checks on the frozen v2-expanded split, frozen
             configuration checks, and a full runtime-pipeline run on
             non-test (validation) fixtures, including one real Qwen Probe
             call on a validation sample compared with its stored output.
             No test inference.
  infer      Runtime on the TEST split. Only each record's DiagnosisInput
             (`input`) and its needs_manual_review flag are kept; target
             labels and metadata are discarded while reading. Writes the
             Diagnosis, Probe and Controller test result files (no labels).
  evaluate   Offline only, after all predictions are saved: reads test target
             labels and computes metrics. PRIMARY excludes the Probe
             prompt-development sample(s) that fall in the test split.

Nothing is trained, tuned or selected here; all thresholds, prompts and
hyperparameters are the frozen validation-stage values.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_diagnosis_v2_expanded_split import DATASET_PATHS, OUTPUT_PATH as SPLIT_PATH
from scripts.evaluate_probe_safe_override_v2 import OverrideRule, safe_label
from scripts.run_controller_validation import OUTPUT_PATH as CONTROLLER_VALIDATION
from scripts.run_probe_smoke import SAMPLES_PATH
from scripts.run_probe_v2_expanded_validation import OUTPUT_PATH as PROBE_VALIDATION
from scripts.run_probe_validation import error_counts
from scripts.train_diagnosis_classifier import evaluate_by_dataset, input_from_dict, write_json
from scripts.train_diagnosis_v2_expanded import MODEL_PATH, OUTPUT_PATH as DIAGNOSIS_VALIDATION
from src.controller.controller import (
    ACTION_BY_LABEL,
    SAFE_OVERRIDE_MAX_STAGE1,
    SAFE_OVERRIDE_MAX_STAGE2,
    decide_from_probed,
    run_controller,
)
from src.controller.schema import ControllerAction
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosisClassifier
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.probe.pipeline import apply_probe
from src.probe.probe import EvidenceProbe
from src.probe.prompts import build_probe_prompt
from src.probe.schema import ProbeLabel
from src.probe.trigger import V2_EXPANDED_TRIGGER

DIAGNOSIS_TEST = ROOT / "results" / "diagnosis" / "diagnosis_v2_expanded_test.json"
PROBE_TEST = ROOT / "results" / "probe" / "probe_v2_expanded_test.json"
CONTROLLER_TEST = ROOT / "results" / "controller" / "controller_v1_test.json"
EVALUATION_PATH = ROOT / "results" / "final" / "diagnosis_probe_controller_test_evaluation.json"

DATASETS = ("nq", "fever")
EXPECTED_TEST = {"nq": 403, "fever": 71}
EXPECTED_TEST_PROMPT_DEVELOPMENT = {("nq", "3724356943291190911")}
INPUT_KEYS = {"sample_id", "dataset", "query", "evidence", "initial_answer"}
FROZEN = {
    "diagnosis_model": str(MODEL_PATH.relative_to(ROOT).as_posix()),
    "stage1": {"c": 10.0, "class_weight": "balanced", "threshold": 0.5},
    "stage2": {"c": 0.01, "class_weight": "balanced"},
    "trigger": {"low": 0.50, "high": 0.65, "stage2_threshold": 0.50},
    "probe": "Qwen/Qwen2.5-7B-Instruct, 4-bit NF4, do_sample=False, one call per triggered sample, src/probe prompts",
    "safe_binary": {"max_stage1_failure_probability": 0.70, "max_stage2_score": 0.50,
                    "override": "Diagnosis failure -> correct only if Probe says correct"},
    "controller": {"correct": "no_action", "evidence_insufficiency": "retrieval_enhancement",
                   "evidence_accessibility": "evidence_refinement", "evidence_utilisation": "generation_repair"},
}


# ---------------------------------------------------------------- data access

def read_split() -> dict:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["splits"]


def load_runtime_inputs(split_name: str) -> tuple[list[dict], dict[tuple[str, str], bool]]:
    """Return DiagnosisInput dicts in split order and needs_manual_review flags; labels are dropped on read."""
    split = read_split()
    inputs, review = [], {}
    for dataset, path in DATASET_PATHS.items():
        wanted = split[dataset][split_name]
        keep, by_id = set(wanted), {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = record["input"]["sample_id"]
                if sample_id in keep:
                    by_id[sample_id] = record["input"]
                    review[(dataset, sample_id)] = bool(record["metadata"]["needs_manual_review"])
                del record
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise RuntimeError(f"{dataset} {split_name}: {len(missing)} IDs missing from {path}")
        inputs += [by_id[sample_id] for sample_id in wanted]
    return inputs, review


def prompt_development_ids() -> set[tuple[str, str]]:
    return {(s["dataset"], s["sample_id"]) for s in json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]}


# ------------------------------------------------------------------ preflight

def check_split_and_inputs() -> dict:
    split = read_split()
    counts = {d: len(split[d]["test"]) for d in DATASETS}
    overlaps = {}
    for d in DATASETS:
        parts = {name: set(ids) for name, ids in split[d].items()}
        overlaps[d] = {f"{a}&{b}": len(parts[a] & parts[b])
                       for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
        overlaps[d]["duplicates_within_test"] = len(split[d]["test"]) - len(parts["test"])
    test_keys = {(d, i) for d in DATASETS for i in split[d]["test"]}
    dev_in_test = prompt_development_ids() & test_keys
    inputs, review = load_runtime_inputs("test")
    bad_keys = [i["sample_id"] for i in inputs if set(i) != INPUT_KEYS]
    items = [input_from_dict(i) for i in inputs]
    checks = {
        "test_counts": counts,
        "test_total": len(items),
        "overlaps": overlaps,
        "prompt_development_in_test": sorted(dev_in_test),
        "primary_n": len(items) - len(dev_in_test),
        "needs_manual_review_in_test": sum(review.values()),
        "inputs_with_unexpected_keys": len(bad_keys),
        "diagnosis_input_schema_ok": len(items),
    }
    ok = (counts == EXPECTED_TEST
          and all(v == 0 for o in overlaps.values() for v in o.values())
          and dev_in_test == EXPECTED_TEST_PROMPT_DEVELOPMENT
          and not any(review.values()) and not bad_keys and len(items) == sum(EXPECTED_TEST.values()))
    if not ok:
        raise RuntimeError(f"Split / input checks failed; not running any model: {checks}")
    return checks


def check_frozen_configuration(classifier: HierarchicalDiagnosisClassifier) -> dict:
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    observed = {
        "stage1": {"c": artifact["stage1"]["c"], "class_weight": artifact["stage1"]["class_weight"],
                   "threshold": artifact["stage1_threshold"]},
        "stage2": {"c": artifact["stage2"]["c"], "class_weight": artifact["stage2"]["class_weight"]},
        "trigger": {"low": V2_EXPANDED_TRIGGER.low, "high": V2_EXPANDED_TRIGGER.high,
                    "stage2_threshold": V2_EXPANDED_TRIGGER.stage2_threshold},
        "safe_binary": (SAFE_OVERRIDE_MAX_STAGE1, SAFE_OVERRIDE_MAX_STAGE2),
        "controller": {k.value: v.value for k, v in ACTION_BY_LABEL.items()},
        "classifier_stage1_threshold": classifier.stage1_threshold,
    }
    expected_safe = (FROZEN["safe_binary"]["max_stage1_failure_probability"], FROZEN["safe_binary"]["max_stage2_score"])
    ok = (observed["stage1"] == FROZEN["stage1"] and observed["stage2"] == FROZEN["stage2"]
          and observed["trigger"] == FROZEN["trigger"] and observed["safe_binary"] == expected_safe
          and observed["controller"] == FROZEN["controller"]
          and observed["classifier_stage1_threshold"] == FROZEN["stage1"]["threshold"])
    if not ok:
        raise RuntimeError(f"Frozen configuration mismatch: {observed}")
    return observed


def check_validation_fixture(classifier: HierarchicalDiagnosisClassifier) -> dict:
    """Full runtime pipeline on validation (non-test) inputs, replaying stored Probe outputs."""
    inputs, _ = load_runtime_inputs("validation")
    items = [input_from_dict(i) for i in inputs]
    diagnoses = classifier.diagnose_batch(items)
    stored_diag = {(p["dataset"], p["sample_id"]): p
                   for p in json.loads(DIAGNOSIS_VALIDATION.read_text(encoding="utf-8"))["predictions"]}
    diag_mismatch = sum(
        stored_diag[(i.dataset.value, i.sample_id)]["final_label"] != d.final_label.value
        or not math.isclose(stored_diag[(i.dataset.value, i.sample_id)]["stage1_failure_probability"],
                            d.stage1_failure_probability, abs_tol=1e-9)
        for i, d in zip(items, diagnoses))

    stored_probe = {(r["dataset"], r["sample_id"]): r
                    for r in json.loads(PROBE_VALIDATION.read_text(encoding="utf-8"))["samples"]}
    replay = {build_probe_prompt(i): stored_probe[(i.dataset.value, i.sample_id)]["probe_result"]["raw_output"]
              for i in items if stored_probe[(i.dataset.value, i.sample_id)]["probe_triggered"]}
    calls = Counter()

    def replay_generate(prompt: str) -> str:
        calls["n"] += 1
        return replay[prompt]

    probe = EvidenceProbe(replay_generate)
    stored_decisions = {(d["dataset"], d["sample_id"]): d
                        for d in json.loads(CONTROLLER_VALIDATION.read_text(encoding="utf-8"))["decisions"]}
    decision_mismatch, run_controller_mismatch = 0, 0
    for idx, (item, diagnosis) in enumerate(zip(items, diagnoses)):
        decision = decide_from_probed(apply_probe(item, diagnosis, probe, V2_EXPANDED_TRIGGER), item.sample_id)
        stored = stored_decisions[(item.dataset.value, item.sample_id)]
        decision_mismatch += (decision.action.value, decision.diagnosis_label.value) != (
            stored["action"], stored["diagnosis_label"])
        if idx < 20:
            single = run_controller(item, classifier, probe, V2_EXPANDED_TRIGGER)
            run_controller_mismatch += (
                (single.action, single.diagnosis_label, single.used_probe, single.probe_overrode_diagnosis)
                != (decision.action, decision.diagnosis_label, decision.used_probe, decision.probe_overrode_diagnosis)
                or not math.isclose(single.diagnosis_confidence, decision.diagnosis_confidence, abs_tol=1e-9))
    checks = {
        "validation_n": len(items),
        "diagnosis_mismatches_vs_stored": diag_mismatch,
        "replayed_probe_calls": calls["n"],
        "stored_triggered": len(replay),
        "controller_mismatches_vs_stored": decision_mismatch,
        "run_controller_vs_batch_path_mismatches_first20": run_controller_mismatch,
    }
    if diag_mismatch or decision_mismatch or run_controller_mismatch or calls["n"] < len(replay):
        raise RuntimeError(f"Validation fixture checks failed: {checks}")
    return checks


def check_no_enhancement_modules() -> dict:
    loaded = sorted(m for m in sys.modules if m.startswith(("src.compensation", "src.retrieval")))
    if loaded:
        raise RuntimeError(f"Enhancement / retrieval modules loaded: {loaded}")
    return {"compensation_or_retrieval_modules_loaded": loaded}


def check_real_probe_on_validation(probe: EvidenceProbe) -> dict:
    """One real Qwen call on a stored, triggered validation sample; output must match exactly."""
    stored = next(r for r in json.loads(PROBE_VALIDATION.read_text(encoding="utf-8"))["samples"]
                  if r["probe_triggered"] and r["dataset"] == "nq")
    inputs, _ = load_runtime_inputs("validation")
    item = input_from_dict(next(i for i in inputs if i["sample_id"] == stored["sample_id"]
                                and i["dataset"] == stored["dataset"]))
    start = time.perf_counter()
    result = probe.run(item)
    latency = time.perf_counter() - start
    same = result.raw_output == stored["probe_result"]["raw_output"]
    checks = {"validation_sample": [stored["dataset"], stored["sample_id"]], "raw_output_identical": same,
              "probe_label": result.probe_label.value, "latency_seconds": latency}
    if not same:
        raise RuntimeError(f"Real Probe output differs from stored validation output: {checks}")
    return checks


def preflight(load_qwen: bool) -> dict:
    report = {"frozen_configuration": FROZEN}
    print("FROZEN CONFIGURATION:\n" + json.dumps(FROZEN, indent=1), flush=True)
    report["split_and_inputs"] = check_split_and_inputs()
    print("split/input checks:", report["split_and_inputs"], flush=True)
    classifier = HierarchicalDiagnosisClassifier.load(MODEL_PATH)
    report["configuration_checks"] = check_frozen_configuration(classifier)
    print("configuration checks:", report["configuration_checks"], flush=True)
    report["validation_fixture"] = check_validation_fixture(classifier)
    print("validation fixture:", report["validation_fixture"], flush=True)
    report["modules"] = check_no_enhancement_modules()
    if load_qwen:
        probe = EvidenceProbe.from_qwen()
        report["real_probe_validation_call"] = check_real_probe_on_validation(probe)
        print("real probe check:", report["real_probe_validation_call"], flush=True)
        report["modules_after_qwen"] = check_no_enhancement_modules()
    print("PREFLIGHT PASSED", flush=True)
    return report


# ---------------------------------------------------------------------- infer

def infer() -> int:
    for path in (DIAGNOSIS_TEST, PROBE_TEST, CONTROLLER_TEST):
        if path.exists():
            raise FileExistsError(f"{path.relative_to(ROOT)} already exists; refusing to overwrite")
    checks = preflight(load_qwen=False)
    inputs, _ = load_runtime_inputs("test")
    items = [input_from_dict(i) for i in inputs]
    classifier = HierarchicalDiagnosisClassifier.load(MODEL_PATH)
    diagnoses = classifier.diagnose_batch(items)

    qwen = EvidenceProbe.from_qwen()
    calls = Counter()

    def counted_generate(prompt: str) -> str:
        calls["total"] += 1
        return qwen._generate(prompt)

    probe = EvidenceProbe(counted_generate)
    rows, decisions = [], []
    for item, diagnosis in zip(items, diagnoses):
        before = calls["total"]
        start = time.perf_counter()
        probed = apply_probe(item, diagnosis, probe, V2_EXPANDED_TRIGGER)
        latency = time.perf_counter() - start
        decision = decide_from_probed(probed, sample_id=item.sample_id)
        n_calls = calls["total"] - before
        if n_calls != int(probed.probe_triggered):
            raise RuntimeError(f"{item.sample_id}: {n_calls} Probe calls for triggered={probed.probe_triggered}")
        rows.append({
            "sample_id": item.sample_id,
            "dataset": item.dataset.value,
            "diagnosis_label_before_probe": diagnosis.final_label.value,
            "stage1_failure_probability": diagnosis.stage1_failure_probability,
            "stage2_class_scores": None if diagnosis.stage2_class_scores is None
            else {k.value: v for k, v in diagnosis.stage2_class_scores.items()},
            "probe_triggered": probed.probe_triggered,
            "probe_calls": n_calls,
            "probe_result": None if probed.probe_result is None else probed.probe_result.to_dict(),
            "probe_latency_seconds": latency if probed.probe_triggered else None,
            "final_diagnosis_label": decision.diagnosis_label.value,
            "probe_overrode_diagnosis": decision.probe_overrode_diagnosis,
            "action": decision.action.value,
        })
        decisions.append(decision.to_dict() | {"dataset": item.dataset.value})
        if probed.probe_triggered:
            print(f"[{calls['total']:3d}] {item.dataset.value:5s} {item.sample_id:>22s} "
                  f"{diagnosis.final_label.value:24s} probe={probed.probe_result.probe_label.value:24s} "
                  f"-> {decision.diagnosis_label.value:24s} ({latency:.1f}s)", flush=True)

    note = "Frozen final TEST runtime outputs; no target labels or gold fields are stored or were read."
    write_json(DIAGNOSIS_TEST, {
        "note": note, "model": FROZEN["diagnosis_model"], "n": len(items),
        "predictions": [{"sample_id": i.sample_id, "dataset": i.dataset.value,
                         "stage1_failure_probability": d.stage1_failure_probability, "stage1_label": d.stage1_label,
                         "stage2_class_scores": None if d.stage2_class_scores is None
                         else {k.value: v for k, v in d.stage2_class_scores.items()},
                         "final_label": d.final_label.value, "confidence": d.confidence}
                        for i, d in zip(items, diagnoses)],
    })
    write_json(PROBE_TEST, {
        "note": note, "model": FROZEN["probe"], "trigger": FROZEN["trigger"],
        "preflight": checks, "probe_calls_total": calls["total"], "samples": rows,
    })
    write_json(CONTROLLER_TEST, {
        "note": note + " Routing only; no compensation is executed.",
        "safe_binary": FROZEN["safe_binary"], "mapping": FROZEN["controller"], "decisions": decisions,
    })
    print(f"probe calls: {calls['total']}; wrote test runtime files", flush=True)
    return 0


# ------------------------------------------------------------------- evaluate

def rate(k: int, n: int) -> float | None:
    return k / n if n else None


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": p95, "total": sum(values)}


def label_metrics(rows: list[dict], labels: dict, key: str) -> dict:
    records = [{"input": {"dataset": r["dataset"]}, "target_label": labels[(r["dataset"], r["sample_id"])]}
               for r in rows]
    predicted = [parse_label(r[key]) for r in rows]
    out = {}
    for subset, m in evaluate_by_dataset(records, predicted).items():
        out[subset] = m | {"correct_recall": m["per_class"]["correct"]["recall"], **error_counts(m)}
    return out


def deltas(before: dict, after: dict) -> dict:
    return {subset: {
        "fp_reduction": before[subset]["correct_to_failure_false_positives"]
        - after[subset]["correct_to_failure_false_positives"],
        "fn_change": after[subset]["failure_to_correct_false_negatives"]
        - before[subset]["failure_to_correct_false_negatives"],
        "failure_type_error_change": after[subset]["failure_type_errors"] - before[subset]["failure_type_errors"],
        "accuracy_change": after[subset]["accuracy"] - before[subset]["accuracy"],
        "macro_f1_change": after[subset]["macro_f1"] - before[subset]["macro_f1"],
        "failure_detection_recall_change": (after[subset]["failure_detection_recall"] or 0)
        - (before[subset]["failure_detection_recall"] or 0),
    } for subset in before}


def probe_report(rows: list[dict], labels: dict) -> dict:
    triggered = [r for r in rows if r["probe_triggered"]]
    overrides = [r for r in rows if r["probe_overrode_diagnosis"]]
    true = lambda r: parse_label(labels[(r["dataset"], r["sample_id"])])
    override_true_correct = sum(true(r) is DiagnosisLabel.CORRECT for r in overrides)
    outcomes = Counter()
    for r in triggered:
        target, before, after = true(r), parse_label(r["diagnosis_label_before_probe"]), parse_label(
            r["final_diagnosis_label"])
        if before is target:
            outcomes["kept_correct" if after is target else "worsened"] += 1
        else:
            outcomes["fixed" if after is target else "still_wrong"] += 1
    by_dataset = {d: {"n": sum(r["dataset"] == d for r in rows),
                      "triggered": sum(r["probe_triggered"] for r in rows if r["dataset"] == d)} for d in DATASETS}
    for v in by_dataset.values():
        v["coverage"] = rate(v["triggered"], v["n"])
    return {
        "n": len(rows),
        "triggered": len(triggered),
        "coverage": rate(len(triggered), len(rows)),
        "by_dataset": by_dataset,
        "probe_label_distribution": {label.value: sum(r["probe_result"]["probe_label"] == label.value
                                                      for r in triggered) for label in ProbeLabel},
        "parse_errors": sum(bool(r["probe_result"]["parse_error"]) for r in triggered),
        "safe_binary": {
            "override_to_correct": len(overrides),
            "triggered_no_override": len(triggered) - len(overrides),
            "not_triggered": len(rows) - len(triggered),
            "override_by_dataset": {d: sum(r["dataset"] == d for r in overrides) for d in DATASETS},
        },
        "override_offline": {
            "true_correct": override_true_correct,
            "true_failure": len(overrides) - override_true_correct,
            "precision": rate(override_true_correct, len(overrides)),
        },
        "triggered_outcomes": {k: outcomes[k] for k in ("fixed", "kept_correct", "worsened", "still_wrong")},
    }


def routing_report(rows: list[dict]) -> dict:
    out = {}
    for subset in ("all",) + DATASETS:
        sub = [r for r in rows if subset == "all" or r["dataset"] == subset]
        counts = {a.value: sum(r["action"] == a.value for r in sub) for a in ControllerAction}
        out[subset] = {"n": len(sub), "counts": counts,
                       "percent": {a: rate(c, len(sub)) for a, c in counts.items()}}
    return out


def cost_report(rows: list[dict], total_calls: int) -> dict:
    triggered = [r for r in rows if r["probe_triggered"]]
    return {
        "test_samples": len(rows),
        "qwen_calls": total_calls,
        "sum_per_sample_calls": sum(r["probe_calls"] for r in rows),
        "each_triggered_exactly_once": all(r["probe_calls"] == 1 for r in triggered)
        and all(r["probe_calls"] == 0 for r in rows if not r["probe_triggered"]),
        "latency_seconds": latency_stats([r["probe_latency_seconds"] for r in triggered]),
        "calls_per_test_sample": rate(total_calls, len(rows)),
    }


def evaluate() -> int:
    for path in (DIAGNOSIS_TEST, PROBE_TEST, CONTROLLER_TEST):
        if not path.exists():
            raise FileNotFoundError(f"{path.relative_to(ROOT)} missing; run infer first")
    if EVALUATION_PATH.exists():
        raise FileExistsError(f"{EVALUATION_PATH.relative_to(ROOT)} already exists; refusing to overwrite")
    probe_data = json.loads(PROBE_TEST.read_text(encoding="utf-8"))
    rows = probe_data["samples"]
    decisions = json.loads(CONTROLLER_TEST.read_text(encoding="utf-8"))["decisions"]
    diag = {(p["dataset"], p["sample_id"]): p
            for p in json.loads(DIAGNOSIS_TEST.read_text(encoding="utf-8"))["predictions"]}

    rule = OverrideRule("S3", t1=SAFE_OVERRIDE_MAX_STAGE1, t2=SAFE_OVERRIDE_MAX_STAGE2)
    consistency = {
        "rows": len(rows),
        "decisions_match_rows": all((d["dataset"], d["sample_id"], d["action"], d["diagnosis_label"]) ==
                                    (r["dataset"], r["sample_id"], r["action"], r["final_diagnosis_label"])
                                    for d, r in zip(decisions, rows)),
        "diagnosis_file_matches_rows": all(diag[(r["dataset"], r["sample_id"])]["final_label"]
                                           == r["diagnosis_label_before_probe"] for r in rows),
        "safe_binary_s3_mismatches": sum(safe_label(rule, r).value != r["final_diagnosis_label"] for r in rows),
        "action_mapping_mismatches": sum(ACTION_BY_LABEL[parse_label(r["final_diagnosis_label"])].value
                                         != r["action"] for r in rows),
    }
    if not (consistency["decisions_match_rows"] and consistency["diagnosis_file_matches_rows"]
            and consistency["safe_binary_s3_mismatches"] == 0 and consistency["action_mapping_mismatches"] == 0):
        raise RuntimeError(f"Stored runtime outputs are inconsistent: {consistency}")

    # Offline: test target labels are read only now, after all runtime outputs were saved.
    split = read_split()
    labels = {}
    for dataset, path in DATASET_PATHS.items():
        wanted = set(split[dataset]["test"])
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record["input"]["sample_id"] in wanted:
                        labels[(dataset, record["input"]["sample_id"])] = record["target_label"]

    dev = prompt_development_ids()
    scopes = {"primary_unseen_test": [r for r in rows if (r["dataset"], r["sample_id"]) not in dev],
              "full_test": rows}
    report = {
        "note": ("Final frozen TEST evaluation. Runtime outputs were produced and saved before any test target "
                 "label was read. PRIMARY excludes the Probe prompt-development sample(s) in the test split."),
        "frozen_configuration": FROZEN,
        "excluded_from_primary": sorted(k for k in dev if k in labels),
        "runtime_consistency": consistency,
    }
    for scope, sub in scopes.items():
        diagnosis_only = label_metrics(sub, labels, "diagnosis_label_before_probe")
        safe_binary = label_metrics(sub, labels, "final_diagnosis_label")
        report[scope] = {
            "diagnosis_only": diagnosis_only,
            "diagnosis_probe_safe_binary": safe_binary,
            "change_vs_diagnosis_only": deltas(diagnosis_only, safe_binary),
            "probe": probe_report(sub, labels),
            "routing": routing_report(sub),
            "cost": cost_report(sub, sum(r["probe_calls"] for r in sub)),
            "true_label_distribution": {
                s: dict(Counter(labels[(r["dataset"], r["sample_id"])] for r in sub
                                if s == "all" or r["dataset"] == s)) for s in ("all",) + DATASETS},
        }
    report["cost_full_run"] = cost_report(rows, probe_data["probe_calls_total"])
    write_json(EVALUATION_PATH, report)

    for scope in scopes:
        r = report[scope]
        for name in ("diagnosis_only", "diagnosis_probe_safe_binary"):
            for subset, m in r[name].items():
                print(f"[{scope} {name} {subset}] n={m['n']} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
                      f"FDR={m['failure_detection_recall']:.3f} correctR={m['correct_recall']:.3f} "
                      f"FP={m['correct_to_failure_false_positives']} FN={m['failure_to_correct_false_negatives']} "
                      f"typeErr={m['failure_type_errors']}")
    print(f"Wrote {EVALUATION_PATH.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "infer", "evaluate"))
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight(load_qwen=True)
        return 0
    return infer() if args.stage == "infer" else evaluate()


if __name__ == "__main__":
    sys.exit(main())
