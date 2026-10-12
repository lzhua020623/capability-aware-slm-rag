"""Run the frozen Evidence Probe on Diagnosis v2-expanded VALIDATION samples selected by V2_EXPANDED_TRIGGER.

Only validation rows are kept from the development files (rows of other
splits are skipped by sample ID and never used); no test inference is run.
Before any model call it checks that:
- validation has EXPECTED_VALIDATION samples;
- the live v2 Diagnosis reproduces the stored v2 validation predictions;
- V2_EXPANDED_TRIGGER equals the configuration chosen by the trigger search
  and the live trigger set equals the set derived from stored predictions.
The Probe prompt is unchanged and receives only the DiagnosisInput. Target
labels appear only in offline_evaluation fields. Policy metrics are computed
by scripts/evaluate_probe_v2_expanded_policies.py.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_diagnosis_v2_expanded_split import DATASET_PATHS, OUTPUT_PATH as SPLIT_PATH
from scripts.run_probe_smoke import SAMPLES_PATH, offline_category, outcome
from scripts.search_probe_trigger_v2_expanded import OUTPUT_PATH as SEARCH_PATH
from scripts.train_diagnosis_classifier import input_from_dict, write_json
from scripts.train_diagnosis_v2_expanded import MODEL_PATH, OUTPUT_PATH as DIAGNOSIS_RESULTS
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosisClassifier
from src.diagnosis.labels import parse_label
from src.probe.pipeline import apply_probe
from src.probe.probe import EvidenceProbe
from src.probe.trigger import V2_EXPANDED_TRIGGER

OUTPUT_PATH = ROOT / "results" / "probe" / "probe_v2_expanded_validation.json"
EXPECTED_VALIDATION = 477
EXPECTED_TRIGGERED = 165


def load_validation_records() -> list[dict]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["splits"]
    records = []
    for dataset, path in DATASET_PATHS.items():
        wanted = split[dataset]["validation"]
        keep = set(wanted)
        by_id = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record["input"]["sample_id"] in keep:
                        by_id[record["input"]["sample_id"]] = record
        records += [by_id[sample_id] for sample_id in wanted]
    return records


def pre_run_checks(records: list[dict], diagnoses: list, inputs: list) -> dict:
    stored = {(r["dataset"], r["sample_id"]): r
              for r in json.loads(DIAGNOSIS_RESULTS.read_text(encoding="utf-8"))["predictions"]}
    mismatched = [
        item.sample_id for item, d in zip(inputs, diagnoses)
        if (row := stored.get((item.dataset.value, item.sample_id))) is None
        or row["final_label"] != d.final_label.value
        or not math.isclose(row["stage1_failure_probability"], d.stage1_failure_probability, abs_tol=1e-9)
    ]
    chosen = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))["step2_search"]["chosen"]
    policy_matches = (chosen["low"], chosen["high"], chosen["stage2_threshold"]) == (
        V2_EXPANDED_TRIGGER.low, V2_EXPANDED_TRIGGER.high, V2_EXPANDED_TRIGGER.stage2_threshold)
    live = {(i.dataset.value, i.sample_id) for i, d in zip(inputs, diagnoses)
            if V2_EXPANDED_TRIGGER.should_probe(d.stage1_failure_probability, d.stage2_class_scores)}
    from_stored = {key for key, row in stored.items()
                   if V2_EXPANDED_TRIGGER.should_probe(row["stage1_failure_probability"], row["stage2_class_scores"])}
    checks = {
        "validation_count": len(records),
        "diagnosis_mismatches": len(mismatched),
        "trigger_matches_search": policy_matches,
        "triggered_count": len(live),
        "live_trigger_equals_stored": live == from_stored,
    }
    ok = (len(records) == EXPECTED_VALIDATION and not mismatched and policy_matches
          and len(live) == EXPECTED_TRIGGERED and live == from_stored)
    if not ok:
        raise RuntimeError(f"Pre-run checks failed; not running the model: {checks}")
    return checks


def main() -> int:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"{OUTPUT_PATH.relative_to(ROOT)} already exists; refusing to overwrite")
    records = load_validation_records()
    classifier = HierarchicalDiagnosisClassifier.load(MODEL_PATH)
    inputs = [input_from_dict(r["input"]) for r in records]
    diagnoses = classifier.diagnose_batch(inputs)
    checks = pre_run_checks(records, diagnoses, inputs)
    print("pre-run checks:", checks)
    dev_ids = {(s["dataset"], s["sample_id"]) for s in json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]}

    probe = EvidenceProbe.from_qwen()
    rows, calls = [], 0
    for item, diagnosis, record in zip(inputs, diagnoses, records):
        start = time.perf_counter()
        result = apply_probe(item, diagnosis, probe, V2_EXPANDED_TRIGGER)
        latency = time.perf_counter() - start
        calls += result.probe_triggered
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
            "probe_latency_seconds": round(latency, 3) if result.probe_triggered else None,
            "final_diagnosis_label": after.value,
            "offline_evaluation": {
                "target_label": target.value,
                "prompt_development_sample": (item.dataset.value, item.sample_id) in dev_ids,
                "category_before_probe": offline_category(target, before),
                "correct_before_probe": before is target,
                "correct_after_probe": after is target,
                "outcome": outcome(target, before, after, result.probe_result.probe_label)
                if result.probe_triggered else None,
            },
        }
        rows.append(row)
        if result.probe_triggered:
            pr = row["probe_result"]
            print(f"[{calls:3d}] {item.sample_id:>22s} {item.dataset.value:5s} {before.value:24s} -> {after.value:24s} "
                  f"{pr['probe_label']:24s} target={target.value:24s} {row['offline_evaluation']['outcome']} "
                  f"({latency:.1f}s)", flush=True)

    latencies = [r["probe_latency_seconds"] for r in rows if r["probe_triggered"]]
    write_json(OUTPUT_PATH, {
        "note": (
            "Frozen v2-expanded Hierarchical Diagnosis + frozen Evidence Probe prompts on validation only; no test "
            "inference. Only offline_evaluation fields contain target labels; they never reach the Probe prompt."
        ),
        "model": "Qwen/Qwen2.5-7B-Instruct (4-bit NF4, do_sample=False), one call per triggered sample",
        "trigger": {"low": V2_EXPANDED_TRIGGER.low, "high": V2_EXPANDED_TRIGGER.high,
                    "stage2_threshold": V2_EXPANDED_TRIGGER.stage2_threshold},
        "pre_run_checks": checks,
        "cost": {
            "validation_samples": len(rows),
            "probe_calls": calls,
            "probe_coverage": calls / len(rows),
            "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
            "total_latency_seconds": sum(latencies),
        },
        "samples": rows,
    })
    print(f"probe calls: {calls}; wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
