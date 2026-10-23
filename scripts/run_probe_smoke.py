"""Evidence Probe v1 smoke test on the fixed prompt-development VALIDATION samples.

The sample IDs come from configs/probe/prompt_development_samples.json, so
reruns compare the same samples before and after prompt changes. Only
validation IDs are read from the frozen split; the test split is never
loaded. Target labels are used offline only to score outcomes; the Probe
itself receives only the DiagnosisInput.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_diagnosis_classifier import DATASET_PATHS, SPLIT_PATH, input_from_dict, write_json
from src.diagnosis.dev_dataset import load_jsonl
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosisClassifier
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.probe.pipeline import apply_probe
from src.probe.probe import EvidenceProbe
from src.probe.schema import ProbeLabel
from src.probe.trigger import should_probe

MODEL_PATH = ROOT / "models" / "diagnosis" / "diagnosis_hierarchical_v1.json"
SAMPLES_PATH = ROOT / "configs" / "probe" / "prompt_development_samples.json"
DEFAULT_OUTPUT = ROOT / "results" / "probe" / "probe_smoke_validation_v2.json"
BEFORE_PATH = ROOT / "results" / "probe" / "probe_smoke_validation.json"


def offline_category(target: DiagnosisLabel, predicted: DiagnosisLabel) -> str:
    if target is predicted:
        return "diagnosis_correct"
    if target is DiagnosisLabel.CORRECT:
        return "correct_to_failure"
    if predicted is DiagnosisLabel.CORRECT:
        return "failure_to_correct"
    return "failure_type_error"


def outcome(target: DiagnosisLabel, original: DiagnosisLabel, refined: DiagnosisLabel, probe_label: ProbeLabel) -> str:
    if probe_label is ProbeLabel.UNCERTAIN:
        return "uncertain"
    if original is target:
        return "kept_correct" if refined is target else "worsened"
    return "fixed" if refined is target else "still_wrong"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.resolve() == BEFORE_PATH.resolve():
        raise ValueError("Refusing to overwrite the pre-fix smoke results")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]
    by_dataset = {dataset: {r["input"]["sample_id"]: r for r in load_jsonl(path)} for dataset, path in DATASET_PATHS.items()}
    records = []
    for sample in samples:
        dataset, sample_id = sample["dataset"], sample["sample_id"]
        if sample_id not in split["splits"][dataset]["validation"]:
            raise ValueError(f"{dataset}:{sample_id} is not a validation sample")
        records.append(by_dataset[dataset][sample_id])

    classifier = HierarchicalDiagnosisClassifier.load(MODEL_PATH)
    inputs = [input_from_dict(r["input"]) for r in records]
    diagnoses = classifier.diagnose_batch(inputs)
    for item, diagnosis in zip(inputs, diagnoses):
        if not should_probe(diagnosis.stage1_failure_probability, diagnosis.stage2_class_scores):
            raise RuntimeError(f"{item.sample_id} is not probe-triggered")

    probe = EvidenceProbe.from_qwen()
    rows = []
    for item, diagnosis, record in zip(inputs, diagnoses, records):
        start = time.perf_counter()
        result = apply_probe(item, diagnosis, probe)
        latency = time.perf_counter() - start
        target = parse_label(record["target_label"])
        original = diagnosis.final_label
        probe_result = result.probe_result
        rows.append({
            "sample_id": item.sample_id,
            "dataset": item.dataset.value,
            "initial_answer": item.initial_answer,
            "original_diagnosis": original.value,
            "probe": probe_result.to_dict(),
            "refined_label": result.final_label.value,
            "latency_seconds": round(latency, 2),
            "offline_only": {
                "target_label": target.value,
                "category": offline_category(target, original),
                "outcome": outcome(target, original, result.final_label, probe_result.probe_label),
            },
        })
        verdict = probe_result.evidence_verdict.value if probe_result.evidence_verdict else "-"
        print(f"{item.sample_id:>22s} {item.dataset.value:5s} init={item.initial_answer[:30]!r} verdict={verdict} "
              f"orig={original.value} probe={probe_result.answer_supported.value}/"
              f"{probe_result.evidence_sufficiency.value}/{probe_result.evidence_accessibility.value} "
              f"-> {probe_result.probe_label.value} target={target.value} "
              f"{rows[-1]['offline_only']['outcome']} ({latency:.1f}s)")
        print(f"    raw: {probe_result.raw_output!r}")

    counts: dict[str, int] = {}
    for row in rows:
        key = row["offline_only"]["outcome"]
        counts[key] = counts.get(key, 0) + 1
    print("outcomes:", counts)
    write_json(args.output, {
        "note": "Smoke rerun on the fixed prompt-development validation samples; offline_only never reaches the prompt.",
        "samples_config": str(SAMPLES_PATH.relative_to(ROOT)),
        "outcome_counts": counts,
        "samples": rows,
    })
    print(f"Wrote {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
