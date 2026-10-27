"""Mac-only offline answer evaluation; never copied into the GPU runtime bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.retrieval_enhancement.generation_replay import (
    load_generation_inputs, read_results, sha256_file, validate_result,
)
from src.evaluation.qa import best_em_f1
from scripts.run_nq_p320_gpu_generation import frozen_metadata

LABELS = ("correct", "evidence_insufficiency", "evidence_accessibility", "evidence_utilisation")
WIN2_MANIFEST_SHA256 = "3c72eb6d5ba62714711b18ae71c7a1c78943f5f18565ab6e1342005a55d79f7d"


def accepted_metadata(package_manifest=None):
    """Explicitly opt into the reviewed Windows package, never arbitrary hashes."""
    expected = frozen_metadata()
    audit = {"mode": "checkout", "source_sha256": expected["source_sha256"]}
    if package_manifest is not None:
        if sha256_file(package_manifest) != WIN2_MANIFEST_SHA256:
            raise ValueError("unapproved package manifest SHA256")
        package = json.loads(Path(package_manifest).read_text())
        expected["source_sha256"] = {p: package["files"][p] for p in expected["source_sha256"]}
        audit = {"mode": "explicit_win2_manifest", "manifest_sha256": WIN2_MANIFEST_SHA256,
                 "source_sha256": expected["source_sha256"],
                 "reason": "Reviewed Windows locking and commit-receipt compatibility fixes; prompt, generation and guard unchanged"}
    return expected, audit


def _rate(k, n):
    return k / n if n else None


def evaluate_generation(reference_path, input_path, gpu_output_dir, output_dir, *, expected_count=2381,
                        package_manifest=None):
    """All 2381 references are scored, including unchanged and explicit errors.

    No partial-score success: all changed inputs must have exactly one terminal
    result in input order. Gold and labels first enter here, never the worker.
    """
    sources = load_generation_inputs(input_path)
    by_input = {r["sample_id"]: r for r in sources}
    gpu_output_dir = Path(gpu_output_dir)
    manifest = json.loads((gpu_output_dir / "run_manifest.json").read_text())
    expected_metadata, source_audit = accepted_metadata(package_manifest)
    if (manifest["input_sha256"], manifest["n"], manifest["run_metadata"]) != (
        sha256_file(input_path), len(sources), expected_metadata
    ):
        raise ValueError("GPU input/config/source hashes differ from this evaluator")
    results = read_results(gpu_output_dir / "results.jsonl")
    if [r["sample_id"] for r in results] != [r["sample_id"] for r in sources]:
        raise ValueError("GPU results incomplete, duplicated, extra or reordered")
    by_result = {}
    for result, source in zip(results, sources):
        validate_result(result, source)
        by_result[result["sample_id"]] = result
    with Path(reference_path).open(encoding="utf-8") as handle:
        references = [json.loads(line) for line in handle]
    ids = [r["sample_id"] for r in references]
    if len(ids) != len(set(ids)) or len(ids) != expected_count:
        raise ValueError("offline reference sample count/uniqueness mismatch")
    if {r["sample_id"] for r in references if r["generation_required"]} != set(by_input):
        raise ValueError("changed input/reference membership differs")
    records = []
    for ref in references:
        sid = ref["sample_id"]
        if (not isinstance(sid, str) or not isinstance(ref["initial_answer"], str)
            or type(ref["generation_required"]) is not bool or ref["label"] not in LABELS
            or not isinstance(ref["gold_answers"], list) or not ref["gold_answers"] or not all(
            isinstance(a, str) and a.strip() for a in ref["gold_answers"]
        )):
            raise ValueError("invalid offline labels/answer aliases")
        if sid in by_input and ref["initial_answer"] != by_input[sid]["initial_answer"]:
            raise ValueError("initial answer differs between GPU input and offline reference")
        result = by_result.get(sid)
        final = result["final_answer"] if result else ref["initial_answer"]
        before_em, before_f1 = best_em_f1(ref["initial_answer"], ref["gold_answers"])
        after_em, after_f1 = best_em_f1(final, ref["gold_answers"])
        raw_em, raw_f1 = best_em_f1(result["raw_answer"], ref["gold_answers"]) if result and isinstance(result["raw_answer"], str) else (before_em, before_f1)
        records.append({"sample_id": sid, "label": ref["label"], "base_em": before_em,
                        "final_em": after_em, "base_f1": before_f1, "final_f1": after_f1,
                        "raw_em": raw_em, "raw_f1": raw_f1,
                        "latency": result["latency"] if result else None,
                        "status": result["status"] if result else "no_change"})

    def aggregate(rows):
        n = len(rows)
        generated = [r for r in rows if r["status"] in {"answer_replaced", "kept_original"}]
        rejected = [r for r in rows if r["status"] == "kept_original"]
        baseline_correct = [r for r in rows if r["base_em"] == 1]
        harmed = sum(r["final_em"] == 0 for r in baseline_correct)
        return {
            "n": n, "base_em": _rate(sum(r["base_em"] for r in rows), n),
            "final_em": _rate(sum(r["final_em"] for r in rows), n),
            "em_delta": _rate(sum(r["final_em"] - r["base_em"] for r in rows), n),
            "base_f1": _rate(sum(r["base_f1"] for r in rows), n),
            "final_f1": _rate(sum(r["final_f1"] for r in rows), n),
            "f1_delta": _rate(sum(r["final_f1"] - r["base_f1"] for r in rows), n),
            "generated_successfully": len(generated), "guard_rejected": len(rejected),
            "guard_rejection_rate": _rate(len(rejected), len(generated)),
            "rejected_but_raw_em_correct": sum(r["raw_em"] == 1 for r in rejected),
            "rejected_raw_em_correct_rate": _rate(sum(r["raw_em"] == 1 for r in rejected), len(rejected)),
            "generation_errors": sum(r["status"] == "generation_failed" for r in rows),
            "unchanged": sum(r["status"] == "no_change" for r in rows),
            "baseline_em_correct": len(baseline_correct), "em_harmed": harmed,
            "harm_rate_among_baseline_em_correct": _rate(harmed, len(baseline_correct)),
            "em_harm_over_all_label_samples": _rate(harmed, n),
        }

    report = {"n": len(records), "generated_inputs": len(sources), "source_verification": source_audit,
              "overall": aggregate(records),
              "by_label": {label: aggregate([r for r in records if r["label"] == label]) for label in LABELS},
              "denominators": {
                  "correct_class_harm": "Within label=correct: Base EM=1 and Final EM=0 / Base EM=1",
                  "rejection_rate": "guard-rejected / successful string generations; errors reported separately",
                  "rejected_raw_em_correct_rate": "raw EM=1 among guard-rejected / all guard-rejected",
              },
              "label_caveat": "AI initial Diagnosis annotations; descriptive strata, not verified human Gold labels",
              "sha256": {"offline_reference": sha256_file(reference_path), "gpu_input": sha256_file(input_path),
                         "gpu_results": sha256_file(gpu_output_dir / "results.jsonl")}}
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    with (target / "metrics.json").open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (target / "per_sample.jsonl").open("x") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--gpu-output-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--package-manifest", type=Path,
                        help="Explicit acceptance of the SHA256-pinned, reviewed win2 manifest")
    args = parser.parse_args()
    print(json.dumps(evaluate_generation(args.reference, args.input, args.gpu_output_dir, args.output_dir,
                                         package_manifest=args.package_manifest), indent=2))


if __name__ == "__main__":
    main()
