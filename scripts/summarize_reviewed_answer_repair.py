"""Report frozen metrics by reviewed group; prepare separate paired human review."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_answer_repair_experiment import encode_json, sha256, write_frozen
from scripts.run_answer_repair import prepare_records, read_existing, read_jsonl, run_fingerprint
from src.diagnosis import DatasetName
from src.evaluation.answer_repair import summarize_repairs


def result_digest(row: dict) -> str:
    return sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())


def summarize_semantic_pairs(results: list[dict], human_pairs: list[dict]) -> dict:
    by_id = {r["sample_id"]: r for r in results}
    seen, completed = set(), []
    for row in human_pairs:
        sid = row["sample_id"]
        if sid in seen or sid not in by_id or row["result_sha256"] != result_digest(by_id[sid]):
            raise ValueError("Duplicate, unknown or mismatched paired review")
        seen.add(sid)
        review = row["pair_review"]
        if review["status"] == "pending":
            continue
        if (review["status"] != "completed" or not review["reviewer"].strip() or not review["reason"].strip()
                or any(type(review.get(k)) is not bool for k in
                       ("initial_correct", "final_correct", "initial_supported", "final_supported"))):
            raise ValueError("Completed paired review requires BOTH answers, support and reviewer rationale")
        completed.append(review)
    n = len(completed)
    initial_wrong = sum(not r["initial_correct"] for r in completed)
    initial_correct = n - initial_wrong
    fixed = sum(not r["initial_correct"] and r["final_correct"] for r in completed)
    grounded_fixed = sum(not r["initial_correct"] and r["final_correct"] and r["final_supported"] for r in completed)
    regressed = sum(r["initial_correct"] and not r["final_correct"] for r in completed)
    # Null, rather than 0%, when nobody has reviewed outputs yet.
    return {"completed_pairs": n, "pending_pairs": len(results) - n,
            "review_complete": n == len(results), "fixed": fixed if n else None,
            "grounded_fixed": grounded_fixed if n else None, "regressed": regressed if n else None,
            "repair_rate_reviewed_subset": fixed / initial_wrong if initial_wrong else None,
            "regression_rate_reviewed_subset": regressed / initial_correct if initial_correct else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--c1", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--human-pairs", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_bytes())
    results = read_jsonl(args.results.read_bytes(), str(args.results))
    if not results:
        raise ValueError("No model results")
    ds, settings = results[0]["dataset"], results[0]["settings"]
    if any(r["dataset"] != ds or r["settings"] != settings for r in results):
        raise ValueError("Mixed result datasets or settings")
    protocol = next((p for p in manifest["protocol"] if p["strategy"] == settings["strategy"]), None)
    if (protocol is None or settings["routing"] != "all"
            or settings["max_new_tokens"] != protocol["max_new_tokens"]
            or settings.get("support_check_max_new_tokens") != protocol.get("support_check_tokens", 0)):
        raise ValueError("Results do not follow the frozen reviewed development protocol")
    source_raw = args.c1.read_bytes()
    if sha256(source_raw) != manifest["datasets"][ds]["input_sha256"]:
        raise ValueError("C1 source checksum differs from cohort")
    prepared = prepare_records(read_jsonl(source_raw, str(args.c1)), DatasetName(ds))
    expected_ids = manifest["datasets"][ds]["sample_ids"]
    if [r["sample_id"] for r in results] != expected_ids or [r.sample_id for _, r in prepared] != expected_ids:
        raise ValueError("Results must cover this dataset's full frozen cohort in order")
    read_existing(args.results, prepared, run_fingerprint(source_raw, b"", settings))
    included = {r["sample_id"]: r for r in manifest["included"] if r["dataset"] == ds}
    groups = {use: summarize_repairs([r for r in results if included[r["sample_id"]]["use"] == use])
              for use in ("include_e_core", "correct_control")}
    human = read_jsonl(args.human_pairs.read_bytes(), str(args.human_pairs)) if args.human_pairs else []
    summary = {"dataset": ds, "strategy": settings["strategy"], "cohort_version": manifest["version"],
               "metric_warning": "Frozen-reference metrics are not semantic repair/regression counts. Paired review must score BOTH initial and final answers consistently.",
               "frozen_metrics": summarize_repairs(results), "frozen_metrics_by_human_group": groups,
               "human_vs_frozen_metric_disagreement_ids": manifest["datasets"][ds]["human_vs_frozen_metric_disagreement_ids"],
               "paired_semantic_review": summarize_semantic_pairs(results, human)}
    template = []
    for result, (source, _) in zip(results, prepared):
        template.append({"sample_id": result["sample_id"], "dataset": ds, "strategy": settings["strategy"],
                         "result_sha256": result_digest(result), "query": result["query"],
                         "prior_initial_review": included[result["sample_id"]],
                         "initial_answer": result["initial_answer"], "final_answer": result["final_answer"],
                         "model_result": result, "evidence": source["evidence"],
                         "frozen_reference": source["gold_answer_or_label"],
                         "pair_review": {"status": "pending", "reviewer": "", "reason": "",
                                         "initial_correct": None, "final_correct": None,
                                         "initial_supported": None, "final_supported": None}})
    # Never overwrite an edited human template or a previous report silently.
    write_frozen(Path(str(args.out_prefix) + ".review.jsonl"),
                 ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in template).encode())
    write_frozen(Path(str(args.out_prefix) + ".summary.json"), encode_json(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
