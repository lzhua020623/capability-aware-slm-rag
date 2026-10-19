"""Offline v8 paired report and pending output-review queue; no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_answer_repair import prepare_records, read_existing, read_jsonl, run_fingerprint
from src.diagnosis import DatasetName
from src.evaluation.answer_repair import score_answer, summarize_repairs


def summarize(bundle: Path, results: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    source_raw = (bundle / manifest.get("scored_input", "inputs/nq75_c1.jsonl")).read_bytes()
    sources = read_jsonl(source_raw, "v8 C1")
    prepared = prepare_records(sources, DatasetName.NQ)
    human = {r["sample_id"]: r for r in read_jsonl((bundle / "offline/initial_reviews.jsonl").read_bytes(), "initial reviews")}
    if [r.sample_id for _, r in prepared] != manifest["sample_ids"] or set(human) != set(manifest["sample_ids"]):
        raise ValueError("v8 source/manifest/human ID mismatch")
    baseline = [score_answer(r.initial_answer, s, DatasetName.NQ) for s, r in prepared]
    report = {"version": manifest["version"], "samples": len(prepared), "development_only": True,
              "baseline_c1": {"correct": sum(r["correct"] for r in baseline), "extra_calls": 0,
                              "f1": sum(r["f1"] for r in baseline) / len(baseline)},
              "human_final_gain": None,
              "interpretation": "Frozen EM/F1 and same-model checks are not human semantic truth. New outputs require paired review. Selected development cases do not estimate population benefit.",
              "arms": {}}
    queue = []
    for arm in manifest["protocol"]:
        path = results / (arm["strategy"] + ".jsonl")
        rows = read_jsonl(path.read_bytes(), str(path)) if path.exists() else []
        if rows:
            settings = rows[0]["settings"]
            fingerprint = run_fingerprint(source_raw, b"", settings)
            rows = read_existing(path, prepared, fingerprint)
            if any(row["settings"]["strategy"] != arm["strategy"] or
                   row["settings"]["max_new_tokens"] != arm["max_new_tokens"] or
                   row["settings"] != settings or settings["routing"] != "all" or
                   settings["source_sha256"] != hashlib.sha256(source_raw).hexdigest() for row in rows):
                raise ValueError("Output arm/settings mismatch")
        stats = summarize_repairs(rows)
        stats.update(completed=len(rows), expected=len(prepared), complete=len(rows) == len(prepared),
                     result_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if rows else None,
                     stage_status_counts={}, stage_invalid_counts={})
        stage_counts, invalid_counts = Counter(), Counter()
        for row, (source, request) in zip(rows, prepared):
            h = human[request.sample_id]
            group = h["group"]
            for stage in (row.get("repair_trace") or {}).get("stages", []):
                name = stage["stage"]
                if stage["validation_error"]:
                    invalid_counts[name] += 1
                else:
                    stage_counts[name + ":" + stage["parsed"]["status"]] += 1
            queue.append({"review_id": f"v8:{arm['strategy']}:{request.sample_id}",
                          "review_number": h["review_number"], "sample_id": request.sample_id,
                          "strategy": arm["strategy"], "group": group, "changed": row["changed"],
                          "query": request.query, "initial_answer": request.initial_answer,
                          "final_answer": row["final_answer"], "status": row["status"],
                          "evidence": [{"rank": e.rank, "passage_id": e.passage_id, "text": e.text} for e in request.evidence],
                          "offline_reference": source["gold_answer_or_label"],
                          "initial_human_review": h, "repair_trace": row.get("repair_trace"),
                          "source_result_sha256": stats["result_sha256"],
                          "human_output_review": {"status": "pending", "reviewer": None,
                              "initial_correct": None, "final_correct": None,
                              "evidence_sufficient": None, "reason": None}})
        stats["stage_status_counts"], stats["stage_invalid_counts"] = dict(stage_counts), dict(invalid_counts)
        for group in ("include_e_core", "correct_control"):
            members = [r for r in rows if human[r["sample_id"]]["group"] == group]
            stats[group] = {"samples": len(members), "changed_pending_review": sum(r["changed"] for r in members),
                            "unchanged": sum(not r["changed"] for r in members),
                            "frozen_metrics": summarize_repairs(members), "semantic_net_gain": None}
        report["arms"][arm["strategy"]] = stats
    report["all_arms_complete"] = all(a["complete"] for a in report["arms"].values())
    report["pending_changed_outputs"] = sum(r["changed"] for r in queue)
    results.mkdir(parents=True, exist_ok=True)
    (results / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    queue.sort(key=lambda r: (not r["changed"], r["group"] == "correct_control", r["review_number"], r["strategy"]))
    (results / "output_review_queue.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in queue))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.bundle, args.results)
    print(json.dumps({"all_arms_complete": report["all_arms_complete"],
                      "pending_changed_outputs": report["pending_changed_outputs"],
                      "comparison": str(args.results / "comparison.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
