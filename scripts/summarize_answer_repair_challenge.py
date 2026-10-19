"""Report the frozen E-only stratified challenge without assuming human truth labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_answer_repair import prepare_records, read_existing, read_jsonl, run_fingerprint
from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from src.diagnosis import DatasetName
from src.evaluation.answer_repair import score_answer, summarize_repairs


def summarize(bundle: Path, results: Path):
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    if manifest["samples"] != 125 or manifest["groups"] != {"screen_candidate": 100, "em_match_control": 25}:
        raise ValueError("Expected the fixed 125-row stratified challenge")
    source_raw = (bundle / manifest["scored_input"]).read_bytes()
    if hashlib.sha256(source_raw).hexdigest() != manifest["source_input_sha256"]["scored_c1"]:
        raise ValueError("Challenge source fingerprint mismatch")
    prepared = prepare_records(read_jsonl(source_raw, "challenge C1"), DatasetName.NQ)
    strata_rows = read_jsonl((bundle / "offline/strata.jsonl").read_bytes(), "strata")
    strata = {x["sample_id"]: x for x in strata_rows}
    if ([r.sample_id for _, r in prepared] != manifest["sample_ids"] or len(strata_rows) != len(strata)
            or [x["sample_id"] for x in strata_rows] != manifest["sample_ids"]
            or dict(Counter(x["screening_group"] for x in strata_rows)) != manifest["groups"]
            or any(x["human_verified"] or x["diagnosis_label"] is not None for x in strata_rows)):
        raise ValueError("Challenge identities, strata or unknown human labels changed")
    baseline = [score_answer(r.initial_answer, s, DatasetName.NQ) for s, r in prepared]
    report = {"version": manifest["version"], "samples": 125, "scope": manifest["scope"],
              "baseline_c1": {"correct": sum(x["correct"] for x in baseline),
                              "f1": sum(x["f1"] for x in baseline) / 125, "extra_calls": 0},
              "selected_final_revision": manifest["selected_final_revision"], "human_final_gain": None,
              "semantic_correctness_verified": False, "arms": {},
              "interpretation": "EM screening is not human diagnosis. Report strata separately; changed-only review gives paired net changes, not full semantic accuracy or full repair recall."}
    for arm in manifest["protocol"]:
        path = results / (arm["strategy"] + ".jsonl")
        rows = read_jsonl(path.read_bytes(), str(path)); settings = rows[0]["settings"] if rows else {}
        if (len(rows) != 125 or settings.get("strategy") != arm["strategy"]
                or settings.get("routing") != "all" or settings.get("max_new_tokens") != arm["max_new_tokens"]
                or settings.get("model_revision") != manifest["model_revision"]
                or any(row["settings"] != settings for row in rows)):
            raise ValueError("Incomplete challenge arm or changed model/settings")
        rows = read_existing(path, prepared, run_fingerprint(source_raw, b"", settings))
        stats = summarize_repairs(rows)
        valid, rejected = Counter(), Counter()
        for row in rows:
            for stage in (row.get("repair_trace") or {}).get("stages", []):
                if stage["validation_error"]:
                    rejected[stage["stage"]] += 1
                else:
                    valid[stage["stage"] + ":" + stage["parsed"]["status"]] += 1
        stats.update(completed=125, expected=125, complete=True,
                     result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     stage_status_counts=dict(valid), stage_invalid_counts=dict(rejected),
                     human_final_gain=None, by_screening_group={})
        for group in manifest["groups"]:
            members = [r for r in rows if strata[r["sample_id"]]["screening_group"] == group]
            stats["by_screening_group"][group] = {"samples": len(members),
                "changed_pending_review": sum(r["changed"] for r in members),
                "frozen_metrics": summarize_repairs(members), "human_final_gain": None}
        report["arms"][arm["strategy"]] = stats
    report["all_arms_complete"] = True
    report["pending_changed_outputs"] = sum(x["changed"] for x in report["arms"].values())
    write_frozen(results / "comparison.json", encode_json(report))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True); p.add_argument("--results", type=Path, required=True)
    args = p.parse_args(argv); report = summarize(args.bundle, args.results)
    print(json.dumps({"all_arms_complete": True, "pending_changed_outputs": report["pending_changed_outputs"],
                      "human_final_gain": None}, indent=2))


if __name__ == "__main__":
    main()
