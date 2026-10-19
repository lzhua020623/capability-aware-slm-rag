"""Evaluate the frozen engineering gate; correctness metrics cannot open the gate."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_answer_repair_v8 import summarize
from src.compensation.answer_repair import V8_STRATEGIES


def evaluate_known_cases(manifest: dict, arms: dict[str, list[dict]]) -> dict | None:
    policy = manifest.get("semantic_regressions")
    if policy is None:
        return None
    indexed = {r["sample_id"]: r for r in arms.get(policy["strategy"], [])}
    cases = []
    for expected in policy["cases"]:
        row = indexed.get(expected["sample_id"])
        stages = {s["stage"]: s for s in ((row or {}).get("repair_trace") or {}).get("stages", [])}
        failures, observed, facts = [], {}, {}
        if row is None:
            failures.append("sample_missing")
        for name, wanted in expected["expected_stage_status"].items():
            stage = stages.get(name) or {}
            parsed = stage.get("parsed") or {}
            observed[name], facts[name] = parsed.get("status"), parsed.get("fact")
            if stage.get("validation_error") is not None or observed[name] != wanted:
                failures.append("stage_status:" + name)
        candidate = (row or {}).get("candidate_answer")
        if "expected_candidate_answer" in expected and candidate != expected["expected_candidate_answer"]:
            failures.append("candidate_identity")
        if expected.get("keep_original") and (row is None or row["final_answer"] != row["initial_answer"]):
            failures.append("original_not_preserved")
        cases.append({"case": expected["case"], "sample_id": expected["sample_id"],
                      "passed": not failures, "failures": failures,
                      "observed_stage_status": observed, "facts": facts,
                      "candidate_answer": candidate, "final_status": (row or {}).get("status")})
    return {"passed": all(c["passed"] for c in cases), "cases": cases,
            "metric_used_for_selection": False,
            "interpretation": "Predeclared development-stage regressions only, not independent semantic evaluation."}


def evaluate_smoke(manifest: dict, arms: dict[str, list[dict]]) -> dict:
    policy = manifest["smoke_acceptance"]
    required_ids = manifest["sample_ids"]
    valid, invalid, observed, rejection_kinds = Counter(), Counter(), Counter(), Counter()
    full_paths, raw_changed, calls = 0, 0, 0
    complete, arm_stats = True, {}
    for arm in manifest["protocol"]:
        strategy = arm["strategy"]
        rows = arms.get(strategy, [])
        ready = [r["sample_id"] for r in rows] == required_ids
        complete = complete and ready
        arm_stats[strategy] = {"completed": len(rows), "complete": ready,
                               "status_counts": dict(Counter(r["status"] for r in rows))}
        for row in rows:
            calls += row["extra_calls"]
            if strategy not in V8_STRATEGIES:
                continue
            stages = (row.get("repair_trace") or {}).get("stages", [])
            if [s["stage"] for s in stages] == ["original_check", "minimal_repair", "candidate_check"]:
                full_paths += 1
            for stage in stages:
                if stage["validation_error"] is None and stage["parsed"] is not None:
                    valid[stage["stage"]] += 1
                else:
                    invalid[stage["stage"]] += 1
                    kind = stage.get("validation_kind")
                    rejection_kinds[kind if kind in ("format", "grounding", "constraint") else "unclassified"] += 1
                generation = stage["generation"]
                flag = generation.get("contract_stopped")
                observed["boundary" if flag is True else "model_end_or_limit" if flag is False else "missing"] += 1
                raw_changed += generation.get("raw_text") != generation["text"]
    failures = []
    if not complete:
        failures.append("incomplete_or_reordered_outputs")
    if "max_format_invalid_v8_stages" in policy:
        # A NEW prospective policy, never applied to the frozen r2 gate.
        if rejection_kinds["format"] > policy["max_format_invalid_v8_stages"]:
            failures.append("invalid_v8_response_format")
        if rejection_kinds["unclassified"]:
            failures.append("unclassified_v8_validation_errors")
        allowed = set(policy.get("allowed_guard_rejections", []))
        if any(rejection_kinds[k] and k not in allowed for k in ("grounding", "constraint")):
            failures.append("disallowed_guard_rejection")
    elif sum(invalid.values()) > policy["max_invalid_v8_stages"]:
        failures.append("invalid_v8_stage_outputs")
    for stage, count in policy["minimum_valid_stage_calls"].items():
        if valid[stage] < count:
            failures.append("stage_not_exercised:" + stage)
    if full_paths < policy["minimum_full_three_stage_paths"]:
        failures.append("full_three_stage_path_not_exercised")
    if policy["require_observed_contract_backend"] and observed["missing"]:
        failures.append("contract_backend_observation_missing")
    if policy["require_raw_text_unchanged"] and raw_changed:
        failures.append("generation_was_cropped_afterwards")
    if calls > manifest["max_generation_calls"]:
        failures.append("generation_call_budget_exceeded")
    known = evaluate_known_cases(manifest, arms)
    ready = not failures and (known is None or known["passed"])
    report = {"version": manifest["version"], "engineering_gate_passed": not failures,
            "ready_for_full_nq75_run": ready, "failures": failures,
            "arms": arm_stats, "valid_v8_stage_calls": dict(valid), "invalid_v8_stage_calls": dict(invalid),
            "validation_rejection_kinds": dict(rejection_kinds),
            "full_three_stage_paths": full_paths, "generation_stop_observations": dict(observed),
            "raw_text_changed_stages": raw_changed, "generation_calls": calls,
            "semantic_correctness_verified": False, "human_final_gain": None,
            "interpretation": "Engineering gate only. Greedy same-model checks and EM do not establish semantic gain."}
    if known is not None:
        report.update(known_case_regressions=known, selected_final_revision="r4" if ready else "r3",
                      selection_policy=manifest["failure_policy"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    # This validates fingerprints, stage decisions, settings and costs first.
    summarize(args.bundle, args.results)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    arms = {a["strategy"]: [json.loads(s) for s in (args.results / (a["strategy"] + ".jsonl")).read_text().splitlines()]
            for a in manifest["protocol"]}
    report = evaluate_smoke(manifest, arms)
    (args.results / "smoke_acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # The experiment can finish successfully while the engineering gate fails.
    # Export all outputs in either case; no full run is launched by this command.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
