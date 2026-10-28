"""Production E2E smoke: bare query through load_current_pipeline().run(query).

Thin wrapper only — no duplicated Diagnosis/Selection/Controller logic.
Train/Validation IDs only; never Test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DESIRED_ACTIONS = (
    "no_action",
    "retrieval_enhancement",
    "evidence_refinement",
    "generation_repair",
)

# Offline diagnosis labels used only to rank candidates; online path is authoritative.
OFFLINE_HINT = {
    "no_action": "correct",
    "retrieval_enhancement": "evidence_insufficiency",
    "evidence_refinement": "evidence_accessibility",
    "generation_repair": "evidence_utilisation",
}


def _load_candidates(per_label: int) -> dict[str, list[dict]]:
    from src.data.team_nq import TeamNQBoundary

    boundary = TeamNQBoundary(ROOT)
    allowed = {
        sid: role
        for role in ("train", "validation")
        for sid in boundary.splits[role]
    }
    buckets: dict[str, list[dict]] = {label: [] for label in OFFLINE_HINT.values()}
    path = ROOT / "data/processed/diagnosis/nq_train_74097_diagnosis.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row["input"]
            sid = str(payload["sample_id"])
            role = allowed.get(sid)
            if role is None:
                continue
            label = row.get("target_label")
            if label not in buckets or len(buckets[label]) >= per_label:
                if all(len(v) >= per_label for v in buckets.values()):
                    break
                continue
            buckets[label].append(
                {
                    "sample_id": sid,
                    "role": role,
                    "query": payload["query"],
                    "offline_target_label": label,
                }
            )
    ordered: dict[str, list[dict]] = {}
    for action, label in OFFLINE_HINT.items():
        ordered[action] = buckets[label]
        if not ordered[action]:
            raise SystemExit(f"No Train/Val candidates for offline hint {label!r}")
    return ordered


def _trace(sample: dict, result) -> dict:
    selection = result.selection_decision
    controller = result.controller_decision
    enhancement = result.enhancement_result
    meta = dict(result.metadata or {})
    native = (enhancement.metadata or {}).get("native") if enhancement is not None else None
    adapter = (enhancement.metadata or {}).get("adapter") if enhancement is not None else None
    action = controller.action.value
    executed = False
    if action == "no_action":
        executed = False
    elif isinstance(native, dict):
        executed = bool(native.get("executed"))
    elif enhancement is not None:
        executed = enhancement.status.value not in ("no_action", "unavailable")

    flags = {
        "unavailable": enhancement is not None and enhancement.status.value == "unavailable",
        "mock": isinstance(adapter, str) and "mock" in adapter.lower(),
        "missing_implementation": bool(
            (enhancement.metadata or {}).get("enhancement_not_registered")
        )
        if enhancement is not None
        else False,
        "exception": enhancement is not None and enhancement.status.value == "error",
        "schema_ok": True,
    }
    if enhancement is not None and enhancement.error:
        flags["error_text"] = enhancement.error

    evidence = list(result.initial_evidence or ())
    return {
        "sample_id": sample["sample_id"],
        "role": sample["role"],
        "offline_target_label": sample.get("offline_target_label"),
        "query": sample["query"],
        "top5_ok": len(evidence) == 5
        and [e.rank for e in evidence] == [1, 2, 3, 4, 5]
        and all(str(e.text).strip() for e in evidence),
        "initial_answer": result.initial_answer,
        "final_answer": result.final_answer,
        "diagnosis": {
            "final_label": result.diagnosis.final_label.value,
            "confidence": result.diagnosis.confidence,
            "stage1_failure_probability": result.diagnosis.stage1_failure_probability,
            "stage1_label": result.diagnosis.stage1_label,
        },
        "selection": {
            "needs_enhancement": selection.needs_enhancement,
            "failure_type": selection.failure_type.value,
            "selection_lane": selection.selection_lane.value,
            "selection_score": selection.selection_score,
            "confidence": selection.confidence,
            "p_fail": selection.p_fail,
            "p_insuff": selection.p_insuff,
        },
        "controller": {
            "action": action,
            "reason": controller.reason,
            "used_probe": controller.used_probe,
        },
        "enhancement": {
            "actually_executed": executed,
            "adapter": adapter,
            "status": None if enhancement is None else enhancement.status.value,
            "changed_answer": None if enhancement is None else enhancement.changed_answer,
            "native_status": None if not isinstance(native, dict) else native.get("status"),
            "native_executed": None if not isinstance(native, dict) else native.get("executed"),
        },
        "pipeline_status": result.status.value,
        "latency_ms": result.latency_ms,
        "flags": flags,
        "baseline": meta.get("baseline"),
        "E_mode": meta.get("E_mode"),
        "feature_provider_cost_state": (meta.get("feature_provider") or {}).get("cost_state"),
    }


def _path_ok(row: dict) -> bool:
    if not row["top5_ok"]:
        return False
    if not str(row["initial_answer"]).strip() or not str(row["final_answer"]).strip():
        return False
    flags = row["flags"]
    if flags["unavailable"] or flags["mock"] or flags["missing_implementation"] or flags["exception"]:
        return False
    action = row["controller"]["action"]
    if action == "no_action":
        return (
            row["selection"]["needs_enhancement"] is False
            and row["final_answer"] == row["initial_answer"]
            and row["enhancement"]["status"] in (None, "no_action")
            and row["enhancement"]["actually_executed"] is False
        )
    return row["enhancement"]["actually_executed"] is True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label-candidates", type=int, default=4)
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/controller/production_e2e_smoke_v1/smoke_results.json",
    )
    args = parser.parse_args(argv)

    candidates = _load_candidates(args.per_label_candidates)
    print(json.dumps({
        "phase": "candidates",
        "counts": {k: len(v) for k, v in candidates.items()},
        "preview": {k: v[0] for k, v in candidates.items()},
    }, ensure_ascii=False, indent=2))

    from src.generation.qwen import free_cuda_memory
    from src.pipeline.current import load_current_pipeline

    nq_resources = ROOT / "configs/controller/nq_train_resources_v1.json"
    pipeline, environment = load_current_pipeline(
        root=ROOT,
        nq_resources=nq_resources,
        data_role="train",
        attach_reference_rag=True,
    )
    print(json.dumps({
        "phase": "pipeline_loaded",
        "reference_rag": environment.get("reference_rag"),
        "nq_train_resources": bool(environment.get("nq_train_resources")),
        "reranker_model": environment.get("reranker_model"),
        "E_v2a_runtime": bool(environment.get("E_v2a_runtime")),
        "ready_for_gpu": environment.get("current_baseline", {}).get("ready_for_gpu"),
        "registered_enhancements": sorted(
            a.value for a in pipeline.host.executor.registered_actions()
        ),
    }, indent=2))

    # Round-robin across desired actions so early coverage is diverse.
    queues = {action: list(cands) for action, cands in candidates.items()}
    covered: dict[str, dict] = {}
    all_rows: list[dict] = []
    runs = 0

    try:
        while runs < args.max_runs and len(covered) < len(DESIRED_ACTIONS):
            progressed = False
            for action in DESIRED_ACTIONS:
                if action in covered or runs >= args.max_runs:
                    continue
                if not queues[action]:
                    continue
                sample = queues[action].pop(0)
                pipeline.data_role = sample["role"]
                print(json.dumps({
                    "phase": "run_start",
                    "run_index": runs + 1,
                    "seeking": action,
                    "sample_id": sample["sample_id"],
                    "role": sample["role"],
                    "query": sample["query"],
                }, ensure_ascii=False, indent=2), flush=True)
                try:
                    result = pipeline.run(sample["query"], sample_id=sample["sample_id"])
                    row = _trace(sample, result)
                    row["seeking"] = action
                    row["path_ok"] = _path_ok(row)
                except Exception as exc:  # noqa: BLE001 - smoke must record failures
                    row = {
                        "sample_id": sample["sample_id"],
                        "role": sample["role"],
                        "offline_target_label": sample.get("offline_target_label"),
                        "query": sample["query"],
                        "seeking": action,
                        "path_ok": False,
                        "exception": {"type": type(exc).__name__, "message": str(exc)},
                        "flags": {
                            "unavailable": False,
                            "mock": False,
                            "missing_implementation": False,
                            "exception": True,
                            "schema_ok": False,
                        },
                    }
                runs += 1
                progressed = True
                all_rows.append(row)
                print(json.dumps({"phase": "run_done", **row}, ensure_ascii=False, indent=2), flush=True)
                actual = (row.get("controller") or {}).get("action")
                if row.get("path_ok") and actual in DESIRED_ACTIONS and actual not in covered:
                    covered[actual] = row
            if not progressed:
                break
    finally:
        close = getattr(getattr(pipeline.host, "reference_rag", None), "close", None)
        if callable(close):
            close()
        free_cuda_memory()

    summary = {
        "status": "E2E_SMOKE_OK"
        if set(covered) == set(DESIRED_ACTIONS)
        else "E2E_SMOKE_PARTIAL",
        "runs": runs,
        "covered_actions": sorted(covered),
        "missing_actions": [a for a in DESIRED_ACTIONS if a not in covered],
        "coverage_table": [
            {
                "sample_id": covered[a]["sample_id"],
                "diagnosis": covered[a]["diagnosis"]["final_label"],
                "selected": covered[a]["selection"]["needs_enhancement"],
                "controller_action": a,
                "enhancement_executed": covered[a]["enhancement"]["actually_executed"],
                "enhancement_status": covered[a]["enhancement"]["status"],
                "pipeline_status": covered[a]["pipeline_status"],
                "path_ok": covered[a]["path_ok"],
            }
            for a in DESIRED_ACTIONS
            if a in covered
        ],
        "any_mock": any(r.get("flags", {}).get("mock") for r in all_rows),
        "any_unavailable": any(r.get("flags", {}).get("unavailable") for r in all_rows),
        "any_missing_implementation": any(
            r.get("flags", {}).get("missing_implementation") for r in all_rows
        ),
        "any_exception": any(r.get("flags", {}).get("exception") for r in all_rows),
        "environment": {
            "reference_rag": environment.get("reference_rag"),
            "reranker_model": environment.get("reranker_model"),
            "nq_train_resources": bool(environment.get("nq_train_resources")),
            "E_v2a_runtime": bool(environment.get("E_v2a_runtime")),
        },
        "all_rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "all_rows"}, indent=2))
    print(json.dumps({"wrote": str(args.output)}, indent=2))
    return 0 if summary["status"] == "E2E_SMOKE_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
