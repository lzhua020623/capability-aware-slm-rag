"""Module-level real smokes for the three production Enhancements (Train/Val only).

Bypasses Diagnosis/Selection. Does not run full AdaptiveRAGPipeline E2E.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _train_item(limit_scan: int = 5000):
    from src.data.team_nq import TeamNQBoundary
    from src.diagnosis.schema import DiagnosisInput, EvidenceItem

    boundary = TeamNQBoundary(ROOT)
    allowed = {
        sid: role
        for role in ("train", "validation")
        for sid in boundary.splits[role]
    }
    path = ROOT / "data/processed/diagnosis/nq_train_74097_diagnosis.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= limit_scan:
                break
            row = json.loads(line)
            payload = row["input"]
            sid = str(payload["sample_id"])
            role = allowed.get(sid)
            if role is None:
                continue
            evidence = [EvidenceItem(**e) for e in payload["evidence"]]
            return DiagnosisInput(
                sample_id=sid,
                dataset="nq",
                query=payload["query"],
                evidence=evidence,
                initial_answer=payload["initial_answer"],
            ), role
    raise SystemExit("No Train/Val DiagnosisInput found for module smoke")


def _decision(action: str, sample_id: str, label: str):
    from src.controller.execution import decision_for
    from src.controller.schema import ControllerAction

    return decision_for(ControllerAction(action), sample_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modules",
        default="retrieval_enhancement,evidence_refinement,generation_repair",
        help="Comma-separated module names to smoke",
    )
    args = parser.parse_args(argv)
    wanted = [m.strip() for m in args.modules.split(",") if m.strip()]

    from src.controller.execution import ExecutionConfig
    from src.controller.providers import load_pipeline
    from src.generation.qwen import free_cuda_memory

    cfg = json.loads((ROOT / "configs/controller/current_baseline.json").read_text(encoding="utf-8"))
    exec_cfg = ExecutionConfig(**cfg["execution_config"])
    if exec_cfg.nq_v2a_mode != "allow_repair":
        raise SystemExit(f"expected nq_v2a_mode=allow_repair, got {exec_cfg.nq_v2a_mode!r}")

    item, role = _train_item()
    print(json.dumps({
        "phase": "input",
        "sample_id": item.sample_id,
        "role": role,
        "query": item.query,
        "initial_answer": item.initial_answer,
        "evidence_n": len(item.evidence),
    }, ensure_ascii=False, indent=2))

    nq_resources = ROOT / "configs/controller/nq_train_resources_v1.json"
    backend, environment = load_pipeline(
        exec_cfg,
        ROOT / cfg["classifier"],
        nq_resources=nq_resources,
    )
    print(json.dumps({
        "phase": "resources",
        "nq_train_resources": bool(environment.get("nq_train_resources")),
        "reranker_model": environment.get("reranker_model"),
        "E_v2a_runtime": bool(environment.get("E_v2a_runtime")),
        "nq_v2a_mode": exec_cfg.nq_v2a_mode,
    }, indent=2))

    label_for = {
        "retrieval_enhancement": "evidence_insufficiency",
        "evidence_refinement": "evidence_accessibility",
        "generation_repair": "evidence_utilisation",
    }
    results = []
    try:
        for action in wanted:
            decision = _decision(action, item.sample_id, label_for[action])
            # decision_for already sets the matching label for the action
            native = backend.execute(item, decision)
            row = {
                "action": action,
                "executed": native.get("executed"),
                "status": native.get("status"),
                "answer_changed": native.get("answer_changed"),
                "final_answer": native.get("final_answer"),
                "final_evidence_n": len(native.get("final_evidence") or []),
                "unavailable": native.get("status") in (
                    "verification_unavailable", "unsupported_dataset",
                ),
                "error": native.get("error"),
                "resources": {
                    "retrieval": "nq_train_bge_base matched stack" if action == "retrieval_enhancement" else None,
                    "reranker": environment.get("reranker_model") if action != "generation_repair" else None,
                    "generator": "Qwen2.5-7B-Instruct NF4",
                    "v2a": environment.get("E_v2a_runtime") if action == "generation_repair" else None,
                },
            }
            if action == "generation_repair":
                check = (native.get("module_trace") or {}).get("support_check") or {}
                row["deployment_mode"] = check.get("deployment_mode")
                row["original_verification"] = check.get("original_verification")
            results.append(row)
            print(json.dumps({"phase": "module", **row}, ensure_ascii=False, indent=2))
    finally:
        free_cuda_memory()

    ok = all(
        r["executed"]
        and not r["unavailable"]
        and r["status"] not in ("execution_failed",)
        and isinstance(r["final_answer"], str)
        and r["final_evidence_n"] >= 1
        for r in results
    )
    print(json.dumps({"status": "SMOKE_OK" if ok else "SMOKE_FAIL", "modules": results}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
