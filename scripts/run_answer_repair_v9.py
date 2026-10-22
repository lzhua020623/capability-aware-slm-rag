"""Opt-in v9 CPU preflight, CUDA execution, resume and frozen trace replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_answer_repair import prepare_records, read_jsonl, read_controller_decisions, result_record
from src.compensation.answer_repair import MODEL_NAME, MODEL_REVISION
from src.compensation.evidence_id_repair import (
    VERSION, STRATEGIES, MAX_NEW_TOKENS, evidence_units, max_calls, repair, replay,
)
from src.diagnosis import DatasetName
from src.evaluation.answer_repair import summarize_repairs


def fingerprint(source, controller_source, settings):
    sources = [*sorted((ROOT / "src").rglob("*.py")), ROOT / "scripts/run_answer_repair.py",
               Path(__file__), ROOT / "configs/base.yaml", ROOT / "configs/answer_repair_v9_protocol_v1.json"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    payload = {"input_sha256": hashlib.sha256(source).hexdigest(),
               "controller_sha256": hashlib.sha256(controller_source).hexdigest(),
               "settings": settings, "code_sha256": hashes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=("fever", "nq"))
    p.add_argument("--c1", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--strategy", required=True, choices=STRATEGIES)
    route = p.add_mutually_exclusive_group(required=True)
    route.add_argument("--all", action="store_true", help="Explicit unrouted component experiment")
    route.add_argument("--controller-decisions", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--replay", action="store_true", help="Verify existing output, without model loading")
    args = p.parse_args(argv)
    if args.out.resolve() == args.c1.resolve() or (args.controller_decisions and args.out.resolve() == args.controller_decisions.resolve()):
        p.error("Output cannot replace an input")
    source = args.c1.read_bytes()
    prepared = prepare_records(read_jsonl(source, str(args.c1)), DatasetName(args.dataset))
    controller_source = args.controller_decisions.read_bytes() if args.controller_decisions else b""
    decisions = read_controller_decisions(args.controller_decisions, DatasetName(args.dataset)) if args.controller_decisions else {}
    if not args.all and set(decisions) != {request.sample_id for _, request in prepared}:
        raise ValueError("Controller decisions must match the exact input cohort")
    settings = {"dataset": args.dataset, "strategy": args.strategy, "version": VERSION,
                "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
                "quantization": "nf4", "do_sample": False, "seed": 42,
                "max_new_tokens": MAX_NEW_TOKENS, "max_calls_per_sample": max_calls(args.strategy),
                "routing": "all_component_ablation" if args.all else "resolved_controller",
                "sample_count": len(prepared), "semantic_verifier": "same_Qwen_not_independent"}
    fp, manifest = fingerprint(source, controller_source, settings)
    # Validate routing during preflight as well; it must remain upstream-owned.
    from src.compensation.controller_adapter import controller_route_status
    if not args.all:
        for _, request in prepared:
            controller_route_status(request, decisions[request.sample_id])
    if args.check_only:
        print(json.dumps({"preflight": "passed", "samples": len(prepared),
                          "evidence_units": sum(len(evidence_units(r)) for _, r in prepared),
                          "maximum_generation_calls": len(prepared) * max_calls(args.strategy),
                          "gpu_inference_completed": False, "run_fingerprint": fp,
                          "manifest": manifest}, ensure_ascii=False, indent=2))
        return 0
    rows = read_jsonl(args.out.read_bytes(), str(args.out)) if args.out.exists() else []
    if args.replay and not args.out.exists():
        raise FileNotFoundError("Replay needs a saved result file")
    if len(rows) > len(prepared):
        raise ValueError("Output exceeds frozen cohort")
    for row, (record, request) in zip(rows, prepared):
        if row.get("sample_id") != request.sample_id or row.get("run_fingerprint") != fp or row.get("settings") != settings:
            raise ValueError("Output belongs to another input/code/settings snapshot")
        rebuilt = replay(request, row, strategy=args.strategy,
                         decision=decisions.get(request.sample_id), all_samples=args.all)
        expected = result_record(record, request, rebuilt, fp, settings)
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("Saved metrics or result differ from deterministic replay")
    if args.replay:
        if len(rows) != len(prepared):
            raise ValueError("Replay of a complete experiment requires all records")
        print(json.dumps({"replay": "passed", "records": len(rows), "run_fingerprint": fp}))
        return 0
    if len(rows) < len(prepared):
        from src.compensation.qwen_repair import QwenRepairGenerator
        generator = QwenRepairGenerator(json_output=args.strategy != "qwen_direct")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            for record, request in prepared[len(rows):]:
                result = repair(request, generator, strategy=args.strategy,
                                decision=decisions.get(request.sample_id), all_samples=args.all)
                row = result_record(record, request, result, fp, settings)
                row["runtime_environment"] = generator.runtime_environment
                handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
                rows.append(row)
                print(json.dumps({"completed": len(rows), "total": len(prepared), "status": result.status}), flush=True)
    report = {"version": VERSION, "settings": settings, "run_fingerprint": fp,
              **summarize_repairs(rows), "manifest": manifest,
              "limitations": ["NQ EM is not human semantic correctness", "component ablation is not end-to-end gain",
                              "arms have different call budgets", "citation validity is not entailment"]}
    args.out.with_suffix(".summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
