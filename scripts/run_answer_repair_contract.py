"""Synthetic-only contract comparison; CPU schema preflight, GPU and replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_answer_repair_contract import request_from_case
from scripts.import_nq_train_preliminary import write_new, sha
from src.compensation.answer_repair import MODEL_NAME, MODEL_REVISION
from src.compensation.constrained_evidence_id import (VERSION, STRATEGIES, repair, replay, plan_schema,
    fact_schema, decision_schema)
from src.compensation.evidence_id_repair import evidence_units


def serialize(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def fingerprint(cases_path):
    files = ["scripts/run_answer_repair_contract.py", "scripts/prepare_answer_repair_contract.py",
        "src/compensation/constrained_evidence_id.py", "src/compensation/qwen_constrained.py",
        "src/compensation/evidence_id_repair.py", "src/compensation/qwen_repair.py",
        "src/compensation/answer_repair.py", "src/compensation/short_answer_repair.py",
        "src/compensation/controller_adapter.py", "src/compensation/line_contract.py",
        "src/generation/qwen.py", "src/config.py", "src/diagnosis/schema.py", "src/diagnosis/labels.py",
        "configs/base.yaml", "configs/answer_repair_contract_v1.json", "requirements-answer-repair-contract.txt"]
    payload = {"version": VERSION, "cases_sha256": sha(cases_path), "source_sha256": {p: sha(ROOT / p) for p in files}}
    return hashlib.sha256(serialize(payload)).hexdigest(), payload


def compile_schemas(cases):
    """Compile with the actual frozen Qwen tokenizer; no model weights/inference."""
    import xgrammar as xgr
    from transformers import AutoConfig, AutoTokenizer
    from jsonschema import Draft202012Validator
    if importlib.metadata.version("xgrammar") != "0.2.8":
        raise RuntimeError("Expected xgrammar 0.2.8")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    config = AutoConfig.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
    compiler = xgr.GrammarCompiler(info, max_threads=4)
    results = []
    for case in cases:
        request = request_from_case(case); units = evidence_units(request)
        samples = [
            ("plan", plan_schema(), {"questions": [request.query]}),
            ("fact", fact_schema(units), {"status": "answerable", "fact": "Fictional test fact.", "evidence_ids": [units[0].id]}),
            ("decision", decision_schema(request, units), {"decision": "keep", "answer": request.initial_answer,
                "evidence_ids": [units[0].id], "reason": "Structural acceptance test only."})]
        for stage, schema, sample in samples:
            Draft202012Validator.check_schema(schema)
            started = time.perf_counter()
            compiled = compiler.compile_json_schema(json.dumps(schema, ensure_ascii=False))
            seconds = time.perf_counter() - started
            if not xgr.GrammarMatcher(compiled).accept_string(json.dumps(sample, ensure_ascii=False)):
                raise RuntimeError(f"Grammar rejected valid {case['case_id']}/{stage}")
            Draft202012Validator(schema).validate(sample)
            if stage == "decision":
                wrong = dict(sample, answer="CHANGED_KEEP_ANSWER")
                if xgr.GrammarMatcher(compiled).accept_string(json.dumps(wrong)):
                    raise RuntimeError("Grammar accepted changed keep answer")
                wrong = dict(sample, evidence_ids=["R999S999"])
                if xgr.GrammarMatcher(compiled).accept_string(json.dumps(wrong)):
                    raise RuntimeError("Grammar accepted unknown ID")
            results.append({"case_id": case["case_id"], "stage": stage, "compiled": True,
                            "compile_seconds": seconds})
            print(json.dumps(results[-1]), flush=True)
    return {"mode": "cpu_grammar_compilation_only_no_model_weights", "compiled_schemas": len(results),
            "valid_examples_accepted": len(results), "bad_keep_or_id_examples_rejected": len(cases) * 2,
            "model_name": MODEL_NAME, "model_revision": MODEL_REVISION, "vocab_size": config.vocab_size,
            "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in
                ("xgrammar", "torch", "transformers", "apache-tvm-ffi", "jsonschema")}, "results": results,
            "gpu_nf4_compatibility_verified": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=ROOT / "artifacts/answer_repair/v9_contract_v1/synthetic_cases.jsonl")
    p.add_argument("--out", type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = p.parse_args(argv)
    if args.out.resolve() == args.cases.resolve():
        p.error("Output cannot replace input")
    cases = [json.loads(line) for line in args.cases.read_bytes().splitlines() if line.strip()]
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Synthetic cases must be nonempty and unique")
    requests = [request_from_case(case) for case in cases]
    fp, manifest = fingerprint(args.cases)
    if args.check_only:
        report = {"preflight": "passed", "synthetic_cases": len(cases), "predictions": len(cases) * 4,
                  "maximum_calls": len(cases) * 12, "gpu_inference_completed": False,
                  "fingerprint": fp, "manifest": manifest}
        write_new(args.out, serialize(report)); print(json.dumps(report, indent=2)); return 0
    if args.compile_only:
        report = {"fingerprint": fp, "manifest": manifest, **compile_schemas(cases)}
        write_new(args.out, serialize(report)); return 0
    cohort = [(c, r, strategy, decoder) for c, r in zip(cases, requests) for strategy in STRATEGIES
              for decoder in ("prompt_only", "xgrammar")]
    rows = [json.loads(line) for line in args.out.read_bytes().splitlines() if line.strip()] if args.out.exists() else []
    if len(rows) > len(cohort):
        raise ValueError("Results exceed synthetic cohort")
    for row, (case, request, strategy, decoder) in zip(rows, cohort):
        if (row.get("fingerprint") != fp or row["case_id"] != case["case_id"]
                or row["strategy"] != strategy or row["decoder"] != decoder):
            raise ValueError("Saved result belongs to a different frozen run")
        replay(request, row["result"], strategy=strategy, all_samples=True)
    if args.replay:
        if len(rows) != len(cohort):
            raise ValueError("Complete replay requires every prediction")
        print(json.dumps({"replay": "passed", "predictions": len(rows), "fingerprint": fp})); return 0
    if len(rows) < len(cohort):
        from src.compensation.qwen_constrained import QwenConstrainedGenerator
        generator = QwenConstrainedGenerator()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            for case, request, strategy, decoder in cohort[len(rows):]:
                class Backend:
                    def generate_schema(self, prompt, limit, schema):
                        method = generator.generate_schema if decoder == "xgrammar" else generator.generate_unconstrained_schema
                        return method(prompt, limit, schema)
                result = repair(request, Backend(), strategy=strategy, all_samples=True)
                row = {"case_id": case["case_id"], "synthetic": True, "strategy": strategy,
                       "decoder": decoder, "fingerprint": fp, "result": asdict(result),
                       "runtime_environment": generator.runtime_environment}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush(); rows.append(row)
                print(json.dumps({"completed": len(rows), "total": len(cohort), "status": result.status}), flush=True)
    groups = {}
    for strategy in STRATEGIES:
        for decoder in ("prompt_only", "xgrammar"):
            subset = [r for r in rows if r["strategy"] == strategy and r["decoder"] == decoder]
            groups[strategy + "/" + decoder] = {"predictions": len(subset),
                "valid": sum(not r["result"]["status"].startswith("invalid_") for r in subset),
                "calls": sum(r["result"]["extra_calls"] for r in subset),
                "generation_seconds": sum(r["result"]["latency"] for r in subset),
                "schema_compile_seconds": sum(c["backend"].get("schema_compile_seconds", 0.) for r in subset
                    for c in (r["result"].get("repair_trace") or {}).get("constraints", []))}
    constrained = [r for r in rows if r["decoder"] == "xgrammar"]
    valid = sum(not r["result"]["status"].startswith("invalid_") for r in constrained)
    errors = [r["result"].get("validation_error") or "" for r in constrained]
    unknown_ids = sum("Unknown evidence ID" in error for error in errors)
    keep_rewrites = sum("Keep cannot rewrite" in error for error in errors)
    for row, (_, request, strategy, _) in zip(rows, cohort):
        replay(request, row["result"], strategy=strategy, all_samples=True)
    report = {"version": VERSION, "fingerprint": fp, "manifest": manifest, "groups": groups,
              "constrained_valid": valid, "constrained_total": len(constrained),
              "unknown_ids": unknown_ids, "keep_rewrites": keep_rewrites, "replay": "passed",
              "engineering_gate_passed": len(rows) == len(cohort) and valid * 10 >= len(constrained) * 9
                                         and unknown_ids == 0 and keep_rewrites == 0,
              "benchmark_gain_claim": False, "model_training": False}
    args.out.with_suffix(".summary.json").write_bytes(serialize(report))
    print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
