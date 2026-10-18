"""Evaluate E on archived C1 evidence; preflight is CPU-only and makes no calls.

Example: python3.11 scripts/run_answer_repair.py --dataset nq \
    --source-ref origin/feature/-ce-70B --all --check-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.answer_repair import (
    DEFAULT_MAX_NEW_TOKENS, MODEL_NAME, MODEL_REVISION, REPAIR_PROMPT_VERSION,
    GROUNDED_REANSWER_PROMPT_VERSION, RepairResult, build_grounded_reanswer_prompt, build_reanswer_prompt,
    build_repair_prompt, repair_answer,
    QUERY_GROUNDED_PROMPT_VERSION, GUARDED_REANSWER_PROMPT_VERSION,
    SUPPORT_CHECK_MAX_NEW_TOKENS, build_query_grounded_prompt,
    LINE_OUTPUT_STRATEGIES, NLI_STRATEGIES, STRATEGIES, V8_STRATEGIES, NQ_ONLY_STRATEGIES,
)
from src.compensation.anchors import ANCHOR_VERSION, FEVER_POLICY, NQ_MIN_COVERAGE
from src.compensation.extractive_repair import EXTRACTIVE_PROMPT_VERSION, build_extractive_prompt
from src.compensation.nli_repair import NLI_ANCHORED_VERSION, NLI_MODEL, NLI_REVISION, NLI_VERSION, NLI_THRESHOLD
from src.compensation.short_answer_repair import SHORT_ANSWER_PROMPT_VERSION, build_short_answer_prompt
from src.compensation.conservative_repair import (
    V8_VERSION, build_check_prompt, build_minimal_repair_prompt, max_calls, repair_from_v8_trace,
)
from src.diagnosis import DatasetName, DiagnosisInput, DiagnosisLabel, DiagnosisOutput, EvidenceItem
from src.evaluation.answer_repair import score_answer, summarize_repairs
from src.compensation.controller_adapter import controller_route_status


def read_jsonl(raw: bytes, name: str) -> list[dict]:
    records = []
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {name}:{number}; original file preserved") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {name}:{number}")
        records.append(record)
    return records


def runtime_input(record: dict, dataset: DatasetName | str) -> DiagnosisInput:
    """Explicit allowlist: no golds, metrics, C3, or judge fields reach E."""
    dataset = DatasetName(dataset)
    source_dataset = {"natural_questions": "nq", "nq": "nq", "fever": "fever"}.get(record.get("dataset"))
    if source_dataset != dataset.value or record.get("condition") != "C1":
        raise ValueError("Input must contain C1 records from the requested dataset only")
    if record.get("model_name") != MODEL_NAME:
        raise ValueError("C1 source must use the shared Qwen2.5-7B-Instruct model")
    evidence = record["evidence"]
    if not isinstance(evidence, list) or any(not isinstance(e, dict) for e in evidence):
        raise ValueError("C1 evidence must be a list of passage objects")
    return DiagnosisInput(
        sample_id=record["sample_id"], dataset=dataset,
        query=record["question_or_claim"], initial_answer=record["prediction"],
        evidence=[EvidenceItem(rank=i, passage_id=e["passage_id"], text=e["text"], score=e["score"])
                  for i, e in enumerate(evidence, 1)],
    )


def prepare_records(records: list[dict], dataset: DatasetName) -> list[tuple[dict, DiagnosisInput]]:
    prepared, seen = [], set()
    for record in records:
        request = runtime_input(record, dataset)
        if request.sample_id in seen:
            raise ValueError(f"Duplicate C1 sample ID: {request.sample_id}")
        seen.add(request.sample_id)
        # The original metric is recomputed offline before running the model.
        metrics = score_answer(request.initial_answer, record, dataset)
        if not isinstance(record.get("correct"), bool) or metrics["correct"] != record["correct"]:
            raise ValueError(f"Stored C1 correctness disagrees with the frozen scorer: {request.sample_id}")
        prepared.append((record, request))
    if not prepared:
        raise ValueError("No C1 records")
    return prepared


def read_diagnoses(path: Path, dataset: DatasetName) -> dict[str, DiagnosisOutput]:
    output = {}
    for row in read_jsonl(path.read_bytes(), str(path)):
        if row.get("dataset") != dataset.value:
            continue
        sample_id = str(row["sample_id"]).strip()
        if not sample_id or sample_id in output:
            raise ValueError(f"Empty or duplicate diagnosis ID: {sample_id!r}")
        output[sample_id] = DiagnosisOutput(
            diagnosis_label=row["diagnosis_label"], confidence=row["confidence"],
            needs_probe=row["needs_probe"],
        )
    return output


def route_status(diagnosis: DiagnosisOutput | None, all_samples: bool) -> str | None:
    """Minimal interface adapter. Confidence/Probe policy remains with Controller."""
    if all_samples:
        return None
    if diagnosis is None:
        raise ValueError("Missing diagnosis; use --all only for the explicit unrouted ablation")
    if diagnosis.needs_probe:
        return "skipped_pending_probe"
    if diagnosis.diagnosis_label != DiagnosisLabel.EVIDENCE_UTILISATION:
        return "skipped_other_diagnosis"
    return None


def read_controller_decisions(path: Path, dataset: DatasetName) -> dict:
    required = {"sample_id", "action", "diagnosis_label", "diagnosis_confidence", "used_probe",
                "probe_overrode_diagnosis", "needs_compensation"}
    output = {}
    for row in read_jsonl(path.read_bytes(), str(path)):
        if not required <= row.keys() or row.keys() - required - {"dataset"}:
            raise ValueError("Expected only resolved ControllerDecision.to_dict() runtime fields")
        if row.get("dataset", dataset.value) != dataset.value:
            continue
        sample_id = str(row["sample_id"]).strip() if row["sample_id"] is not None else ""
        if not sample_id or sample_id in output:
            raise ValueError("Empty or duplicate Controller sample ID")
        output[sample_id] = SimpleNamespace(**{key: row[key] for key in required})
    return output


def run_fingerprint(source: bytes, diagnosis_source: bytes, settings: dict) -> str:
    h = hashlib.sha256()
    for part in (source, diagnosis_source, json.dumps(settings, sort_keys=True).encode()):
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    for path in (
        "scripts/run_answer_repair.py", "src/compensation/answer_repair.py",
        "src/compensation/qwen_repair.py", "src/evaluation/answer_repair.py",
        "src/evaluation/qa.py", "src/evaluation/fever.py", "src/diagnosis/schema.py",
        "src/diagnosis/labels.py", "src/generation/qwen.py", "src/generation/__init__.py",
        "src/config.py", "configs/base.yaml",
        "src/compensation/controller_adapter.py",
        "src/compensation/support_check.py",
        "src/compensation/extractive_repair.py", "src/compensation/nli_repair.py",
        "src/compensation/anchors.py", "src/compensation/short_answer_repair.py",
        "src/compensation/conservative_repair.py",
    ):
        h.update(path.encode())
        h.update((ROOT / path).read_bytes())
    return h.hexdigest()


def result_record(
    source: dict, request: DiagnosisInput, result: RepairResult, fingerprint: str, settings: dict,
) -> dict:
    return {
        "sample_id": request.sample_id, "dataset": request.dataset.value, "condition": "E",
        "model_name": settings.get("model_name", MODEL_NAME), "run_fingerprint": fingerprint, "settings": settings,
        "query": request.query,
        "retrieved_evidence_ids": [item.passage_id for item in request.evidence],
        **asdict(result), "changed": result.changed,
        "initial_metrics": score_answer(request.initial_answer, source, request.dataset),
        "final_metrics": score_answer(result.final_answer, source, request.dataset),
    }


def read_existing(path: Path, prepared: list[tuple[dict, DiagnosisInput]], fingerprint: str) -> list[dict]:
    if not path.exists():
        return []
    rows = read_jsonl(path.read_bytes(), str(path))
    if len(rows) > len(prepared):
        raise ValueError("Output contains more samples than this run")
    for row, (source, request) in zip(rows, prepared):
        if (row.get("run_fingerprint") != fingerprint or row.get("sample_id") != request.sample_id
                or row.get("dataset") != request.dataset.value or row.get("condition") != "E"
                or row.get("initial_answer") != request.initial_answer):
            raise ValueError("Output fingerprint/ID/order differs; use a separate output file")
        if (row.get("initial_metrics") != score_answer(request.initial_answer, source, request.dataset)
                or row.get("final_metrics") != score_answer(row["final_answer"], source, request.dataset)
                or row.get("changed") != (row["final_answer"] != request.initial_answer)):
            raise ValueError("Saved output metrics are inconsistent; original file preserved")
        strategy = row.get("settings", {}).get("strategy")
        call_limit = max_calls(strategy) if strategy in V8_STRATEGIES else (2 if strategy == "guarded_reanswer" else 1)
        if (type(row.get("extra_calls")) is not int or not 0 <= row["extra_calls"] <= call_limit
                or not math.isfinite(row["latency"]) or row["latency"] < 0):
            raise ValueError("Invalid saved cost record")
        if strategy not in V8_STRATEGIES and (row.get("support_check") is not None) != (row["extra_calls"] == 2):
            raise ValueError("Saved support check and call count disagree")
        if strategy in V8_STRATEGIES:
            trace = row.get("repair_trace")
            if trace is not None:
                if trace.get("strategy") != strategy or trace.get("max_new_tokens") != row["settings"]["max_new_tokens"]:
                    raise ValueError("Saved v8 settings/trace mismatch")
                expected = asdict(repair_from_v8_trace(request, trace))
                if any(row.get(k) != v for k, v in json.loads(json.dumps(expected)).items()):
                    raise ValueError("Saved v8 decision/trace/cost mismatch")
            elif row["extra_calls"] or not row["status"].startswith("skipped_"):
                raise ValueError("Missing v8 trace")
        if strategy in NLI_STRATEGIES:
            trace = row.get("verifier_trace")
            if trace is not None:
                from src.compensation.nli_repair import repair_from_nli_trace, repair_from_nli_trace_anchored
                decide = repair_from_nli_trace if strategy == "fever_nli" else repair_from_nli_trace_anchored
                expected = asdict(decide(request, trace))
                if any(row.get(k) != v for k, v in expected.items()):
                    raise ValueError("Saved NLI decision/trace/cost mismatch")
            elif row.get("verifier_calls", 0) != 0 or not row["status"].startswith("skipped_"):
                raise ValueError("Missing NLI trace")
            if row["extra_calls"] != 0:
                raise ValueError("NLI must not claim Qwen generation calls")
    return rows


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A complete final JSON line may lack a newline after interruption.
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def validate_runtime_environment(saved: list[dict], current: dict | None) -> None:
    """Prevent resumed model calls from mixing revisions or GPU/package settings."""
    current = json.loads(json.dumps(current, sort_keys=True, allow_nan=False))
    if any(row.get("runtime_environment") != current for row in saved if row["extra_calls"] > 0 or row.get("verifier_calls", 0) > 0):
        raise ValueError("Runtime environment/model revision differs from saved calls; use a separate output file")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("nq", "fever"), required=True)
    source_args = parser.add_mutually_exclusive_group(required=True)
    source_args.add_argument("--c1", type=Path)
    source_args.add_argument("--source-ref", help="Read archived C1 directly from a local Git ref")
    route_args = parser.add_mutually_exclusive_group(required=True)
    route_args.add_argument("--all", action="store_true", help="Explicit development ablation; no diagnosis")
    route_args.add_argument("--diagnoses", type=Path, help="Legacy diagnosis-only development ablation; not final Controller routing")
    route_args.add_argument("--controller-decisions", type=Path, help="Resolved ControllerDecision.to_dict() JSONL, covering evaluated IDs")
    parser.add_argument("--strategy", choices=STRATEGIES, default="verify_repair")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, help="Evaluate a fixed prefix; use a separate output for each cohort")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Preflight plus first prompt; saves no predictions")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Token budget and limit must be positive")
    dataset = DatasetName(args.dataset)
    is_nli = args.strategy in NLI_STRATEGIES
    if (is_nli and dataset != DatasetName.FEVER or
            args.strategy in NQ_ONLY_STRATEGIES and dataset != DatasetName.NQ):
        parser.error("The selected strategy does not support this dataset")
    if args.c1:
        raw, source_name = args.c1.read_bytes(), str(args.c1.resolve())
    else:
        # Separate argv elements prevent shell interpretation of ref names.
        ref = subprocess.check_output(["git", "rev-parse", "--verify", args.source_ref + "^{commit}"], cwd=ROOT, text=True).strip()
        source_name = f"{ref}:results/preliminary/{args.dataset}_c1_7b.jsonl"
        raw = subprocess.check_output(["git", "show", source_name], cwd=ROOT)
    prepared = prepare_records(read_jsonl(raw, source_name), dataset)
    if args.limit:
        prepared = prepared[:args.limit]
    route_file = args.controller_decisions or args.diagnoses
    diagnosis_source = route_file.read_bytes() if route_file else b""
    diagnoses = read_diagnoses(args.diagnoses, dataset) if args.diagnoses else {}
    if args.controller_decisions:
        decisions = read_controller_decisions(args.controller_decisions, dataset)
        statuses = {request.sample_id: controller_route_status(request, decisions.get(request.sample_id))
                    for _, request in prepared}
    else:
        statuses = {request.sample_id: route_status(diagnoses.get(request.sample_id), args.all)
                    for _, request in prepared}
    settings = {
        "dataset": dataset.value, "strategy": args.strategy,
        "routing": "all" if args.all else ("controller" if args.controller_decisions else "diagnosis_legacy_ablation"),
        "max_new_tokens": args.max_new_tokens, "model_name": NLI_MODEL if is_nli else MODEL_NAME,
        "quantization": None if is_nli else "nf4", "do_sample": False, "seed": 42,
        "model_revision": NLI_REVISION if is_nli else MODEL_REVISION,
        "prompt_version": {"verify_repair": REPAIR_PROMPT_VERSION,
                           "grounded_reanswer": GROUNDED_REANSWER_PROMPT_VERSION,
                           "query_grounded": QUERY_GROUNDED_PROMPT_VERSION,
                           "guarded_reanswer": GUARDED_REANSWER_PROMPT_VERSION,
                           "reanswer": "c1_reanswer", "extractive_reanswer": EXTRACTIVE_PROMPT_VERSION,
                           "fever_nli": NLI_VERSION, "short_answer_quote": SHORT_ANSWER_PROMPT_VERSION,
                           "fever_nli_anchored": NLI_ANCHORED_VERSION,
                           **{strategy: V8_VERSION for strategy in V8_STRATEGIES}}[args.strategy],
        "support_check_max_new_tokens": SUPPORT_CHECK_MAX_NEW_TOKENS if args.strategy == "guarded_reanswer" else 0,
        "max_calls_per_sample": (max_calls(args.strategy) if args.strategy in V8_STRATEGIES else
                                 0 if is_nli else (2 if args.strategy == "guarded_reanswer" else 1)),
        "nli_threshold": NLI_THRESHOLD if is_nli else None,
        "cost_policy": "separate_classifier_forward_batches" if is_nli else "qwen_generation_calls",
        "json_stop": None if is_nli or args.strategy in LINE_OUTPUT_STRATEGIES else "complete_object_final_token_framing",
        "citation_matching": ("exact_source_character_window" if is_nli else
                              "whitespace_insensitive_quote_normalised_token_boundaries" if args.strategy in ("short_answer_quote", *V8_STRATEGIES)
                              else "whole_words_and_punctuation_ignore_spacing"),
        "answer_span_matching": ("canonical_source_slices" if args.strategy == "extractive_reanswer" else
                                "whitespace_insensitive_slice_inside_quote" if args.strategy in ("short_answer_quote", *V8_STRATEGIES) else
                                None if is_nli else "whole_words_and_punctuation_ignore_spacing" if args.strategy in
                                ("query_grounded", "guarded_reanswer") else "legacy_whitespace_casefold"),
        "anchor_policy": ({"version": ANCHOR_VERSION, "fever": FEVER_POLICY}
                          if args.strategy == "fever_nli_anchored" else
                          {"version": ANCHOR_VERSION, "nq_min_coverage_inclusive": NQ_MIN_COVERAGE}
                          if args.strategy == "short_answer_quote" else None),
        "sample_count": len(prepared),
        "sample_ids_sha256": hashlib.sha256(json.dumps([r.sample_id for _, r in prepared]).encode()).hexdigest(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source": source_name,
    }
    fingerprint = run_fingerprint(raw, diagnosis_source, settings)
    out = args.out or ROOT / "results" / "answer_repair" / f"{args.dataset}_{args.strategy}_{settings['routing']}.jsonl"
    if args.c1 and out.resolve() == args.c1.resolve():
        raise ValueError("Output must not overwrite C1 input")
    if route_file and out.resolve() == route_file.resolve():
        raise ValueError("Output must not overwrite routing input")
    if out.resolve().is_relative_to((ROOT / "results" / "preliminary").resolve()):
        raise ValueError("E output must be outside the frozen results/preliminary archive")
    saved = read_existing(out, prepared, fingerprint)
    plan = {
        "source": source_name, "dataset": dataset.value, "strategy": args.strategy,
        "routing": settings["routing"], "samples": len(prepared), "completed": len(saved),
        "routed_samples": sum(status is None for status in statuses.values()),
        "remaining_call_upper_bound": settings["max_calls_per_sample"] * sum(
            statuses[r.sample_id] is None for _, r in prepared[len(saved):]),
        "remaining_classifier_samples": sum(statuses[r.sample_id] is None for _, r in prepared[len(saved):]) if is_nli else 0,
        "fingerprint": fingerprint, "output": str(out),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.check_only or args.dry_run:
        if args.dry_run:
            build = {"extractive_reanswer": build_extractive_prompt, "fever_nli": None, "fever_nli_anchored": None,
                     "verify_repair": build_repair_prompt, "grounded_reanswer": build_grounded_reanswer_prompt,
                     "query_grounded": build_query_grounded_prompt, "guarded_reanswer": build_query_grounded_prompt,
                     "reanswer": build_reanswer_prompt, "short_answer_quote": build_short_answer_prompt,
                     **{strategy: (build_minimal_repair_prompt if strategy == "conservative_no_original_gate"
                                    else build_check_prompt) for strategy in V8_STRATEGIES}}[args.strategy]
            print(build(prepared[0][1]) if build else "NLI: full evidence windows as premises; unmodified claim as hypothesis.")
        return 0
    # Lock before rereading output or loading a model. One writer per file.
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_suffix(out.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"Output lock exists: {lock}. Confirm no writer is active before removing it.") from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        saved = read_existing(out, prepared, fingerprint)
        generator = None
        verifier = None
        for source, request in prepared[len(saved):]:
            status = statuses[request.sample_id]
            if status is not None:
                result = RepairResult(request.initial_answer, request.initial_answer, status)
            else:
                if is_nli:
                    if verifier is None:
                        from src.compensation.nli_repair import MNLIVerifier
                        verifier = MNLIVerifier()
                        validate_runtime_environment(saved, verifier.runtime_environment)
                elif generator is None:
                    from src.compensation.qwen_repair import QwenRepairGenerator
                    generator = QwenRepairGenerator(json_output=args.strategy not in LINE_OUTPUT_STRATEGIES)
                    validate_runtime_environment(saved, getattr(generator, "runtime_environment", None))
                result = repair_answer(request, generator, verifier=verifier, strategy=args.strategy, max_new_tokens=args.max_new_tokens)
            row = result_record(source, request, result, fingerprint, settings)
            row["runtime_environment"] = getattr(verifier if is_nli else generator, "runtime_environment", None) if (result.extra_calls or result.verifier_calls) else None
            append_record(out, row)
            saved.append(row)
            print(f"[{len(saved)}/{len(prepared)}] {request.sample_id}: {result.status}", flush=True)
        summary = {"run_fingerprint": fingerprint, "routing": settings["routing"], "strategy": args.strategy,
                   **summarize_repairs(saved)}
        summary_path = out.with_suffix(".summary.json")
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(summary_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        lock.unlink()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
