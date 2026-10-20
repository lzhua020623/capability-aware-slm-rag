"""Generate frozen C1 answers for all NQ Train Top-5 retrieval records.

Frozen protocol (must match Dev Reference RAG C1):
  model: Qwen/Qwen2.5-7B-Instruct
  quantization: 4-bit NF4 (bitsandbytes)
  prompt: src.generation.qwen.build_rag_prompt / PROMPT_INSTRUCTION
  chat template: user-only apply_chat_template(add_generation_prompt=True)
  decoding: generate_greedy (do_sample=False, max_new_tokens=64)

Gold answers, gold passage ids, and labels never enter the Qwen prompt.
Supports resume by skipping sample_ids already present in the output JSONL.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_generator_config  # noqa: E402
from src.generation.qwen import (  # noqa: E402
    PROMPT_INSTRUCTION,
    build_rag_prompt,
    generate_greedy,
    load_qwen_nf4,
)

INPUT_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_train_bge_top5.jsonl"
OUTPUT_PATH = ROOT / "data" / "processed" / "diagnosis" / "nq_train_c1_outputs.jsonl"
FAILED_PATH = ROOT / "data" / "processed" / "diagnosis" / "nq_train_c1_failed.jsonl"
PROGRESS_PATH = ROOT / "data" / "processed" / "diagnosis" / "nq_train_c1_progress.json"
ANNOTATION_PATH = ROOT / "data" / "annotations" / "nq_train_annotation_candidates.jsonl"
STATS_PATH = ROOT / "data" / "annotations" / "nq_train_annotation_candidates_stats.json"
RETRIEVAL_STATS = ROOT / "data" / "processed" / "retrieval" / "nq_train_bge_top5_stats.json"

EXPECTED = 74_097
SMOKE_SEED = 42
SMOKE_N = 100
# Structural field-name leakage only. Natural language in evidence (e.g. the
# word "correctness" in a Wikipedia passage) must not block generation.
FORBIDDEN_FIELD_PATTERNS = (
    r"(?i)\bgold[_\s-]?answer\b\s*:",
    r"(?i)\bshort[_\s-]?answer\b\s*:",
    r"(?i)\bgold[_\s-]?evidence\b\s*:",
    r"(?i)\bgold[_\s-]?passage[_\s-]?id\b\s*:",
    r"(?i)\bgold[_\s-]?in[_\s-]?top\d*\b\s*:",
    r"(?i)\btarget[_\s-]?label\b\s*:",
    r"(?i)\bcorrectness\b\s*:",
    r"(?i)\bexact[_\s-]?match\b\s*:",
    r"(?i)\b\bem\b\s*:",
    r"(?i)\b\bf1\b\s*:",
)


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in done:
                raise RuntimeError(f"Duplicate sample_id {sample_id} in {path} line {line_no}")
            done.add(sample_id)
    return done


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def evidence_for_prompt(retrieved_passages: list[dict]) -> list[dict]:
    """Runtime evidence for the generator: text only, rank order preserved."""
    if len(retrieved_passages) != 5:
        raise ValueError(f"expected 5 passages, got {len(retrieved_passages)}")
    ordered = sorted(retrieved_passages, key=lambda item: int(item["rank"]))
    if [int(item["rank"]) for item in ordered] != [1, 2, 3, 4, 5]:
        raise ValueError(f"bad ranks {[item.get('rank') for item in ordered]}")
    return [{"text": item["text"]} for item in ordered]


def top5_for_output(retrieved_passages: list[dict]) -> list[dict]:
    ordered = sorted(retrieved_passages, key=lambda item: int(item["rank"]))
    return [
        {
            "rank": int(item["rank"]),
            "passage_id": item["passage_id"],
            "score": float(item["score"]),
            "text": item["text"],
        }
        for item in ordered
    ]


def assert_no_leakage(prompt: str, record: dict) -> None:
    """Reject structural gold/eval field injection; allow coincidental evidence text."""
    for pattern in FORBIDDEN_FIELD_PATTERNS:
        match = re.search(pattern, prompt)
        if match:
            raise RuntimeError(f"Forbidden field leakage matched {pattern!r}: {match.group(0)!r}")
    # Gold short answers may appear verbatim inside retrieved evidence; that is not leakage.
    if "gold_passage_id" in prompt or "gold_in_top" in prompt:
        raise RuntimeError("Gold metadata key leaked into prompt")
    _ = record


def generation_config_snapshot() -> dict:
    generator = load_generator_config()
    return {
        "model_name": str(generator["model_name"]),
        "quantization": str(generator.get("quantization") or "nf4"),
        "load_in_4bit": bool(generator.get("load_in_4bit", True)),
        "do_sample": False,
        "max_new_tokens": int(generator.get("max_new_tokens", 64)),
        "prompt_instruction": PROMPT_INSTRUCTION,
        "prompt_builder": "src.generation.qwen.build_rag_prompt",
        "chat_template": "tokenizer.apply_chat_template(user-only, add_generation_prompt=True)",
        "batch_size": 1,
    }


def write_progress(completed: int, failed: int, started: float) -> None:
    elapsed = time.time() - started
    remaining = max(EXPECTED - completed, 0)
    rate = completed / elapsed if elapsed > 0 and completed else 0.0
    eta = remaining / rate if rate > 0 else None
    payload = {
        "completed": completed,
        "failed": failed,
        "expected": EXPECTED,
        "remaining": remaining,
        "elapsed_seconds": round(elapsed, 2),
        "rate_per_second": round(rate, 4) if rate else None,
        "eta_seconds": round(eta, 1) if eta is not None else None,
    }
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def validate_input_record(record: dict) -> None:
    if not str(record.get("query") or "").strip():
        raise ValueError("empty query")
    passages = record.get("retrieved_passages")
    if not isinstance(passages, list) or len(passages) != 5:
        raise ValueError("retrieved_passages must have length 5")
    for item in passages:
        if not str(item.get("text") or "").strip():
            raise ValueError("empty evidence text")


def run_generation(
    *,
    limit: int | None,
    sample_ids: set[str] | None,
    smoke: bool,
    retry_failed: bool,
) -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: missing {INPUT_PATH}")
        return 1

    cfg = generation_config_snapshot()
    print("Frozen C1 configuration:")
    print(json.dumps(cfg, indent=2))
    if cfg["model_name"] != "Qwen/Qwen2.5-7B-Instruct":
        print("ERROR: unexpected generator model")
        return 1
    if cfg["quantization"].lower() != "nf4" or not cfg["load_in_4bit"]:
        print("ERROR: expected NF4 4-bit loading")
        return 1

    done = set() if smoke else load_completed_ids(OUTPUT_PATH)
    if not smoke and FAILED_PATH.exists():
        # Drop stale failure rows whose sample_id later succeeded.
        remaining_failures = []
        with FAILED_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                err = json.loads(line)
                if str(err["sample_id"]) not in done:
                    remaining_failures.append(err)
        with FAILED_PATH.open("w", encoding="utf-8", newline="\n") as handle:
            for err in remaining_failures:
                handle.write(json.dumps(err, ensure_ascii=False) + "\n")
        if retry_failed:
            print(f"Retry mode: pending failed ids={len(remaining_failures)}")

    # Optional smoke sample selection.
    selected_ids: set[str] | None = sample_ids
    if smoke and selected_ids is None:
        all_ids = []
        with INPUT_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    all_ids.append(str(json.loads(line)["sample_id"]))
        if len(all_ids) != EXPECTED:
            print(f"WARNING: input has {len(all_ids)} records, expected {EXPECTED}")
        selected_ids = set(random.Random(SMOKE_SEED).sample(all_ids, min(SMOKE_N, len(all_ids))))
        print(f"Smoke sample_ids={len(selected_ids)} seed={SMOKE_SEED}")

    out_path = (
        ROOT / "data" / "processed" / "diagnosis" / "nq_train_c1_smoke.jsonl"
        if smoke
        else OUTPUT_PATH
    )
    if smoke and out_path.exists():
        out_path.unlink()
    smoke_done: set[str] = set()

    print("Loading Qwen2.5-7B-Instruct NF4 ...")
    tokenizer, model = load_qwen_nf4(cfg["model_name"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    started = time.time()
    generated = 0
    failed = 0
    leakage_checked = 0
    empty_answers = 0

    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if selected_ids is not None and sample_id not in selected_ids:
                continue
            if not smoke and sample_id in done:
                continue
            if limit is not None and generated >= limit:
                break

            try:
                validate_input_record(record)
                passages = evidence_for_prompt(record["retrieved_passages"])
                prompt = build_rag_prompt(record["query"], passages)
                assert_no_leakage(prompt, record)
                leakage_checked += 1
                prediction, latency = generate_greedy(
                    tokenizer,
                    model,
                    prompt,
                    max_new_tokens=cfg["max_new_tokens"],
                )
                if not isinstance(prediction, str):
                    raise TypeError("prediction must be str")
                if prediction.strip() == "":
                    empty_answers += 1
                latencies.append(float(latency))
                output = {
                    "sample_id": record["sample_id"],
                    "query": record["query"],
                    "c1_prediction": prediction,
                    "top5_evidence": top5_for_output(record["retrieved_passages"]),
                    "generation_metadata": {
                        **cfg,
                        "latency_seconds": float(latency),
                        "prompt_char_len": len(prompt),
                    },
                }
                append_jsonl(out_path, output)
                if smoke:
                    smoke_done.add(sample_id)
                else:
                    done.add(sample_id)
                generated += 1
                completed_total = len(smoke_done) if smoke else len(done)
                if generated % 10 == 0 or smoke:
                    peak = (
                        torch.cuda.max_memory_allocated() / (1024**2)
                        if torch.cuda.is_available()
                        else None
                    )
                    print(
                        f"[{completed_total}/{'100' if smoke else EXPECTED}] "
                        f"id={sample_id} latency={latency:.2f}s "
                        f"pred_len={len(prediction)} peak_vram_mib={peak}",
                        flush=True,
                    )
                if not smoke:
                    write_progress(len(done), failed, started)
            except Exception as exc:  # noqa: BLE001 - per-sample isolation required
                failed += 1
                err = {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "line_no": line_no,
                }
                if smoke:
                    print(f"SMOKE FAIL {sample_id}: {exc}")
                    return 1
                append_jsonl(FAILED_PATH, err)
                print(f"FAIL {sample_id}: {type(exc).__name__}: {exc}", flush=True)
                write_progress(len(done), failed, started)

    elapsed = time.time() - started
    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
    )
    summary = {
        "mode": "smoke" if smoke else "full",
        "generated_this_run": generated,
        "failed_this_run": failed,
        "completed_total": len(smoke_done) if smoke else len(done),
        "empty_answers_this_run": empty_answers,
        "leakage_checks": leakage_checked,
        "elapsed_seconds": round(elapsed, 2),
        "mean_latency": (sum(latencies) / len(latencies)) if latencies else None,
        "median_latency": (sorted(latencies)[len(latencies) // 2] if latencies else None),
        "p95_latency": (
            sorted(latencies)[max(int(math.ceil(0.95 * len(latencies)) - 1), 0)]
            if latencies
            else None
        ),
        "peak_vram_mib": peak_vram,
        "batch_size": 1,
        "output_path": str(out_path.relative_to(ROOT).as_posix()),
        "config": cfg,
    }
    if smoke and latencies:
        mean = summary["mean_latency"]
        summary["estimated_seconds_for_74097"] = round(mean * EXPECTED, 1)
        summary["estimated_hours_for_74097"] = round(mean * EXPECTED / 3600, 2)
    print(json.dumps(summary, indent=2))

    if smoke:
        if generated != min(SMOKE_N, EXPECTED) or failed:
            print("SMOKE FAILED")
            return 1
        print("SMOKE PASSED")
        return 0

    # Drop stale failure rows for IDs that succeeded (including prior false positives).
    if FAILED_PATH.exists():
        remaining_failures = []
        with FAILED_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                err = json.loads(line)
                if str(err["sample_id"]) not in done:
                    remaining_failures.append(err)
        with FAILED_PATH.open("w", encoding="utf-8", newline="\n") as handle:
            for err in remaining_failures:
                handle.write(json.dumps(err, ensure_ascii=False) + "\n")
        write_progress(len(done), len(remaining_failures), started)
        return 0 if not remaining_failures else 1
    return 0 if failed == 0 else 1


def build_annotation_ready() -> int:
    if not OUTPUT_PATH.exists():
        print(f"ERROR: missing {OUTPUT_PATH}")
        return 1
    if ANNOTATION_PATH.exists() or STATS_PATH.exists():
        print("ERROR: annotation outputs already exist; refusing to overwrite")
        return 1

    # Map gold/retrieval reference from Top-5 file.
    ref_by_id: dict[str, dict] = {}
    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            ref_by_id[str(rec["sample_id"])] = rec

    n = 0
    empty = 0
    long_out = 0
    preds: Counter[str] = Counter()
    with OUTPUT_PATH.open("r", encoding="utf-8") as src, ANNOTATION_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            sample_id = str(rec["sample_id"])
            top5 = ref_by_id[sample_id]
            pred = rec["c1_prediction"]
            preds[pred] += 1
            if pred.strip() == "":
                empty += 1
            if len(pred) > 200:
                long_out += 1
            evidence = [
                {
                    "rank": int(item["rank"]),
                    "passage_id": item["passage_id"],
                    "text": item["text"],
                    "score": float(item["score"]),
                }
                for item in sorted(rec["top5_evidence"], key=lambda x: int(x["rank"]))
            ]
            gold_pid = top5.get("gold_passage_id")
            gold_text = None
            for item in evidence:
                if item["passage_id"] == gold_pid:
                    gold_text = item["text"]
                    break
            out = {
                "sample_id": rec["sample_id"],
                "dataset": "nq",
                "query": rec["query"],
                "initial_answer": pred,
                "evidence": evidence,
                "annotation_reference": {
                    "gold_answer": top5.get("short_answer"),
                    "gold_evidence_text": gold_text,
                    "gold_passage_id": gold_pid,
                    "gold_in_top1": top5.get("gold_in_top1"),
                    "gold_in_top3": top5.get("gold_in_top3"),
                    "gold_in_top5": top5.get("gold_in_top5"),
                },
                "label": None,
            }
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    if n != EXPECTED:
        print(f"ERROR: annotation candidates={n}, expected {EXPECTED}")
        return 1

    retrieval_stats = {}
    if RETRIEVAL_STATS.exists():
        retrieval_stats = json.loads(RETRIEVAL_STATS.read_text(encoding="utf-8"))
    cfg = generation_config_snapshot()
    completed = load_completed_ids(OUTPUT_PATH)
    latencies = []
    with OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            meta = json.loads(line).get("generation_metadata") or {}
            if "latency_seconds" in meta:
                latencies.append(float(meta["latency_seconds"]))

    stats = {
        "raw_nq_train_count": 307_373,
        "controlled_count": EXPECTED,
        "retrieval_corpus_count": 1_224_119,
        "retrieval_recall_at_1": retrieval_stats.get("recall_at_1"),
        "retrieval_recall_at_3": retrieval_stats.get("recall_at_3"),
        "retrieval_recall_at_5": retrieval_stats.get("recall_at_5"),
        "c1_generated_count": len(completed),
        "generation_model": cfg["model_name"],
        "quantization": cfg["quantization"],
        "decoding": "greedy do_sample=False",
        "max_new_tokens": cfg["max_new_tokens"],
        "mean_generation_latency": (sum(latencies) / len(latencies)) if latencies else None,
        "total_generation_time": sum(latencies) if latencies else None,
        "failed_count": len(load_completed_ids(FAILED_PATH)),
        "annotation_candidate_count": n,
        "empty_answer_count": empty,
        "very_long_output_count_gt_200_chars": long_out,
        "exact_duplicate_prediction_groups": sum(1 for c in preds.values() if c > 1),
        "c1_output_path": str(OUTPUT_PATH.relative_to(ROOT).as_posix()),
        "annotation_path": str(ANNOTATION_PATH.relative_to(ROOT).as_posix()),
        "diagnosis_input_compatible_fields": [
            "sample_id",
            "dataset",
            "query",
            "initial_answer",
            "evidence.rank",
            "evidence.passage_id",
            "evidence.text",
            "evidence.score",
        ],
        "annotation_reference_fields": [
            "gold_answer",
            "gold_evidence_text",
            "gold_passage_id",
            "gold_in_top1",
            "gold_in_top3",
            "gold_in_top5",
        ],
    }
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {ANNOTATION_PATH}")
    print(f"Wrote {STATS_PATH}")
    return 0


def integrity_check() -> int:
    done = load_completed_ids(OUTPUT_PATH)
    failed = load_completed_ids(FAILED_PATH) if FAILED_PATH.exists() else set()
    expected_ids = set()
    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                expected_ids.add(str(json.loads(line)["sample_id"]))
    missing = sorted(expected_ids - done)
    extra = sorted(done - expected_ids)
    report = {
        "expected": len(expected_ids),
        "generated": len(done),
        "failed_file_ids": len(failed),
        "missing": len(missing),
        "extra": len(extra),
        "missing_sample": missing[:10],
    }
    print(json.dumps(report, indent=2))
    return 0 if len(done) == EXPECTED and not missing and not extra else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="100-sample smoke test")
    parser.add_argument("--run", action="store_true", help="full/resumable generation")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--build-annotation-ready", action="store_true")
    parser.add_argument("--integrity-check", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    if args.print_config:
        print(json.dumps(generation_config_snapshot(), indent=2))
        return 0
    if args.build_annotation_ready:
        return build_annotation_ready()
    if args.integrity_check:
        return integrity_check()
    if args.smoke:
        return run_generation(limit=None, sample_ids=None, smoke=True, retry_failed=False)
    if args.run:
        return run_generation(
            limit=args.limit,
            sample_ids=None,
            smoke=False,
            retry_failed=args.retry_failed,
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
