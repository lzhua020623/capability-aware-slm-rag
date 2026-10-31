#!/usr/bin/env python3
"""Formal Adaptive Product evaluation runner.

Inference path (only):
  load_current_pipeline → CurrentPipeline.run(query, sample_id=…) → final_answer

Scoring / split / metrics I/O are reused from baselines/_lib (same as baseline-evaluation).
Does not call run_baseline_evaluation.py for product answers.
Does not inject Diagnosis / Selection / Controller decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines._lib.answer_scoring import (  # noqa: E402
    aggregate_answer_metrics,
    answer_in_context,
    score_answer,
)
from baselines._lib.metrics_io import (  # noqa: E402
    append_prediction,
    load_done_ids,
    read_predictions,
    write_json,
)
from baselines._lib.records_io import load_records_by_ids  # noqa: E402
from baselines._lib.split_io import (  # noqa: E402
    SplitError,
    extract_split_ids,
    load_split_json,
    validate_eval_split_file,
)

BASELINES = Path(__file__).resolve().parent
MODE = "adaptive_product"
DEFAULT_CONFIG = BASELINES / "configs" / "adaptive_product.json"
FORBIDDEN_OUTPUT_MARKERS = (
    "outputs/qwen_only",
    "outputs/rag_matched",
    "outputs/rag/",
    "outputs\\qwen_only",
    "outputs\\rag_matched",
    "outputs\\rag\\",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path


def relpath(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_product_config(path: Path | None = None) -> dict:
    cfg_path = resolve_path(path) if path else DEFAULT_CONFIG
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("mode") != MODE:
        raise ValueError(f"Config mode must be {MODE!r}, got {cfg.get('mode')!r}")
    cfg["_config_path"] = relpath(cfg_path)
    return cfg


def assert_safe_output_dir(output_dir: Path) -> None:
    text = relpath(output_dir).replace("\\", "/")
    for marker in FORBIDDEN_OUTPUT_MARKERS:
        marker_norm = marker.replace("\\", "/")
        if marker_norm in text:
            raise SystemExit(
                f"Refusing to write product evaluation into baseline output dir: {text}"
            )


def evidence_hits(evidence) -> list[dict]:
    hits = []
    for i, item in enumerate(evidence or (), start=1):
        hits.append(
            {
                "passage_id": getattr(item, "passage_id", None),
                "text": getattr(item, "text", "") or "",
                "score": getattr(item, "score", None),
                "rank": int(getattr(item, "rank", i) or i),
            }
        )
    return hits


def build_product_row(
    *,
    result,
    gold: list[str],
    prompt_style: str,
) -> dict:
    """Map AdaptiveRAGResult → baseline-compatible prediction row + product trace."""
    prediction = "" if result.final_answer is None else str(result.final_answer)
    scores = score_answer(prediction, gold)
    initial_hits = evidence_hits(result.initial_evidence)
    final_hits = evidence_hits(result.final_evidence)
    selection = result.selection_decision
    controller = result.controller_decision
    enhancement = result.enhancement_result
    diagnosis = result.diagnosis
    row = {
        "sample_id": str(result.sample_id) if result.sample_id is not None else None,
        "query": result.query,
        "prediction": prediction,
        "gold": gold,
        "mode": MODE,
        **scores,
        "initial_answer": result.initial_answer,
        "diagnosis_label": getattr(diagnosis.final_label, "value", str(diagnosis.final_label)),
        "selected": bool(selection.needs_enhancement),
        "controller_action": getattr(controller.action, "value", str(controller.action)),
        "enhancement_status": getattr(enhancement.status, "value", str(enhancement.status)),
        "retrieved_passage_ids": [h["passage_id"] for h in initial_hits],
        "retrieved_ranks": [int(h["rank"]) for h in initial_hits],
        "answer_in_context": answer_in_context(gold, initial_hits),
        "answer_in_context_final": answer_in_context(gold, final_hits),
        "rag_prompt_style": prompt_style,
    }
    return row


def aggregate_product_metrics(rows: list[dict]) -> dict:
    base = aggregate_answer_metrics(rows, mode=MODE)
    # Formal AIC@5 comes only from answer_in_context (initial_evidence).
    actions = Counter(str(r.get("controller_action") or "") for r in rows)
    selected_n = sum(1 for r in rows if r.get("selected"))
    final_yes = sum(1 for r in rows if int(r.get("answer_in_context_final") or 0) == 1)
    out = dict(base)
    out["selected_rate"] = (selected_n / len(rows)) if rows else 0.0
    out["selected_n"] = selected_n
    out["no_action_n"] = int(actions.get("no_action", 0))
    out["retrieval_enhancement_n"] = int(actions.get("retrieval_enhancement", 0))
    out["evidence_refinement_n"] = int(actions.get("evidence_refinement", 0))
    out["generation_repair_n"] = int(actions.get("generation_repair", 0))
    if rows:
        out["answer_in_context_at_5_final"] = final_yes / len(rows)
        out["answer_in_context_final_yes_n"] = final_yes
        out["answer_in_context_final_no_n"] = len(rows) - final_yes
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive Product formal evaluation")
    p.add_argument("--config", default=None, help="baselines/configs/adaptive_product.json")
    p.add_argument("--split-file", default=None)
    p.add_argument(
        "--eval-split",
        choices=("train", "validation", "test"),
        default=None,
        help="Frozen split role. Formal Test requires --allow-sealed-test.",
    )
    p.add_argument(
        "--allow-sealed-test",
        action="store_true",
        help="Explicit unlock for sealed Test execution (required when --eval-split test).",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dataset-file", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only with --smoke (Train/Val engineering smoke). Forbidden on formal Test.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Engineering smoke on Train/Val IDs (not formal Test). Writes smoke_dir by default.",
    )
    p.add_argument(
        "--validate-split-only",
        action="store_true",
        help="Validate frozen split + assets; no model inference.",
    )
    p.add_argument(
        "--nq-resources",
        default=None,
        help="Override NQ resources registry path.",
    )
    p.add_argument(
        "--feature-batch-size",
        type=int,
        default=8,
        help="P26/feature micro-batch size (Qwen remains serial). Default 8.",
    )
    return p.parse_args(argv)


def resolve_eval_split(args: argparse.Namespace, cfg: dict) -> str:
    if args.smoke:
        return args.eval_split or "train"
    if args.eval_split:
        return args.eval_split
    return str(cfg["split"].get("default_eval_split") or "test")


def enforce_sealed_gate(eval_split: str, allow_sealed_test: bool, *, smoke: bool) -> None:
    if eval_split == "test":
        if smoke:
            raise SystemExit("Refusing --smoke with --eval-split test; use formal Test path.")
        if not allow_sealed_test:
            raise SystemExit(
                "Sealed Test is locked. Pass --eval-split test --allow-sealed-test "
                "explicitly for formal product evaluation."
            )


def load_eval_ids(cfg: dict, split_path: Path, eval_split: str) -> list[str]:
    split_cfg = cfg["split"]
    prefix = list(split_cfg.get("split_key_prefix") or ["splits", "nq"])
    expected_map = split_cfg["expected_n"]
    if eval_split not in expected_map:
        raise SystemExit(f"Config expected_n missing {eval_split!r}")
    expected_n = int(expected_map[eval_split])
    split = load_split_json(split_path)
    return extract_split_ids(
        split,
        id_key=prefix + [eval_split],
        expected_n=expected_n,
        version_prefix=split_cfg.get("version_prefix"),
        split_name=eval_split,
    )


def formal_readiness(cfg: dict, split_path: Path, eval_split: str) -> dict:
    split_cfg = cfg["split"]
    prefix = list(split_cfg.get("split_key_prefix") or ["splits", "nq"])
    expected_n = int(split_cfg["expected_n"][eval_split])
    summary = validate_eval_split_file(
        split_path,
        id_key=prefix + [eval_split],
        expected_n=expected_n,
        split_name=eval_split,
        version_prefix=split_cfg.get("version_prefix"),
    )
    gaps = []
    ret = cfg.get("retrieval") or {}
    for key in ("index_path", "ids_path", "offsets_path", "corpus_path"):
        if key not in ret:
            continue
        path = resolve_path(ret[key])
        if not path.is_file():
            gaps.append(f"missing_{key}:{relpath(path)}")
    records_path = resolve_path(cfg["dataset"]["records_path"])
    if not records_path.is_file():
        gaps.append(f"missing_dataset:{relpath(records_path)}")
    ref_cfg = resolve_path(cfg["production"]["reference_rag_config"])
    if not ref_cfg.is_file():
        gaps.append(f"missing_reference_rag_config:{relpath(ref_cfg)}")
    status = "READY_FOR_FROZEN_TEST" if not gaps else "PRODUCT_ASSETS_MISSING"
    return {
        "status": status,
        "split": summary,
        "split_sha256": sha256_file(split_path),
        "eval_split": eval_split,
        "expected_n": expected_n,
        "gaps": gaps,
    }


def default_output_dir(args: argparse.Namespace, cfg: dict) -> Path:
    if args.output_dir:
        return resolve_path(args.output_dir)
    key = "smoke_dir" if args.smoke else "default_dir"
    return resolve_path(cfg["output"][key])


def build_run_config(
    *,
    cfg: dict,
    args: argparse.Namespace,
    eval_split: str,
    eval_n: int,
    split_path: Path,
    output_dir: Path,
    status: str,
) -> dict:
    ret = cfg.get("retrieval") or {}
    gen = cfg.get("generator") or {}
    return {
        "mode": MODE,
        "smoke": bool(args.smoke),
        "generated_at": utc_now(),
        "config_path": cfg.get("_config_path"),
        "split_file": relpath(split_path),
        "split_sha256": sha256_file(split_path),
        "dataset_file": relpath(resolve_path(args.dataset_file or cfg["dataset"]["records_path"])),
        "eval_split": eval_split,
        "eval_n": eval_n,
        "allow_sealed_test": bool(args.allow_sealed_test and eval_split == "test"),
        "rag_prompt_style": (cfg.get("rag_prompt") or {}).get("default_style"),
        "generator": {
            "model_name": gen.get("model_name"),
            "quantization": gen.get("quantization"),
            "do_sample": gen.get("do_sample"),
            "max_new_tokens": gen.get("max_new_tokens"),
        },
        "retrieval": {
            "embedding_model": ret.get("embedding_model"),
            "top_k": ret.get("top_k"),
            "backend": ret.get("backend"),
            "index_path": ret.get("index_path"),
            "ids_path": ret.get("ids_path"),
            "offsets_path": ret.get("offsets_path"),
            "corpus_path": ret.get("corpus_path"),
            "expected_ntotal": ret.get("expected_ntotal"),
        },
        "production": dict(cfg.get("production") or {}),
        "scoring": dict(cfg.get("scoring") or {}),
        "aic_definition": "answer_in_context on AdaptiveRAGResult.initial_evidence (Reference RAG Top-5)",
        "feature_batch_size": int(getattr(args, "feature_batch_size", 8) or 8),
        "output_dir": relpath(output_dir),
        "status": status,
    }


def run_validate_only(args: argparse.Namespace, cfg: dict) -> int:
    eval_split = resolve_eval_split(args, cfg)
    enforce_sealed_gate(eval_split, args.allow_sealed_test, smoke=False)
    split_path = resolve_path(args.split_file or cfg["split"]["default_path"])
    receipt = formal_readiness(cfg, split_path, eval_split)
    ids = load_eval_ids(cfg, split_path, eval_split) if receipt["status"] != "FROZEN_SPLIT_INVALID" else []
    payload = {
        **receipt,
        "id_count": len(ids),
        "inference": False,
        "mode": MODE,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "READY_FOR_FROZEN_TEST" and len(ids) == receipt["expected_n"] else 2


def run_inference(args: argparse.Namespace, cfg: dict) -> int:
    from src.data.team_nq import TeamNQBoundary
    from src.generation.qwen import free_cuda_memory
    from src.pipeline.current import load_current_pipeline

    eval_split = resolve_eval_split(args, cfg)
    enforce_sealed_gate(eval_split, args.allow_sealed_test, smoke=bool(args.smoke))
    if args.smoke and eval_split == "test":
        raise SystemExit("internal guard: smoke cannot target test")
    if (not args.smoke) and eval_split == "test" and args.limit is not None:
        raise SystemExit("Refusing --limit on formal Test; use full 11114 or --smoke.")

    split_path = resolve_path(args.split_file or cfg["split"]["default_path"])
    # Asset readiness always checks Test split schema + on-disk retrieval/dataset files.
    asset_receipt = formal_readiness(cfg, split_path, "test")
    if asset_receipt["status"] != "READY_FOR_FROZEN_TEST" and not args.smoke:
        raise SystemExit(json.dumps(asset_receipt, ensure_ascii=False, indent=2))

    ids = load_eval_ids(cfg, split_path, eval_split)
    allow_sealed = bool(args.allow_sealed_test and eval_split == "test")
    boundary = TeamNQBoundary(ROOT, allow_sealed_test=allow_sealed)
    if eval_split in boundary.eligible:
        allowed = boundary.eligible[eval_split]
        ids = [sid for sid in ids if sid in allowed]
    if not ids:
        raise SystemExit(f"No eligible IDs for role {eval_split!r} after boundary filter")
    if args.smoke:
        limit = args.limit if args.limit is not None else 3
        if limit < 1:
            raise SystemExit("--limit must be >= 1 for smoke")
        ids = ids[:limit]

    records_path = resolve_path(args.dataset_file or cfg["dataset"]["records_path"])
    ds = cfg["dataset"]
    records = load_records_by_ids(
        records_path,
        ids,
        id_field=ds.get("id_field", "input.sample_id"),
        query_field=ds.get("query_field", "input.query"),
        gold_field=ds.get("gold_field", "metadata.annotation_reference.gold_answer"),
    )

    output_dir = default_output_dir(args, cfg)
    assert_safe_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / cfg["output"]["predictions"]
    metrics_path = output_dir / cfg["output"]["metrics"]
    run_config_path = output_dir / cfg["output"]["run_config"]

    if not args.resume and pred_path.exists():
        pred_path.unlink()

    done = load_done_ids(pred_path) if args.resume else set()
    prompt_style = str((cfg.get("rag_prompt") or {}).get("default_style") or "context_fallback")
    nq_resources = resolve_path(args.nq_resources or cfg["production"]["nq_resources"])

    # Map eval_split → production data_role. Train/val smoke uses that role;
    # sealed Test uses data_role=test with allow_sealed_test.
    data_role = eval_split if eval_split in ("train", "validation", "test") else "validation"

    feature_batch_size = int(args.feature_batch_size) if args.feature_batch_size else 8
    if feature_batch_size < 1:
        raise SystemExit("--feature-batch-size must be >= 1")

    pipeline, env = load_current_pipeline(
        registration=cfg["production"]["baseline_registration"],
        root=ROOT,
        nq_resources=nq_resources,
        data_role=data_role,
        attach_reference_rag=True,
        allow_sealed_test=allow_sealed,
    )
    try:
        pending = [item for item in records if str(item["sample_id"]) not in done]
        gold_by_id = {str(item["sample_id"]): item["gold"] for item in pending}
        for offset in range(0, len(pending), feature_batch_size):
            chunk = pending[offset : offset + feature_batch_size]
            queries = [(rec["query"], str(rec["sample_id"])) for rec in chunk]
            results = pipeline.run_batch(queries, feature_batch_size=len(queries))
            for result in results:
                sid = str(result.sample_id)
                row = build_product_row(
                    result=result,
                    gold=gold_by_id[sid],
                    prompt_style=prompt_style,
                )
                append_prediction(pred_path, row)
                done.add(sid)
    finally:
        close = getattr(getattr(pipeline.host, "reference_rag", None), "close", None)
        if callable(close):
            close()
        free_cuda_memory()

    rows = read_predictions(pred_path)
    metrics = aggregate_product_metrics(rows)
    metrics["eval_split"] = eval_split
    metrics["rag_prompt_style"] = prompt_style
    metrics["smoke"] = bool(args.smoke)
    write_json(metrics_path, metrics)

    status = "PRODUCT_SMOKE_OK" if args.smoke else "READY_FOR_FROZEN_TEST"
    run_cfg = build_run_config(
        cfg=cfg,
        args=args,
        eval_split=eval_split,
        eval_n=len(rows),
        split_path=split_path,
        output_dir=output_dir,
        status=status,
    )
    ref_env = dict((env or {}).get("reference_rag") or {})
    if "config" in ref_env and ref_env["config"]:
        ref_env["config"] = relpath(Path(ref_env["config"]))
    run_cfg["production_env_reference_rag"] = ref_env or None
    run_cfg["allow_sealed_test_effective"] = (env or {}).get("allow_sealed_test")
    write_json(run_config_path, run_cfg)

    print(
        json.dumps(
            {
                "status": status,
                "n": len(rows),
                "eval_split": eval_split,
                "output_dir": relpath(output_dir),
                "metrics": metrics,
                "mode": MODE,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_product_config(args.config)
    if args.validate_split_only:
        return run_validate_only(args, cfg)
    if args.smoke or args.eval_split or args.allow_sealed_test:
        return run_inference(args, cfg)
    # Default without flags: refuse silent Test.
    raise SystemExit(
        "Specify --smoke for Train/Val engineering smoke, "
        "or --eval-split test --allow-sealed-test for formal Test, "
        "or --validate-split-only."
    )


if __name__ == "__main__":
    raise SystemExit(main())
