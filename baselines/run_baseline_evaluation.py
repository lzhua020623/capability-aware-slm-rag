#!/usr/bin/env python3
"""Unified baseline evaluation runner (qwen_only | rag).

Reuses main-tree APIs only:
  - src.generation.qwen (load_qwen_nf4, generate_greedy, build_rag_prompt)
  - src.retrieval.e5_faiss.E5FaissRetriever
  - src.evaluation.qa.best_em_f1

Does not import controller / compensation / diagnosis / probe.
Does not regenerate frozen Test IDs. Formal Test requires the original split file.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines._lib.metrics_io import (  # noqa: E402
    aggregate_metrics,
    append_prediction,
    load_done_ids,
    read_predictions,
    score_row,
    write_json,
)
from baselines._lib.prompts import build_closed_book_prompt  # noqa: E402
from baselines._lib.records_io import (  # noqa: E402
    load_records_by_ids,
    synthetic_smoke_records,
)
from baselines._lib.split_io import SplitError, validate_split_file  # noqa: E402
from src.config import load_generator_config  # noqa: E402
from src.evaluation.qa import best_em_f1  # noqa: E402
from src.generation.qwen import (  # noqa: E402
    build_rag_prompt,
    free_cuda_memory,
    generate_greedy,
    load_qwen_nf4,
)
from src.retrieval.e5_faiss import E5FaissRetriever  # noqa: E402

BASELINES = Path(__file__).resolve().parent
MODES = ("qwen_only", "rag")
STATUS_FROZEN_SPLIT_MISSING = "FROZEN_SPLIT_MISSING"
STATUS_READY = "READY_FOR_FROZEN_TEST"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_mode_config(mode: str) -> dict:
    path = BASELINES / "configs" / f"{mode}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mode config: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("mode") != mode:
        raise ValueError(f"Config mode mismatch: {cfg.get('mode')!r} vs {mode!r}")
    return cfg


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path


def retrieve_top_k(
    retriever: E5FaissRetriever, questions: list[str], top_k: int
) -> list[list[dict]]:
    """Same Top-k construction pattern as scripts/run_nq_base_rag.py."""
    query_embeddings = retriever.encode_queries(questions, batch_size=16)
    scores, indices = retriever.search(query_embeddings, top_k=top_k)
    all_hits: list[list[dict]] = []
    for row_scores, row_indices in zip(scores, indices):
        hits = []
        for rank, (score, idx) in enumerate(zip(row_scores, row_indices), start=1):
            faiss_id = int(idx)
            if faiss_id < 0:
                continue
            meta = retriever.metadata[faiss_id]
            hits.append(
                {
                    "passage_id": meta["passage_id"],
                    "text": meta["text"],
                    "score": float(score),
                    "rank": rank,
                }
            )
        all_hits.append(hits)
    return all_hits


def build_output_row(
    *,
    mode: str,
    sample_id: str,
    query: str,
    prediction: str,
    gold: list[str],
    hits: list[dict] | None = None,
) -> dict:
    em, f1 = score_row(prediction, gold)
    row = {
        "sample_id": sample_id,
        "query": query,
        "prediction": prediction,
        "gold": gold,
        "mode": mode,
        "em": int(em),
        "f1": float(f1),
    }
    if mode == "rag":
        hits = hits or []
        row["retrieved_passage_ids"] = [h["passage_id"] for h in hits]
        row["retrieved_ranks"] = [int(h.get("rank", i + 1)) for i, h in enumerate(hits)]
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline evaluation (qwen_only | rag)")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Frozen diagnosis_nq_train_74097_split.json (required for formal Test)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to baselines/outputs/<mode>/",
    )
    p.add_argument("--resume", action="store_true", help="Skip sample_ids already in predictions.jsonl")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only for --smoke (synthetic / non-Test). Rejected on formal frozen Test.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Synthetic non-Test smoke (no frozen Test IDs, optional --dry-run).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="With --smoke: stub predictions without loading Qwen/FAISS.",
    )
    p.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Override records JSONL (formal path uses mode config dataset.records_path).",
    )
    p.add_argument(
        "--validate-split-only",
        action="store_true",
        help="Validate split schema and exit (no inference).",
    )
    return p.parse_args(argv)


def formal_readiness(split_path: Path, cfg: dict) -> tuple[str, dict]:
    """Return (status, details). Never invents IDs."""
    split_cfg = cfg["split"]
    if not split_path.is_file():
        return STATUS_FROZEN_SPLIT_MISSING, {
            "split_path": str(split_path),
            "message": "Original frozen split file not found; do not regenerate IDs.",
        }
    try:
        summary = validate_split_file(
            split_path,
            test_key=list(split_cfg["test_key"]),
            expected_test_n=int(split_cfg["expected_test_n"]),
            version_prefix=split_cfg.get("version_prefix"),
        )
    except (OSError, SplitError, json.JSONDecodeError) as exc:
        return "FROZEN_SPLIT_INVALID", {"split_path": str(split_path), "error": str(exc)}

    ret = cfg.get("retrieval")
    gaps = []
    if cfg["mode"] == "rag" and ret:
        for key in ("index_path", "meta_path"):
            path = resolve_path(ret[key])
            if not path.is_file():
                gaps.append(f"missing_{key}:{path}")
    records_path = resolve_path(cfg["dataset"]["records_path"])
    if not records_path.is_file():
        gaps.append(f"missing_dataset:{records_path}")
    if gaps:
        return "BASELINE_ASSETS_MISSING", {"split": summary, "gaps": gaps}
    return STATUS_READY, {"split": summary}


def run_smoke(args: argparse.Namespace, cfg: dict, output_dir: Path) -> int:
    mode = args.mode
    pred_path = output_dir / cfg["output"]["predictions"]
    if not args.resume and pred_path.exists():
        pred_path.unlink()

    n = args.limit if args.limit is not None else 3
    if n < 1:
        raise SystemExit("--limit must be >= 1 for smoke")
    records = synthetic_smoke_records(n)
    done = load_done_ids(pred_path) if args.resume else set()

    tokenizer = model = retriever = None
    if not args.dry_run:
        tokenizer, model = load_qwen_nf4()
        if mode == "rag":
            ret = cfg["retrieval"]
            retriever = E5FaissRetriever(
                ret["embedding_model"],
                passage_prefix=ret.get("passage_prefix", ""),
                query_prefix=ret.get("query_prefix", ""),
            )
            retriever.load(resolve_path(ret["index_path"]), resolve_path(ret["meta_path"]))

    try:
        for item in records:
            sid = item["sample_id"]
            if sid in done:
                continue
            query = item["query"]
            gold = item["gold"]
            hits: list[dict] = []
            if args.dry_run:
                prediction = gold[0] if gold else "smoke"
            elif mode == "qwen_only":
                prompt = build_closed_book_prompt(query)
                prediction, _ = generate_greedy(tokenizer, model, prompt)
            else:
                assert retriever is not None
                hits = retrieve_top_k(retriever, [query], int(cfg["retrieval"]["top_k"]))[0]
                prompt = build_rag_prompt(query, hits)
                prediction, _ = generate_greedy(tokenizer, model, prompt)
            row = build_output_row(
                mode=mode,
                sample_id=sid,
                query=query,
                prediction=prediction,
                gold=gold,
                hits=hits if mode == "rag" else None,
            )
            append_prediction(pred_path, row)
            done.add(sid)
    finally:
        if retriever is not None:
            retriever.unload()
        free_cuda_memory()

    rows = read_predictions(pred_path)
    metrics = aggregate_metrics(rows, mode=mode)
    metrics["smoke"] = True
    metrics["dry_run"] = bool(args.dry_run)
    write_json(output_dir / cfg["output"]["metrics"], metrics)
    run_cfg = {
        "mode": mode,
        "smoke": True,
        "dry_run": bool(args.dry_run),
        "generated_at": utc_now(),
        "n": len(rows),
        "generator": load_generator_config(),
        "retrieval": cfg.get("retrieval"),
        "evaluator": "src.evaluation.qa.best_em_f1",
    }
    write_json(output_dir / cfg["output"]["run_config"], run_cfg)
    print(json.dumps({"status": "SMOKE_OK", "metrics": metrics}, indent=2))
    return 0


def run_formal(args: argparse.Namespace, cfg: dict, output_dir: Path, split_path: Path) -> int:
    if args.limit is not None:
        raise SystemExit(
            "Refusing --limit on formal frozen Test. Use --smoke for limited synthetic runs."
        )
    status, details = formal_readiness(split_path, cfg)
    if status != STATUS_READY:
        print(json.dumps({"status": status, "details": details}, indent=2))
        return 2

    split_cfg = cfg["split"]
    from baselines._lib.split_io import extract_test_ids, load_split_json

    split = load_split_json(split_path)
    test_ids = extract_test_ids(
        split,
        test_key=list(split_cfg["test_key"]),
        expected_test_n=int(split_cfg["expected_test_n"]),
        version_prefix=split_cfg.get("version_prefix"),
    )
    records_path = resolve_path(args.dataset_file or cfg["dataset"]["records_path"])
    records = load_records_by_ids(
        records_path,
        test_ids,
        id_field=cfg["dataset"]["id_field"],
        query_field=cfg["dataset"]["query_field"],
        gold_field=cfg["dataset"]["gold_field"],
    )

    pred_path = output_dir / cfg["output"]["predictions"]
    done = load_done_ids(pred_path) if args.resume else set()
    if not args.resume and pred_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing {pred_path}. Pass --resume or remove the file."
        )

    write_json(
        output_dir / cfg["output"]["run_config"],
        {
            "mode": args.mode,
            "smoke": False,
            "generated_at": utc_now(),
            "split_file": str(split_path),
            "dataset_file": str(records_path),
            "test_n": len(records),
            "generator": load_generator_config(),
            "retrieval": cfg.get("retrieval"),
            "evaluator": "src.evaluation.qa.best_em_f1",
            "status": STATUS_READY,
        },
    )

    tokenizer, model = load_qwen_nf4()
    retriever = None
    if args.mode == "rag":
        ret = cfg["retrieval"]
        retriever = E5FaissRetriever(
            ret["embedding_model"],
            passage_prefix=ret.get("passage_prefix", ""),
            query_prefix=ret.get("query_prefix", ""),
        )
        retriever.load(resolve_path(ret["index_path"]), resolve_path(ret["meta_path"]))

    try:
        pending = [r for r in records if r["sample_id"] not in done]
        # Optional batch retrieve for RAG
        hits_by_id: dict[str, list[dict]] = {}
        if args.mode == "rag" and pending:
            assert retriever is not None
            questions = [r["query"] for r in pending]
            all_hits = retrieve_top_k(retriever, questions, int(cfg["retrieval"]["top_k"]))
            for rec, hits in zip(pending, all_hits):
                hits_by_id[rec["sample_id"]] = hits

        for rec in pending:
            sid = rec["sample_id"]
            query = rec["query"]
            gold = rec["gold"]
            if args.mode == "qwen_only":
                prompt = build_closed_book_prompt(query)
                prediction, _ = generate_greedy(tokenizer, model, prompt)
                hits = None
            else:
                hits = hits_by_id[sid]
                prompt = build_rag_prompt(query, hits)
                prediction, _ = generate_greedy(tokenizer, model, prompt)
            row = build_output_row(
                mode=args.mode,
                sample_id=sid,
                query=query,
                prediction=prediction,
                gold=gold,
                hits=hits,
            )
            append_prediction(pred_path, row)
    finally:
        if retriever is not None:
            retriever.unload()
        free_cuda_memory()

    rows = read_predictions(pred_path)
    metrics = aggregate_metrics(rows, mode=args.mode)
    write_json(output_dir / cfg["output"]["metrics"], metrics)
    print(json.dumps({"status": "FORMAL_COMPLETE", "metrics": metrics}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_mode_config(args.mode)
    output_dir = resolve_path(args.output_dir) if args.output_dir else (BASELINES / "outputs" / args.mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_split = resolve_path(cfg["split"]["default_path"])
    split_path = resolve_path(args.split_file) if args.split_file else default_split

    if args.validate_split_only:
        status, details = formal_readiness(split_path, cfg)
        print(json.dumps({"status": status, "details": details}, indent=2))
        return 0 if status == STATUS_READY else 2

    if args.smoke:
        return run_smoke(args, cfg, output_dir)

    # Formal path: require frozen split; do not invent IDs.
    if not split_path.is_file():
        payload = {
            "status": STATUS_FROZEN_SPLIT_MISSING,
            "split_path": str(split_path),
            "message": (
                "Place the original diagnosis_nq_train_74097_split.json under "
                "baselines/splits/ (or pass --split-file). Do not regenerate Test IDs."
            ),
            "mode": args.mode,
        }
        print(json.dumps(payload, indent=2))
        write_json(output_dir / "readiness.json", payload)
        return 2

    return run_formal(args, cfg, output_dir, split_path)


if __name__ == "__main__":
    # best_em_f1 imported for clarity / test discovery of evaluator dependency
    _ = best_em_f1
    raise SystemExit(main())
