#!/usr/bin/env python3
"""Unified baseline evaluation runner (qwen_only | rag | fever_qwen_only).

Reuses main-tree APIs only:
  - src.generation.qwen (load_qwen_nf4, generate_greedy, build_rag_prompt)
  - src.retrieval.e5_faiss.E5FaissRetriever
  - src.evaluation.qa.best_em_f1 (NQ modes)
  - baselines._lib.fever_scoring (FEVER mode)

Does not import controller / compensation / diagnosis / probe.
Does not regenerate frozen Test IDs. Formal NQ Test requires the original split file.
FEVER uses independent frozen bundle:
  baselines/data/fever_dev_13332.jsonl + baselines/splits/fever_dev_13332_split.json
  → baselines/outputs/fever_qwen_only/ (never NQ output dirs).
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
    answer_in_context,
    apply_scores_to_row,
    score_answer,
)
from baselines._lib.fever_scoring import (  # noqa: E402
    aggregate_fever_metrics,
    build_fever_output_row,
)
from baselines._lib.matched_retrieval import (  # noqa: E402
    load_matched_retriever,
    retrieve_top_k_matched,
)
from baselines._lib.metrics_io import (  # noqa: E402
    aggregate_metrics,
    append_prediction,
    atomic_write_jsonl,
    backup_file,
    load_done_ids,
    read_predictions,
    write_json,
)
from baselines._lib.prompts import (  # noqa: E402
    RAG_PROMPT_STYLES,
    build_baseline_rag_prompt,
    build_closed_book_prompt,
    build_fever_closed_book_prompt,
)
from baselines._lib.records_io import (  # noqa: E402
    load_fever_dev_records,
    load_records_by_ids,
    sample_fever_smoke_records,
    synthetic_smoke_records,
)
from baselines._lib.split_io import (  # noqa: E402
    SplitError,
    extract_split_ids,
    load_split_json,
    validate_eval_split_file,
    validate_split_file,
)
from src.config import load_generator_config  # noqa: E402
from src.generation.qwen import (  # noqa: E402
    free_cuda_memory,
    generate_greedy,
    load_qwen_nf4,
)
from src.retrieval.e5_faiss import E5FaissRetriever  # noqa: E402

BASELINES = Path(__file__).resolve().parent
MODES = ("qwen_only", "rag", "fever_qwen_only")
STATUS_FROZEN_SPLIT_MISSING = "FROZEN_SPLIT_MISSING"
STATUS_READY = "READY_FOR_FROZEN_TEST"
STATUS_READY_FEVER = "READY_FOR_FEVER_QWEN_ONLY_FULL"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_mode_config(mode: str, config_path: str | Path | None = None) -> dict:
    # Formal RAG defaults to matched train-index freeze (never legacy nq_bge_base).
    if config_path:
        path = resolve_path(config_path)
    elif mode == "rag":
        path = BASELINES / "configs" / "rag_matched.json"
    else:
        path = BASELINES / "configs" / f"{mode}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mode config: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("mode") != mode:
        raise ValueError(f"Config mode mismatch: {cfg.get('mode')!r} vs {mode!r}")
    cfg["_config_path"] = str(path)
    return cfg


def assert_not_legacy_dev_index(cfg: dict) -> None:
    """Hard refuse invalidated Dev-index path for formal / matched RAG."""
    ret = cfg.get("retrieval") or {}
    index_path = str(ret.get("index_path") or "").replace("\\", "/")
    if "nq_bge_base.index" in index_path and "nq_train_bge" not in index_path:
        raise SystemExit(
            "Refusing legacy Dev-index nq_bge_base for diagnosis_nq_train_74097. "
            "Use baselines/configs/rag_matched.json (nq_train_bge_base.index)."
        )


def resolve_rag_prompt_style(args: argparse.Namespace, cfg: dict) -> str:
    if args.rag_prompt:
        return args.rag_prompt
    default = (cfg.get("rag_prompt") or {}).get("default_style")
    if default:
        return str(default)
    return "context_only"


def retrieval_backend(cfg: dict) -> str:
    ret = cfg.get("retrieval") or {}
    return str(ret.get("backend") or "meta_jsonl")


def retrieve_top_k(
    retriever: E5FaissRetriever, questions: list[str], top_k: int
) -> list[list[dict]]:
    """Same Top-k construction pattern as scripts/run_nq_base_rag.py (meta JSONL)."""
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
    prompt_style: str | None = None,
) -> dict:
    scores = score_answer(prediction, gold)
    row = {
        "sample_id": sample_id,
        "query": query,
        "prediction": prediction,
        "gold": gold,
        "mode": mode,
        **scores,
    }
    if mode == "rag":
        hits = hits or []
        row["retrieved_passage_ids"] = [h["passage_id"] for h in hits]
        row["retrieved_ranks"] = [int(h.get("rank", i + 1)) for i, h in enumerate(hits)]
        row["answer_in_context"] = answer_in_context(gold, hits)
        if prompt_style is not None:
            row["rag_prompt_style"] = prompt_style
    return row


def resolve_eval_split(cfg: dict, eval_split: str | None) -> str:
    split_cfg = cfg["split"]
    if eval_split:
        return eval_split
    if "default_eval_split" in split_cfg:
        return str(split_cfg["default_eval_split"])
    return "test"


def split_id_key_and_n(cfg: dict, eval_split: str) -> tuple[list[str], int]:
    split_cfg = cfg["split"]
    if "expected_n" in split_cfg:
        expected_map = split_cfg["expected_n"]
        if eval_split not in expected_map:
            raise ValueError(f"Config split.expected_n missing key {eval_split!r}")
        prefix = list(split_cfg.get("split_key_prefix") or ["splits", "nq"])
        return prefix + [eval_split], int(expected_map[eval_split])
    # Legacy rag/qwen_only configs: Test only.
    if eval_split != "test":
        raise ValueError(
            f"Config {cfg.get('_config_path')} only defines Test; "
            f"refusing --eval-split {eval_split}"
        )
    return list(split_cfg["test_key"]), int(split_cfg["expected_test_n"])


def default_output_dir(args: argparse.Namespace, cfg: dict, eval_split: str) -> Path:
    if args.output_dir:
        return resolve_path(args.output_dir)
    if args.mode == "fever_qwen_only":
        if args.smoke:
            return BASELINES / "outputs" / "fever_qwen_only_smoke"
        return BASELINES / "outputs" / "fever_qwen_only"
    if args.mode == "rag" and retrieval_backend(cfg) == "corpus_offsets":
        style = resolve_rag_prompt_style(args, cfg)
        return BASELINES / "outputs" / f"rag_matched_{eval_split}_{style}"
    return BASELINES / "outputs" / args.mode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline evaluation (qwen_only | rag | fever_qwen_only)"
    )
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config JSON (e.g. baselines/configs/rag_matched.json).",
    )
    p.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Frozen diagnosis_nq_train_74097_split.json",
    )
    p.add_argument(
        "--eval-split",
        choices=("validation", "test"),
        default=None,
        help="Which frozen split list to evaluate (default: config default_eval_split or test).",
    )
    p.add_argument(
        "--rag-prompt",
        choices=RAG_PROMPT_STYLES,
        default=None,
        help="RAG prompt style (required for matched Val prompt comparison).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to baselines/outputs/<mode>/ (or rag_matched_<split>_<prompt>/).",
    )
    p.add_argument("--resume", action="store_true", help="Skip sample_ids already in predictions.jsonl")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only for --smoke (synthetic / non-Test). Rejected on formal runs.",
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
    p.add_argument(
        "--reevaluate-only",
        action="store_true",
        help=(
            "Offline rescore of existing predictions.jsonl only. "
            "No Qwen/BGE/FAISS/inference; preserves sample_id/query/prediction/gold."
        ),
    )
    return p.parse_args(argv)


def _apply_fever_scores_to_row(row: dict) -> dict:
    """Rescore FEVER row in place of identity fields; returns a new dict."""
    claim = row.get("claim") or row.get("query") or ""
    scored = build_fever_output_row(
        sample_id=str(row.get("sample_id")),
        claim=str(claim),
        prediction=str(row.get("prediction") or ""),
        gold=row.get("gold"),
        mode=str(row.get("mode") or "fever_qwen_only"),
    )
    out = dict(row)
    out.update(scored)
    return out


def run_reevaluate_only(args: argparse.Namespace, cfg: dict, output_dir: Path) -> int:
    """Rescore existing predictions without any model loading."""
    pred_path = output_dir / cfg["output"]["predictions"]
    metrics_path = output_dir / cfg["output"]["metrics"]
    if not pred_path.is_file():
        raise SystemExit(f"No predictions to reevaluate: {pred_path}")

    rows = read_predictions(pred_path)
    if not rows:
        raise SystemExit(f"Empty predictions file: {pred_path}")

    fever_mode = args.mode == "fever_qwen_only"
    # Identity snapshot before rewrite (string equality, not new inference).
    identity_before = [
        (
            r.get("sample_id"),
            r.get("claim") if fever_mode else r.get("query"),
            r.get("prediction"),
            json.dumps(r.get("gold"), ensure_ascii=False),
        )
        for r in rows
    ]

    if fever_mode:
        rescored = [_apply_fever_scores_to_row(r) for r in rows]
        identity_after = [
            (
                r.get("sample_id"),
                r.get("claim"),
                r.get("prediction"),
                json.dumps(r.get("gold"), ensure_ascii=False),
            )
            for r in rescored
        ]
    else:
        rescored = [apply_scores_to_row(r) for r in rows]
        identity_after = [
            (
                r.get("sample_id"),
                r.get("query"),
                r.get("prediction"),
                json.dumps(r.get("gold"), ensure_ascii=False),
            )
            for r in rescored
        ]
    if identity_before != identity_after:
        raise SystemExit("Refusing write: identity fields changed during rescoring")

    backup_dir = output_dir / "_backup"
    bak_pred = backup_file(pred_path, backup_dir)
    bak_metrics = backup_file(metrics_path, backup_dir) if metrics_path.is_file() else None

    atomic_write_jsonl(pred_path, rescored)
    if fever_mode:
        metrics = aggregate_fever_metrics(rescored, mode=args.mode)
        evaluator = "baselines._lib.fever_scoring.score_fever_prediction"
        em_f1_backend = None
    else:
        metrics = aggregate_metrics(rescored, mode=args.mode)
        evaluator = "baselines._lib.answer_scoring.score_answer"
        em_f1_backend = "src.evaluation.qa.best_em_f1"
    metrics["reevaluate_only"] = True
    metrics["inference"] = False
    metrics["backup_predictions"] = str(bak_pred) if bak_pred else None
    metrics["backup_metrics"] = str(bak_metrics) if bak_metrics else None
    write_json(metrics_path, metrics)

    # Light run_config note (does not load generator models).
    run_cfg = {
        "mode": args.mode,
        "reevaluate_only": True,
        "inference": False,
        "generated_at": utc_now(),
        "n": len(rescored),
        "predictions": str(pred_path),
        "evaluator": evaluator,
    }
    if em_f1_backend is not None:
        run_cfg["em_f1_backend"] = em_f1_backend
    write_json(output_dir / cfg["output"]["run_config"], run_cfg)
    print(
        json.dumps(
            {
                "status": "REEVALUATE_OK",
                "n": len(rescored),
                "metrics": metrics,
                "qwen_called": False,
                "retriever_called": False,
            },
            indent=2,
        )
    )
    return 0


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_fever_bundle_records(cfg: dict, *, dataset_file: str | None = None) -> list[dict]:
    """Load frozen FEVER evaluation records (compact fever_dev_13332 schema)."""
    ds = cfg["dataset"]
    records_path = resolve_path(dataset_file or ds["records_path"])
    return load_fever_dev_records(
        records_path,
        id_field=ds.get("id_field", "sample_id"),
        claim_field=ds.get("claim_field", "claim"),
        gold_field=ds.get("gold_field", "gold"),
        expected_n=ds.get("expected_n", 13332),
        legal_labels=ds.get("legal_labels") or ("SUPPORTS", "REFUTES"),
    )


def fever_split_ids(cfg: dict, split_path: Path | None = None) -> tuple[Path, list[str], dict]:
    """Load ordered Test IDs from frozen FEVER split (NQ-parallel shape)."""
    split_cfg = cfg.get("split") or {}
    path = split_path or resolve_path(split_cfg["default_path"])
    if not path.is_file():
        raise FileNotFoundError(f"FEVER split missing: {path}")
    prefix = list(split_cfg.get("split_key_prefix") or ["splits", "fever"])
    eval_split = str(split_cfg.get("default_eval_split") or "test")
    expected_n = int((split_cfg.get("expected_n") or {}).get(eval_split, cfg["dataset"]["expected_n"]))
    split = load_split_json(path)
    ids = extract_split_ids(
        split,
        id_key=prefix + [eval_split],
        expected_n=expected_n,
        version_prefix=split_cfg.get("version_prefix"),
        split_name=f"fever_{eval_split}",
    )
    return path, ids, split


def fever_readiness(cfg: dict) -> tuple[str, dict]:
    """Validate frozen FEVER evaluation bundle (records + split + SHA; no inference)."""
    ds = cfg.get("dataset") or {}
    split_cfg = cfg.get("split") or {}
    records_path = resolve_path(ds["records_path"])
    split_path = resolve_path(split_cfg.get("default_path") or "")
    details: dict = {
        "records_path": str(records_path),
        "split_path": str(split_path),
    }

    if not records_path.is_file():
        return "FEVER_DATASET_MISSING", {
            **details,
            "message": "Frozen fever_dev_13332 evaluation records not found.",
        }
    if not split_path.is_file():
        return "FEVER_SPLIT_MISSING", {
            **details,
            "message": "Frozen fever_dev_13332_split.json not found.",
        }

    try:
        rows = load_fever_bundle_records(cfg)
        _, split_ids, split = fever_split_ids(cfg, split_path)
    except (OSError, ValueError, SplitError, json.JSONDecodeError, KeyError) as exc:
        return "FEVER_BUNDLE_INVALID", {**details, "error": str(exc)}

    record_ids = [r["sample_id"] for r in rows]
    if record_ids != split_ids:
        return "FEVER_ID_ORDER_MISMATCH", {
            **details,
            "records_n": len(record_ids),
            "split_n": len(split_ids),
            "first_mismatch": next(
                (
                    {"index": i, "record": a, "split": b}
                    for i, (a, b) in enumerate(zip(record_ids, split_ids))
                    if a != b
                ),
                {"index": None, "record_extra": record_ids[len(split_ids) :], "split_extra": split_ids[len(record_ids) :]},
            ),
        }

    counts = Counter(r["gold"] for r in rows)
    if set(counts) - {"SUPPORTS", "REFUTES"}:
        return "FEVER_LABELS_INVALID", {**details, "label_counts": dict(counts)}
    if counts.get("SUPPORTS") != 6666 or counts.get("REFUTES") != 6666:
        return "FEVER_LABEL_COUNTS_UNEXPECTED", {
            **details,
            "label_counts": dict(counts),
            "expected": {"SUPPORTS": 6666, "REFUTES": 6666},
        }

    records_sha = sha256_file(records_path)
    expected_rec_sha = (split.get("sha256") or {}).get("evaluation_records")
    if expected_rec_sha and records_sha != expected_rec_sha:
        return "FEVER_RECORDS_SHA_MISMATCH", {
            **details,
            "records_sha256": records_sha,
            "expected_sha256": expected_rec_sha,
        }

    source_path = resolve_path(ds["source_path"]) if ds.get("source_path") else None
    source_sha = None
    if source_path and source_path.is_file():
        source_sha = sha256_file(source_path)
        expected_src = (split.get("sha256") or {}).get("source_fever_dev")
        if expected_src and source_sha != expected_src:
            return "FEVER_SOURCE_SHA_MISMATCH", {
                **details,
                "source_path": str(source_path),
                "source_sha256": source_sha,
                "expected_sha256": expected_src,
            }

    resources_info = None
    res_cfg = cfg.get("resources") or {}
    manifest_rel = res_cfg.get("manifest_path")
    if manifest_rel:
        manifest_path = resolve_path(manifest_rel)
        if not manifest_path.is_file():
            return "FEVER_RESOURCES_MANIFEST_MISSING", {
                **details,
                "manifest_path": str(manifest_path),
            }
        try:
            resources_info = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return "FEVER_RESOURCES_MANIFEST_INVALID", {
                **details,
                "manifest_path": str(manifest_path),
                "error": str(exc),
            }
        details["resources_manifest"] = str(manifest_path)
        details["resources_summary"] = resources_info.get("summary")

    out_default = (cfg.get("output") or {}).get("default_dir") or "baselines/outputs/fever_qwen_only"
    if "fever_qwen_only" not in str(out_default).replace("\\", "/"):
        return "FEVER_OUTPUT_DIR_INVALID", {
            **details,
            "output_default_dir": out_default,
            "message": "Formal FEVER output must remain under fever_qwen_only.",
        }

    return STATUS_READY_FEVER, {
        **details,
        "n": len(rows),
        "unique_ids": len(set(record_ids)),
        "label_counts": dict(counts),
        "records_sha256": records_sha,
        "source_sha256": source_sha,
        "eval_split": "test",
        "split_version": split.get("version"),
        "output_default_dir": out_default,
    }

def formal_readiness(
    split_path: Path, cfg: dict, *, eval_split: str = "test"
) -> tuple[str, dict]:
    """Return (status, details). Never invents IDs."""
    split_cfg = cfg["split"]
    if not split_path.is_file():
        return STATUS_FROZEN_SPLIT_MISSING, {
            "split_path": str(split_path),
            "message": "Original frozen split file not found; do not regenerate IDs.",
        }
    try:
        id_key, expected_n = split_id_key_and_n(cfg, eval_split)
        if "expected_n" in split_cfg:
            summary = validate_eval_split_file(
                split_path,
                id_key=id_key,
                expected_n=expected_n,
                split_name=eval_split,
                version_prefix=split_cfg.get("version_prefix"),
            )
        else:
            summary = validate_split_file(
                split_path,
                test_key=list(split_cfg["test_key"]),
                expected_test_n=int(split_cfg["expected_test_n"]),
                version_prefix=split_cfg.get("version_prefix"),
            )
    except (OSError, SplitError, json.JSONDecodeError, ValueError) as exc:
        return "FROZEN_SPLIT_INVALID", {"split_path": str(split_path), "error": str(exc)}

    ret = cfg.get("retrieval")
    gaps = []
    if cfg["mode"] == "rag" and ret:
        backend = retrieval_backend(cfg)
        if backend == "corpus_offsets":
            for key in ("index_path", "ids_path", "offsets_path", "corpus_path"):
                path = resolve_path(ret[key])
                if not path.is_file():
                    gaps.append(f"missing_{key}:{path}")
        else:
            for key in ("index_path", "meta_path"):
                path = resolve_path(ret[key])
                if not path.is_file():
                    gaps.append(f"missing_{key}:{path}")
    records_path = resolve_path(cfg["dataset"]["records_path"])
    if not records_path.is_file():
        gaps.append(f"missing_dataset:{records_path}")
    if gaps:
        return "BASELINE_ASSETS_MISSING", {"split": summary, "gaps": gaps}
    return STATUS_READY, {"split": summary, "eval_split": eval_split}


def run_fever_smoke(args: argparse.Namespace, cfg: dict, output_dir: Path) -> int:
    """Seeded balanced subsample from frozen fever_dev_13332 (no RAG)."""
    mode = "fever_qwen_only"
    smoke_cfg = cfg.get("smoke") or {}
    n = args.limit if args.limit is not None else int(smoke_cfg.get("n", 30))
    seed = int(smoke_cfg.get("seed", 42))
    balanced = bool(smoke_cfg.get("balanced", True))
    if n < 1:
        raise SystemExit("--limit must be >= 1 for FEVER smoke")

    records_path = resolve_path(args.dataset_file or cfg["dataset"]["records_path"])
    all_rows = load_fever_bundle_records(cfg, dataset_file=args.dataset_file)
    records = sample_fever_smoke_records(
        all_rows, n=n, seed=seed, balanced=balanced
    )

    pred_path = output_dir / cfg["output"]["predictions"]
    if not args.resume and pred_path.exists():
        pred_path.unlink()
    done = load_done_ids(pred_path) if args.resume else set()

    tokenizer = model = None
    if not args.dry_run:
        tokenizer, model = load_qwen_nf4()

    try:
        for item in records:
            sid = item["sample_id"]
            if sid in done:
                continue
            claim = item["claim"]
            gold = item["gold"]
            if args.dry_run:
                prediction = str(gold)
            else:
                prompt = build_fever_closed_book_prompt(claim)
                prediction, _ = generate_greedy(tokenizer, model, prompt)
            row = build_fever_output_row(
                sample_id=sid,
                claim=claim,
                prediction=prediction,
                gold=gold,
                mode=mode,
            )
            append_prediction(pred_path, row)
            done.add(sid)
    finally:
        free_cuda_memory()

    rows = read_predictions(pred_path)
    metrics = aggregate_fever_metrics(rows, mode=mode)
    metrics["smoke"] = True
    metrics["dry_run"] = bool(args.dry_run)
    metrics["smoke_seed"] = seed
    metrics["smoke_balanced"] = balanced
    write_json(output_dir / cfg["output"]["metrics"], metrics)
    run_cfg = {
        "mode": mode,
        "dataset": "fever",
        "smoke": True,
        "dry_run": bool(args.dry_run),
        "generated_at": utc_now(),
        "n": len(rows),
        "smoke_seed": seed,
        "smoke_balanced": balanced,
        "dataset_file": str(records_path),
        "generator": load_generator_config(),
        "retrieval": None,
        "evaluator": "baselines._lib.fever_scoring.score_fever_prediction",
        "primary_metrics": ["accuracy", "macro_f1", "invalid_n"],
    }
    write_json(output_dir / cfg["output"]["run_config"], run_cfg)
    print(
        json.dumps(
            {
                "status": "FEVER_SMOKE_OK",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "invalid_n": metrics["invalid_n"],
                "metrics": metrics,
            },
            indent=2,
        )
    )
    return 0


def run_fever_formal(args: argparse.Namespace, cfg: dict, output_dir: Path) -> int:
    """Full frozen fever_dev_13332 Qwen-only formal eval (independent of NQ outputs)."""
    if args.limit is not None:
        raise SystemExit(
            "Refusing --limit on formal FEVER eval. Use --smoke for limited runs."
        )
    # Hard guard: never write into NQ formal dirs.
    out_s = str(output_dir.resolve()).replace("\\", "/")
    for banned in ("/outputs/qwen_only", "/outputs/rag_matched_test_context_fallback"):
        if out_s.rstrip("/").endswith(banned) or banned + "/" in out_s + "/":
            raise SystemExit(f"Refusing FEVER write into NQ output dir: {output_dir}")

    mode = "fever_qwen_only"
    status, details = fever_readiness(cfg)
    if status != STATUS_READY_FEVER:
        print(json.dumps({"status": status, "details": details}, indent=2))
        return 2

    records_path = resolve_path(args.dataset_file or cfg["dataset"]["records_path"])
    records = load_fever_bundle_records(cfg, dataset_file=args.dataset_file)
    split_path, split_ids, _split = fever_split_ids(cfg)
    if [r["sample_id"] for r in records] != split_ids:
        raise SystemExit("FEVER records/split ID order mismatch at formal start")

    pred_path = output_dir / cfg["output"]["predictions"]
    done = load_done_ids(pred_path) if args.resume else set()
    if not args.resume and pred_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing {pred_path}. Pass --resume or remove the file."
        )

    write_json(
        output_dir / cfg["output"]["run_config"],
        {
            "mode": mode,
            "dataset": "fever",
            "smoke": False,
            "generated_at": utc_now(),
            "config_path": cfg.get("_config_path"),
            "dataset_file": str(records_path),
            "split_file": str(split_path),
            "eval_split": "test",
            "eval_n": len(records),
            "records_sha256": details.get("records_sha256"),
            "generator": load_generator_config(),
            "retrieval": None,
            "evaluator": "baselines._lib.fever_scoring.score_fever_prediction",
            "primary_metrics": ["accuracy", "macro_f1", "invalid_n"],
            "status": STATUS_READY_FEVER,
        },
    )

    tokenizer, model = load_qwen_nf4()
    try:
        pending = [r for r in records if r["sample_id"] not in done]
        for i, rec in enumerate(pending, start=1):
            prompt = build_fever_closed_book_prompt(rec["claim"])
            prediction, _ = generate_greedy(tokenizer, model, prompt)
            row = build_fever_output_row(
                sample_id=rec["sample_id"],
                claim=rec["claim"],
                prediction=prediction,
                gold=rec["gold"],
                mode=mode,
            )
            append_prediction(pred_path, row)
            if i % 100 == 0 or i == len(pending):
                print(
                    f"[progress] fever_qwen_only: {i}/{len(pending)} "
                    f"this_run done_total≈{len(done)+i}",
                    flush=True,
                )
    finally:
        free_cuda_memory()

    rows = read_predictions(pred_path)
    metrics = aggregate_fever_metrics(rows, mode=mode)
    metrics["eval_split"] = "test"
    write_json(output_dir / cfg["output"]["metrics"], metrics)
    # Finalize run_config status now that inference + metrics are complete.
    run_cfg_path = output_dir / cfg["output"]["run_config"]
    if run_cfg_path.is_file():
        run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        run_cfg["status"] = "FEVER_FORMAL_COMPLETE"
        run_cfg["completed_at"] = utc_now()
        run_cfg["metrics_n"] = metrics.get("n")
        write_json(run_cfg_path, run_cfg)
    print(
        json.dumps(
            {"status": "FEVER_FORMAL_COMPLETE", "metrics": metrics},
            indent=2,
        )
    )
    return 0


def run_smoke(args: argparse.Namespace, cfg: dict, output_dir: Path) -> int:
    if args.mode == "fever_qwen_only":
        return run_fever_smoke(args, cfg, output_dir)

    mode = args.mode
    prompt_style = resolve_rag_prompt_style(args, cfg)
    pred_path = output_dir / cfg["output"]["predictions"]
    if not args.resume and pred_path.exists():
        pred_path.unlink()

    n = args.limit if args.limit is not None else 3
    if n < 1:
        raise SystemExit("--limit must be >= 1 for smoke")
    records = synthetic_smoke_records(n)
    done = load_done_ids(pred_path) if args.resume else set()

    tokenizer = model = retriever = None
    lookup = None
    passage_ids: list[str] | None = None
    if not args.dry_run:
        tokenizer, model = load_qwen_nf4()
        if mode == "rag":
            ret = cfg["retrieval"]
            if retrieval_backend(cfg) == "corpus_offsets":
                raise SystemExit(
                    "Smoke with matched corpus_offsets index is unsupported; "
                    "use --dry-run or legacy meta_jsonl rag.json for smoke."
                )
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
                prompt = build_baseline_rag_prompt(query, hits, prompt_style=prompt_style)
                prediction, _ = generate_greedy(tokenizer, model, prompt)
            row = build_output_row(
                mode=mode,
                sample_id=sid,
                query=query,
                prediction=prediction,
                gold=gold,
                hits=hits if mode == "rag" else None,
                prompt_style=prompt_style if mode == "rag" else None,
            )
            append_prediction(pred_path, row)
            done.add(sid)
    finally:
        if lookup is not None:
            lookup.close()
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
        "rag_prompt_style": prompt_style if mode == "rag" else None,
        "evaluator": "src.evaluation.qa.best_em_f1",
    }
    write_json(output_dir / cfg["output"]["run_config"], run_cfg)
    print(json.dumps({"status": "SMOKE_OK", "metrics": metrics}, indent=2))
    return 0


def run_formal(
    args: argparse.Namespace,
    cfg: dict,
    output_dir: Path,
    split_path: Path,
    *,
    eval_split: str,
) -> int:
    if args.limit is not None:
        raise SystemExit(
            "Refusing --limit on formal frozen eval. Use --smoke for limited synthetic runs."
        )
    assert_not_legacy_dev_index(cfg)
    prompt_style = resolve_rag_prompt_style(args, cfg)
    if args.mode == "rag" and retrieval_backend(cfg) == "corpus_offsets":
        if prompt_style not in RAG_PROMPT_STYLES:
            raise SystemExit(
                "Matched-index RAG requires rag prompt "
                f"{RAG_PROMPT_STYLES} (got {prompt_style!r})."
            )

    status, details = formal_readiness(split_path, cfg, eval_split=eval_split)
    if status != STATUS_READY:
        print(json.dumps({"status": status, "details": details}, indent=2))
        return 2

    split_cfg = cfg["split"]
    id_key, expected_n = split_id_key_and_n(cfg, eval_split)
    split = load_split_json(split_path)
    eval_ids = extract_split_ids(
        split,
        id_key=id_key,
        expected_n=expected_n,
        version_prefix=split_cfg.get("version_prefix"),
        split_name=eval_split,
    )
    # Hard guard: never silently mix Test when Validation was requested.
    if eval_split == "validation":
        test_key = list(split_cfg.get("split_key_prefix") or ["splits", "nq"]) + ["test"]
        try:
            from baselines._lib.split_io import _dig

            test_ids = {str(x) for x in _dig(split, test_key)}
        except SplitError:
            test_ids = set()
        overlap = set(eval_ids) & test_ids
        if overlap:
            raise SystemExit(f"Validation/Test ID overlap detected (n={len(overlap)})")

    records_path = resolve_path(args.dataset_file or cfg["dataset"]["records_path"])
    records = load_records_by_ids(
        records_path,
        eval_ids,
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
            "config_path": cfg.get("_config_path"),
            "split_file": str(split_path),
            "dataset_file": str(records_path),
            "eval_split": eval_split,
            "eval_n": len(records),
            "rag_prompt_style": prompt_style if args.mode == "rag" else None,
            "generator": load_generator_config(),
            "retrieval": cfg.get("retrieval"),
            "evaluator": "src.evaluation.qa.best_em_f1",
            "status": STATUS_READY,
        },
    )

    tokenizer, model = load_qwen_nf4()
    retriever = None
    lookup = None
    passage_ids: list[str] | None = None
    if args.mode == "rag":
        ret = cfg["retrieval"]
        if retrieval_backend(cfg) == "corpus_offsets":
            retriever, lookup, passage_ids = load_matched_retriever(
                embedding_model=ret["embedding_model"],
                index_path=resolve_path(ret["index_path"]),
                ids_path=resolve_path(ret["ids_path"]),
                offsets_path=resolve_path(ret["offsets_path"]),
                corpus_path=resolve_path(ret["corpus_path"]),
                passage_prefix=ret.get("passage_prefix", ""),
                query_prefix=ret.get("query_prefix", ""),
                expected_ntotal=ret.get("expected_ntotal"),
            )
        else:
            retriever = E5FaissRetriever(
                ret["embedding_model"],
                passage_prefix=ret.get("passage_prefix", ""),
                query_prefix=ret.get("query_prefix", ""),
            )
            retriever.load(resolve_path(ret["index_path"]), resolve_path(ret["meta_path"]))

    try:
        pending = [r for r in records if r["sample_id"] not in done]
        hits_by_id: dict[str, list[dict]] = {}
        if args.mode == "rag" and pending:
            assert retriever is not None
            questions = [r["query"] for r in pending]
            top_k = int(cfg["retrieval"]["top_k"])
            if retrieval_backend(cfg) == "corpus_offsets":
                assert lookup is not None and passage_ids is not None
                all_hits = retrieve_top_k_matched(
                    retriever, lookup, passage_ids, questions, top_k
                )
            else:
                all_hits = retrieve_top_k(retriever, questions, top_k)
            for rec, hits in zip(pending, all_hits):
                hits_by_id[rec["sample_id"]] = hits

        for i, rec in enumerate(pending, start=1):
            sid = rec["sample_id"]
            query = rec["query"]
            gold = rec["gold"]
            if args.mode == "qwen_only":
                prompt = build_closed_book_prompt(query)
                prediction, _ = generate_greedy(tokenizer, model, prompt)
                hits = None
            else:
                hits = hits_by_id[sid]
                prompt = build_baseline_rag_prompt(
                    query, hits, prompt_style=prompt_style
                )
                prediction, _ = generate_greedy(tokenizer, model, prompt)
            row = build_output_row(
                mode=args.mode,
                sample_id=sid,
                query=query,
                prediction=prediction,
                gold=gold,
                hits=hits,
                prompt_style=prompt_style if args.mode == "rag" else None,
            )
            append_prediction(pred_path, row)
            if i % 100 == 0 or i == len(pending):
                print(
                    f"[progress] {eval_split} {prompt_style if args.mode == 'rag' else args.mode}: "
                    f"{i}/{len(pending)} this_run done_total≈{len(done)+i}",
                    flush=True,
                )
    finally:
        if lookup is not None:
            lookup.close()
        if retriever is not None:
            retriever.unload()
        free_cuda_memory()

    rows = read_predictions(pred_path)
    metrics = aggregate_metrics(rows, mode=args.mode)
    metrics["eval_split"] = eval_split
    metrics["rag_prompt_style"] = prompt_style if args.mode == "rag" else None
    write_json(output_dir / cfg["output"]["metrics"], metrics)
    print(json.dumps({"status": "FORMAL_COMPLETE", "metrics": metrics}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_mode_config(args.mode, args.config)

    # FEVER path: independent of NQ frozen split / outputs.
    if args.mode == "fever_qwen_only":
        # eval_split unused for FEVER; keep CLI compatibility without touching NQ defaults.
        output_dir = default_output_dir(args, cfg, eval_split="fever_dev")
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.validate_split_only:
            status, details = fever_readiness(cfg)
            print(json.dumps({"status": status, "details": details}, indent=2))
            return 0 if status == STATUS_READY_FEVER else 2

        if args.reevaluate_only:
            if args.smoke or args.dry_run:
                raise SystemExit("--reevaluate-only cannot combine with --smoke/--dry-run")
            return run_reevaluate_only(args, cfg, output_dir)

        if args.smoke:
            return run_fever_smoke(args, cfg, output_dir)

        return run_fever_formal(args, cfg, output_dir)

    eval_split = resolve_eval_split(cfg, args.eval_split)
    output_dir = default_output_dir(args, cfg, eval_split)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_split = resolve_path(cfg["split"]["default_path"])
    split_path = resolve_path(args.split_file) if args.split_file else default_split

    if args.validate_split_only:
        status, details = formal_readiness(split_path, cfg, eval_split=eval_split)
        print(json.dumps({"status": status, "details": details}, indent=2))
        return 0 if status == STATUS_READY else 2

    if args.reevaluate_only:
        if args.smoke or args.dry_run:
            raise SystemExit("--reevaluate-only cannot combine with --smoke/--dry-run")
        return run_reevaluate_only(args, cfg, output_dir)

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

    return run_formal(args, cfg, output_dir, split_path, eval_split=eval_split)


if __name__ == "__main__":
    raise SystemExit(main())
