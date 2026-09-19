"""Run the frozen 500-sample NQ C0/C1 preliminary experiment.

The model weights remain fixed. Results are appended after every generation so
an interrupted run can be resumed safely by running the same command again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_base_config, load_preliminary_config
from src.evaluation.qa import best_em_f1, gold_answers_from_record
from src.generation.qwen import (
    cpu_offload_enabled,
    free_cuda_memory,
    generate_greedy,
    load_qwen_nf4,
)
from src.retrieval.e5_faiss import E5FaissRetriever

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_frozen_queries(config: dict) -> list[dict]:
    sample_config = config["samples"]["nq"]
    manifest_path = ROOT / sample_config["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_ids = [str(item) for item in manifest["ids"]]
    expected_n = int(sample_config["n"])
    if len(frozen_ids) != expected_n or len(set(frozen_ids)) != expected_n:
        raise RuntimeError(f"Expected {expected_n} unique frozen NQ IDs")

    # Retrieval records contain the same controlled questions plus passage IDs.
    records = load_jsonl(ROOT / "data/processed/retrieval/nq_dev.jsonl")
    by_id = {str(record["id"]): record for record in records}
    missing = [sample_id for sample_id in frozen_ids if sample_id not in by_id]
    if missing:
        raise RuntimeError(f"Frozen NQ IDs missing from nq_dev.jsonl: {missing[:5]}")
    return [by_id[sample_id] for sample_id in frozen_ids]


def validate_frozen_settings(config: dict) -> None:
    base_generator = load_base_config()["generator"]
    primary = config["primary_model"]
    for key in (
        "model_name",
        "quantization",
        "load_in_4bit",
        "do_sample",
        "max_new_tokens",
    ):
        if base_generator.get(key) != primary.get(key):
            raise RuntimeError(f"Generator setting {key!r} differs between base and preliminary configs")
    if primary["do_sample"] is not False:
        raise RuntimeError("The frozen experiment requires deterministic decoding")
    if config["retrieval"]["top_k"] != 5:
        raise RuntimeError("The frozen experiment requires retrieval Top-5")


def retrieve(config: dict, queries: list[dict]) -> list[list[dict]]:
    retrieval = config["retrieval"]
    nq_config = retrieval["nq"]
    retriever = E5FaissRetriever(
        retrieval["embedding_model"],
        passage_prefix="",
        query_prefix=BGE_QUERY_INSTRUCTION,
    )
    retriever.load(ROOT / nq_config["index_path"], ROOT / nq_config["meta_path"])
    embeddings = retriever.encode_queries(
        [query["question"] for query in queries], batch_size=16
    )
    scores, indices = retriever.search(embeddings, top_k=int(retrieval["top_k"]))
    all_hits: list[list[dict]] = []
    for row_scores, row_indices in zip(scores, indices):
        hits = []
        for score, index in zip(row_scores, row_indices):
            faiss_id = int(index)
            if faiss_id < 0:
                continue
            item = retriever.metadata[faiss_id]
            hits.append(
                {
                    "passage_id": item["passage_id"],
                    "text": item["text"],
                    "score": float(score),
                }
            )
        all_hits.append(hits)
    retriever.unload()
    free_cuda_memory()
    return all_hits


def completed_ids(path: Path, condition: str) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for record in load_jsonl(path):
        if record.get("condition") != condition:
            raise RuntimeError(f"Unexpected condition in existing result file {path}")
        sample_id = str(record["sample_id"])
        if sample_id in completed:
            raise RuntimeError(f"Duplicate sample ID {sample_id} in {path}")
        completed.add(sample_id)
    return completed


def build_prompt(instruction: str, question: str, hits: list[dict] | None) -> str:
    if hits is None:
        return f"{instruction}\n\nQuestion: {question}"
    context = "\n\n".join(
        f"[{number}] {hit['text']}" for number, hit in enumerate(hits, start=1)
    )
    return f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}"


def write_result(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def summarize(path: Path) -> None:
    records = load_jsonl(path) if path.exists() else []
    n = len(records)
    em = sum(int(item["em"]) for item in records) / n if n else 0.0
    f1 = sum(float(item["f1"]) for item in records) / n if n else 0.0
    print(f"{path.name}: n={n}, EM={em:.4f}, F1={f1:.4f}")


def preflight(config: dict, queries: list[dict]) -> None:
    nq_config = config["retrieval"]["nq"]
    required = [
        ROOT / nq_config["index_path"],
        ROOT / nq_config["meta_path"],
        ROOT / config["samples"]["nq"]["manifest"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing required files:\n" + "\n".join(missing))
    if len(queries) != 500:
        raise RuntimeError(f"Expected 500 frozen queries, found {len(queries)}")
    print("Preflight passed")
    print(f"Frozen NQ queries: {len(queries)} unique IDs")
    print(f"FAISS index: {required[0]}")
    print(f"FAISS metadata: {required[1]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only", action="store_true", help="validate inputs without loading models"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one frozen query under C0/C1 without writing formal results",
    )
    args = parser.parse_args()

    config = load_preliminary_config()
    validate_frozen_settings(config)
    queries = load_frozen_queries(config)
    preflight(config, queries)
    if args.check_only:
        return 0

    if args.smoke_test:
        query = queries[0]
        hits = retrieve(config, [query])[0]
        primary = config["primary_model"]
        tokenizer, model = load_qwen_nf4(primary["model_name"])
        golds = gold_answers_from_record(query)
        for condition in ("C0", "C1"):
            prompt = build_prompt(
                config["prompts"]["nq"][condition],
                query["question"],
                None if condition == "C0" else hits,
            )
            prediction, latency = generate_greedy(
                tokenizer,
                model,
                prompt,
                max_new_tokens=int(primary["max_new_tokens"]),
            )
            em, f1 = best_em_f1(prediction, golds)
            print(
                f"SMOKE {condition}: prediction={prediction!r}, EM={em}, "
                f"F1={f1:.3f}, latency={latency:.2f}s"
            )
        print("Smoke test passed; no formal result files were written.")
        return 0

    result_paths = config["result_paths"]
    c0_path = ROOT / result_paths["nq_c0_7b"]
    c1_path = ROOT / result_paths["nq_c1_7b"]
    done = {"C0": completed_ids(c0_path, "C0"), "C1": completed_ids(c1_path, "C1")}
    frozen_ids = {str(query["id"]) for query in queries}
    for condition in ("C0", "C1"):
        unknown = done[condition] - frozen_ids
        if unknown:
            raise RuntimeError(f"Existing {condition} results contain non-frozen IDs")
    if len(done["C0"]) == 500 and len(done["C1"]) == 500:
        print("C0 and C1 are already complete; nothing to run.")
        summarize(c0_path)
        summarize(c1_path)
        return 0

    print("Retrieving frozen Top-5 evidence for all 500 queries...")
    all_hits = retrieve(config, queries)
    print("Retriever unloaded; loading the frozen 7B NF4 generator...")
    primary = config["primary_model"]
    tokenizer, model = load_qwen_nf4(primary["model_name"])
    prompts = config["prompts"]["nq"]
    max_new_tokens = int(primary["max_new_tokens"])

    total_generations = 1000
    already_done = len(done["C0"]) + len(done["C1"])
    for position, (query, hits) in enumerate(zip(queries, all_hits), start=1):
        sample_id = str(query["id"])
        golds = gold_answers_from_record(query)
        if not golds:
            raise RuntimeError(f"No gold answer for NQ sample {sample_id}")
        gold_answer = golds[0]
        gold_passage_id = query.get("gold_passage_id")

        for condition, path in (("C0", c0_path), ("C1", c1_path)):
            if sample_id in done[condition]:
                continue
            evidence = [] if condition == "C0" else hits
            prompt = build_prompt(
                prompts[condition], query["question"], None if condition == "C0" else hits
            )
            prediction, latency = generate_greedy(
                tokenizer, model, prompt, max_new_tokens=max_new_tokens
            )
            em, f1 = best_em_f1(prediction, golds)
            record = {
                "sample_id": query["id"],
                "dataset": "natural_questions",
                "condition": condition,
                "model_name": primary["model_name"],
                "cpu_offload": cpu_offload_enabled(),
                "cpu_offload_modules": ["lm_head"] if cpu_offload_enabled() else [],
                "question_or_claim": query["question"],
                "gold_answer_or_label": gold_answer,
                "evidence": evidence,
                "prediction": prediction,
                "correct": bool(em),
                "latency": latency,
                "em": em,
                "f1": f1,
            }
            if condition == "C1":
                record["retrieved_top5_ids"] = [hit["passage_id"] for hit in hits]
                record["gold_evidence_in_top5"] = bool(gold_passage_id) and any(
                    hit["passage_id"] == gold_passage_id for hit in hits
                )
            write_result(path, record)
            done[condition].add(sample_id)
            already_done += 1
            print(
                f"[{already_done}/{total_generations}] sample={position}/500 "
                f"{condition} EM={em} F1={f1:.3f} latency={latency:.2f}s"
            )

    summarize(c0_path)
    summarize(c1_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
