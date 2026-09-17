"""NQ Base RAG smoke test v2: shorter-span prompt and NQ-style evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_generator_config
from src.evaluation.qa import best_em_f1, gold_answers_from_record
from src.generation.qwen import (
    build_rag_prompt,
    free_cuda_memory,
    generate_greedy,
    load_qwen_nf4,
)
from src.retrieval.e5_faiss import E5FaissRetriever

QUERY_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_dev.jsonl"
INDEX_PATH = ROOT / "data" / "processed" / "indexes" / "nq_bge_base.index"
META_PATH = ROOT / "data" / "processed" / "indexes" / "nq_bge_base_meta.jsonl"
BASELINE_3B_PATH = ROOT / "results" / "base_rag" / "nq_smoke_test_v2.jsonl"
OUT_PATH = ROOT / "results" / "base_rag" / "nq_smoke_test_7b.jsonl"

RETRIEVER_MODEL = "BAAI/bge-base-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_K = 5
SMOKE_N = 20


def safe_print(*args) -> None:
    text = " ".join(str(arg) for arg in args)
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def gold_in_top5(hits: list[dict], gold_passage_id: str | None) -> bool:
    if not gold_passage_id:
        return False
    return any(hit.get("passage_id") == gold_passage_id for hit in hits)


def retrieve_top_k(
    retriever: E5FaissRetriever, questions: list[str], top_k: int
) -> list[list[dict]]:
    query_embeddings = retriever.encode_queries(questions, batch_size=16)
    scores, indices = retriever.search(query_embeddings, top_k=top_k)
    all_hits = []
    for row_scores, row_indices in zip(scores, indices):
        hits = []
        for score, idx in zip(row_scores, row_indices):
            faiss_id = int(idx)
            if faiss_id < 0:
                continue
            meta = retriever.metadata[faiss_id]
            hits.append(
                {
                    "passage_id": meta["passage_id"],
                    "text": meta["text"],
                    "score": float(score),
                }
            )
        all_hits.append(hits)
    return all_hits


def unload_retriever(retriever: E5FaissRetriever) -> None:
    retriever.unload()
    free_cuda_memory()


def summarize(records: list[dict]) -> dict:
    n = len(records)
    gold_hits = sum(1 for item in records if item["gold_in_top5"])
    retrieval_miss = n - gold_hits
    gold_but_wrong = sum(
        1 for item in records if item["gold_in_top5"] and item["em"] == 0
    )
    return {
        "n": n,
        "em": sum(item["em"] for item in records) / n if n else 0.0,
        "f1": sum(item["f1"] for item in records) / n if n else 0.0,
        "gold_in_top5": gold_hits,
        "retrieval_miss": retrieval_miss,
        "gold_in_top5_but_wrong": gold_but_wrong,
    }


def summarize_saved(records: list[dict]) -> dict:
    n = len(records)
    gold_hits = sum(1 for item in records if item.get("gold_in_top5"))
    em_total = 0.0
    f1_total = 0.0
    gold_but_wrong = 0
    for item in records:
        em = int(item.get("em", item.get("exact_match", 0)))
        f1 = float(item.get("f1", item.get("token_level_f1", 0.0)))
        in_top5 = bool(item.get("gold_in_top5"))
        em_total += em
        f1_total += f1
        if in_top5 and em == 0:
            gold_but_wrong += 1
    return {
        "n": n,
        "em": em_total / n if n else 0.0,
        "f1": f1_total / n if n else 0.0,
        "gold_in_top5": gold_hits,
        "retrieval_miss": n - gold_hits,
        "gold_in_top5_but_wrong": gold_but_wrong,
    }


def print_comparison(old: dict, new: dict) -> None:
    print()
    print(f"{'Metric':<22} {'3B':>10} {'7B':>10}")
    print(f"{'EM':<22} {old['em']:10.4f} {new['em']:10.4f}")
    print(f"{'F1':<22} {old['f1']:10.4f} {new['f1']:10.4f}")
    print(f"{'Retrieval hit':<22} {old['gold_in_top5']:10d} {new['gold_in_top5']:10d}")
    print(
        f"{'Generator failures':<22} "
        f"{old['gold_in_top5_but_wrong']:10d} {new['gold_in_top5_but_wrong']:10d}"
    )


def main() -> int:
    generator = load_generator_config()
    generator_model = str(generator["model_name"])
    max_new_tokens = int(generator.get("max_new_tokens", 64))
    queries = load_jsonl(QUERY_PATH)[:SMOKE_N]
    print(f"NQ Reference RAG smoke test: first {len(queries)} queries, same order as 3B")
    print(f"Retriever: {RETRIEVER_MODEL}, top_k={TOP_K}")
    print(f"Generator: {generator_model}, 4-bit NF4, greedy, max_new_tokens={max_new_tokens}")
    print(f"Writing {OUT_PATH} without overwriting 3B results")

    retriever = E5FaissRetriever(
        RETRIEVER_MODEL,
        passage_prefix="",
        query_prefix=BGE_QUERY_INSTRUCTION,
    )
    retriever.load(INDEX_PATH, META_PATH)
    questions = [item["question"] for item in queries]
    retrieved = retrieve_top_k(retriever, questions, TOP_K)
    unload_retriever(retriever)
    print("Retriever unloaded and CUDA cache cleared before loading the generator.")

    tokenizer, model = load_qwen_nf4(generator_model)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for query, hits in zip(queries, retrieved):
            golds = gold_answers_from_record(query)
            gold_text = golds[0] if golds else ""
            prompt = build_rag_prompt(query["question"], hits)
            answer, _latency = generate_greedy(
                tokenizer,
                model,
                prompt,
                max_new_tokens=max_new_tokens,
            )
            em, f1 = best_em_f1(answer, golds)
            in_top5 = gold_in_top5(hits, query.get("gold_passage_id"))
            record = {
                "id": query["id"],
                "question": query["question"],
                "gold_answer": gold_text,
                "model_answer": answer,
                "gold_in_top5": in_top5,
                "em": em,
                "f1": f1,
            }
            results.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            safe_print(
                f"[{len(results)}/{len(queries)}] gold_in_top5={in_top5} "
                f"EM={em} F1={f1:.3f} pred={answer!r}"
            )

    new_metrics = summarize(results)
    print()
    print(f"7B EM: {new_metrics['em']:.4f}")
    print(f"7B F1: {new_metrics['f1']:.4f}")
    print(f"gold in Top-5: {new_metrics['gold_in_top5']}")
    print(f"retrieval miss: {new_metrics['retrieval_miss']}")
    print(f"gold in Top-5 but answer wrong: {new_metrics['gold_in_top5_but_wrong']}")
    if BASELINE_3B_PATH.exists():
        old_metrics = summarize_saved(load_jsonl(BASELINE_3B_PATH))
        print_comparison(old_metrics, new_metrics)
    else:
        print(f"3B baseline not found at {BASELINE_3B_PATH}")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
