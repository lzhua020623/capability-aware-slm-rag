"""FEVER Reference RAG smoke test: 20 claims, seed=42, frozen BGE + Qwen."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_generator_config
from src.evaluation.fever import gold_evidence_in_top5, parse_fever_prediction
from src.generation.qwen import (
    build_fever_rag_prompt,
    free_cuda_memory,
    generate_greedy,
    load_qwen_nf4,
)
from src.retrieval.e5_faiss import E5FaissRetriever

QUERY_PATH = ROOT / "data" / "processed" / "retrieval" / "fever_dev.jsonl"
CORPUS_PATH = ROOT / "data" / "processed" / "retrieval" / "fever_corpus.jsonl"
OUT_PATH = ROOT / "results" / "base_rag" / "fever_smoke_test.jsonl"
CKPT_PATH = ROOT / "results" / "base_rag" / "fever_smoke_retrieval_ckpt.json"

RETRIEVER_MODEL = "BAAI/bge-base-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_K = 5
SMOKE_N = 20
SEED = 42
ENCODE_BATCH = 64
TEXT_LIMIT = 2000
SMOKE_DISTRACTORS = 100_000
CORPUS_EXPECTED = 5_396_106


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


def sample_queries(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if n >= len(records):
        return list(records)
    return rng.sample(records, n)


def build_smoke_pool(corpus_path: Path, gold_page_ids: set[str], n_distractors: int, seed: int) -> list[dict]:
    golds: dict[str, dict] = {}
    distractors: list[dict] = []
    n_non_gold = 0
    rng = random.Random(seed)
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=CORPUS_EXPECTED, desc="Scanning FEVER corpus", mininterval=5.0):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            passage_id = str(record.get("passage_id") or "")
            text = (record.get("text") or "").strip()
            if not passage_id or not text:
                continue
            item = {"passage_id": passage_id, "text": text[:TEXT_LIMIT]}
            if passage_id in gold_page_ids:
                golds[passage_id] = item
                continue
            n_non_gold += 1
            if len(distractors) < n_distractors:
                distractors.append(item)
            else:
                j = rng.randint(1, n_non_gold)
                if j <= n_distractors:
                    distractors[j - 1] = item
    missing = sorted(gold_page_ids - set(golds))
    if missing:
        print(f"WARNING: {len(missing)} gold pages missing from corpus, e.g. {missing[:5]}")
    pool = list(golds.values()) + distractors
    print(
        f"Smoke retrieval pool: {len(golds)} gold pages + {len(distractors)} "
        f"random Wikipedia pages from fever_corpus.jsonl (seed={seed})"
    )
    return pool


def retrieve_top_k(
    retriever: E5FaissRetriever,
    claims: list[str],
    pool: list[dict],
    top_k: int,
) -> list[list[dict]]:
    passage_texts = [item["text"] for item in pool]
    print("Encoding smoke-pool passages with L2 normalization")
    passage_embeddings = retriever.encode(
        passage_texts,
        prefix=retriever.passage_prefix,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
    )
    retriever.set_metadata(pool)
    retriever.build_index(passage_embeddings)
    print("Encoding claims with BGE retrieval instruction and L2 normalization")
    query_embeddings = retriever.encode(
        claims,
        prefix=retriever.query_prefix,
        batch_size=16,
        show_progress_bar=False,
    )
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
    retriever.index = None
    free_cuda_memory()


def main() -> int:
    if not QUERY_PATH.exists() or not CORPUS_PATH.exists():
        print(f"Missing FEVER retrieval files. Need {QUERY_PATH} and {CORPUS_PATH}")
        return 1
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()

    queries_all = load_jsonl(QUERY_PATH)
    queries = sample_queries(queries_all, SMOKE_N, SEED)
    sample_ids = [item["id"] for item in queries]
    gold_page_ids = {
        str(page_id)
        for item in queries
        for page_id in (item.get("gold_page_ids") or [])
    }
    generator = load_generator_config()
    generator_model = str(generator["model_name"])
    max_new_tokens = int(generator.get("max_new_tokens", 64))
    print(f"FEVER Reference RAG smoke test: {len(queries)} claims, seed={SEED}")
    print(f"Retriever: {RETRIEVER_MODEL}, FAISS inner-product, L2, top_k={TOP_K}")
    print(f"Generator: {generator_model}, 4-bit NF4, greedy, max_new_tokens={max_new_tokens}")
    print(f"Writing {OUT_PATH}")
    print(f"Sample ids: {sample_ids}")

    pool = build_smoke_pool(CORPUS_PATH, gold_page_ids, SMOKE_DISTRACTORS, SEED)
    retriever = E5FaissRetriever(
        RETRIEVER_MODEL,
        passage_prefix="",
        query_prefix=BGE_QUERY_INSTRUCTION,
    )
    retrieved = retrieve_top_k(
        retriever,
        [item["claim"] for item in queries],
        pool,
        TOP_K,
    )
    unload_retriever(retriever)
    print("Retriever unloaded and CUDA cache cleared before loading the generator.")

    tokenizer, model = load_qwen_nf4(generator_model)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for query, hits in zip(queries, retrieved):
            prompt = build_fever_rag_prompt(query["claim"], hits)
            raw_answer, latency = generate_greedy(
                tokenizer,
                model,
                prompt,
                max_new_tokens=max_new_tokens,
            )
            prediction = parse_fever_prediction(raw_answer)
            gold_label = query["label"]
            in_top5 = gold_evidence_in_top5(hits, query.get("gold_page_ids"))
            correct = prediction == gold_label
            record = {
                "id": query["id"],
                "claim": query["claim"],
                "gold_label": gold_label,
                "retrieved_top5": hits,
                "gold_evidence_in_top5": in_top5,
                "model_prediction": prediction,
                "prediction_correct": correct,
                "latency": latency,
            }
            results.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            safe_print(
                f"[{len(results)}/{len(queries)}] gold_in_top5={in_top5} "
                f"gold={gold_label} pred={prediction} correct={correct} "
                f"latency={latency:.3f}s raw={raw_answer!r}"
            )

    n = len(results)
    gold_hits = sum(1 for item in results if item["gold_evidence_in_top5"])
    retrieval_miss = n - gold_hits
    gold_but_wrong = sum(
        1
        for item in results
        if item["gold_evidence_in_top5"] and not item["prediction_correct"]
    )
    accuracy = sum(1 for item in results if item["prediction_correct"]) / n if n else 0.0
    print()
    print(f"sample count: {n}")
    print(f"classification accuracy: {accuracy:.4f}")
    print(f"gold evidence in Top-5 数量: {gold_hits}")
    print(f"retrieval miss 数量: {retrieval_miss}")
    print(f"gold evidence 在 Top-5 但 prediction 错误数量: {gold_but_wrong}")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
