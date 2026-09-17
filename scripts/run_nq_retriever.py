"""Build an E5 + FAISS NQ base retriever and evaluate Recall@k."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.e5_faiss import E5FaissRetriever

CORPUS_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_corpus.jsonl"
QUERY_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_dev.jsonl"
INDEX_DIR = ROOT / "data" / "processed" / "indexes"
INDEX_PATH = INDEX_DIR / "nq_e5_base.index"
META_PATH = INDEX_DIR / "nq_e5_base_meta.jsonl"
METRICS_PATH = ROOT / "results" / "retrieval" / "nq_e5_metrics.json"

MODEL_NAME = "intfloat/e5-base-v2"
TOP_KS = (3, 5, 10)
MAX_K = max(TOP_KS)
SEED = 42
N_PREVIEW = 5
BATCH_SIZE = 64


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def recall_at_k(retrieved_sets: list[set[str]], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    n_hit = sum(1 for retrieved, gold in zip(retrieved_sets, gold_ids) if gold in retrieved)
    return n_hit / len(gold_ids)


def main() -> int:
    corpus = load_jsonl(CORPUS_PATH)
    queries = load_jsonl(QUERY_PATH)
    print(f"Loaded {len(corpus)} NQ passages from {CORPUS_PATH}")
    print(f"Loaded {len(queries)} NQ queries from {QUERY_PATH}")

    retriever = E5FaissRetriever(MODEL_NAME)
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Embedding device: {retriever.device}")
    print("FAISS device: cpu")

    passage_texts = [item["text"] for item in corpus]
    print("Encoding passages with prefix 'passage:' and L2 normalization")
    passage_embeddings = retriever.encode_passages(passage_texts, batch_size=BATCH_SIZE)
    retriever.set_metadata(corpus)
    retriever.build_index(passage_embeddings)
    retriever.save(INDEX_PATH, META_PATH)
    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved passage metadata to {META_PATH}")

    questions = [item["question"] for item in queries]
    gold_ids = [item["gold_passage_id"] for item in queries]
    print("Encoding queries with prefix 'query:' and L2 normalization")
    query_embeddings = retriever.encode_queries(questions, batch_size=BATCH_SIZE)
    scores, indices = retriever.search(query_embeddings, top_k=MAX_K)

    retrieved_ids: list[list[str]] = []
    hit_sets: dict[int, list[set[str]]] = {k: [] for k in TOP_KS}
    for row in indices:
        ids = [retriever.passage_ids[int(idx)] for idx in row if int(idx) >= 0]
        retrieved_ids.append(ids)
        for k in TOP_KS:
            hit_sets[k].append(set(ids[:k]))

    metrics = {
        "model": MODEL_NAME,
        "index": str(INDEX_PATH),
        "num_passages": len(corpus),
        "num_queries": len(queries),
        "embedding_device": retriever.device,
        "faiss_device": "cpu",
        "normalization": "l2",
        "index_type": "IndexFlatIP",
    }
    for k in TOP_KS:
        metrics[f"recall@{k}"] = recall_at_k(hit_sets[k], gold_ids)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("NQ E5 base retriever metrics:")
    for k in TOP_KS:
        print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
    print(f"Wrote {METRICS_PATH}")

    rng = random.Random(SEED)
    preview_idx = rng.sample(range(len(queries)), k=min(N_PREVIEW, len(queries)))
    meta_by_id = {item["passage_id"]: item for item in corpus}
    print(f"\nRandom preview ({len(preview_idx)} queries, seed={SEED}):")
    for i, qid in enumerate(preview_idx, start=1):
        query = queries[qid]
        gold_id = query["gold_passage_id"]
        top5_ids = retrieved_ids[qid][:5]
        hit = gold_id in top5_ids
        gold_passage = meta_by_id.get(gold_id, {})
        print(f"\n[{i}] query_id={query['id']}")
        print(f"  question: {query['question']}")
        print(f"  gold_passage_id: {gold_id}")
        print(f"  gold passage: {gold_passage.get('text', query.get('gold_evidence_text'))}")
        print(f"  gold passage 是否命中 Top-5: {hit}")
        print("  Top-5 retrieved passages:")
        for rank, pid in enumerate(top5_ids, start=1):
            passage = meta_by_id.get(pid, {})
            marker = "  [GOLD]" if pid == gold_id else ""
            print(f"    {rank}. {pid}{marker}")
            print(f"       {passage.get('text', '')}")
            print(f"       score={float(scores[qid][rank - 1]):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
