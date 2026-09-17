"""Build a BGE + FAISS NQ retriever and compare Recall@k with E5."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.e5_faiss import E5FaissRetriever

CORPUS_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_corpus.jsonl"
QUERY_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_dev.jsonl"
INDEX_DIR = ROOT / "data" / "processed" / "indexes"
INDEX_PATH = INDEX_DIR / "nq_bge_base.index"
META_PATH = INDEX_DIR / "nq_bge_base_meta.jsonl"
METRICS_PATH = ROOT / "results" / "retrieval" / "nq_bge_metrics.json"
E5_METRICS_PATH = ROOT / "results" / "retrieval" / "nq_e5_metrics.json"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_KS = (3, 5, 10)
MAX_K = max(TOP_KS)
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


def load_e5_metrics() -> dict | None:
    if not E5_METRICS_PATH.exists():
        return None
    return json.loads(E5_METRICS_PATH.read_text(encoding="utf-8"))


def main() -> int:
    corpus = load_jsonl(CORPUS_PATH)
    queries = load_jsonl(QUERY_PATH)
    print(f"Loaded {len(corpus)} NQ passages from {CORPUS_PATH}")
    print(f"Loaded {len(queries)} NQ queries from {QUERY_PATH}")
    if len(corpus) != 99922 or len(queries) != 3216:
        print(
            f"WARNING: expected 3216 queries and 99922 passages, "
            f"got {len(queries)} queries and {len(corpus)} passages"
        )

    retriever = E5FaissRetriever(
        MODEL_NAME,
        passage_prefix="",
        query_prefix=BGE_QUERY_INSTRUCTION,
    )
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Embedding device: {retriever.device}")
    print("FAISS device: cpu")
    print(f"BGE query instruction: {BGE_QUERY_INSTRUCTION!r}")
    print("BGE passages are encoded without an instruction prefix")

    passage_texts = [item["text"] for item in corpus]
    print("Encoding passages with L2 normalization")
    passage_embeddings = retriever.encode_passages(passage_texts, batch_size=BATCH_SIZE)
    retriever.set_metadata(corpus)
    retriever.build_index(passage_embeddings)
    retriever.save(INDEX_PATH, META_PATH)
    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved passage metadata to {META_PATH}")

    questions = [item["question"] for item in queries]
    gold_ids = [item["gold_passage_id"] for item in queries]
    print("Encoding queries with BGE retrieval instruction and L2 normalization")
    query_embeddings = retriever.encode_queries(questions, batch_size=BATCH_SIZE)
    _scores, indices = retriever.search(query_embeddings, top_k=MAX_K)

    hit_sets: dict[int, list[set[str]]] = {k: [] for k in TOP_KS}
    for row in indices:
        ids = [retriever.passage_ids[int(idx)] for idx in row if int(idx) >= 0]
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
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "passage_prefix": "",
    }
    for k in TOP_KS:
        metrics[f"recall@{k}"] = recall_at_k(hit_sets[k], gold_ids)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {METRICS_PATH}")

    e5_metrics = load_e5_metrics()
    print("\nNQ retriever comparison")
    print(f"{'model':<24} {'Recall@3':>10} {'Recall@5':>10} {'Recall@10':>10}")
    if e5_metrics:
        print(
            f"{'intfloat/e5-base-v2':<24} "
            f"{e5_metrics['recall@3']:10.4f} "
            f"{e5_metrics['recall@5']:10.4f} "
            f"{e5_metrics['recall@10']:10.4f}"
        )
    else:
        print("E5 metrics file not found; skipped E5 row.")
    print(
        f"{MODEL_NAME:<24} "
        f"{metrics['recall@3']:10.4f} "
        f"{metrics['recall@5']:10.4f} "
        f"{metrics['recall@10']:10.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
