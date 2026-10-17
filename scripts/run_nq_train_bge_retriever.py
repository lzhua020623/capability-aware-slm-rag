"""Run frozen BGE + FAISS Top-5 retrieval over all NQ Train queries.

Uses the Train IndexFlatIP built by scripts/build_nq_train_bge_index.py and the
same query instruction / L2 normalization as Dev Reference RAG.
Does not call Qwen, Diagnosis, Probe, or Controller.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.e5_faiss import E5FaissRetriever  # noqa: E402

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_K = 5
BATCH_SIZE = 32
EXPECTED_QUERIES = 74_097
EXPECTED_CORPUS = 1_224_119

QUERY_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_train.jsonl"
CORPUS_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_train_corpus.jsonl"
INDEX_PATH = ROOT / "data" / "processed" / "indexes" / "nq_train_bge_base.index"
OFFSETS_PATH = ROOT / "data" / "processed" / "indexes" / "nq_train_bge_base_offsets.npy"
IDS_PATH = ROOT / "data" / "processed" / "indexes" / "nq_train_bge_base_ids.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_train_bge_top5.jsonl"
STATS_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_train_bge_top5_stats.json"
PROGRESS_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_train_bge_top5_progress.json"


def load_passage_ids() -> list[str]:
    ids = []
    with IDS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line)["passage_id"])
    return ids


class CorpusLookup:
    def __init__(self, corpus_path: Path, offsets: np.ndarray):
        self.path = corpus_path
        self.offsets = offsets
        self.handle = corpus_path.open("r", encoding="utf-8")

    def get(self, faiss_id: int) -> dict:
        self.handle.seek(int(self.offsets[faiss_id]))
        return json.loads(self.handle.readline())

    def close(self) -> None:
        self.handle.close()


def completed_count(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def dev_regression_sample(n: int = 20, seed: int = 42) -> int:
    """Re-query frozen Dev index without modifying Dev files.

    Checks: Top-5 shape, finite scores, deterministic repeat, and compares to
    preliminary C1 Top-5 ids when that artifact is present.
    """
    import random

    dev_index = ROOT / "data" / "processed" / "indexes" / "nq_bge_base.index"
    dev_meta = ROOT / "data" / "processed" / "indexes" / "nq_bge_base_meta.jsonl"
    c1_path = ROOT / "results" / "preliminary" / "nq_c1_7b.jsonl"
    if not dev_index.exists() or not dev_meta.exists():
        print("SKIP Dev regression: frozen Dev index/meta missing")
        return 0

    meta = []
    with dev_meta.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                meta.append(json.loads(line))
    if len(meta) != 99922:
        print(f"FAIL: Dev meta size {len(meta)} != 99922")
        return 1

    retriever = E5FaissRetriever(MODEL_NAME, passage_prefix="", query_prefix=BGE_QUERY_INSTRUCTION)
    retriever.index = faiss.read_index(str(dev_index))
    retriever.set_metadata(meta)
    if retriever.index.ntotal != 99922 or retriever.index.d != 768:
        print(f"FAIL: Dev index ntotal/d = {retriever.index.ntotal}/{retriever.index.d}")
        retriever.unload()
        return 1

    queries = []
    with (ROOT / "data/processed/retrieval/nq_dev.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                queries.append(json.loads(line))
    rng = random.Random(seed)
    chosen = rng.sample(queries, min(n, len(queries)))

    def top5_for(question: str) -> list[str]:
        emb = retriever.encode_queries([question], batch_size=1)
        scores, indices = retriever.search(emb, top_k=5)
        ids = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) < 0 or not np.isfinite(score):
                raise RuntimeError("non-finite or negative hit")
            ids.append(meta[int(idx)]["passage_id"])
        if len(ids) != 5 or len(set(ids)) != 5:
            raise RuntimeError(f"bad top5 {ids}")
        return ids

    gold_hits = 0
    for rec in chosen:
        ids = top5_for(rec["question"])
        ids2 = top5_for(rec["question"])
        if ids != ids2:
            print(f"FAIL nondeterministic {rec['id']}")
            retriever.unload()
            return 1
        if rec["gold_passage_id"] in ids:
            gold_hits += 1

    print(
        f"Dev regression protocol check: n={len(chosen)} "
        f"exact_top5_ok deterministic_ok gold_in_top5={gold_hits}/{len(chosen)}"
    )

    if c1_path.exists():
        c1 = [json.loads(line) for line in c1_path.open(encoding="utf-8") if line.strip()]
        by_id = {str(r["sample_id"]): r for r in c1}
        qmap = {str(r["id"]): r["question"] for r in queries}
        sample_ids = [sid for sid in by_id if sid in qmap]
        picked = rng.sample(sample_ids, min(n, len(sample_ids)))
        mismatches = 0
        for sid in picked:
            got = top5_for(qmap[sid])
            expected = list(by_id[sid]["retrieved_top5_ids"])
            if got != expected:
                mismatches += 1
                if mismatches <= 3:
                    print(f"MISMATCH {sid}: got={got} expected={expected}")
        print(f"Dev regression vs C1 Top-5: checked={len(picked)} mismatches={mismatches}")
        retriever.unload()
        return 0 if mismatches == 0 else 1

    print("NOTE: preliminary C1 file absent on this branch; protocol checks passed")
    retriever.unload()
    return 0


def run_full(batch_size: int) -> int:
    for required in (QUERY_PATH, CORPUS_PATH, INDEX_PATH, OFFSETS_PATH, IDS_PATH):
        if not required.exists():
            print(f"ERROR: missing {required}")
            return 1

    passage_ids = load_passage_ids()
    if len(passage_ids) != EXPECTED_CORPUS:
        print(f"ERROR: ids={len(passage_ids)} expected {EXPECTED_CORPUS}")
        return 1
    offsets = np.load(OFFSETS_PATH)
    if len(offsets) != EXPECTED_CORPUS:
        print(f"ERROR: offsets={len(offsets)} expected {EXPECTED_CORPUS}")
        return 1

    index = faiss.read_index(str(INDEX_PATH))
    if index.ntotal != EXPECTED_CORPUS or index.d != 768:
        print(f"ERROR: index ntotal={index.ntotal} d={index.d}")
        return 1

    done = completed_count(OUT_PATH)
    mode = "a" if done else "w"
    if done and OUT_PATH.exists():
        print(f"Resuming retrieval from query {done}")
    elif OUT_PATH.exists() and done == 0:
        print(f"ERROR: {OUT_PATH} exists but empty/unreadable; refusing")
        return 1
    if done == 0 and OUT_PATH.exists():
        # fresh
        pass
    if done == EXPECTED_QUERIES and STATS_PATH.exists():
        print("Already complete")
        return 0
    if done == 0 and OUT_PATH.exists():
        print(f"ERROR: refusing to overwrite existing {OUT_PATH}")
        return 1

    retriever = E5FaissRetriever(MODEL_NAME, passage_prefix="", query_prefix=BGE_QUERY_INSTRUCTION)
    retriever.index = index
    retriever.passage_ids = passage_ids
    lookup = CorpusLookup(CORPUS_PATH, offsets)

    hit1 = hit3 = hit5 = 0
    errors = []
    t0 = time.time()
    latencies = []
    processed = 0

    # If resuming, accumulate prior recall from existing lines.
    if done:
        with OUT_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                hit1 += int(rec["gold_in_top1"])
                hit3 += int(rec["gold_in_top3"])
                hit5 += int(rec["gold_in_top5"])

    batch_queries: list[dict] = []
    out_handle = OUT_PATH.open(mode, encoding="utf-8", newline="\n")

    def flush_batch() -> None:
        nonlocal hit1, hit3, hit5, processed, batch_queries
        if not batch_queries:
            return
        t_batch = time.time()
        emb = retriever.encode_queries([q["question"] for q in batch_queries], batch_size=batch_size)
        scores, indices = retriever.search(emb, top_k=TOP_K)
        latencies.append((time.time() - t_batch) / len(batch_queries))
        for qi, query in enumerate(batch_queries):
            hits = []
            seen_pids = set()
            for rank, (score, idx) in enumerate(zip(scores[qi], indices[qi]), start=1):
                idx = int(idx)
                if idx < 0 or not np.isfinite(score):
                    errors.append(f"{query['id']}: bad hit rank={rank}")
                    continue
                meta = lookup.get(idx)
                pid = meta["passage_id"]
                if pid != passage_ids[idx]:
                    errors.append(f"{query['id']}: id mismatch at {idx}")
                if pid in seen_pids:
                    errors.append(f"{query['id']}: duplicate {pid}")
                seen_pids.add(pid)
                if not (meta.get("text") or "").strip() or not (query.get("question") or "").strip():
                    errors.append(f"{query['id']}: empty text/query")
                hits.append(
                    {
                        "rank": rank,
                        "passage_id": pid,
                        "document_id": meta.get("document_id"),
                        "document_title": meta.get("document_title"),
                        "text": meta["text"],
                        "score": float(score),
                    }
                )
            if [h["rank"] for h in hits] != [1, 2, 3, 4, 5]:
                errors.append(f"{query['id']}: ranks={[h['rank'] for h in hits]}")
            gold = query["gold_passage_id"]
            ids = [h["passage_id"] for h in hits]
            g1, g3, g5 = gold in ids[:1], gold in ids[:3], gold in ids[:5]
            hit1 += int(g1)
            hit3 += int(g3)
            hit5 += int(g5)
            out_handle.write(
                json.dumps(
                    {
                        "sample_id": query["id"],
                        "query": query["question"],
                        "short_answer": query.get("short_answer"),
                        "gold_passage_id": gold,
                        "retrieved_passages": hits,
                        "gold_in_top1": g1,
                        "gold_in_top3": g3,
                        "gold_in_top5": g5,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            processed += 1
        out_handle.flush()
        total_done = done + processed
        if total_done % 500 == 0 or total_done == EXPECTED_QUERIES:
            print(
                f"retrieved {total_done}/{EXPECTED_QUERIES} "
                f"R@5={(hit5 / total_done):.4f} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
            PROGRESS_PATH.write_text(
                json.dumps({"completed": total_done, "errors": len(errors)}) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        batch_queries = []

    with QUERY_PATH.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if not line.strip():
                continue
            if line_idx < done:
                continue
            batch_queries.append(json.loads(line))
            if len(batch_queries) >= batch_size:
                flush_batch()
                if errors and len(errors) > 50:
                    break
        flush_batch()

    out_handle.close()
    lookup.close()
    retriever.unload()

    total = done + processed
    if total != EXPECTED_QUERIES:
        print(f"ERROR: retrieved {total}, expected {EXPECTED_QUERIES}")
        return 1
    if errors:
        print(f"ERROR: {len(errors)} integrity issues; sample={errors[:10]}")
        return 1

    # Final integrity: unique sample ids
    ids = set()
    with OUT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            ids.add(str(rec["sample_id"]))
    if len(ids) != EXPECTED_QUERIES:
        print(f"ERROR: unique sample_id={len(ids)}")
        return 1

    stats = {
        "raw_train_count": 307_373,
        "controlled_query_count": EXPECTED_QUERIES,
        "corpus_passage_count": EXPECTED_CORPUS,
        "retrieved_query_count": EXPECTED_QUERIES,
        "recall_at_1": hit1 / EXPECTED_QUERIES,
        "recall_at_3": hit3 / EXPECTED_QUERIES,
        "recall_at_5": hit5 / EXPECTED_QUERIES,
        "model_name": MODEL_NAME,
        "index_type": "IndexFlatIP",
        "top_k": TOP_K,
        "normalized_embeddings": True,
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "passage_prefix": "",
        "retrieval_output_path": str(OUT_PATH.relative_to(ROOT).as_posix()),
        "index_path": str(INDEX_PATH.relative_to(ROOT).as_posix()),
        "corpus_path": str(CORPUS_PATH.relative_to(ROOT).as_posix()),
        "query_path": str(QUERY_PATH.relative_to(ROOT).as_posix()),
        "index_bytes": INDEX_PATH.stat().st_size,
        "output_bytes": OUT_PATH.stat().st_size,
        "mean_latency_seconds_per_query": float(np.mean(latencies)) if latencies else None,
        "elapsed_seconds_this_run": round(time.time() - t0, 2),
        "integrity_errors": 0,
        "exactly_top5": True,
        "called_qwen": False,
        "called_diagnosis_probe_controller": False,
        "read_test": False,
    }
    if STATS_PATH.exists():
        print(f"ERROR: refusing to overwrite {STATS_PATH}")
        return 1
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
    print(json.dumps(stats, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-regression", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.dev_regression:
        return dev_regression_sample()
    if args.run:
        return run_full(args.batch_size)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
