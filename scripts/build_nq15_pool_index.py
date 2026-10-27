"""Index the passages the NQ_15 records already contain, as a stand-in candidate pool.

Testing the full reranking recipe -- retrieve Top-N, rerank, keep Top-k -- needs
a corpus to retrieve from. The project's own corpus is 1,224,119 passages built
from a 51.7 GB download that is not on this machine. The annotated records
nonetheless carry 197,964 distinct passages between them (the Top-5 of 74,097
samples), which is 16.2% of that corpus and costs nothing to index.

What this buys and what it does not:

* it does test "choose 5 out of 50" with a real deeper pool, which reranking
  the frozen Top-5 cannot;
* it cannot surface a passage that no sample's Top-5 ever contained, so recall
  is strictly below the full corpus and any gain measured here is a lower bound.

That limitation matters less than it sounds for this evaluation set: every one
of the 482 held-out accessibility samples whose gold passage is missing from
its own Top-5 still has the gold answer string somewhere in that Top-5, so the
"find the missing answer" upside of a deeper pool is close to absent here
either way.

The encoder, normalisation and query instruction come from the frozen config,
so this is the project's own retriever over a smaller corpus. Artifacts are
named `nq15_pool_*` so they can never be mistaken for the frozen full index.

Usage:
    python scripts/build_nq15_pool_index.py
    python scripts/build_nq15_pool_index.py --limit-records 2000   # smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_base_config  # noqa: E402

DEFAULT_RECORDS = ROOT / "data" / "processed" / "diagnosis" / "nq15_train_records.jsonl"
OUT_DIR = ROOT / "data" / "processed" / "indexes"
CORPUS_OUT = ROOT / "data" / "processed" / "retrieval" / "nq15_pool_corpus.jsonl"
INDEX_OUT = OUT_DIR / "nq15_pool_bge_base.index"
IDS_OUT = OUT_DIR / "nq15_pool_bge_base_ids.jsonl"
META_OUT = OUT_DIR / "nq15_pool_bge_metadata.json"
FULL_CORPUS_SIZE = 1_224_119


def collect_passages(records_path: Path, limit: int | None = None) -> dict[str, str]:
    """passage_id -> text over every record's evidence, first text wins on a repeat."""
    passages: dict[str, str] = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if not line.strip():
                continue
            for passage in json.loads(line)["evidence"]:
                passages.setdefault(str(passage["passage_id"]), passage["text"])
    return passages


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--corpus-out", type=Path, default=CORPUS_OUT)
    parser.add_argument("--index-out", type=Path, default=INDEX_OUT)
    parser.add_argument("--ids-out", type=Path, default=IDS_OUT)
    parser.add_argument("--meta-out", type=Path, default=META_OUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.index_out, args.ids_out):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite to rebuild")

    passages = collect_passages(args.records, args.limit_records)
    ids = sorted(passages)  # a stable row order, independent of record order
    print(f"{len(ids)} distinct passages ({len(ids) / FULL_CORPUS_SIZE:.1%} of the full corpus)")

    import faiss
    from src.retrieval.e5_faiss import E5FaissRetriever

    retrieval = load_base_config()["retrieval"]
    model_name = str(retrieval["embedding_model"])
    # BGE puts its instruction on the query side only; passages go in bare.
    encoder = E5FaissRetriever(model_name=model_name, passage_prefix="", query_prefix="")
    print(f"encoding with {model_name} on {encoder.device}")

    start = time.perf_counter()
    embeddings = encoder.encode_passages([passages[key] for key in ids], batch_size=args.batch_size)
    elapsed = time.perf_counter() - start
    print(f"encoded {len(ids)} passages in {elapsed / 60:.1f} min")

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index_out))
    write_jsonl(args.ids_out, ({"passage_id": key} for key in ids))
    write_jsonl(args.corpus_out, ({"passage_id": key, "text": passages[key]} for key in ids))

    meta = {
        "note": "A stand-in pool built from the passages inside the annotated records, not the project's "
        "frozen corpus. Recall is strictly below the full index; gains measured on it are a lower bound.",
        "source_records": str(args.records.relative_to(ROOT)) if args.records.is_relative_to(ROOT) else str(args.records),
        "passages": len(ids),
        "full_corpus_passages": FULL_CORPUS_SIZE,
        "fraction_of_full_corpus": round(len(ids) / FULL_CORPUS_SIZE, 4),
        "embedding_model": model_name,
        "normalize_embeddings": bool(retrieval.get("normalize_embeddings", True)),
        "index_type": "IndexFlatIP",
        "dimension": int(embeddings.shape[1]),
        "encode_seconds": round(elapsed, 1),
    }
    args.meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"wrote {args.index_out}")


if __name__ == "__main__":
    main()
