"""Build the formal FEVER BGE + FAISS IndexIVFPQ index.

Encodes data/processed/retrieval/fever_corpus.jsonl. Does not load Qwen
and does not use the 100k smoke-test candidate pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_preliminary_config
from src.retrieval.e5_faiss import E5FaissRetriever
from src.retrieval.fever_ivfpq import (
    load_index,
    make_ivfpq_index,
    sample_training_vectors,
    save_index,
    set_nprobe,
)

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
CORPUS_DEFAULT = ROOT / "data" / "processed" / "retrieval" / "fever_corpus.jsonl"
FORMAL_EXPECTED = 5_396_106


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fever_cfg(config: dict) -> dict:
    retrieval = config["retrieval"]
    fever = dict(retrieval.get("fever") or {})
    fever.setdefault("embedding_model", retrieval["embedding_model"])
    fever.setdefault("top_k", retrieval["top_k"])
    fever.setdefault("normalize_embeddings", retrieval["normalize_embeddings"])
    fever.setdefault("seed", config.get("seed", 42))
    return fever


def paths_from_cfg(fever: dict, engineering_test: bool) -> dict[str, Path]:
    if engineering_test:
        work = ROOT / "data" / "processed" / "indexes" / "_fever_ivfpq_engineering_test"
        return {
            "corpus": ROOT / fever["corpus"],
            "work_dir": work,
            "shard_dir": work / "shards",
            "progress": work / "progress.json",
            "index": work / "fever_bge_ivfpq.index",
            "ids": work / "fever_bge_ivfpq_ids.jsonl",
            "metadata": work / "fever_bge_ivfpq_meta.json",
        }
    return {
        "corpus": ROOT / fever["corpus"],
        "work_dir": ROOT / fever["work_dir"],
        "shard_dir": ROOT / fever["work_dir"] / "shards",
        "progress": ROOT / fever["work_dir"] / "progress.json",
        "index": ROOT / fever["index_path"],
        "ids": ROOT / fever["ids_path"],
        "metadata": ROOT / fever["metadata_path"],
    }


def new_progress(dim: int | None = None) -> dict:
    return {
        "stage": "encode",
        "encode_lines_read": 0,
        "n_encoded": 0,
        "next_shard": 0,
        "dim": dim,
        "buffer_ids": [],
    }


def shard_paths(shard_dir: Path) -> tuple[list[Path], list[Path]]:
    embs = sorted(shard_dir.glob("shard_*.npy"))
    ids = [path.with_suffix(".jsonl") for path in embs]
    return embs, ids


def flush_shard(
    shard_dir: Path,
    shard_idx: int,
    vectors: list[np.ndarray],
    ids: list[str],
) -> None:
    if not vectors:
        return
    shard_dir.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
    np.save(shard_dir / f"shard_{shard_idx:05d}.npy", array)
    with (shard_dir / f"shard_{shard_idx:05d}.jsonl").open("w", encoding="utf-8") as handle:
        for passage_id in ids:
            handle.write(json.dumps({"passage_id": passage_id}, ensure_ascii=False) + "\n")


def encode_corpus(
    retriever: E5FaissRetriever,
    corpus_path: Path,
    progress_path: Path,
    shard_dir: Path,
    shard_size: int,
    encode_batch: int,
    max_passages: int | None,
) -> dict:
    progress = load_json(progress_path) if progress_path.exists() else new_progress()
    buffer_vecs: list[np.ndarray] = []
    buffer_ids: list[str] = []
    already_encoded = int(progress.get("n_encoded") or 0)
    n_encoded = already_encoded
    next_shard = int(progress.get("next_shard") or 0)
    dim = progress.get("dim")
    seen_valid = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []

    def persist() -> None:
        write_json(
            progress_path,
            {
                "stage": "encode",
                "n_encoded": n_encoded,
                "next_shard": next_shard,
                "dim": dim,
            },
        )

    def encode_batch_texts() -> None:
        nonlocal dim, n_encoded, next_shard
        if not batch_texts:
            return
        embeddings = retriever.encode(
            batch_texts,
            prefix=retriever.passage_prefix,
            batch_size=encode_batch,
            show_progress_bar=False,
        )
        if dim is None:
            dim = int(embeddings.shape[1])
        elif int(embeddings.shape[1]) != int(dim):
            raise RuntimeError(
                f"Embedding dimension changed from {dim} to {embeddings.shape[1]}"
            )
        for row, passage_id in zip(embeddings, batch_ids):
            if max_passages is not None and n_encoded >= max_passages:
                break
            buffer_vecs.append(np.asarray(row, dtype=np.float32))
            buffer_ids.append(passage_id)
            n_encoded += 1
            if len(buffer_vecs) >= shard_size:
                flush_shard(shard_dir, next_shard, buffer_vecs, buffer_ids)
                next_shard += 1
                buffer_vecs.clear()
                buffer_ids.clear()
                persist()
        batch_ids.clear()
        batch_texts.clear()

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Encoding FEVER corpus", mininterval=5.0):
            if max_passages is not None and n_encoded >= max_passages:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            passage_id = str(record.get("passage_id") or "").strip()
            text = (record.get("text") or "").strip()
            if not passage_id or not text:
                continue
            if seen_valid < already_encoded:
                seen_valid += 1
                continue
            batch_ids.append(passage_id)
            batch_texts.append(text)
            if len(batch_texts) >= encode_batch:
                encode_batch_texts()

    encode_batch_texts()
    if buffer_vecs:
        flush_shard(shard_dir, next_shard, buffer_vecs, buffer_ids)
        next_shard += 1
    progress = {
        "stage": "train_add",
        "n_encoded": n_encoded,
        "next_shard": next_shard,
        "dim": dim,
    }
    write_json(progress_path, progress)
    return progress


def train_and_add(
    progress: dict,
    fever: dict,
    paths: dict[str, Path],
    nlist: int,
    nprobe: int,
    seed: int,
) -> faiss.Index:
    emb_paths, id_paths = shard_paths(paths["shard_dir"])
    if not emb_paths:
        raise RuntimeError("No embedding shards found; encode stage did not finish")
    dim = int(progress["dim"] or np.load(emb_paths[0], mmap_mode="r").shape[1])
    n_encoded = int(progress["n_encoded"])
    if n_encoded < nlist:
        raise RuntimeError(
            f"Need at least nlist={nlist} vectors to train IVF, have {n_encoded}. "
            "Use --engineering-test for a small run."
        )
    n_train = min(n_encoded, max(nlist * int(fever.get("train_size_per_list", 256)), nlist))
    print(f"Sampling {n_train} training vectors with seed={seed}")
    train = sample_training_vectors(emb_paths, n_train, seed)
    print(f"Training IndexIVFPQ dim={dim} nlist={nlist} m={fever['m']} nbits={fever['nbits']}")
    index = make_ivfpq_index(dim, nlist, int(fever["m"]), int(fever["nbits"]))
    index.verbose = True
    index.train(train)
    set_nprobe(index, nprobe)
    del train

    ids_out = paths["ids"]
    ids_out.parent.mkdir(parents=True, exist_ok=True)
    with ids_out.open("w", encoding="utf-8") as writer:
        for emb_path, id_path in zip(emb_paths, id_paths):
            shard = np.ascontiguousarray(np.load(emb_path), dtype=np.float32)
            index.add(shard)
            with id_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    writer.write(line if line.endswith("\n") else line + "\n")
            print(f"Added {shard.shape[0]} vectors from {emb_path.name}; ntotal={index.ntotal}")
            save_index(index, paths["index"])

    if int(index.ntotal) != n_encoded:
        raise RuntimeError(f"Index ntotal={index.ntotal} != encoded {n_encoded}")
    progress["stage"] = "done"
    progress["ntotal"] = int(index.ntotal)
    write_json(paths["progress"], progress)
    save_index(index, paths["index"])
    return index


def write_metadata(
    paths: dict[str, Path],
    fever: dict,
    nlist: int,
    nprobe: int,
    dim: int,
    ntotal: int,
    engineering_test: bool,
) -> None:
    payload = {
        "corpus": str(paths["corpus"].relative_to(ROOT)).replace("\\", "/"),
        "corpus_size": ntotal,
        "embedding_model": fever["embedding_model"],
        "embedding_dimension": dim,
        "normalization": "l2" if fever.get("normalize_embeddings", True) else "none",
        "faiss_index_type": "IndexIVFPQ",
        "metric": fever.get("metric", "inner_product"),
        "nlist": nlist,
        "m": int(fever["m"]),
        "nbits": int(fever["nbits"]),
        "nprobe": nprobe,
        "top_k": int(fever.get("top_k", 5)),
        "seed": int(fever.get("seed", 42)),
        "engineering_test": engineering_test,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "index_path": str(paths["index"].relative_to(ROOT)).replace("\\", "/"),
        "ids_path": str(paths["ids"].relative_to(ROOT)).replace("\\", "/"),
        "ntotal": ntotal,
        "valid_for_formal_c1": (not engineering_test) and ntotal == int(fever.get("expected_corpus_size", FORMAL_EXPECTED)),
    }
    write_json(paths["metadata"], payload)
    print(f"Wrote {paths['metadata']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build formal FEVER IndexIVFPQ")
    parser.add_argument(
        "--engineering-test",
        action="store_true",
        help="Encode a tiny subset into a separate test directory. Not valid for formal C1.",
    )
    parser.add_argument("--max-passages", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_preliminary_config()
    fever = fever_cfg(config)
    engineering_test = bool(args.engineering_test)
    max_passages = args.max_passages
    nlist = int(fever["nlist"])
    nprobe = int(fever["nprobe"])
    if engineering_test:
        max_passages = max_passages or int(fever.get("engineering_test_passages", 256))
        nlist = int(fever.get("engineering_test_nlist", 8))
        nprobe = min(nprobe, nlist)
        print(
            f"ENGINEERING TEST only: max_passages={max_passages}, nlist={nlist}. "
            "Not valid for formal FEVER C1."
        )

    paths = paths_from_cfg(fever, engineering_test)
    corpus_path = paths["corpus"]
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)
    paths["work_dir"].mkdir(parents=True, exist_ok=True)
    paths["shard_dir"].mkdir(parents=True, exist_ok=True)

    if (
        not engineering_test
        and paths["index"].exists()
        and paths["ids"].exists()
        and paths["metadata"].exists()
    ):
        meta = load_json(paths["metadata"])
        if meta.get("valid_for_formal_c1"):
            print(f"Formal FEVER index already exists at {paths['index']}")
            return 0

    if not args.resume and paths["progress"].exists():
        paths["progress"].unlink()

    progress = load_json(paths["progress"]) if paths["progress"].exists() else new_progress()
    if progress.get("stage") not in {"train_add", "done"}:
        print(f"Loading encoder {fever['embedding_model']} (no Qwen)")
        retriever = E5FaissRetriever(
            fever["embedding_model"],
            passage_prefix="",
            query_prefix=BGE_QUERY_INSTRUCTION,
        )
        print(f"Embedding device: {retriever.device}")
        progress = encode_corpus(
            retriever,
            corpus_path,
            paths["progress"],
            paths["shard_dir"],
            shard_size=int(fever.get("shard_size", 20000)),
            encode_batch=int(fever.get("encode_batch", 64)),
            max_passages=max_passages,
        )
        retriever.unload()
    else:
        print("Skipping encode; resuming from existing shards")

    if progress.get("stage") != "done":
        index = train_and_add(
            progress,
            fever,
            paths,
            nlist=nlist,
            nprobe=nprobe,
            seed=int(fever.get("seed", 42)),
        )
    else:
        index = load_index(paths["index"], nprobe)

    write_metadata(
        paths,
        fever,
        nlist=nlist,
        nprobe=nprobe,
        dim=int(progress["dim"]),
        ntotal=int(index.ntotal),
        engineering_test=engineering_test,
    )

    if engineering_test:
        query = np.ascontiguousarray(
            np.load(next(paths["shard_dir"].glob("shard_*.npy")))[:1], dtype=np.float32
        )
        set_nprobe(index, nprobe)
        scores, indices = index.search(query, int(fever.get("top_k", 5)))
        print(f"Engineering self-search top-5 indices={indices[0].tolist()} scores={scores[0].tolist()}")
        print("Engineering test passed. Formal full-corpus build was NOT run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
