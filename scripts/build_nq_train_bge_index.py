"""Build a resumable BGE + FAISS IndexFlatIP index for NQ Train corpus.

Frozen protocol (must match Dev Reference RAG):
  model: BAAI/bge-base-en-v1.5
  query instruction: \"Represent this sentence for searching relevant passages: \"
  passage prefix: \"\"
  normalize_embeddings: L2 (SentenceTransformer + faiss.normalize_L2)
  index: faiss.IndexFlatIP
  top_k used downstream: 5

Does not overwrite Dev indexes. Does not call Qwen / Diagnosis / Probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch


def move_file(src: Path, dst: Path) -> None:
    """Atomic-ish move that works across Windows drives."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    src.unlink()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.e5_faiss import E5FaissRetriever  # noqa: E402

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBED_DIM = 768
EXPECTED_CORPUS = 1_224_119
SHARD_SIZE = 10_000
BATCH_SIZE = 32

CORPUS_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_train_corpus.jsonl"
QUERY_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_train.jsonl"
INDEX_DIR = ROOT / "data" / "processed" / "indexes"
INDEX_PATH = INDEX_DIR / "nq_train_bge_base.index"
META_JSON = INDEX_DIR / "nq_train_bge_metadata.json"
OFFSETS_PATH = INDEX_DIR / "nq_train_bge_base_offsets.npy"
IDS_PATH = INDEX_DIR / "nq_train_bge_base_ids.jsonl"
WORK_DIR = Path(os.environ.get("NQ_TRAIN_BGE_WORK", r"E:\nq_train_bge_work"))
SHARD_DIR = WORK_DIR / "shards"
PROGRESS_PATH = WORK_DIR / "progress.json"


def file_fingerprint(path: Path) -> dict:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return {"path": str(path.as_posix()), "sha256": h.hexdigest(), "bytes": size}


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {"completed_shards": [], "passages_encoded": 0}
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))


def save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(PROGRESS_PATH)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)


def iter_corpus(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        offset = handle.tell()
        for line_no, line in enumerate(handle):
            if not line.strip():
                offset = handle.tell()
                continue
            yield line_no, offset, json.loads(line)
            offset = handle.tell()


def encode_corpus_v2(retriever: E5FaissRetriever, batch_size: int) -> int:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()
    completed = set(progress.get("completed_shards", []))
    ids_partial = WORK_DIR / "ids.partial.jsonl"
    offsets_partial = WORK_DIR / "offsets.partial.npy"

    # Determine resume point: number of completed shards * SHARD_SIZE (last shard may be short).
    skip_passages = int(progress.get("passages_encoded", 0))
    if skip_passages and not ids_partial.exists():
        raise RuntimeError("Cannot resume: missing ids.partial.jsonl")

    ids_handle = ids_partial.open("a" if skip_passages else "w", encoding="utf-8", newline="\n")
    offsets: list[np.int64]
    if skip_passages:
        offsets = list(np.load(offsets_partial))
        if len(offsets) != skip_passages:
            raise RuntimeError(f"offsets length {len(offsets)} != passages_encoded {skip_passages}")
    else:
        offsets = []

    shard_idx = len(completed)
    buf_texts: list[str] = []
    buf_ids: list[str] = []
    buf_offs: list[int] = []
    seen = 0
    t0 = time.time()

    def write_shard() -> None:
        nonlocal shard_idx, buf_texts, buf_ids, buf_offs
        if not buf_texts:
            return
        path = SHARD_DIR / f"shard_{shard_idx:05d}.npy"
        print(
            f"Encoding shard {shard_idx}: n={len(buf_texts)} total_will_be={seen} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        emb = retriever.encode_passages(buf_texts, batch_size=batch_size)
        if emb.dtype != np.float32 or emb.shape[1] != EMBED_DIM:
            raise RuntimeError(f"Unexpected embeddings: dtype={emb.dtype} shape={emb.shape}")
        atomic_save_npy(path, emb)
        for pid in buf_ids:
            ids_handle.write(json.dumps({"passage_id": pid}, ensure_ascii=False) + "\n")
        ids_handle.flush()
        offsets.extend(buf_offs)
        np.save(offsets_partial, np.asarray(offsets, dtype=np.int64))
        completed.add(shard_idx)
        progress.update(
            {
                "completed_shards": sorted(completed),
                "passages_encoded": len(offsets),
                "last_shard": shard_idx,
                "elapsed_seconds": round(time.time() - t0, 2),
            }
        )
        save_progress(progress)
        shard_idx += 1
        buf_texts, buf_ids, buf_offs = [], [], []

    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            if seen < skip_passages:
                seen += 1
                continue
            record = json.loads(line)
            buf_texts.append(record["text"])
            buf_ids.append(record["passage_id"])
            buf_offs.append(offset)
            seen += 1
            if len(buf_texts) >= SHARD_SIZE:
                write_shard()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_shard()
    ids_handle.close()

    if len(offsets) != EXPECTED_CORPUS:
        raise RuntimeError(f"Encoded {len(offsets)} passages, expected {EXPECTED_CORPUS}")

    # Finalize sidecars into indexes dir (works across drives on Windows).
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if IDS_PATH.exists() or OFFSETS_PATH.exists():
        raise FileExistsError("Refusing to overwrite existing train id/offset artifacts")
    move_file(ids_partial, IDS_PATH)
    # Write offsets directly to final path via temp on the destination drive.
    atomic_save_npy(OFFSETS_PATH, np.asarray(offsets, dtype=np.int64))
    if offsets_partial.exists():
        offsets_partial.unlink()
    return len(offsets)


def assemble_index() -> None:
    if INDEX_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite {INDEX_PATH}")
    progress = load_progress()
    shard_ids = sorted(progress["completed_shards"])
    if not shard_ids:
        raise RuntimeError("No completed embedding shards")
    # Build on the work drive first, then move (cross-drive safe).
    work_index = WORK_DIR / "nq_train_bge_base.index.tmp"
    if work_index.exists():
        work_index.unlink()
    index = faiss.IndexFlatIP(EMBED_DIM)
    for shard_idx in shard_ids:
        path = SHARD_DIR / f"shard_{shard_idx:05d}.npy"
        emb = np.load(path)
        if emb.ndim != 2 or emb.shape[1] != EMBED_DIM:
            raise RuntimeError(f"Bad shard {path}: {emb.shape}")
        if not emb.flags["C_CONTIGUOUS"]:
            emb = np.ascontiguousarray(emb)
        index.add(emb)
        del emb
        if shard_idx % 10 == 0 or shard_idx == shard_ids[-1]:
            print(f"Added shard {shard_idx}: ntotal={index.ntotal}", flush=True)
    if index.ntotal != EXPECTED_CORPUS:
        raise RuntimeError(f"index.ntotal={index.ntotal} expected {EXPECTED_CORPUS}")
    faiss.write_index(index, str(work_index))
    del index
    move_file(work_index, INDEX_PATH)
    print(f"Wrote {INDEX_PATH} ntotal={EXPECTED_CORPUS}")


def write_metadata(elapsed: float) -> None:
    if META_JSON.exists():
        raise FileExistsError(f"Refusing to overwrite {META_JSON}")
    payload = {
        "model_name": MODEL_NAME,
        "embedding_dimension": EMBED_DIM,
        "normalize_embeddings": True,
        "similarity_metric": "inner_product",
        "index_type": "IndexFlatIP",
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "passage_prefix": "",
        "corpus_size": EXPECTED_CORPUS,
        "corpus_path": str(CORPUS_PATH.relative_to(ROOT).as_posix()),
        "corpus_fingerprint": file_fingerprint(CORPUS_PATH),
        "index_path": str(INDEX_PATH.relative_to(ROOT).as_posix()),
        "ids_path": str(IDS_PATH.relative_to(ROOT).as_posix()),
        "offsets_path": str(OFFSETS_PATH.relative_to(ROOT).as_posix()),
        "shard_size": SHARD_SIZE,
        "batch_size": BATCH_SIZE,
        "work_dir": str(WORK_DIR),
        "elapsed_seconds_encode_and_assemble": round(elapsed, 2),
    }
    META_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {META_JSON}")


def smoke_test(batch_size: int, seed: int = 42, n_queries: int = 100, n_extra: int = 8000) -> int:
    """Engineering smoke test on a mini corpus; does not write formal train artifacts."""
    rng = random.Random(seed)
    queries = []
    with QUERY_PATH.open("r", encoding="utf-8") as handle:
        all_q = [json.loads(line) for line in handle if line.strip()]
    queries = rng.sample(all_q, n_queries)
    gold_ids = {q["gold_passage_id"] for q in queries}

    # Mini corpus: gold passages + random extras (by line reservoir over first pass is heavy);
    # stream corpus once collecting golds and reservoir sample.
    reservoir: list[dict] = []
    gold_recs: dict[str, dict] = {}
    seen = 0
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec["passage_id"]
            if pid in gold_ids:
                gold_recs[pid] = rec
            seen += 1
            if len(reservoir) < n_extra:
                reservoir.append(rec)
            else:
                j = rng.randrange(seen)
                if j < n_extra:
                    reservoir[j] = rec

    # Dedup by passage_id, golds first.
    mini: list[dict] = []
    used = set()
    for rec in list(gold_recs.values()) + reservoir:
        if rec["passage_id"] in used:
            continue
        used.add(rec["passage_id"])
        mini.append(rec)

    print(f"Smoke mini corpus size={len(mini)} queries={len(queries)}")
    retriever = E5FaissRetriever(MODEL_NAME, passage_prefix="", query_prefix=BGE_QUERY_INSTRUCTION)
    emb = retriever.encode_passages([r["text"] for r in mini], batch_size=batch_size)
    retriever.set_metadata(mini)
    retriever.build_index(emb)
    q_emb = retriever.encode_queries([q["question"] for q in queries], batch_size=batch_size)
    scores, indices = retriever.search(q_emb, top_k=5)

    errors = []
    hit1 = hit3 = hit5 = 0
    for qi, query in enumerate(queries):
        row_ids = []
        for rank, (score, idx) in enumerate(zip(scores[qi], indices[qi]), start=1):
            idx = int(idx)
            if idx < 0:
                errors.append(f"{query['id']}: negative index")
                continue
            if not np.isfinite(score):
                errors.append(f"{query['id']}: non-finite score")
            pid = retriever.passage_ids[idx]
            if pid in row_ids:
                errors.append(f"{query['id']}: duplicate {pid}")
            row_ids.append(pid)
            if rank != len(row_ids):
                errors.append(f"{query['id']}: rank hole")
            if not mini[idx]["text"].strip():
                errors.append(f"{query['id']}: empty text")
        if len(row_ids) != 5:
            errors.append(f"{query['id']}: got {len(row_ids)} hits")
        gold = query["gold_passage_id"]
        if gold in row_ids[:1]:
            hit1 += 1
        if gold in row_ids[:3]:
            hit3 += 1
        if gold in row_ids[:5]:
            hit5 += 1

    print(
        json.dumps(
            {
                "smoke_queries": n_queries,
                "mini_corpus": len(mini),
                "errors": len(errors),
                "recall@1": hit1 / n_queries,
                "recall@3": hit3 / n_queries,
                "recall@5": hit5 / n_queries,
                "sample_errors": errors[:10],
            },
            indent=2,
        )
    )
    retriever.unload()
    return 0 if not errors else 1


def preflight() -> None:
    import shutil

    import psutil

    bytes_f32 = EXPECTED_CORPUS * EMBED_DIM * 4
    print("=== Resource preflight ===")
    print(f"corpus_count: {EXPECTED_CORPUS}")
    print(f"embedding_dimension: {EMBED_DIM}")
    print(f"float32_matrix_GB: {bytes_f32 / 1e9:.3f}")
    print(f"IndexFlatIP_RAM_GB_approx: {bytes_f32 / 1e9:.3f}")
    print(f"checkpoint_shards_disk_GB_approx: {bytes_f32 / 1e9:.3f}")
    vm = psutil.virtual_memory()
    print(f"RAM_total_GB: {vm.total / 1e9:.1f}")
    print(f"RAM_available_GB: {vm.available / 1e9:.1f}")
    for drive in ("C:/", "D:/", "E:/"):
        usage = shutil.disk_usage(drive)
        print(f"disk_free_GB[{drive}]: {usage.free / 1e9:.1f}")
    print(f"cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"vram_GB: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
    print(f"recommended_batch_size: {BATCH_SIZE}")
    print(f"work_dir: {WORK_DIR}")
    if vm.available < bytes_f32 + 1.5e9:
        print(
            "WARNING: available RAM is tight for assembling IndexFlatIP. "
            "Encoding can proceed with shards; assemble may need more free RAM."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--encode", action="store_true", help="Encode corpus shards (resumable)")
    parser.add_argument("--assemble", action="store_true", help="Assemble FAISS index from shards")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.preflight:
        preflight()
        return 0
    if args.smoke:
        preflight()
        return smoke_test(args.batch_size)
    if args.encode:
        preflight()
        if not CORPUS_PATH.exists():
            print(f"ERROR: missing {CORPUS_PATH}")
            return 1
        # Count lines quickly
        n = sum(1 for line in CORPUS_PATH.open(encoding="utf-8") if line.strip())
        if n != EXPECTED_CORPUS:
            print(f"ERROR: corpus has {n} lines, expected {EXPECTED_CORPUS}")
            return 1
        t0 = time.time()
        retriever = E5FaissRetriever(MODEL_NAME, passage_prefix="", query_prefix=BGE_QUERY_INSTRUCTION)
        print(f"device={retriever.device} model={MODEL_NAME}")
        encode_corpus_v2(retriever, args.batch_size)
        retriever.unload()
        print(f"Encode done in {time.time() - t0:.1f}s")
        return 0
    if args.assemble:
        preflight()
        t0 = time.time()
        assemble_index()
        write_metadata(time.time() - t0)
        # Verify ntotal
        index = faiss.read_index(str(INDEX_PATH))
        print(f"Verify ntotal={index.ntotal} dim={index.d}")
        if index.ntotal != EXPECTED_CORPUS or index.d != EMBED_DIM:
            return 1
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
