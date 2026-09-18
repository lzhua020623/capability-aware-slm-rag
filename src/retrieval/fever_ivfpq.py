"""FAISS IndexIVFPQ helpers for formal FEVER retrieval.

NQ continues to use IndexFlatIP in E5FaissRetriever. This module is FEVER-only.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def make_ivfpq_index(dim: int, nlist: int, m: int, nbits: int) -> faiss.IndexIVFPQ:
    if dim % m != 0:
        raise RuntimeError(
            f"IVFPQ parameter m={m} must divide embedding dimension {dim}"
        )
    quantizer = faiss.IndexFlatIP(dim)
    try:
        index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
        )
    except TypeError:
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
    if hasattr(index, "by_residual"):
        index.by_residual = False
    return index


def set_nprobe(index: faiss.Index, nprobe: int) -> None:
    index.nprobe = nprobe


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path, nprobe: int) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(
            f"Formal FEVER FAISS index not found: {path}. "
            "Build it with scripts/build_fever_index.py. "
            "The 100k smoke-test pool is not valid for formal C1."
        )
    index = faiss.read_index(str(path))
    set_nprobe(index, nprobe)
    return index


def sample_training_vectors(
    shard_paths: list[Path],
    n_train: int,
    seed: int,
) -> np.ndarray:
    counts = [int(np.load(path, mmap_mode="r").shape[0]) for path in shard_paths]
    total = int(sum(counts))
    if total == 0:
        raise RuntimeError("No FEVER embedding shards to train on")
    n_train = min(n_train, total)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(total, size=n_train, replace=False))
    parts = []
    offset = 0
    cursor = 0
    for path, count in zip(shard_paths, counts):
        shard_end = offset + count
        take = []
        while cursor < len(chosen) and chosen[cursor] < shard_end:
            take.append(int(chosen[cursor] - offset))
            cursor += 1
        if take:
            shard = np.load(path, mmap_mode="r")
            parts.append(np.asarray(shard[take], dtype=np.float32))
        offset = shard_end
        if cursor >= len(chosen):
            break
    train = np.concatenate(parts, axis=0)
    if not train.flags["C_CONTIGUOUS"]:
        train = np.ascontiguousarray(train, dtype=np.float32)
    return train
