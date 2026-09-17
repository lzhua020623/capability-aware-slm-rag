"""E5 + FAISS inner-product retriever."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def apply_prefix(prefix: str, text: str) -> str:
    if not prefix:
        return text
    if prefix.endswith(" "):
        return prefix + text
    return f"{prefix} {text}".strip()


class E5FaissRetriever:
    """Dense retriever using a SentenceTransformer encoder and a FAISS IP index."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: str | None = None,
        passage_prefix: str = "passage:",
        query_prefix: str = "query:",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.passage_prefix = passage_prefix
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name, device=device)
        self.index: faiss.Index | None = None
        self.passage_ids: list[str] = []
        self.metadata: list[dict] = []

    def encode(self, texts: list[str], prefix: str, batch_size: int = 64) -> np.ndarray:
        prefixed = [apply_prefix(prefix, text) for text in texts]
        current_batch = max(1, batch_size)
        while True:
            try:
                embeddings = self.model.encode(
                    prefixed,
                    batch_size=current_batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not is_cuda_oom(exc) or current_batch <= 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                next_batch = max(1, current_batch // 2)
                print(
                    f"CUDA OOM at batch_size={current_batch}; "
                    f"retrying with batch_size={next_batch}"
                )
                current_batch = next_batch
        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.encode(texts, prefix=self.passage_prefix, batch_size=batch_size)

    def encode_queries(self, questions: list[str], batch_size: int = 64) -> np.ndarray:
        return self.encode(questions, prefix=self.query_prefix, batch_size=batch_size)

    def build_index(self, passage_embeddings: np.ndarray) -> faiss.Index:
        dim = int(passage_embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(passage_embeddings)
        self.index = index
        return index

    def set_metadata(self, metadata: list[dict]) -> None:
        self.metadata = metadata
        self.passage_ids = [item["passage_id"] for item in metadata]

    def search(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("FAISS index has not been built or loaded.")
        scores, indices = self.index.search(query_embeddings, top_k)
        return scores, indices

    def save(self, index_path: Path, meta_path: Path) -> None:
        if self.index is None:
            raise RuntimeError("FAISS index has not been built or loaded.")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with meta_path.open("w", encoding="utf-8") as handle:
            for item in self.metadata:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load(self, index_path: Path, meta_path: Path) -> None:
        self.index = faiss.read_index(str(index_path))
        metadata = []
        with meta_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    metadata.append(json.loads(line))
        self.set_metadata(metadata)
