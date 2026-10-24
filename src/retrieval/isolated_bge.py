"""Pinned CPU query encoding in a separate Torch-only process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

from src.compensation.retrieval_enhancement.adapters import BGE_MODEL_NAME, BGE_QUERY_INSTRUCTION


def encode_queries(queries, revision):
    if (not isinstance(queries, list) or not queries
            or any(not isinstance(q, str) or not q.strip() for q in queries)
            or not re.fullmatch('[0-9a-f]{40}', revision)):
        raise ValueError('Nonempty queries and a pinned full model revision required')
    with tempfile.TemporaryDirectory(prefix='controller_bge_') as folder:
        folder = Path(folder)
        source, result, info = [folder/n for n in ('queries.json', 'vectors.npy', 'environment.json')]
        source.write_text(json.dumps(queries, ensure_ascii=False), encoding='utf-8')
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        root = Path(__file__).resolve().parents[2]
        subprocess.run([sys.executable, '-X', 'faulthandler', '-m', 'src.retrieval.isolated_bge',
                        str(source), str(result), str(info), revision], cwd=root, env=env, check=True)
        vectors = np.load(result, allow_pickle=False)
        provenance = json.loads(info.read_text())
    if (vectors.shape != (len(queries), 768) or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
            or provenance.get('revision') != revision
            or provenance.get('query_instruction') != BGE_QUERY_INSTRUCTION):
        raise ValueError('Invalid isolated BGE result/provenance')
    return vectors, provenance


def worker(source, output, info, revision):
    # FAISS is deliberately never imported into this process.
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1)
    model = SentenceTransformer(BGE_MODEL_NAME, revision=revision, device='cpu', trust_remote_code=False)
    queries = json.loads(Path(source).read_text(encoding='utf-8'))
    vectors = model.encode([BGE_QUERY_INSTRUCTION+q for q in queries], batch_size=16,
                          normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    with Path(output).open('xb') as handle:
        np.save(handle, vectors, allow_pickle=False)
    Path(info).write_text(json.dumps({'model_name': BGE_MODEL_NAME, 'revision': revision,
        'query_instruction': BGE_QUERY_INSTRUCTION, 'passage_prefix': '', 'device': 'cpu',
        'batch_size': 16, 'max_seq_length': model.max_seq_length,
        'execution': 'isolated pinned CPU encoder; no FAISS'}), encoding='utf-8')


if __name__ == '__main__':
    worker(*sys.argv[1:])
