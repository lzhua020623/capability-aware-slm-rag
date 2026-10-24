"""Explicit NQ Train resource binding; validation never loads a model or gold."""
from pathlib import Path
import json
import re

from src.compensation.retrieval_enhancement.adapters import BGE_MODEL_NAME, BGE_QUERY_INSTRUCTION
from src.retrieval.offset_corpus import OffsetCorpus, file_sha, stamp

RESOURCE_NAMES = {'corpus', 'ids', 'offsets', 'index', 'recovery_metadata', 'mapping_validation'}


def validate_resources(manifest_path, *, root):
    manifest_path, root = Path(manifest_path), Path(root)
    registry = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (registry.get('schema_version') != 'controller_nq_train_resources_v1'
            or registry.get('dataset') != 'nq_train'
            or registry.get('model_name') != BGE_MODEL_NAME
            or registry.get('query_instruction') != BGE_QUERY_INSTRUCTION
            or registry.get('passage_prefix') != ''
            or registry.get('embedding_dimension') != 768
            or type(registry.get('corpus_size')) is not int or registry['corpus_size'] < 1
            or not re.fullmatch(r'[0-9a-f]{40}', registry.get('verified_model_revision', ''))
            or set(registry.get('files', {})) != RESOURCE_NAMES):
        raise ValueError('Invalid NQ Train resource registry')
    paths, fingerprints = {}, {}
    for name, info in registry['files'].items():
        if set(info) != {'path', 'bytes', 'sha256'}:
            raise ValueError(f'Invalid resource declaration: {name}')
        path = root / info['path']
        before = stamp(path)
        actual = file_sha(path)
        if actual != info['sha256'] or before[0] != info['bytes'] or before != stamp(path):
            raise ValueError(f'Resource fingerprint mismatch: {name}')
        paths[name] = path
        fingerprints[name] = dict(info)
    audit = json.loads(paths['mapping_validation'].read_text(encoding='utf-8'))
    metadata = json.loads(paths['recovery_metadata'].read_text(encoding='utf-8'))
    n = registry['corpus_size']
    if (audit.get('schema_version') != 'nq_complete_mapping_validation_v1'
            or audit.get('passed') is not True or audit.get('records_checked') != n
            or audit.get('actual_seek_reads') != n or audit.get('index_sha256_verified') is not True):
        raise ValueError('NQ mapping audit is incomplete')
    for name in RESOURCE_NAMES - {'mapping_validation'}:
        expected = audit.get('fingerprints', {}).get(name, {})
        if any(expected.get(key) != fingerprints[name][key] for key in ('bytes', 'sha256')):
            raise ValueError(f'Audit/resource mismatch: {name}')
    check = metadata.get('embedding_alignment', {})
    if (metadata.get('schema_version') != 'nq_train_index_recovery_v1'
            or metadata.get('corpus_size') != n or metadata.get('index_type') != 'IndexFlatIP'
            or check.get('performed') is not True or check.get('passed') is not True
            or check.get('model', {}).get('revision') != registry['verified_model_revision']):
        raise ValueError('Missing compatible sampled BGE check')
    return registry, paths, {'registry_sha256': file_sha(manifest_path), 'files': fingerprints,
                             'verified_model_revision': registry['verified_model_revision'],
                             'historical_model_revision': metadata.get('original_model_revision'),
                             'mapping_records_checked': n, 'metadata_loading': 'bounded_offset_corpus'}


class OffsetBGERetriever:
    """Same prefix/normalization/search contract as the existing BGE runner.

    Dependencies are supplied by the loader; CPU unit checks need no model,
    CUDA, or FAISS import. Corpus text remains on disk with a bounded cache.
    """
    model_name, query_prefix, passage_prefix = BGE_MODEL_NAME, BGE_QUERY_INSTRUCTION, ''

    def __init__(self, model, index, store, normalizer):
        self.model, self.index, self.metadata = model, index, store
        self.passage_ids, self.normalizer = store.passage_ids, normalizer

    def encode_queries(self, questions, batch_size=16):
        import numpy as np
        embeddings = self.model.encode([self.query_prefix + q for q in questions],
            batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.normalizer(embeddings)
        return embeddings

    def search(self, embeddings, top_k):
        return self.index.search(embeddings, top_k)

    def close(self):
        self.metadata.close()


def load_backend(manifest_path, *, root):
    registry, paths, receipt = validate_resources(manifest_path, root=root)
    # Model and index ownership remains outside the algorithm adapter. Full real
    # Controller runs require the CUDA environment; no heavy imports in preflight.
    import faiss
    from sentence_transformers import SentenceTransformer
    from src.compensation.retrieval_enhancement.adapters import NQFaissBackend
    model = SentenceTransformer(BGE_MODEL_NAME, revision=registry['verified_model_revision'], device='cpu')
    if not hasattr(faiss, 'IO_FLAG_MMAP_IFC'):
        raise RuntimeError('This resource loader requires FAISS Flat read-only mmap support')
    index = faiss.read_index(str(paths['index']), faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY)
    if type(index).__name__ != 'IndexFlatIP' or index.d != 768 or index.ntotal != registry['corpus_size']:
        raise ValueError('Loaded NQ index structure differs from registry')
    store = OffsetCorpus(paths['corpus'], paths['ids'], paths['offsets'])
    dense = OffsetBGERetriever(model, index, store, faiss.normalize_L2)
    try:
        backend = NQFaissBackend(dense, max_top_k=20)
    except BaseException:
        dense.close()
        raise
    return backend, receipt
