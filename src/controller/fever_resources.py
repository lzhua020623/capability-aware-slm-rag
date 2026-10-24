"""Verified FEVER IVFPQ resources and explicit dataset dispatch."""
import json
from pathlib import Path
import re

from src.compensation.retrieval_enhancement.adapters import BGE_MODEL_NAME, BGE_QUERY_INSTRUCTION, NQFaissBackend
from src.controller.nq_train_resources import OffsetBGERetriever
from src.retrieval.offset_corpus import OffsetCorpus, file_sha, stamp


def validate_index(index, *, count):
    expected = {'ntotal': count, 'd': 768, 'nlist': 4096, 'nprobe': 16,
                'metric_type': 0, 'is_trained': True, 'by_residual': False}
    if type(index).__name__ != 'IndexIVFPQ':
        raise ValueError('Expected FEVER IndexIVFPQ')
    for key, value in expected.items():
        if getattr(index, key, None) != value:
            raise ValueError(f'Unexpected FEVER index {key}')
    if index.pq.M != 64 or index.pq.nbits != 8:
        raise ValueError('Unexpected FEVER PQ encoding')


def verify_import(resource_dir, import_report, structure_report):
    """Read-only bootstrap validation before the query replay is registered."""
    resource_dir = Path(resource_dir)
    imported = json.loads(Path(import_report).read_text())
    structure = json.loads(Path(structure_report).read_text())
    if (imported.get('passed') is not True or imported['mapping'].get('full_order_verified') is not True
            or imported['mapping']['records_checked'] != 5_396_106
            or structure.get('passed') is not True
            or structure.get('import_report_sha256') != file_sha(import_report)):
        raise ValueError('FEVER import/structural validation is incomplete')
    names = {'corpus': 'fever_corpus.jsonl', 'index': 'fever_bge_ivfpq.index',
             'ids': 'fever_bge_ivfpq_ids.jsonl', 'metadata': 'fever_bge_ivfpq_meta.json',
             'offsets': 'fever_bge_ivfpq_offsets.npy'}
    paths, fingerprints = {}, {}
    for key, name in names.items():
        info = (imported['mapping']['fingerprints']['offsets'] if key == 'offsets'
                else imported['extracted'][name])
        path = resource_dir/name
        before = stamp(path)
        digest = file_sha(path)
        if before != stamp(path) or before[0] != info['bytes'] or digest != info['sha256']:
            raise ValueError(f'FEVER resource fingerprint mismatch: {key}')
        paths[key] = path
        fingerprints[key] = {'path': str(path), 'bytes': before[0], 'sha256': digest}
    return paths, fingerprints


class FeverFaissBackend(NQFaissBackend):
    """Shared numerical/query guard; FEVER has an audited lazy corpus store.

    The loader hashes the full import/order audit before construction. Unlike
    the historical full-text NQ metadata API, OffsetCorpus also checks source
    stamps and each selected ID/text on every lookup. No repeated 5M-row text
    materialization is needed during construction or each retrieval.
    """
    def __init__(self, retriever, *, expected_count):
        self.expected_count = expected_count
        super().__init__(retriever, max_top_k=20)

    def _check_resources(self):
        r = self.retriever
        if (r.model_name != BGE_MODEL_NAME or r.query_prefix != BGE_QUERY_INSTRUCTION
                or r.passage_prefix != '' or r.model is None):
            raise ValueError('FEVER requires the pinned BGE query contract')
        validate_index(r.index, count=self.expected_count)
        store = r.metadata
        if (not isinstance(store, OffsetCorpus) or len(store) != self.expected_count
                or r.passage_ids is not store.passage_ids or store.handle.closed):
            raise ValueError('FEVER requires the matching live OffsetCorpus')
        if any(stamp(path) != old for path, old in store.stamps.items()):
            raise ValueError('Loaded FEVER corpus or sidecars have changed')
        return self.expected_count, 768

    def retrieve(self, query, *, dataset, top_k):
        if dataset != 'fever':
            raise ValueError('FeverFaissBackend supports fever only')
        # Parent implementation owns the unchanged single-query normalization,
        # score/position guards and evidence whitelist. Dataset dispatch occurs
        # here; the parent uses its legacy name only for its own entry guard.
        return super().retrieve(query, dataset='nq', top_k=top_k)


class DatasetRetriever:
    def __init__(self, backends):
        if set(backends) - {'nq', 'fever'}:
            raise ValueError('Unknown retrieval dataset')
        self.backends = dict(backends)
        self.supported_datasets = frozenset(backends)

    def retrieve(self, query, *, dataset, top_k):
        if dataset not in self.backends:
            raise ValueError(f'No verified retrieval resources for {dataset}')
        return self.backends[dataset].retrieve(query, dataset=dataset, top_k=top_k)


def validate_resources(manifest_path, *, root):
    registry = json.loads(Path(manifest_path).read_text())
    if (registry.get('schema_version') != 'controller_fever_resources_v1'
            or registry.get('dataset') != 'fever'
            or registry.get('model_name') != BGE_MODEL_NAME
            or registry.get('query_instruction') != BGE_QUERY_INSTRUCTION
            or registry.get('passage_prefix') != ''
            or not re.fullmatch('[0-9a-f]{40}', registry.get('verified_model_revision', ''))
            or registry.get('corpus_size') != 5_396_106):
        raise ValueError('Invalid FEVER resource registry')
    names = {'corpus', 'index', 'ids', 'metadata', 'offsets', 'import_report', 'structure_report', 'query_replay'}
    if set(registry.get('files', {})) != names:
        raise ValueError('Incomplete FEVER resource registry')
    paths = {}
    for name, info in registry['files'].items():
        path = Path(root)/info['path']; before = stamp(path)
        if before[0] != info['bytes'] or file_sha(path) != info['sha256'] or before != stamp(path):
            raise ValueError(f'FEVER registry fingerprint mismatch: {name}')
        paths[name] = path
    imported_paths, _ = verify_import(paths['corpus'].parent, paths['import_report'], paths['structure_report'])
    if any(paths[key].resolve() != path.resolve() for key, path in imported_paths.items()):
        raise ValueError('FEVER registry and import paths disagree')
    replay = json.loads(paths['query_replay'].read_text())
    if (replay.get('passed') is not True or replay.get('n_queries') != 10
            or replay.get('same_ordered_top5_count') != 10
            or replay.get('encoding', {}).get('revision') != registry['verified_model_revision']
            or replay.get('import_report_sha256') != file_sha(paths['import_report'])
            or replay.get('structure_report_sha256') != file_sha(paths['structure_report'])):
        raise ValueError('FEVER pinned query replay has not passed')
    return registry, paths, {'registry_sha256': file_sha(manifest_path), 'files': registry['files'],
        'verified_model_revision': registry['verified_model_revision'], 'historical_model_revision': None,
        'query_compatibility_scope': '10 historical queries, ordered Top5 IDs; not all vector embeddings',
        'metadata_loading': 'bounded_offset_corpus', 'index_nprobe': 16}


def load_backend(manifest_path, *, root):
    registry, paths, receipt = validate_resources(manifest_path, root=root)
    import faiss
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL_NAME, revision=registry['verified_model_revision'], device='cpu')
    index = faiss.read_index(str(paths['index']))
    validate_index(index, count=registry['corpus_size'])
    store = OffsetCorpus(paths['corpus'], paths['ids'], paths['offsets'])
    try:
        dense = OffsetBGERetriever(model, index, store, faiss.normalize_L2)
        backend = FeverFaissBackend(dense, expected_count=registry['corpus_size'])
    except BaseException:
        store.close()
        raise
    return backend, receipt
