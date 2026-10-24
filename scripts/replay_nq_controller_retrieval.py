"""Resource-only replay on frozen development queries; no generation or scoring."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recover_nq_train_index_sidecars import encode_bge_in_child
from src.compensation.retrieval_enhancement.adapters import BGE_QUERY_INSTRUCTION, NQFaissBackend
from src.controller.nq_train_resources import OffsetBGERetriever, validate_resources
from src.retrieval.offset_corpus import OffsetCorpus, file_sha


def development_queries(path, limit):
    if type(limit) is not int or limit < 1:
        raise ValueError('Positive query count required')
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('dataset') != 'nq' or not isinstance(row.get('query'), str) or not row['query'].strip():
                raise ValueError('Expected NQ runtime development input')
            rows.append({'sample_id': row['sample_id'], 'query': row['query'],
                         'reference_top5': [e['passage_id'] for e in row['evidence']]})
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError('Insufficient development queries')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate development sample ID')
    return rows


class PreencodedQueryModel:
    """Replay genuine child-process BGE vectors through the normal adapter."""
    def __init__(self, queries, vectors):
        if len(queries) != len(vectors):
            raise ValueError('Query/vector counts differ')
        self.vectors = {BGE_QUERY_INSTRUCTION + q: v for q, v in zip(queries, vectors)}

    def encode(self, texts, **kwargs):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path, default=ROOT/'configs/controller/nq_train_resources_v1.json')
    parser.add_argument('--input', type=Path, default=ROOT/'artifacts/controller/data_v1/nq/development/runtime.jsonl')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Preserve old replay output; choose a new path')
    registry, paths, receipt = validate_resources(args.resources, root=ROOT)
    rows = development_queries(args.input, args.limit)
    provenance = {}
    vectors = encode_bge_in_child([BGE_QUERY_INSTRUCTION + r['query'] for r in rows], provenance)
    if provenance['revision'] != registry['verified_model_revision']:
        raise ValueError('Encoding snapshot differs from verified registry; no replay performed')
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(str(paths['index']), faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY)
    checks = []
    with OffsetCorpus(paths['corpus'], paths['ids'], paths['offsets']) as store:
        dense = OffsetBGERetriever(PreencodedQueryModel([r['query'] for r in rows], vectors),
                                  index, store, faiss.normalize_L2)
        backend = NQFaissBackend(dense, max_top_k=20)
        for row in rows:
            top20 = backend.retrieve(row['query'], dataset='nq', top_k=20)
            top5 = backend.retrieve(row['query'], dataset='nq', top_k=5)
            if len(top20) != 20 or top20[:5] != top5:
                raise ValueError('Top20/Top5 consistency failure')
            checks.append({'sample_id': row['sample_id'], 'reference_top5': row['reference_top5'],
                           'actual_top5': [r['passage_id'] for r in top5],
                           'same_ordered_ids': row['reference_top5'] == [r['passage_id'] for r in top5],
                           'top20_ids': [r['passage_id'] for r in top20],
                           'top5_scores': [r['score'] for r in top5]})
    report = {'schema_version': 'nq_controller_retrieval_replay_v1', 'resource_checks_passed': True,
              'n_queries': len(rows), 'input_sha256': file_sha(args.input), 'resources': receipt,
              'encoding': {**provenance, 'effective_query_instruction': BGE_QUERY_INSTRUCTION},
              'execution': 'BGE child process; FAISS and lazy Controller backend in parent; no Qwen',
              'same_ordered_top5_count': sum(r['same_ordered_ids'] for r in checks),
              'quality_scored': False, 'checks': checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2); handle.write('\n')
    print(json.dumps({k: report[k] for k in ('resource_checks_passed', 'n_queries', 'same_ordered_top5_count')}))


if __name__ == '__main__':
    main()
