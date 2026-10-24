"""Read-only structural/search check of imported FEVER assets, without Torch."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.offset_corpus import OffsetCorpus, file_sha


def validate_index(index, metadata):
    expected = {'ntotal': metadata['corpus_size'], 'd': 768, 'nlist': 4096,
                'nprobe': 16, 'metric_type': 0, 'is_trained': True, 'by_residual': False}
    if type(index).__name__ != 'IndexIVFPQ':
        raise ValueError('Expected IndexIVFPQ')
    for key, value in expected.items():
        if getattr(index, key, None) != value:
            raise ValueError(f'Unexpected FEVER index {key}')
    if index.pq.M != 64 or index.pq.nbits != 8:
        raise ValueError('Unexpected FEVER PQ encoding')
    return {**expected, 'type': 'IndexIVFPQ', 'm': index.pq.M, 'nbits': index.pq.nbits}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--import-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Choose a new validation output')
    receipt = json.loads(args.import_report.read_text())
    if receipt.get('passed') is not True or receipt['mapping'].get('full_order_verified') is not True:
        raise ValueError('Import/mapping has not passed')
    files = dict(receipt['extracted'])
    files['fever_bge_ivfpq_offsets.npy'] = receipt['mapping']['fingerprints']['offsets']
    for name, info in files.items():
        path = args.resource_dir/name
        if path.stat().st_size != info['bytes'] or file_sha(path) != info['sha256']:
            raise ValueError(f'Resource changed after import: {name}')
    metadata = json.loads((args.resource_dir/'fever_bge_ivfpq_meta.json').read_text())
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(str(args.resource_dir/'fever_bge_ivfpq.index'))
    structure = validate_index(index, metadata)
    checks = []
    # Reconstructed PQ vectors test the actual FAISS -> ID -> original-text
    # lookup. They do not prove semantic BGE compatibility or QA improvement.
    with OffsetCorpus(args.resource_dir/'fever_corpus.jsonl', args.resource_dir/'fever_bge_ivfpq_ids.jsonl',
                      args.resource_dir/'fever_bge_ivfpq_offsets.npy') as store:
        if len(store) != index.ntotal:
            raise ValueError('Index and mapping counts differ')
        index.make_direct_map()
        for position in (0, index.ntotal//2, index.ntotal-1):
            query = np.asarray(index.reconstruct(position), dtype=np.float32).reshape(1, -1)
            if not np.isfinite(query).all() or np.linalg.norm(query) == 0:
                raise ValueError('Invalid PQ reconstruction')
            faiss.normalize_L2(query)
            scores, positions = index.search(query, 20)
            valid = [int(p) for p in positions[0] if p >= 0]
            if len(valid) != 20 or not np.isfinite(scores).all():
                raise ValueError('Invalid FEVER Top20 result')
            checks.append({'source_position': position,
                           'source_passage_id': store[position]['passage_id'],
                           'retrieved_top20_ids': [store[p]['passage_id'] for p in valid]})
    report = {'schema_version': 'fever_resource_structure_v1', 'passed': True,
              'import_report_sha256': file_sha(args.import_report), 'structure': structure,
              'full_mapping_records': receipt['mapping']['records_checked'],
              'faiss_version': faiss.__version__, 'search_checks': checks,
              'search_scope': '3 reconstructed PQ vectors, engineering lookup only; no question/gold read',
              'semantic_bge_alignment_verified': False, 'quality_scored': False,
              'formal_execution_ready': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2); handle.write('\n')
    print(json.dumps({'passed': True, 'full_mapping_records': report['full_mapping_records']}))


if __name__ == '__main__':
    main()
