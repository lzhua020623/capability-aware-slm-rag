"""Check the pinned BGE query snapshot against 10 already-exposed FEVER C1s."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controller.fever_resources import FeverFaissBackend, verify_import
from src.controller.nq_train_resources import OffsetBGERetriever
from src.retrieval.isolated_bge import encode_queries
from src.retrieval.offset_corpus import OffsetCorpus, file_sha
from scripts.replay_nq_controller_retrieval import PreencodedQueryModel


def historical_queries(path, limit=10):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            # Explicit whitelist; predictions and offline reference labels ignored.
            rows.append({'sample_id': str(row['sample_id']), 'query': row['question_or_claim'],
                         'top5_ids': [e['passage_id'] for e in row['evidence']]})
            if len(rows) == limit:
                break
    if len(rows) != limit or len({r['sample_id'] for r in rows}) != limit:
        raise ValueError('Expected the fixed number of unique historical queries')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-dir', type=Path, default=ROOT/'data/processed/controller_resources/fever_v1')
    parser.add_argument('--import-report', type=Path, default=ROOT/'artifacts/controller/fever_resources_v1/import_validation.json')
    parser.add_argument('--structure-report', type=Path, default=ROOT/'artifacts/controller/fever_resources_v1/structure_validation.json')
    parser.add_argument('--historical-c1', type=Path, default=ROOT/'results/preliminary/fever_c1_7b.jsonl')
    parser.add_argument('--revision', default='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Preserve historical replay outputs')
    paths, fingerprints = verify_import(args.resource_dir, args.import_report, args.structure_report)
    rows = historical_queries(args.historical_c1)
    queries = [r['query'] for r in rows]
    vectors, provenance = encode_queries(queries, args.revision)
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(str(paths['index']))
    checks = []
    with OffsetCorpus(paths['corpus'], paths['ids'], paths['offsets']) as store:
        dense = OffsetBGERetriever(PreencodedQueryModel(queries, vectors), index, store, faiss.normalize_L2)
        backend = FeverFaissBackend(dense, expected_count=5_396_106)
        for row in rows:
            top20 = backend.retrieve(row['query'], dataset='fever', top_k=20)
            top5 = backend.retrieve(row['query'], dataset='fever', top_k=5)
            if len(top20) != 20 or top20[:5] != top5:
                raise ValueError('FEVER Top20/Top5 consistency failure')
            actual = [r['passage_id'] for r in top5]
            checks.append({'sample_id': row['sample_id'], 'reference_top5': row['top5_ids'],
                           'actual_top5': actual, 'same_ordered_ids': actual == row['top5_ids'],
                           'top5_scores': [r['score'] for r in top5]})
    matched = sum(r['same_ordered_ids'] for r in checks)
    report = {'schema_version': 'fever_controller_query_replay_v1', 'passed': matched == 10,
              'n_queries': 10, 'same_ordered_top5_count': matched, 'encoding': provenance,
              'historical_c1_sha256': file_sha(args.historical_c1),
              'import_report_sha256': file_sha(args.import_report),
              'structure_report_sha256': file_sha(args.structure_report),
              'resource_fingerprints': fingerprints, 'checks': checks, 'quality_scored': False,
              'scope': 'Fixed first 10 historical C1 queries; no gold/prediction use, not final cohort'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2); handle.write('\n')
    print(json.dumps({'passed': report['passed'], 'same_ordered_top5_count': matched}), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
