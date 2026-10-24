"""Import the team's two FEVER ZIPs without overwriting frozen T2 data."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.offset_corpus import file_sha
from src.retrieval.resource_import import build_ordered_offsets, extract_member


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c1-zip', type=Path, required=True)
    parser.add_argument('--corpus-zip', type=Path, required=True)
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.resource_dir.exists() or args.report.exists():
        parser.error('Choose new resource/report paths; existing imports are preserved')
    args.resource_dir.mkdir(parents=True)
    receipt = {'schema_version': 'fever_resource_import_v1', 'passed': False,
               'frozen_t2_data_modified': False, 'archives': {}, 'extracted': {}}
    for name, path in [('c1', args.c1_zip), ('corpus', args.corpus_zip)]:
        receipt['archives'][name] = {'name': path.name, 'bytes': path.stat().st_size, 'sha256': file_sha(path)}
    for archive, member, name in [
        (args.c1_zip, 'for_fever_c1/fever_bge_ivfpq.index', 'fever_bge_ivfpq.index'),
        (args.c1_zip, 'for_fever_c1/fever_bge_ivfpq_ids.jsonl', 'fever_bge_ivfpq_ids.jsonl'),
        (args.c1_zip, 'for_fever_c1/fever_bge_ivfpq_meta.json', 'fever_bge_ivfpq_meta.json'),
        (args.corpus_zip, 'fever_corpus.jsonl', 'fever_corpus.jsonl'),
    ]:
        receipt['extracted'][name] = extract_member(archive, member, args.resource_dir / name)
        print(json.dumps({'extracted': name}), flush=True)
    meta = json.loads((args.resource_dir / 'fever_bge_ivfpq_meta.json').read_text())
    if (meta['corpus_size'], meta['ntotal'], meta['embedding_dimension'], meta['faiss_index_type']) != (
            5_396_106, 5_396_106, 768, 'IndexIVFPQ'):
        raise ValueError('Unexpected FEVER index metadata')
    tracked_meta = ROOT / 'results/retrieval/fever_bge_ivfpq_meta.json'
    receipt['metadata_matches_tracked'] = file_sha(tracked_meta) == receipt['extracted']['fever_bge_ivfpq_meta.json']['sha256']
    if not receipt['metadata_matches_tracked']:
        raise ValueError('Downloaded metadata differs from tracked provenance')
    receipt['mapping'] = build_ordered_offsets(
        args.resource_dir / 'fever_corpus.jsonl', args.resource_dir / 'fever_bge_ivfpq_ids.jsonl',
        args.resource_dir / 'fever_bge_ivfpq_offsets.npy', expected_count=meta['corpus_size'],
        progress=lambda n: print(json.dumps({'mapped': n}), flush=True))
    # Hash the historical dev copy for provenance only; never replace T2's dev.
    with zipfile.ZipFile(args.corpus_zip) as source:
        import hashlib
        digest = hashlib.sha256()
        with source.open('fever_dev.jsonl') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        receipt['historical_dev_sha256'] = digest.hexdigest()
    receipt['passed'] = True
    receipt['remaining'] = ['FAISS structural/search validation', 'FEVER Controller provider and C1 generation',
                            'Original BGE model revision not recorded in team metadata']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2); handle.write('\n')
    print(json.dumps({'passed': True, 'report': str(args.report)}))


if __name__ == '__main__':
    main()
