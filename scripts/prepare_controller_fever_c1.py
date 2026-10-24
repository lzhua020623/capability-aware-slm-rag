"""Prepare Top5 for all frozen 900 FEVER tasks on CPU; no answer/gold use."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.answer_repair import MODEL_NAME, MODEL_REVISION
from src.controller.fever_c1 import ROLES, COUNTS, read_jsonl, validate_task, validate_retrieval
from src.controller.fever_resources import FeverFaissBackend, validate_resources
from src.controller.nq_train_resources import OffsetBGERetriever
from src.generation.prompts import FEVER_PROMPT_INSTRUCTION
from src.retrieval.isolated_bge import encode_queries
from src.retrieval.offset_corpus import OffsetCorpus, file_sha
from scripts.replay_nq_controller_retrieval import PreencodedQueryModel


def source_manifest(root=ROOT):
    paths = sorted((root/'src').rglob('*.py'))
    paths += [root/p for p in ('scripts/run_controller_fever_c1.py', 'scripts/prepare_controller_fever_c1.py',
              'configs/base.yaml', 'configs/preliminary.yaml', 'requirements-answer-repair-v8.txt')]
    return {str(p.relative_to(root)): file_sha(p) for p in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path, default=ROOT/'configs/controller/fever_resources_v1.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('Use a new collection directory')
    registry, paths, receipt = validate_resources(args.resources, root=ROOT)
    tasks, ordered = {}, []
    for role in ROLES:
        path = ROOT/f'artifacts/controller/data_v1/fever/{role}/c1_tasks.jsonl'
        rows = read_jsonl(path)
        if len(rows) != COUNTS[role]:
            raise ValueError('Frozen task count mismatch')
        for row in rows:
            validate_task(row)
        tasks[role] = rows; ordered.extend(rows)
    if len({r['sample_id'] for r in ordered}) != 900:
        raise ValueError('Duplicate frozen FEVER ID')
    sources = source_manifest()
    vectors, encoding = encode_queries([r['query'] for r in ordered], registry['verified_model_revision'])
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(str(paths['index']))
    args.output_dir.mkdir(parents=True)
    manifest = {'schema_version': 'controller_fever_c1_collection_v1', 'counts': COUNTS,
                'ordered_ids': {role:[r['sample_id'] for r in rows] for role,rows in tasks.items()},
                'source_files': sources, 'model_name': MODEL_NAME, 'model_revision': MODEL_REVISION,
                'max_new_tokens': 64, 'prompt_instruction': FEVER_PROMPT_INSTRUCTION,
                'data_manifest_sha256': file_sha(ROOT/'artifacts/controller/data_v1/manifest.json'),
                'protocol_sha256': file_sha(ROOT/'configs/controller/experiment_protocol_v1.json'),
                'task_sha256': {r:file_sha(ROOT/f'artifacts/controller/data_v1/fever/{r}/c1_tasks.jsonl') for r in ROLES},
                'retrieval_resources': receipt, 'encoding': encoding, 'files': {},
                'gold_included': False, 'generation_performed': False, 'quality_scored': False}
    with OffsetCorpus(paths['corpus'], paths['ids'], paths['offsets']) as store:
        dense = OffsetBGERetriever(PreencodedQueryModel([r['query'] for r in ordered],vectors),
                                  index, store, faiss.normalize_L2)
        backend = FeverFaissBackend(dense, expected_count=registry['corpus_size'])
        done = 0
        for role in ROLES:
            path = args.output_dir/role/'retrieval.jsonl'; path.parent.mkdir(parents=True)
            with path.open('x',encoding='utf-8') as handle:
                for task in tasks[role]:
                    hits = backend.retrieve(task['query'],dataset='fever',top_k=5)
                    row = {**task, 'evidence':[{**e, 'rank':i} for i,e in enumerate(hits,1)]}
                    validate_retrieval(row)
                    handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    done += 1
                    if done % 50 == 0:
                        print(json.dumps({'retrieved':done,'total':900}),flush=True)
            manifest['files'][role] = {'path':f'{role}/retrieval.jsonl', 'sha256':file_sha(path)}
    if sources != source_manifest():
        raise ValueError('Source changed during C1 collection')
    with (args.output_dir/'manifest.json').open('x') as handle:
        json.dump(manifest,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps({'retrieval_ready':True,'n':900,'generation_performed':False}),flush=True)


if __name__ == '__main__':
    main()
