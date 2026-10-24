"""Runtime-only FEVER C1 collection contracts; no gold or quality evaluation."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from src.compensation.answer_repair import MODEL_NAME, MODEL_REVISION
from src.diagnosis.schema import DiagnosisInput, EvidenceItem
from src.evaluation.fever import parse_fever_prediction
from src.generation.prompts import FEVER_PROMPT_INSTRUCTION, build_fever_rag_prompt
from src.retrieval.offset_corpus import file_sha

ROLES = ('development', 'selection', 'final_candidate')
COUNTS = {'development': 200, 'selection': 200, 'final_candidate': 500}
REQUIRED_SOURCES = {'src/controller/fever_c1.py', 'src/generation/prompts.py',
                    'src/generation/qwen.py', 'src/compensation/qwen_repair.py',
                    'src/compensation/answer_repair.py', 'scripts/run_controller_fever_c1.py',
                    'configs/base.yaml', 'requirements-answer-repair-v8.txt'}


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_task(row):
    if (set(row) != {'dataset', 'sample_id', 'query'} or row['dataset'] != 'fever'
            or not isinstance(row['sample_id'], str) or not row['sample_id'].strip()
            or not isinstance(row['query'], str) or not row['query'].strip()):
        raise ValueError('Expected a strict runtime-only FEVER task')


def validate_retrieval(row):
    if set(row) != {'dataset', 'sample_id', 'query', 'evidence'}:
        raise ValueError('Unexpected retrieval fields; offline references must be separate')
    validate_task({k: row[k] for k in ('dataset', 'sample_id', 'query')})
    evidence = row['evidence']
    if not isinstance(evidence, list) or len(evidence) != 5:
        raise ValueError('C1 needs exactly five passages')
    for i, item in enumerate(evidence, 1):
        if set(item) != {'rank', 'passage_id', 'text', 'score'} or item['rank'] != i:
            raise ValueError('Invalid C1 evidence schema/order')
        EvidenceItem(**item)  # Schema checks text/ID/finite score.
        if not item['text'].strip() or not item['passage_id'].strip() or not math.isfinite(item['score']):
            raise ValueError('Empty evidence or non-finite C1 retrieval score')
    if len({e['passage_id'] for e in evidence}) != 5:
        raise ValueError('Duplicate evidence ID')
    if any(a['score'] < b['score'] for a,b in zip(evidence, evidence[1:])):
        raise ValueError('C1 evidence must preserve BGE retrieval order')


def verify_collection(directory, *, root):
    directory, root = Path(directory), Path(root)
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest.get('schema_version') != 'controller_fever_c1_collection_v1'
            or manifest.get('counts') != COUNTS
            or manifest.get('model_name') != MODEL_NAME or manifest.get('model_revision') != MODEL_REVISION
            or manifest.get('max_new_tokens') != 64
            or manifest.get('prompt_instruction') != FEVER_PROMPT_INSTRUCTION
            or set(manifest.get('files', {})) != set(ROLES)
            or not REQUIRED_SOURCES <= set(manifest.get('source_files', {}))
            or manifest.get('gold_included') is not False
            or manifest.get('generation_performed') is not False):
        raise ValueError('C1 collection differs from the frozen plan')
    for relative, expected in manifest['source_files'].items():
        if file_sha(root/relative) != expected:
            raise ValueError(f'C1 source fingerprint mismatch: {relative}')
    rows_by_role, seen = {}, set()
    for role in ROLES:
        info = manifest['files'][role]
        if info['path'] != f'{role}/retrieval.jsonl':
            raise ValueError('Unexpected collection file path')
        path = directory/info['path']
        if file_sha(path) != info['sha256']:
            raise ValueError(f'C1 retrieval fingerprint mismatch: {role}')
        rows = read_jsonl(path)
        if len(rows) != COUNTS[role] or [r['sample_id'] for r in rows] != manifest['ordered_ids'][role]:
            raise ValueError('Missing/reordered frozen C1 tasks')
        for row in rows:
            validate_retrieval(row)
            if row['sample_id'] in seen:
                raise ValueError('C1 sample ID repeated across roles')
            seen.add(row['sample_id'])
        rows_by_role[role] = rows
    return manifest, rows_by_role


def generate_one(row, generator):
    validate_retrieval(row)
    prompt = build_fever_rag_prompt(row['query'], row['evidence'])
    generation = generator.generate(prompt, 64)
    # Preserve raw observed output in runtime, just as the old C1 adapter does.
    # Empty output is an explicit UNKNOWN observation; never invent a binary label.
    observed = generation.text.strip()
    runtime = {**row, 'initial_answer': observed or 'UNKNOWN'}
    DiagnosisInput(runtime['sample_id'], 'fever', runtime['query'],
                   [EvidenceItem(**e) for e in runtime['evidence']], runtime['initial_answer'])
    trace = {'sample_id': row['sample_id'], 'dataset': 'fever', 'prompt': prompt,
             'generation': asdict(generation), 'parsed_label': parse_fever_prediction(generation.text),
             'empty_output_as_unknown': not observed}
    return runtime, trace
