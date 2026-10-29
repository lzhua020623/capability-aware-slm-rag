"""Quality-round GPU pass on the registered Validation sweep IDs; no gold is read.

Per item: C1 through the production Reference RAG with full prefill, Diagnosis, the 78
P60 features and the frozen selection (raw scores kept for offline recalibration), then
every registered candidate action run through the production executor as if P60 had
selected the item. One JSON line per item, appended as it finishes, so a run resumes.
The first IDs are then prepared a second time to measure run-to-run drift.
"""
import argparse
from dataclasses import asdict, replace
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

REGISTRATION = 'configs/controller/quality_sweep_v1.json'
NQ_RESOURCES = 'configs/controller/nq_train_resources_v1.json'
OWN_ACTION = {'evidence_insufficiency': 'retrieval_enhancement',
              'evidence_accessibility': 'evidence_refinement',
              'evidence_utilisation': 'generation_repair'}
LABEL_FOR_ACTION = {action: label for label, action in OWN_ACTION.items()}
REPRO_N = 200
SOURCES = ('scripts/precompute_quality_sweep.py', 'src/pipeline/reference_rag.py', 'src/pipeline/current.py',
           'src/controller/selection.py', 'src/controller/execution.py', REGISTRATION)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_registration(root=ROOT):
    registration = json.loads((Path(root) / REGISTRATION).read_text(encoding='utf-8'))
    spec = registration['split']['sweep']
    if file_sha(Path(root) / spec['path']) != spec['sha256']:
        raise ValueError('Sweep ID list differs from its registration')
    ids = json.loads((Path(root) / spec['path']).read_text(encoding='utf-8'))
    if len(ids) != spec['n'] or len(set(ids)) != len(ids):
        raise ValueError('Sweep ID list has the wrong size or duplicates')
    return registration, ids


def read_queries(path, ids):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    if any(set(row) != {'sample_id', 'query'} for row in rows):
        raise ValueError('The query file must hold sample_id and query only')
    by_id = {row['sample_id']: row['query'] for row in rows}
    missing = [sid for sid in ids if sid not in by_id]
    if missing:
        raise ValueError(f'{len(missing)} registered IDs lack a query')
    return [(sid, by_id[sid]) for sid in ids]


def gated_correct(diagnosis):
    return diagnosis['final_label'] == 'correct' or diagnosis['stage1_label'] == 'correct'


def actions_for(diagnosis, correct_cut):
    """Registered candidates: P3-20 for every failure plus the label's own module; P3-20 for high-p_fail correct."""
    if gated_correct(diagnosis):
        return ['retrieval_enhancement'] if diagnosis['stage1_failure_probability'] >= correct_cut else []
    own = OWN_ACTION[diagnosis['final_label']]
    return ['retrieval_enhancement'] + ([own] if own != 'retrieval_enhancement' else [])


def label_scores(scores):
    return None if scores is None else {getattr(k, 'value', k): float(v) for k, v in scores.items()}


def diagnosis_dict(d):
    return {'final_label': d.final_label.value, 'stage1_label': d.stage1_label,
            'stage1_failure_probability': float(d.stage1_failure_probability),
            'stage2_class_scores': label_scores(d.stage2_class_scores),
            'class_scores': label_scores(d.class_scores), 'confidence': float(d.confidence)}


def selection_dict(s):
    number = lambda v: None if v is None else float(v)
    return {'needs_enhancement': s.needs_enhancement, 'failure_type': s.failure_type.value,
            'selection_lane': s.selection_lane.value, 'selection_score': number(s.selection_score),
            'p_fail': number(s.p_fail), 'p_insuff': number(s.p_insuff), 'd_combo': number(s.d_combo),
            'rec': number(s.rec), 'p_corr_3c': number(s.p_corr_3c)}


def forced_decision(selection, action):
    """The production Controller decision for `action`, as if P60 had selected the item."""
    from src.controller.controller import decide
    from src.controller.schema import SelectionLane
    from src.diagnosis.labels import DiagnosisLabel
    # The lane only explains a selection; routing never reads it.
    lane = selection.selection_lane if selection.needs_enhancement else SelectionLane.RESIDUAL_COMBO
    decision = decide(replace(selection, needs_enhancement=True, selection_lane=lane),
                      failure_type_override=DiagnosisLabel(LABEL_FOR_ACTION[action]))
    if decision.action.value != action:
        raise RuntimeError(f'Forced decision routed to {decision.action.value}, not {action}')
    return decision


def run_action(executor, item, selection, action):
    from src.compensation.schema import EnhancementRequest
    request = EnhancementRequest.from_runtime(query=item.query, initial_answer=item.initial_answer,
        evidence=item.evidence, controller_decision=forced_decision(selection, action),
        sample_id=item.sample_id, dataset=item.dataset.value)
    started = time.perf_counter()
    result = executor.execute(request)
    return {'action': action, 'final_answer': result.final_answer, 'changed_answer': result.changed_answer,
            'status': result.status.value, 'error': result.error,
            'final_evidence': [asdict(e) for e in result.evidence],
            'native': result.metadata.get('native'), 'enhancement_cost': result.metadata.get('enhancement_cost'),
            'seconds': time.perf_counter() - started}


def prepare(pipeline, batch):
    """C1 (full prefill) -> Diagnosis -> 78 features -> frozen selection, as in CurrentPipeline.run_batch."""
    from src.controller.runtime import canonical_input
    rag, rows = pipeline.host.reference_rag, []
    for sid, query in batch:
        started = time.perf_counter()
        item = rag.run(query, sample_id=sid, dataset='nq')
        rows.append((canonical_input(item), item.initial_answer, time.perf_counter() - started))
    items = [row[0] for row in rows]
    diagnoses = [pipeline.host.classifier.diagnose(item) for item in items]
    features = pipeline.features.features_for_batch(items, diagnoses)
    selections = [pipeline.host.selection_adapter.select(d, f, sample_id=i.sample_id)
                  for i, d, f in zip(items, diagnoses, features)]
    return [(item, raw, seconds, d, f, s) for (item, raw, seconds), d, f, s in zip(rows, diagnoses, features, selections)]


def prepared_record(item, raw, seconds, diagnosis, features, selection):
    return {'sample_id': item.sample_id, 'query': item.query, 'initial_answer': item.initial_answer,
            'initial_answer_raw': raw, 'evidence': [asdict(e) for e in item.evidence],
            'diagnosis': diagnosis_dict(diagnosis), 'frozen_selection': selection_dict(selection),
            'features': [float(v) for v in features], 'c1_seconds': seconds}


def done_ids(path):
    """IDs already written; a line cut off by an interrupted write is dropped first."""
    if not path.exists():
        return set()
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        path.write_bytes(data[:data.rfind(b'\n') + 1])
    return {json.loads(line)['sample_id'] for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def compare(first, second):
    """Run-to-run drift between two prepared records of the same ID."""
    feature_gap = max(abs(a - b) for a, b in zip(first['features'], second['features']))
    return {'c1_same': first['initial_answer'] == second['initial_answer'],
            'label_same': first['diagnosis']['final_label'] == second['diagnosis']['final_label'],
            'selected_same': (first['frozen_selection']['needs_enhancement']
                              == second['frozen_selection']['needs_enhancement']),
            'p_fail_gap': abs(first['diagnosis']['stage1_failure_probability']
                              - second['diagnosis']['stage1_failure_probability']),
            'max_feature_gap': feature_gap}


def write(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', type=Path, required=True, help='sample_id/query only export')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=None, help='Engineering run: first N sweep IDs')
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args(argv)
    registration, ids = load_registration()
    if file_sha(args.queries) != registration['split']['query_export']['sha256']:
        raise ValueError('Query export differs from its registration')
    ids = ids[:args.limit] if args.limit else ids
    queries = read_queries(args.queries, ids)
    cut = registration['precompute']['correct_p_fail_cut']
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('The sweep pass requires CUDA')
    from src.pipeline.current import load_current_pipeline
    from src.pipeline.reference_rag import full_prefill_generate

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = args.output_dir / 'records.jsonl'
    finished = done_ids(records)
    pipeline, environment = load_current_pipeline(root=ROOT, nq_resources=ROOT / NQ_RESOURCES,
                                                  data_role='validation', attach_reference_rag=True)
    pipeline.data_boundary.require(ids, 'validation')
    rag = pipeline.host.reference_rag
    rag._generate = full_prefill_generate
    shared_chunk = rag.model.generation_config.prefill_chunk_size
    manifest = {'schema_version': 'quality_sweep_precompute_v1', 'state': 'running', 'started_UTC': now(),
                'registration_sha256': file_sha(ROOT / REGISTRATION), 'query_sha256': file_sha(args.queries),
                'engineering_run': bool(args.limit), 'n_ids': len(ids), 'resumed_with': len(finished),
                'sources': {s: file_sha(ROOT / s) for s in SOURCES}, 'c1_prefill': 'full',
                'shared_prefill_chunk_size': shared_chunk, 'correct_p_fail_cut': cut, 'environment': environment}
    write(args.output_dir / 'manifest.json', manifest)

    todo = [row for row in queries if row[0] not in finished]
    started, counts = time.perf_counter(), {}
    with records.open('a', encoding='utf-8') as handle:
        for offset in range(0, len(todo), args.batch_size):
            for item, raw, seconds, diagnosis, features, selection in prepare(pipeline, todo[offset:offset + args.batch_size]):
                record = prepared_record(item, raw, seconds, diagnosis, features, selection)
                record['actions'] = {}
                for action in actions_for(record['diagnosis'], cut):
                    record['actions'][action] = run_action(pipeline.host.executor, item, selection, action)
                    counts[action] = counts.get(action, 0) + 1
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                handle.flush()
            done = min(offset + args.batch_size, len(todo))
            print(json.dumps({'done': done, 'of': len(todo), 'seconds': round(time.perf_counter() - started, 1),
                              'actions': counts}), flush=True)
    if rag.model.generation_config.prefill_chunk_size != shared_chunk:
        raise RuntimeError('The shared prefill setting changed during the run')

    first = {json.loads(line)['sample_id']: json.loads(line)
             for line in records.read_text(encoding='utf-8').splitlines()}
    repro_ids = queries[:REPRO_N]
    gaps = [compare(first[row[0]], prepared_record(*prepared))
            for offset in range(0, len(repro_ids), args.batch_size)
            for row, prepared in zip(repro_ids[offset:offset + args.batch_size],
                                     prepare(pipeline, repro_ids[offset:offset + args.batch_size]))]
    repro = {'n': len(gaps), 'c1_same': sum(g['c1_same'] for g in gaps),
             'label_same': sum(g['label_same'] for g in gaps), 'selected_same': sum(g['selected_same'] for g in gaps),
             'max_p_fail_gap': max((g['p_fail_gap'] for g in gaps), default=0.0),
             'max_feature_gap': max((g['max_feature_gap'] for g in gaps), default=0.0)}
    write(args.output_dir / 'repro.json', repro)
    manifest.update(state='completed', finished_UTC=now(), records=len(first), actions_run=counts,
                    records_sha256=file_sha(records), repro=repro)
    write(args.output_dir / 'manifest.json', manifest)
    print(json.dumps({'records': len(first), 'repro': repro}), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
