"""Route 1's P3-20 on every sweep item that has no stored P3-20 outcome; no gold is read.

Items are rebuilt from the stored route-1 sweep records (C1 answer, Top-5, Diagnosis,
frozen selection), so C1 is not regenerated. P3-20 runs through the production executor
as a forced decision, as in scripts/precompute_quality_sweep.py. The first items that
already have a stored outcome are run again to measure agreement with the stored run.
One JSON line per (sample, source), appended as it finishes, so a run resumes.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.precompute_quality_sweep import NQ_RESOURCES, file_sha, now, run_action, write  # noqa: E402
from scripts.precompute_quality_sweep import load_registration as load_sweep  # noqa: E402

REGISTRATION = 'configs/controller/l2_round_v1.json'
P3 = 'retrieval_enhancement'
REPRO_N = 200
SOURCES = ('scripts/precompute_p3_coverage.py', 'scripts/precompute_quality_sweep.py', 'src/pipeline/current.py',
           'src/compensation/retrieval_enhancement/p3.py', REGISTRATION)


def item_from_record(record):
    """The canonical NQ item P3-20 received in the sweep."""
    from src.diagnosis.schema import DiagnosisInput, EvidenceItem
    return DiagnosisInput(sample_id=record['sample_id'], dataset='nq', query=record['query'],
                          evidence=[EvidenceItem(**e) for e in record['evidence']],
                          initial_answer=record['initial_answer'])


def selection_from_record(record):
    """The frozen selection; its confidence is the Diagnosis confidence, as in the adapter."""
    from src.controller.schema import SelectionDecision
    s = record['frozen_selection']
    return SelectionDecision(needs_enhancement=s['needs_enhancement'], failure_type=s['failure_type'],
                             selection_score=s['selection_score'], selection_lane=s['selection_lane'],
                             confidence=record['diagnosis']['confidence'], p_fail=s['p_fail'],
                             p_insuff=s['p_insuff'], sample_id=record['sample_id'], d_combo=s['d_combo'],
                             rec=s['rec'], p_corr_3c=s['p_corr_3c'])


def plan(records, repro_n=REPRO_N):
    """(sample_id, source) in sweep order: 'new' for items without P3-20, then 'repro' for the first stored ones."""
    new = [(r['sample_id'], 'new') for r in records if P3 not in r['actions']]
    repro = [(r['sample_id'], 'repro') for r in records if P3 in r['actions']][:repro_n]
    return new + repro


def evidence_ids(action):
    return [e['passage_id'] for e in action['final_evidence']]


def agreement(stored, rerun):
    """Agreement of a re-run with stored P3-20 outcomes, by sample_id."""
    ids = sorted(rerun)
    same = lambda f: sum(f(stored[s]) == f(rerun[s]) for s in ids)
    return {'n': len(ids), 'final_answer': same(lambda a: a['final_answer']), 'status': same(lambda a: a['status']),
            'final_evidence': same(evidence_ids),
            'native_status': same(lambda a: ((a['native'] or {}).get('module_trace') or {}).get('status'))}


def done_keys(path):
    if not path.exists():
        return set()
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        path.write_bytes(data[:data.rfind(b'\n') + 1])
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return {(r['sample_id'], r['source']) for r in rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True, help='stored route-1 sweep records.jsonl')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=None, help='Engineering run: first N planned rows')
    args = p.parse_args(argv)
    registration = json.loads((ROOT / REGISTRATION).read_text(encoding='utf-8'))
    if file_sha(ROOT / registration['base']['registration']) != registration['base']['sha256']:
        raise ValueError('Route 1 registration differs from the L2 round registration')
    _, ids = load_sweep()
    from scripts.quality_sweep_offline import read_records
    records = read_records(args.records, ids)
    by_id = {r['sample_id']: r for r in records}
    todo_all = plan(records)
    todo_all = todo_all[:args.limit] if args.limit else todo_all
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('P3-20 regeneration requires CUDA')
    if torch.get_num_threads() != 16:
        raise RuntimeError(f'Set OMP_NUM_THREADS=16 (torch uses {torch.get_num_threads()})')
    from src.pipeline.current import load_current_pipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / 'p3_records.jsonl'
    finished = done_keys(out)
    pipeline, environment = load_current_pipeline(root=ROOT, nq_resources=ROOT / NQ_RESOURCES,
                                                  data_role='validation', attach_reference_rag=True)
    pipeline.data_boundary.require(ids, 'validation')
    manifest = {'schema_version': 'p3_coverage_v1', 'state': 'running', 'started_UTC': now(),
                'registration_sha256': file_sha(ROOT / REGISTRATION), 'records_sha256': file_sha(args.records),
                'engineering_run': bool(args.limit), 'planned': len(todo_all), 'resumed_with': len(finished),
                'cpu_threads': torch.get_num_threads(), 'sources': {s: file_sha(ROOT / s) for s in SOURCES},
                'environment': environment}
    write(args.output_dir / 'manifest.json', manifest)
    todo = [key for key in todo_all if key not in finished]
    started = time.perf_counter()
    with out.open('a', encoding='utf-8') as handle:
        for i, (sid, source) in enumerate(todo, 1):
            record = by_id[sid]
            action = run_action(pipeline.host.executor, item_from_record(record), selection_from_record(record), P3)
            handle.write(json.dumps({'sample_id': sid, 'source': source, 'action': action},
                                    ensure_ascii=False, default=str) + '\n')
            handle.flush()
            if i % 50 == 0 or i == len(todo):
                print(json.dumps({'done': len(finished) + i, 'of': len(todo_all),
                                  'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    rows = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines() if line.strip()]
    rerun = {r['sample_id']: r['action'] for r in rows if r['source'] == 'repro'}
    repro = agreement({sid: by_id[sid]['actions'][P3] for sid in rerun}, rerun)
    manifest.update(state='completed', finished_UTC=now(), rows=len(rows),
                    counts={s: sum(r['source'] == s for r in rows) for s in ('new', 'repro')},
                    output_sha256=file_sha(out), repro=repro)
    write(args.output_dir / 'manifest.json', manifest)
    print(json.dumps({'rows': len(rows), 'repro': repro}), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
