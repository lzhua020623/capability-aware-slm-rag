"""Run the frozen quality-round pipeline once on the confirm set or the team Test; questions only.

GPU side: `load_current_pipeline(registration=quality_baseline_v1)` and `run_batch`, one result
per line in input order, resumable. Gold never reaches this host; the local scorer
(`scripts/score_quality_eval.py`) builds the team's predictions.jsonl from these results.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

REGISTRATION = 'configs/controller/quality_baseline_v1.json'
NQ_RESOURCES = 'configs/controller/nq_train_resources_v1.json'
ROLES = {'confirm': 'validation', 'test': 'test'}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_queries(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    if any(set(row) != {'sample_id', 'query'} for row in rows):
        raise ValueError('The query file must hold sample_id and query only')
    return [(row['sample_id'], row['query']) for row in rows]


def expected_ids(role, root=ROOT):
    """The registered confirm IDs, or the team Test IDs in the team split order."""
    if role == 'confirm':
        sweep = json.loads((root / 'configs/controller/quality_sweep_v1.json').read_text(encoding='utf-8'))
        return json.loads((root / sweep['split']['confirm']['path']).read_text(encoding='utf-8'))
    from src.data.team_nq import TeamNQBoundary
    boundary = TeamNQBoundary(root, allow_sealed_test=True)
    return [sid for sid, role_of in boundary.membership.items() if role_of == 'test']


def done_ids(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        path.write_bytes(data[:data.rfind(b'\n') + 1])
    return [json.loads(line)['sample_id'] for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=tuple(ROLES), required=True)
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--allow-sealed-test', action='store_true')
    p.add_argument('--limit', type=int, default=None, help='Engineering run on confirm only')
    args = p.parse_args(argv)
    if args.role == 'test' and (not args.allow_sealed_test or args.limit):
        raise SystemExit('The Test run needs --allow-sealed-test and no --limit')
    queries = read_queries(args.queries)
    ids = expected_ids(args.role)
    if sorted(sid for sid, _ in queries) != sorted(ids):
        raise ValueError('Query file does not hold exactly the registered IDs')
    by_id = dict(queries)
    order = ids[:args.limit] if args.limit else ids
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Requires CUDA')
    from src.pipeline.current import load_current_pipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = args.output_dir / 'results.jsonl'
    finished = done_ids(results)
    if finished != order[:len(finished)]:
        raise ValueError('Existing results are not a prefix of the registered order')
    pipeline, environment = load_current_pipeline(
        root=ROOT, nq_resources=ROOT / NQ_RESOURCES, data_role=ROLES[args.role], attach_reference_rag=True,
        allow_sealed_test=args.role == 'test', registration=REGISTRATION)
    manifest = {'schema_version': 'quality_eval_v1', 'state': 'running', 'role': args.role,
                'data_role': ROLES[args.role], 'registration': REGISTRATION,
                'registration_sha256': file_sha(ROOT / REGISTRATION), 'query_sha256': file_sha(args.queries),
                'n_ids': len(order), 'engineering_run': bool(args.limit), 'resumed_with': len(finished),
                'started_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'environment': environment}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    todo, started = order[len(finished):], time.perf_counter()
    with results.open('a', encoding='utf-8') as handle:
        for offset in range(0, len(todo), args.batch_size):
            batch = [(by_id[sid], sid) for sid in todo[offset:offset + args.batch_size]]
            for result in pipeline.run_batch(batch, feature_batch_size=len(batch)):
                handle.write(json.dumps(result.to_dict(), ensure_ascii=False, default=str) + '\n')
            handle.flush()
            print(json.dumps({'done': len(finished) + offset + len(batch), 'of': len(order),
                              'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    manifest.update(state='completed', results=len(done_ids(results)), results_sha256=file_sha(results),
                    finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    return 0


if __name__ == '__main__': raise SystemExit(main())
