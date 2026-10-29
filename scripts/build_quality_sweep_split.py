"""Validation sweep/confirm IDs for the quality round, continuing the E-distillation ordering.

Development500 and Selection500 were the first 1,000 IDs of the allowed Validation pool
ordered by sha256([seed, 'validation', id]). This continues the same ordering: the next
6,000 IDs form the sweep set, the rest the confirm set. Only IDs are read; questions are
exported separately with no gold or annotation field.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.data.team_nq import BoundaryError, TeamNQBoundary

SEED = 'e_distillation_v1_20261027'
PURPOSE = 'validation'
USED = 1000
SWEEP_N = 6000
OUT = 'artifacts/controller/quality_sweep_v1'
QUERY_FIELDS = ('sample_id', 'query')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ordered_validation(boundary):
    return sorted(boundary.allowed_ids('validation'), key=lambda sid: (digest([SEED, PURPOSE, sid]), sid))


def split(boundary):
    order = ordered_validation(boundary)
    if len(order) < USED + SWEEP_N + 1:
        raise BoundaryError('Validation pool too small for the registered split')
    return {'development_selection': order[:USED], 'sweep': order[USED:USED + SWEEP_N],
            'confirm': order[USED + SWEEP_N:]}


def ids_bytes(ids):
    """CRLF bytes, as a core.autocrlf=true checkout writes them, so the registered hash is stable."""
    return (json.dumps(ids, indent=0) + '\n').replace('\n', '\r\n').encode()


def export_queries(boundary, delivery, ids, out):
    """sample_id and query only, in registered order; refuses missing or duplicate IDs."""
    wanted, rows = set(ids), {}
    for path in sorted(Path(delivery).glob('nq_train_annotation_*.jsonl')):
        for line in path.open(encoding='utf-8'):
            row = json.loads(line)
            if row['sample_id'] in wanted:
                if row['sample_id'] in rows:
                    raise BoundaryError('Duplicate sweep ID in delivery')
                rows[row['sample_id']] = {k: row[k] for k in QUERY_FIELDS}
    if set(rows) != wanted:
        raise BoundaryError('Delivery lacks registered IDs')
    boundary.require(ids, 'validation')
    with Path(out).open('x', encoding='utf-8') as handle:
        for sid in ids:
            handle.write(json.dumps(rows[sid], ensure_ascii=False) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('ids')
    q = sub.add_parser('queries')
    q.add_argument('--delivery', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    boundary = TeamNQBoundary(ROOT)
    roles = split(boundary)
    if args.command == 'ids':
        out = ROOT / OUT
        out.mkdir(parents=True, exist_ok=True)
        for role in ('sweep', 'confirm'):
            (out / f'{role}_ids.json').write_bytes(ids_bytes(roles[role]))
        print(json.dumps({role: len(ids) for role, ids in roles.items()}))
    else:
        export_queries(boundary, args.delivery, roles['sweep'] + roles['confirm'], args.output)
    return 0


if __name__ == '__main__': raise SystemExit(main())
