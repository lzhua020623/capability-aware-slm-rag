"""NQ recoverability ladder, oracle levels (configs/controller/recoverability_oracle_v1.json).

contexts: for every sweep item, the L3 context (gold passage first, then C1's Top-5 without it, first four)
          and the L4 context (gold passage alone) as l2_generate.py rows. Gold passage IDs come from the
          team delivery (local only); rows hold sample_id, query and passage_ids, with no marker of which
          passage is gold. A context equal to C1's Top-5 needs no generation (C1's answer).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.quality_sweep_offline import read_jsonl, read_records  # noqa: E402

REGISTRATION = 'configs/controller/recoverability_oracle_v1.json'
LEVELS = ('L3', 'L4')


def load_registration(root=ROOT):
    return json.loads((Path(root) / REGISTRATION).read_text(encoding='utf-8'))


def sweep_ids(root=ROOT):
    return json.loads((Path(root) / load_registration(root)['set']['ids']).read_text(encoding='utf-8'))


def gold_passages(delivery, ids):
    """sample_id -> gold passage ID from the team delivery shards (offline reference fields only)."""
    wanted, out = set(ids), {}
    for path in sorted(Path(delivery).glob('nq_train_annotation_*.jsonl')):
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if row['sample_id'] in wanted:
                    pid = row['annotation_reference']['gold_passage_id']
                    if not isinstance(pid, str) or not pid:
                        raise ValueError('Missing gold passage ID for ' + row['sample_id'])
                    out[row['sample_id']] = pid
    if set(out) != wanted:
        raise ValueError(f'{len(wanted - set(out))} IDs lack a delivery row')
    return out


def oracle_contexts(c1_top5, gold):
    """Registered L3 and L4 passage lists for one item."""
    return {'L3': (gold, *[p for p in c1_top5 if p != gold][:4]), 'L4': (gold,)}


def context_rows(records, gold, corpus_ids):
    """l2_generate.py rows for every oracle context that differs from C1's Top-5; items whose gold passage
    is not in the corpus are returned separately."""
    rows, missing, seen = [], [], set()
    for r in records:
        sid, c1 = r['sample_id'], tuple(e['passage_id'] for e in r['evidence'])
        if gold[sid] not in corpus_ids:
            missing.append(sid)
            continue
        for ids in oracle_contexts(c1, gold[sid]).values():
            if ids != c1 and (sid, ids) not in seen:
                seen.add((sid, ids))
                rows.append({'sample_id': sid, 'query': r['query'], 'passage_ids': list(ids)})
    return rows, missing


def corpus_passage_ids(path):
    with Path(path).open(encoding='utf-8') as handle:
        return {json.loads(line)['passage_id'] for line in handle if line.strip()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('contexts')
    c.add_argument('--records', type=Path, required=True, help='route-1 sweep records.jsonl')
    c.add_argument('--delivery', type=Path, required=True, help='team nq_train_delivery directory (local only)')
    c.add_argument('--corpus', type=Path, required=True, help='NQ Train corpus jsonl')
    c.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    ids = sweep_ids()
    records = read_records(args.records, ids)
    rows, missing = context_rows(records, gold_passages(args.delivery, ids), corpus_passage_ids(args.corpus))
    with args.output.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'items': len(records), 'contexts_to_generate': len(rows), 'gold_not_in_corpus': len(missing),
                      'by_level': {n: sum(len(r['passage_ids']) == (1 if n == 'L4' else 5) for r in rows) for n in LEVELS}}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
