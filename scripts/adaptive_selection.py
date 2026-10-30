"""Adaptive multi-context answer selection (configs/controller/adaptive_selection_v1.json): GPU-side inputs.

`contexts` lists, for every item of a route-1 run (confirm results or Test results), the Top-5 of each
registered context that still needs a 7B answer: Top-5 lists equal to C1's take the C1 answer, and Top-5
lists already answered (route 1's P3-20 traces, the L2 / BM25 generation runs) are reused by (sample, Top-5).
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import l2_offline  # noqa: E402

REGISTRATION = 'configs/controller/adaptive_selection_v1.json'
P3 = 'retrieval_enhancement'


def load_registration(root=ROOT):
    return json.loads((Path(root) / REGISTRATION).read_text(encoding='utf-8'))


def settings(registration):
    out = []
    for name in registration['contexts']['order']:
        bge, rest = name[3:].split('_bm25')
        bm25, fusion = rest.split('_', 1)
        out.append({'k_bge': int(bge), 'k_bm25': int(bm25), 'fusion': fusion})
    if [l2_offline.name(c) for c in out] != registration['contexts']['order']:
        raise ValueError('Context names do not round-trip')
    return out


def read_results(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def trace_of(result):
    return ((result['enhancement_result'].get('metadata') or {}).get('native') or {}).get('module_trace') or {}


def stored_generations(results, generation_files):
    stored = {}
    for r in results:
        t = trace_of(r)
        if r['controller_decision']['action'] == P3 and t.get('generated_answer') is not None:
            stored[(r['sample_id'], tuple(e['passage_id'] for e in t['evidence']))] = (t['generated_answer'], t['reject_reason'])
    for path in generation_files:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                g = json.loads(line)
                stored[(g['sample_id'], tuple(g['passage_ids']))] = (g['generated_answer'], g['reject_reason'])
    return stored


def context_rows(results, candidates, stored, configs):
    rows, seen = [], set()
    for r in results:
        sid = r['sample_id']
        base = tuple(e['passage_id'] for e in r['initial_evidence'])
        for config in configs:
            ids = l2_offline.select_ids(candidates[sid], config)
            if ids == base or (sid, ids) in stored or (sid, ids) in seen:
                continue
            seen.add((sid, ids))
            rows.append({'sample_id': sid, 'query': r['query'], 'passage_ids': list(ids)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('contexts')
    c.add_argument('--results', type=Path, required=True, help='route-1 results.jsonl(.gz) of the role')
    c.add_argument('--candidates', type=Path, nargs='+', required=True, help='l2_candidates.py candidates.jsonl files')
    c.add_argument('--generations', type=Path, nargs='*', default=[], help='earlier l2_generate.py outputs to reuse')
    c.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    configs = settings(load_registration())
    results = list(read_results(args.results))
    candidates = {}
    for path in args.candidates:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                if row['sample_id'] in candidates:
                    raise ValueError('Duplicate candidate row ' + row['sample_id'])
                candidates[row['sample_id']] = row
    if {r['sample_id'] for r in results} != set(candidates):
        raise ValueError('Candidate rows do not cover the results')
    rows = context_rows(results, candidates, stored_generations(results, args.generations), configs)
    with args.output.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'items': len(results), 'contexts_to_generate': len(rows)}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
