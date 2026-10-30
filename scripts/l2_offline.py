"""Local L2 steps (configs/controller/l2_round_v1.json step_2_l2): contexts, judge inputs, grid score.

Every grid setting is selected offline from the stored candidate lists and GPU scores. As in
P3-20, a selection equal to C1's Top-5 keeps C1 (no_change); any other Top-5 needs one
generation, and answer_rejection keeps C1 when the answer is not grounded. Generations are
reused by (sample, Top-5): stored P3-20 traces (route-1 sweep and step-1 coverage) first,
then the L2 generation run. Route-1 routing (items with a stored route-1 P3-20 outcome)
decides which items are scored; the all-items score is reported only.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from baselines._lib.answer_scoring import score_answer  # noqa: E402
from scripts.quality_sweep_offline import judge_id, load_judged, read_jsonl, read_records  # noqa: E402
from scripts.precompute_quality_sweep import load_registration as load_sweep  # noqa: E402
from src.compensation.retrieval_enhancement.p3_l2 import select_l2, union_candidates  # noqa: E402

REGISTRATION = 'configs/controller/l2_round_v1.json'
P3 = 'retrieval_enhancement'


def load_registration(root=ROOT):
    return json.loads((Path(root) / REGISTRATION).read_text(encoding='utf-8'))


def grid(registration):
    g = registration['step_2_l2']['grid']
    return [{'k_bge': a, 'k_bm25': b, 'fusion': f} for a, b, f in itertools.product(g['k_bge'], g['k_bm25'], g['fusion'])]


def name(config):
    return f"bge{config['k_bge']}_bm25{config['k_bm25']}_{config['fusion']}"


def select_ids(row, config):
    """Top-5 passage IDs of one candidate row under one setting; texts are not needed to select."""
    bge = [{'passage_id': pid, 'text': pid} for pid, _ in row['bge'][:config['k_bge']]]
    bm25 = [{'passage_id': pid, 'text': pid} for pid, _ in row['bm25'][:config['k_bm25']]]
    pool = union_candidates(bge, bm25)
    return tuple(r['passage_id'] for r in select_l2(pool, [row['ce'][p['passage_id']] for p in pool], config['fusion']))


def trace_of(action):
    return ((action or {}).get('native') or {}).get('module_trace') or {}


def stored_generations(records, coverage_rows):
    """(sample, Top-5 IDs) -> (generated answer, reject reason) from stored P3-20 traces."""
    out = {}
    actions = [(r['sample_id'], r['actions'].get(P3)) for r in records]
    actions += [(r['sample_id'], r['action']) for r in coverage_rows if r['source'] == 'new']
    for sid, action in actions:
        t = trace_of(action)
        if t.get('generated_answer') is not None:
            out[(sid, tuple(e['passage_id'] for e in t['evidence']))] = (t['generated_answer'], t['reject_reason'])
    return out


def final_answer(c1, ids, base_ids, generations, sid):
    if ids == base_ids:
        return c1
    answer, reason = generations[(sid, ids)]
    return c1 if reason is not None else answer.strip()


def load_inputs(args):
    _, ids = load_sweep()
    records = read_records(args.records, ids)
    candidates = {r['sample_id']: r for r in read_jsonl(args.candidates)}
    if set(candidates) != set(ids):
        raise ValueError('Candidate rows do not cover the sweep IDs')
    coverage = read_jsonl(args.coverage)
    generations = stored_generations(records, coverage)
    for path in args.generations or []:
        for g in read_jsonl(path):
            generations[(g['sample_id'], tuple(g['passage_ids']))] = (g['generated_answer'], g['reject_reason'])
    return records, candidates, generations


def answers(records, candidates, generations, registration):
    """{setting name: {sample_id: final answer}} for every item."""
    base = {r['sample_id']: tuple(e['passage_id'] for e in r['evidence']) for r in records}
    return {name(c): {r['sample_id']: final_answer(r['initial_answer'], select_ids(candidates[r['sample_id']], c),
                                                   base[r['sample_id']], generations, r['sample_id'])
                      for r in records} for c in grid(registration)}


def needed_contexts(records, candidates, generations, registration):
    base = {r['sample_id']: tuple(e['passage_id'] for e in r['evidence']) for r in records}
    queries = {r['sample_id']: r['query'] for r in records}
    seen, rows = set(), []
    for r in records:
        sid = r['sample_id']
        for c in grid(registration):
            ids = select_ids(candidates[sid], c)
            if ids != base[sid] and (sid, ids) not in generations and (sid, ids) not in seen:
                seen.add((sid, ids))
                rows.append({'sample_id': sid, 'query': queries[sid], 'passage_ids': list(ids)})
    return rows


def evaluate(records, finals, gold, judged, routed):
    fixes = harms = changed = unjudged = 0
    for r in records:
        sid, c1 = r['sample_id'], r['initial_answer']
        after = finals[sid] if sid in routed else c1
        if after == c1:
            continue
        changed += 1
        label = lambda a: True if score_answer(a, gold[sid])['normalized_match'] else judged.get(judge_id(sid, a))
        before, now = label(c1), label(after)
        if before is None or now is None:
            unjudged += 1
            continue
        fixes += now and not before
        harms += before and not now
    return {'changed': changed, 'fixes': fixes, 'harms': harms, 'net': fixes - harms, 'unjudged': unjudged}


def choose(results):
    """Registered rule: best net; within 3 items fewer candidates, then rrf before ce_only."""
    best = max(r['net'] for r in results)
    close = [r for r in results if r['net'] >= best - 3]
    return min(close, key=lambda r: (r['k_bge'] + r['k_bm25'], r['fusion'] != 'rrf', -r['net']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ('contexts', 'judge-inputs', 'score'):
        s = sub.add_parser(cmd)
        s.add_argument('--records', type=Path, required=True, help='route-1 sweep records.jsonl')
        s.add_argument('--coverage', type=Path, required=True, help='step-1 p3_records.jsonl')
        s.add_argument('--candidates', type=Path, required=True)
        s.add_argument('--generations', type=Path, nargs='*', default=[])
        if cmd == 'contexts':
            s.add_argument('--output', type=Path, required=True)
        else:
            s.add_argument('--references', type=Path, required=True)
            s.add_argument('--judged', type=Path, nargs='*', default=[])
            s.add_argument('--output-dir' if cmd == 'judge-inputs' else '--output', type=Path, required=True)
            if cmd == 'judge-inputs':
                s.add_argument('--shards', type=int, default=4)
    args = p.parse_args(argv)
    registration = load_registration()
    records, candidates, generations = load_inputs(args)
    if args.command == 'contexts':
        rows = needed_contexts(records, candidates, generations, registration)
        with args.output.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'stored_generations': len(generations), 'contexts_to_generate': len(rows)}))
        return 0
    gold = {r['sample_id']: r['gold'] for r in read_jsonl(args.references)}
    judged = load_judged(args.judged)
    finals = answers(records, candidates, generations, registration)
    if args.command == 'judge-inputs':
        queries = {r['sample_id']: r['query'] for r in records}
        rows, seen = [], set(judged)
        for r in records:
            sid = r['sample_id']
            strings = {r['initial_answer']} | {f[sid] for f in finals.values()}
            if len(strings) == 1:
                continue
            for answer in sorted(strings):
                jid = judge_id(sid, answer)
                if jid not in seen:
                    seen.add(jid)
                    rows.append({'sample_id': jid, 'source_sample_id': sid, 'query': queries[sid], 'prediction': answer,
                                 'gold': gold[sid], 'normalized_match': bool(score_answer(answer, gold[sid])['normalized_match'])})
        args.output_dir.mkdir(parents=True, exist_ok=False)
        shards = [[] for _ in range(args.shards)]
        for row in rows:
            shards[int(hashlib.sha256(row['sample_id'].encode()).hexdigest(), 16) % args.shards].append(row)
        for i, shard in enumerate(shards):
            with (args.output_dir / f'shard_{i}.jsonl').open('x', encoding='utf-8') as handle:
                for row in shard:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'rows': len(rows), 'api_rows': sum(not r['normalized_match'] for r in rows)}))
        return 0
    routed = {r['sample_id'] for r in records if P3 in r['actions']}
    every = {r['sample_id'] for r in records}
    results = [{**c, 'name': name(c), **evaluate(records, finals[name(c)], gold, judged, routed),
                'all_items': evaluate(records, finals[name(c)], gold, judged, every)} for c in grid(registration)]
    route_1 = {r['sample_id']: (r['actions'][P3]['final_answer'] if P3 in r['actions'] else r['initial_answer'])
               for r in records}
    control = finals[name({'k_bge': 20, 'k_bm25': 0, 'fusion': 'rrf'})]
    report = {'schema_version': 'l2_offline_v1', 'records': len(records), 'routed': len(routed),
              'control_equals_route_1': sum(control[s] == route_1[s] for s in routed),
              'complete': not any(r['unjudged'] for r in results),
              'chosen': choose(results) if not any(r['unjudged'] for r in results) else None,
              'results': sorted(results, key=lambda r: -r['net'])}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('records', 'routed', 'control_equals_route_1', 'complete', 'chosen')}, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
