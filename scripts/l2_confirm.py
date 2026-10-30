"""Confirm steps for the chosen L2 setting (configs/controller/l2_round_v1.json step_2_l2 adoption).

The setting is the one the sweep score report chose. Route-1 confirm results (quality_confirm_v1) give C1's
Top-5 and answer, route 1's action and its P3-20 trace. On the items route 1 sends to P3, the L2 answer
replaces the P3-20 answer; that paired comparison against route 1 is the adoption test. L2 is also computed
on the other items (step 3 gate curves) but does not enter the test. Generations are reused by (sample,
Top-5) from the stored P3-20 traces, then from the L2 confirm generation run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from baselines._lib.answer_scoring import score_answer  # noqa: E402
from scripts.l2_candidates import role_ids  # noqa: E402
from scripts.l2_offline import choose, final_answer, grid, load_registration, name, select_ids  # noqa: E402
from scripts.quality_sweep_offline import judge_id, load_judged, read_jsonl  # noqa: E402
from scripts.score_quality_eval import mcnemar_p  # noqa: E402

P3 = 'retrieval_enhancement'
CONTROL = {'k_bge': 20, 'k_bm25': 0, 'fusion': 'rrf'}


def trace_of(result):
    return ((result['enhancement_result'].get('metadata') or {}).get('native') or {}).get('module_trace') or {}


def load_choice(path):
    """The chosen setting of a complete sweep report, checked against the registered rule."""
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if not report['complete'] or report['chosen'] is None:
        raise ValueError('The sweep report is not complete')
    if report['chosen']['name'] != choose(report['results'])['name']:
        raise ValueError('The sweep report choice does not follow the registered rule')
    return {k: report['chosen'][k] for k in ('k_bge', 'k_bm25', 'fusion')}


def load_inputs(args):
    ids = role_ids('confirm')
    results = read_jsonl(args.results)
    if [r['sample_id'] for r in results] != ids:
        raise ValueError('Confirm results are incomplete or out of registered order')
    candidates = {r['sample_id']: r for r in read_jsonl(args.candidates)}
    if set(candidates) != set(ids):
        raise ValueError('Candidate rows do not cover the confirm IDs')
    generations = {}
    for r in results:
        t = trace_of(r)
        if r['controller_decision']['action'] == P3 and t.get('generated_answer') is not None:
            generations[(r['sample_id'], tuple(e['passage_id'] for e in t['evidence']))] = (t['generated_answer'], t['reject_reason'])
    for path in args.generations or []:
        for g in read_jsonl(path):
            generations[(g['sample_id'], tuple(g['passage_ids']))] = (g['generated_answer'], g['reject_reason'])
    return results, candidates, generations


def base_ids(result):
    return tuple(e['passage_id'] for e in result['initial_evidence'])


def control_check(results, candidates):
    """Routed items whose control Top-5 (P3-20 from the GPU scores) equals route 1's P3-20 Top-5."""
    routed = [r for r in results if r['controller_decision']['action'] == P3]
    same = sum(select_ids(candidates[r['sample_id']], CONTROL) ==
               tuple(e['passage_id'] for e in (trace_of(r).get('evidence') or r['initial_evidence'])) for r in routed)
    return {'routed': len(routed), 'same_top5': same}


def needed_contexts(results, candidates, generations, config):
    rows = []
    for r in results:
        sid, ids = r['sample_id'], select_ids(candidates[r['sample_id']], config)
        if ids != base_ids(r) and (sid, ids) not in generations:
            rows.append({'sample_id': sid, 'query': r['query'], 'passage_ids': list(ids)})
    return rows


def answers(results, candidates, generations, config):
    """{'c1', 'route_1', 'l2' (L2 on every item), 'route_1_l2' (L2 on route 1's P3 items)}: {sample_id: answer}."""
    out = {'c1': {}, 'route_1': {}, 'l2': {}, 'route_1_l2': {}}
    for r in results:
        sid = r['sample_id']
        l2 = final_answer(r['initial_answer'], select_ids(candidates[sid], config), base_ids(r), generations, sid)
        out['c1'][sid], out['route_1'][sid], out['l2'][sid] = r['initial_answer'], r['final_answer'], l2
        out['route_1_l2'][sid] = l2 if r['controller_decision']['action'] == P3 else r['initial_answer']
    return out


def paired(before, after, gold, judged):
    """Fixes and harms of `after` against `before`; items with the same answer count zero."""
    fixes = harms = changed = unjudged = 0
    for sid in before:
        if after[sid] == before[sid]:
            continue
        changed += 1
        label = lambda a: True if score_answer(a, gold[sid])['normalized_match'] else judged.get(judge_id(sid, a))
        was, now = label(before[sid]), label(after[sid])
        if was is None or now is None:
            unjudged += 1
            continue
        fixes += now and not was
        harms += was and not now
    return {'changed': changed, 'fixes': fixes, 'harms': harms, 'net': fixes - harms, 'unjudged': unjudged,
            'mcnemar_p': mcnemar_p(fixes, harms)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ('contexts', 'judge-inputs', 'score'):
        s = sub.add_parser(cmd)
        s.add_argument('--results', type=Path, required=True, help='route-1 confirm results.jsonl')
        s.add_argument('--candidates', type=Path, required=True, help='l2_candidates.py --role confirm output')
        s.add_argument('--choice', type=Path, required=True, help='complete l2_offline.py score report')
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
    config = load_choice(args.choice)
    if config not in grid(load_registration()):
        raise ValueError('The chosen setting is not in the registered grid')
    results, candidates, generations = load_inputs(args)
    if args.command == 'contexts':
        rows = needed_contexts(results, candidates, generations, config)
        with args.output.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'setting': name(config), 'stored_generations': len(generations), 'contexts_to_generate': len(rows),
                          'control_check': control_check(results, candidates)}))
        return 0
    gold = {r['sample_id']: r['gold'] for r in read_jsonl(args.references)}
    judged = load_judged(args.judged)
    finals = answers(results, candidates, generations, config)
    if args.command == 'judge-inputs':
        rows, seen = [], set(judged)
        for r in results:
            sid = r['sample_id']
            strings = {f[sid] for f in finals.values()}
            if len(strings) == 1:
                continue
            for answer in sorted(strings):
                jid = judge_id(sid, answer)
                if jid not in seen:
                    seen.add(jid)
                    rows.append({'sample_id': jid, 'source_sample_id': sid, 'query': r['query'], 'prediction': answer,
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
    adoption = paired(finals['route_1'], finals['route_1_l2'], gold, judged)
    vs_c1 = {k: paired(finals['c1'], finals[k], gold, judged) for k in ('route_1', 'route_1_l2', 'l2')}
    complete = not adoption['unjudged'] and not any(v['unjudged'] for v in vs_c1.values())
    report = {'schema_version': 'l2_confirm_v1', 'setting': name(config),
              'choice_report_sha256': hashlib.sha256(args.choice.read_bytes()).hexdigest(),
              'n': len(results), 'routed': sum(r['controller_decision']['action'] == P3 for r in results),
              'control_check': control_check(results, candidates), 'complete': complete,
              'adoption': {**adoption, 'rule': load_registration()['step_2_l2']['adoption'],
                           'passed': complete and adoption['net'] > 0 and adoption['mcnemar_p'] < 0.05},
              'against_c1': vs_c1}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
