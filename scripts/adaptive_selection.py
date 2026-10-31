"""Adaptive multi-context answer selection (configs/controller/adaptive_selection_v1.json).

`contexts` lists, for every item of a route-1 run (confirm results or Test results), the Top-5 of each
registered context that still needs a 7B answer: Top-5 lists equal to C1's take the C1 answer, and Top-5
lists already answered (route 1's P3-20 traces, the L2 / BM25 generation runs) are reused by (sample, Top-5).

The selection itself works on items: C1 and B+BM25 answers, route-1 routing, the stage-1 failure
probability, C1's Top-5, each context's Top-5 and answer, candidate cross-encoder scores and BGE/BM25 ranks.
Answers group by normalize_standard. Every feature is a runtime signal; labels only score outcomes.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import l2_offline  # noqa: E402
# The selection itself is the product module's (src/compensation/answer_selection.py).
from src.compensation.answer_selection import (  # noqa: E402,F401
    FEATURES, STAGES, decide_groups, groups, norm, probability, supports)

REGISTRATION = 'configs/controller/adaptive_selection_v1.json'
POLICY = 'artifacts/controller/adaptive_selection_v1/policy.json'
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


def decide(item, names, texts, decision, model=None):
    return decide_groups(groups(item, names, texts), item['base'], decision, model)


def extra_generations(item, names):
    """New 7B generations over B+BM25: distinct Top-5s that differ from C1's and B+BM25 did not already answer."""
    paid = {tuple(item['c1_top5'])}
    if item['routed']:
        paid.add(tuple(item['contexts']['bge20_bm250_rrf']['top5']))
        if item['bm25_brings']:
            paid.add(tuple(item['contexts']['bge20_bm2520_ce_only']['top5']))
    n = 0
    for name in names:
        top5 = tuple(item['contexts'][name]['top5'])
        if top5 not in paid:
            paid.add(top5)
            n += 1
    return n


def stage_rows(item, texts, order):
    """Per stage: candidate groups with features and the extra generations up to that stage."""
    return {str(k): {'groups': groups(item, order[:k], texts), 'cost': extra_generations(item, order[:k])} for k in STAGES}


def apply_stages(row, policy, models):
    """(answer, stage, extra generations) of a policy on an item's stage rows."""
    if not (row['routed'] or row['p_fail'] >= policy['theta']):
        return row['base'], 0, 0
    for k in STAGES:
        stage = row['stages'][str(k)]
        if max(g['x'][1] for g in stage['groups']) >= policy['tau'] or k == STAGES[-1]:
            break
    return decide_groups(stage['groups'], row['base'], policy['decision'], models.get(str(k))), k, stage['cost']


def apply_fixed(row, k, decision, models):
    stage = row['stages'][str(k)]
    return decide_groups(stage['groups'], row['base'], decision, models.get(str(k))), k, stage['cost']


def apply_policy(item, policy, models, texts, order):
    """(answer, stage, extra generations): pre-gate, then stages 3/5/8 until the largest group's share reaches tau."""
    return apply_stages({**item, 'stages': stage_rows(item, texts, order)}, policy, models)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def build_items(results, base, candidates, stored, corpus, order):
    """Stage rows for every route-1 result; every context answer must already exist (stored generations)."""
    from scripts import bm25_gate
    configs = {l2_offline.name(c): c for c in settings({'contexts': {'order': order}})}
    raw = []
    for r in results:
        sid = r['sample_id']
        cand, c1_top5 = candidates[sid], [e['passage_id'] for e in r['initial_evidence']]
        routed = r['controller_decision']['action'] == P3
        contexts = {}
        for name, config in configs.items():
            top5 = l2_offline.select_ids(cand, config)
            contexts[name] = {'top5': list(top5),
                              'answer': l2_offline.final_answer(r['initial_answer'], top5, tuple(c1_top5), stored, sid)}
        l2 = l2_offline.select_ids(cand, bm25_gate.SETTING)
        raw.append({'sample_id': sid, 'query': r['query'], 'c1': r['initial_answer'], 'base': base[sid], 'routed': routed,
                    'p_fail': r['diagnosis']['stage1_failure_probability'], 'c1_top5': c1_top5,
                    'bm25_brings': routed and bm25_gate.brings_bm25(cand, l2), 'contexts': contexts, 'ce': cand['ce'],
                    'bge_rank': {p: i + 1 for i, (p, _) in enumerate(cand['bge'])},
                    'bm25_rank': {p: i + 1 for i, (p, _) in enumerate(cand['bm25'])}})
    need = {p for it in raw for p in it['c1_top5'] + [q for c in it['contexts'].values() for q in c['top5']]}
    texts = {}
    with Path(corpus).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['passage_id'] in need:
                texts[row['passage_id']] = row['text']
    if len(texts) != len(need):
        raise ValueError('Corpus lacks passages of the candidate lists')
    return [{k: it[k] for k in ('sample_id', 'query', 'c1', 'base', 'routed', 'p_fail')} | {'stages': stage_rows(it, texts, order)}
            for it in raw]


def policies(policy):
    """Name -> function(row, models): the chosen policy, the reported points and the fixed baselines."""
    out = {'chosen': lambda row, m, p=policy['chosen']: apply_stages(row, p, m)}
    for point in policy['reported_points']:
        name = f"theta{point['theta']}_tau{point['tau']}_{point['decision']}"
        out[name] = lambda row, m, p=point: apply_stages(row, p, m)
    for fixed in policy['fixed_baselines']:
        name = f"fixed{fixed['stage']}_{fixed['decision']}"
        out[name] = lambda row, m, f=fixed: apply_fixed(row, f['stage'], f['decision'], m)
    return out


def answers_of(rows, policy):
    return {name: {row['sample_id']: fn(row, policy['models']) for row in rows} for name, fn in policies(policy).items()}


def needed_pairs(rows, outcomes):
    pairs = set()
    for row in rows:
        for result in outcomes.values():
            answer = result[row['sample_id']][0]
            if answer != row['base']:
                pairs.update({(row['sample_id'], answer), (row['sample_id'], row['base']), (row['sample_id'], row['c1'])})
    return pairs


def label_of(labels, gold, sid, answer):
    from baselines._lib.answer_scoring import score_answer
    value = labels.get((sid, answer))
    if value is None and score_answer(answer, gold[sid])['normalized_match']:
        return True
    return value


def summarize(rows, outcomes, labels, gold):
    from collections import Counter
    from scripts.score_quality_eval import mcnemar_p
    report = {}
    for name, result in outcomes.items():
        c, stages = Counter(), Counter()
        for row in rows:
            answer, stage, cost = result[row['sample_id']]
            stages[stage] += 1
            c['gens'] += cost
            for ref, key in ((row['base'], 'base'), (row['c1'], 'c1')):
                if answer == ref:
                    continue
                was, now = label_of(labels, gold, row['sample_id'], ref), label_of(labels, gold, row['sample_id'], answer)
                if was is None or now is None:
                    c[f'{key}_unjudged'] += 1
                    continue
                c[f'{key}_fix'] += now and not was
                c[f'{key}_harm'] += was and not now
        n = len(rows)
        net = c['base_fix'] - c['base_harm']
        report[name] = {'against_b_bm25': {'fixes': c['base_fix'], 'harms': c['base_harm'], 'net': net,
                                           'pp': 100 * net / n, 'mcnemar_p': mcnemar_p(c['base_fix'], c['base_harm']),
                                           'unjudged': c['base_unjudged']},
                        'against_c1': {'fixes': c['c1_fix'], 'harms': c['c1_harm'], 'net': c['c1_fix'] - c['c1_harm'],
                                       'pp': 100 * (c['c1_fix'] - c['c1_harm']) / n, 'unjudged': c['c1_unjudged']},
                        'extra_generations_per_item': c['gens'] / n, 'stages': {str(k): v for k, v in sorted(stages.items())}}
    return report

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('contexts')
    c.add_argument('--results', type=Path, required=True, help='route-1 results.jsonl(.gz) of the role')
    c.add_argument('--candidates', type=Path, nargs='+', required=True, help='l2_candidates.py candidates.jsonl files')
    c.add_argument('--generations', type=Path, nargs='*', default=[], help='earlier l2_generate.py outputs to reuse')
    c.add_argument('--output', type=Path, required=True)
    i = sub.add_parser('items')
    i.add_argument('--results', type=Path, required=True)
    i.add_argument('--base', type=Path, required=True, help='sample_id/base rows: the B+BM25 answers')
    i.add_argument('--candidates', type=Path, nargs='+', required=True)
    i.add_argument('--generations', type=Path, nargs='*', default=[])
    i.add_argument('--corpus', type=Path, required=True)
    i.add_argument('--output', type=Path, required=True)
    for cmd in ('judge-inputs', 'score'):
        s = sub.add_parser(cmd)
        s.add_argument('--items', type=Path, required=True)
        s.add_argument('--policy', type=Path, default=ROOT / POLICY)
        s.add_argument('--references', type=Path, required=True, help='sample_id/gold rows, local only')
        s.add_argument('--labels', type=Path, nargs='*', default=[], help='judge files, later wins')
        if cmd == 'judge-inputs':
            s.add_argument('--output-dir', type=Path, required=True)
            s.add_argument('--shards', type=int, default=4)
        else:
            s.add_argument('--role', choices=('confirm', 'test'), required=True)
            s.add_argument('--team', type=Path, nargs='*', default=[], help='test: team judge files (qwen_only product rag)')
            s.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'contexts':
        configs = settings(load_registration())
        results = list(read_results(args.results))
        candidates = {}
        for path in args.candidates:
            for row in read_jsonl(path):
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
    if args.command == 'items':
        results = list(read_results(args.results))
        candidates = {}
        for path in args.candidates:
            candidates.update({row['sample_id']: row for row in read_jsonl(path)})
        base = {row['sample_id']: row['base'] for row in read_jsonl(args.base)}
        rows = build_items(results, base, candidates, stored_generations(results, args.generations), args.corpus,
                           load_registration()['contexts']['order'])
        with args.output.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'items': len(rows)}))
        return 0
    from scripts.score_quality_eval import prior_labels
    policy = json.loads(args.policy.read_text(encoding='utf-8'))
    rows = read_jsonl(args.items)
    gold = {r['sample_id']: r['gold'] for r in read_jsonl(args.references)}
    labels = prior_labels(args.labels)
    outcomes = answers_of(rows, policy)
    if args.command == 'judge-inputs':
        from baselines._lib.answer_scoring import score_answer
        from scripts.quality_sweep_offline import judge_id
        query = {r['sample_id']: r.get('query', '') for r in rows}
        pending = []
        for sid, answer in sorted(needed_pairs(rows, outcomes)):
            if (sid, answer) in labels:
                continue
            pending.append({'sample_id': judge_id(sid, answer), 'source_sample_id': sid, 'query': query[sid],
                            'prediction': answer, 'gold': gold[sid],
                            'normalized_match': bool(score_answer(answer, gold[sid])['normalized_match'])})
        args.output_dir.mkdir(parents=True, exist_ok=False)
        import hashlib
        shards = [[] for _ in range(args.shards)]
        for row in pending:
            shards[int(hashlib.sha256(row['sample_id'].encode()).hexdigest(), 16) % args.shards].append(row)
        for k, shard in enumerate(shards):
            with (args.output_dir / f'shard_{k}.jsonl').open('x', encoding='utf-8') as handle:
                for row in shard:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'rows': len(pending), 'api_rows': sum(not r['normalized_match'] for r in pending)}))
        return 0
    report = summarize(rows, outcomes, labels, gold)
    chosen = report['chosen']['against_b_bm25']
    registration = load_registration()
    rule = registration['confirm']['rule'] if args.role == 'confirm' else registration['test']['adoption']
    complete = not any(v['against_b_bm25']['unjudged'] for v in report.values())
    passed = complete and chosen['net'] > 0 and (args.role == 'confirm' or chosen['mcnemar_p'] < 0.05)
    out = {'schema_version': 'adaptive_selection_score_v1', 'role': args.role, 'n': len(rows),
           'policy_sha256': __import__('hashlib').sha256(args.policy.read_bytes()).hexdigest(),
           'complete': complete, 'rule': rule, 'passed': passed, 'policies': report}
    if args.team:
        from scripts.score_quality_eval import compare
        chosen_rows = [{'sample_id': r['sample_id'], 'prediction': outcomes['chosen'][r['sample_id']][0]} for r in rows]
        team = dict(zip(('qwen_only', 'product', 'rag'), args.team))
        full = prior_labels([team['qwen_only'], team['product'], *args.labels, team['rag']])
        out['chosen_against_team'] = compare(chosen_rows, full, {'rag': team['rag'], 'product': team['product']})
    args.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'complete': complete, 'passed': passed, 'chosen': report['chosen']}, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
