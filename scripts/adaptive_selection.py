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


FEATURES = ('votes', 'share', 'is_c1', 'is_base', 'n_support', 'max_ce', 'best_bge', 'best_bm25', 'c1_support',
            'tokens', 'p_fail', 'routed', 'n_groups', 'top_votes', 'margin')
STAGES = (3, 5, 8)


def norm(answer):
    from baselines._lib.answer_scoring import normalize_standard
    return normalize_standard(answer)


def supports(answer, text):
    """The answer_rejection grounding test: the normalized answer is a token span of the normalized passage."""
    from src.evaluation.qa import normalize_answer
    needle = normalize_answer(answer)
    return bool(needle) and f' {needle} ' in f' {normalize_answer(text)} '


def groups(item, names, texts):
    """Candidate answer groups of C1 plus the named contexts, with FEATURES; B+BM25's group is always present."""
    from collections import Counter
    pool = [item['c1'], *(item['contexts'][n]['answer'] for n in names)]
    counts = Counter(norm(a) for a in pool)
    rep = {}
    for a in pool:
        rep.setdefault(norm(a), a)
    rep.setdefault(norm(item['base']), item['base'])
    counts.setdefault(norm(item['base']), 0)
    allowed = list(dict.fromkeys(item['c1_top5'] + [p for n in names for p in item['contexts'][n]['top5']]))
    ranked = sorted(counts.values(), reverse=True) + [0]
    out = []
    for g, a in rep.items():
        sup = [p for p in allowed if supports(a, texts[p])]
        ce = [item['ce'][p] for p in sup if p in item['ce']]
        out.append({'answer': a, 'x': [
            counts[g], counts[g] / len(pool), float(g == norm(item['c1'])), float(g == norm(item['base'])),
            len(sup), max(ce) if ce else -12.0, min([item['bge_rank'].get(p, 101) for p in sup] or [101]),
            min([item['bm25_rank'].get(p, 51) for p in sup] or [51]), float(any(p in item['c1_top5'] for p in sup)),
            len(g.split()), item['p_fail'], float(item['routed']), len(counts), ranked[0], ranked[0] - ranked[1]]})
    return out


def probability(model, x):
    """Standardized logistic regression from its stored parameters (sklearn StandardScaler + LogisticRegression)."""
    import math
    z = sum(w * (v - m) / s for w, v, m, s in zip(model['coef'], x, model['mean'], model['scale'])) + model['intercept']
    return 1.0 / (1.0 + math.exp(-z))


def decide(item, names, texts, decision, model=None):
    gs = groups(item, names, texts)
    if len(gs) == 1:
        return item['base']
    if decision == 'vote':
        top = max(g['x'][0] for g in gs)
        winners = [g for g in gs if g['x'][0] == top]
        return item['base'] if any(g['x'][3] for g in winners) or len(winners) > 1 else winners[0]['answer']
    p = [probability(model, g['x']) for g in gs]
    base = next(i for i, g in enumerate(gs) if g['x'][3] == 1.0)
    best = max(range(len(gs)), key=lambda i: (p[i], -i))
    return gs[best]['answer'] if p[best] > p[base] else item['base']


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


def apply_policy(item, policy, models, texts, order):
    """(answer, stage, extra generations): pre-gate, then stages 3/5/8 until the largest group's share reaches tau."""
    if not (item['routed'] or item['p_fail'] >= policy['theta']):
        return item['base'], 0, 0
    for k in STAGES:
        names = order[:k]
        share = max(g['x'][1] for g in groups(item, names, texts))
        if share >= policy['tau'] or k == STAGES[-1]:
            break
    answer = decide(item, names, texts, policy['decision'], models.get(str(k)))
    return answer, k, extra_generations(item, names)


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
