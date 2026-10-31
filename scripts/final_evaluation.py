"""Final NQ evaluation from archived answers and labels; no GPU.

- Test 11,114: judge accuracy, EM, normalized accuracy and token F1 of every level, paired bootstrap 95% CIs,
  and the registered selection policies against each other at equal generation budgets.
- P60 routing ablation: route-1 routing against P3-20 on every item and random routing of the same size,
  on the sweep (labels of artifacts/controller/recoverability_oracle_v1) and on Test (P3-20 and BM25-rule
  answers of every item; the unlabelled ones went to one judge batch).
- Measured per-question latency, calls and input tokens of the product (timing run on confirm questions).

Every number is an evaluation of choices fixed earlier; nothing here selects or tunes the product.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

RELEASE = 'artifacts/controller/final_evaluation_v1'
AD_RELEASE = 'artifacts/controller/adaptive_selection_eval_v1/results.tar.gz'
BM25_RELEASE = 'artifacts/controller/bm25_gate_test_v1/results.tar.gz'
ORACLE_RELEASE = 'artifacts/controller/recoverability_oracle_v1/results.tar.gz'
P3 = 'bge20_bm250_rrf'
SEED = 20261031
N_BOOT = 10000
N_RANDOM = 5000
LATENCY_SEED = 'latency_v1_20261031::'
PAIRS = (('product', 'rag'), ('product', 'c1'), ('product', 'b_bm25'), ('b_bm25', 'b'), ('b', 'c1'),
         ('product', 'team_product'), ('c1', 'rag'))
POLICY_PAIRS = (('chosen', 'fixed5_selector'), ('chosen', 'fixed8_selector'), ('chosen', 'fixed8_vote'),
                ('chosen', 'fixed5_vote'), ('fixed3_selector', 'fixed3_vote'), ('fixed5_selector', 'fixed5_vote'),
                ('fixed8_selector', 'fixed8_vote'), ('theta0.39_tau0.6_selector', 'fixed8_vote'),
                ('theta0.39_tau0.6_selector', 'b_bm25'))


def untar(path):
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def jsonl(data):
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


def write(tmp, name, data):
    path = Path(tmp) / name.replace('/', '__')
    path.write_bytes(data)
    return path


def batch_files(tmp, prefix, members, path):
    return [write(tmp, f'{prefix}_{i}', members[f'{path}/shard_{i}/judge_results.jsonl']) for i in range(4)]


def auc(scores, positive):
    """Mann-Whitney AUC: probability that a positive item scores above a negative one (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    order = np.argsort(s, kind='mergesort')
    _, inverse, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[order] = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap(columns, n_boot=N_BOOT, seed=SEED, chunk=1000):
    """Bootstrap means of every column (n items x k) over the same item resamples."""
    columns = np.asarray(columns, float)
    n = len(columns)
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        weights = np.stack([np.bincount(row, minlength=n) for row in idx]).astype(float)
        out.append(weights @ columns / n)
    return np.vstack(out)


def paired_table(arrays, pairs, metrics):
    """{a-b: {metric: pp, ci95, fixes, harms}} with one shared set of resamples."""
    names = [(a, b, m) for a, b in pairs for m in metrics]
    diffs = np.column_stack([arrays[a][m] - arrays[b][m] for a, b, m in names])
    boot = bootstrap(diffs)
    out = {}
    for j, (a, b, m) in enumerate(names):
        d = diffs[:, j]
        out.setdefault(f'{a}-{b}', {})[m] = {
            'pp': round(100 * float(d.mean()), 4),
            'ci95': [round(100 * float(np.percentile(boot[:, j], q)), 4) for q in (2.5, 97.5)],
            'fixes': int((d > 0).sum()), 'harms': int((d < 0).sum())}
    return out


def routing_ablation(c1, p3, routed, p_fail, *, seed=SEED, n_random=N_RANDOM, shares=None):
    """Net against C1 when P60 routes, when every item goes to P3, and when a random set of the same size does."""
    c1, p3, routed = (np.asarray(x, bool) for x in (c1, p3, routed))
    p_fail = np.asarray(p_fail, float)
    n, k = len(c1), int(routed.sum())
    gain, loss = p3 & ~c1, c1 & ~p3
    net = lambda m: int(gain[m].sum() - loss[m].sum())
    rng = np.random.default_rng(seed)
    rand = np.array([net(rng.permutation(n)[:k]) for _ in range(n_random)])
    out = {'n': n, 'routed': k, 'accuracy': {'C1': float(c1.mean()), 'P60_routed': float(np.where(routed, p3, c1).mean()),
                                             'all_to_P3': float(p3.mean())},
           'net_vs_c1': {'P60_routed': net(routed), 'all_to_P3': net(np.ones(n, bool)), 'unrouted_if_sent': net(~routed)},
           'fix_harm': {'P60_routed': [int(gain[routed].sum()), int(loss[routed].sum())],
                        'unrouted_if_sent': [int(gain[~routed].sum()), int(loss[~routed].sum())]},
           'random_same_size': {'draws': n_random, 'mean': float(rand.mean()),
                                'range95': [float(np.percentile(rand, 2.5)), float(np.percentile(rand, 97.5))],
                                'share_at_or_above_P60': float((rand >= net(routed)).mean())},
           'auc_p_fail': {'c1_wrong': auc(p_fail, ~c1), 'p3_fixes': auc(p_fail, gain),
                          'p3_fixes_among_c1_wrong': auc(p_fail[~c1], gain[~c1]), 'p3_harms': auc(p_fail, loss)}}
    if shares:
        order = np.argsort(-p_fail, kind='mergesort')
        out['by_p_fail_share'] = [{'share': q, 'net': net(order[:int(round(q * n))])} for q in shares]
    return out


def latency_summary(rows):
    """Per question: seconds by stage and cumulative level, LLM calls and input tokens (C1 counted in every level)."""
    stat = lambda xs: {'mean': float(np.mean(xs)), 'p50': float(np.percentile(xs, 50)), 'p95': float(np.percentile(xs, 95))}
    get = lambda r, k: r.get(k, 0.0)
    levels = []
    for r in rows:
        enh, sel = r['enhancement_cost'], r['selection_cost']
        diagnosis = get(r, 'diagnosis_classifier_s') + get(r, 'diagnosis_features_s')
        levels.append({'c1_retrieval': r['c1_s'] - r['c1_generate_s'], 'c1_generation': r['c1_generate_s'], 'diagnosis': diagnosis,
                       'controller': get(r, 'controller_s'), 'answer_selection': r['answer_selection_s'],
                       'rag_s': r['c1_s'], 'b_bm25_s': r['c1_s'] + diagnosis + get(r, 'controller_s'), 'product_s': r['total_s'],
                       'rag_calls': 1, 'b_bm25_calls': 1 + enh['llm_calls'], 'product_calls': 1 + enh['llm_calls'] + sel['llm_calls'],
                       'rag_tokens': r['c1_input_tokens'], 'b_bm25_tokens': r['c1_input_tokens'] + enh['input_tokens'],
                       'product_tokens': r['c1_input_tokens'] + enh['input_tokens'] + sel['input_tokens'],
                       'routed': r['action'] != 'no_action', 'stage': r['selection_stage']})
    col = lambda k, rs=levels: stat([x[k] for x in rs])
    out = {'n': len(rows), 'routed_share': float(np.mean([x['routed'] for x in levels])),
           'stages_s': {k: col(k) for k in ('c1_retrieval', 'c1_generation', 'diagnosis', 'controller', 'answer_selection')},
           'levels': {lvl: {'seconds': col(f'{lvl}_s'), 'llm_calls': col(f'{lvl}_calls'), 'input_tokens': col(f'{lvl}_tokens')}
                      for lvl in ('rag', 'b_bm25', 'product')},
           'product_s_by_route': {name: col('product_s', [x for x in levels if x['routed'] == flag]) for name, flag in
                                  (('routed', True), ('not_routed', False))},
           'answer_selection_s_by_stage': {str(k): col('answer_selection', [x for x in levels if x['stage'] == k]) for k in (3, 5, 8)}}
    return out


def latency_ids(confirm_ids):
    order = sorted(confirm_ids, key=lambda s: hashlib.sha256((LATENCY_SEED + s).encode()).hexdigest())
    return {'warmup': order[:10], 'timed': order[10:310]}


def test_p3_rows(results, candidates, stored, items):
    """Per Test item: C1, the P3-20 context answer, the BM25-rule answer if it were applied to every item, B+BM25."""
    from scripts import bm25_gate, l2_offline
    from scripts import adaptive_selection as ad
    p3_config = dict(zip(ad.load_registration()['contexts']['order'], ad.settings(ad.load_registration())))[P3]
    rows = []
    for r in results:
        sid = r['sample_id']
        c1_top5 = tuple(e['passage_id'] for e in r['initial_evidence'])
        p3 = l2_offline.final_answer(r['initial_answer'], l2_offline.select_ids(candidates[sid], p3_config), c1_top5, stored, sid)
        l2 = l2_offline.select_ids(candidates[sid], bm25_gate.SETTING)
        brings = bm25_gate.brings_bm25(candidates[sid], l2)
        rows.append({'sample_id': sid, 'routed': r['controller_decision']['action'] == ad.P3,
                     'p_fail': r['diagnosis']['stage1_failure_probability'], 'c1': r['initial_answer'], 'p3': p3,
                     'bm25_all': l2_offline.final_answer(r['initial_answer'], l2, c1_top5, stored, sid) if brings else p3,
                     'b_bm25': items[sid]['base']})
    return rows


def load_test(root, tmp, extra_labels=()):
    """(route-1 predictions, team files, labels, adaptive release members) with the registered label order, RAG last."""
    from scripts import bm25_gate as gate
    from scripts import score_quality_eval as scorer
    rel, bm = untar(root / AD_RELEASE), untar(root / BM25_RELEASE)
    preds, ours, team = gate.route_1_test(root / gate.ROUTE_1_TEST, tmp)
    files = [team['qwen_only'], team['product'], *ours, *batch_files(tmp, 'bm25', bm, 'judge/outputs'),
             *batch_files(tmp, 'ad', rel, 'test/judge/outputs'), *extra_labels, team['rag']]
    return preds, team, scorer.prior_labels(files), rel


def test_section(preds, team, labels, rel, policy):
    from baselines._lib.answer_scoring import score_answer
    from scripts import adaptive_selection as ad
    from scripts import score_quality_eval as scorer
    rows = jsonl(rel['test/items.jsonl'])
    outcomes = ad.answers_of(rows, policy)
    gold = {r['sample_id']: r['gold'] for r in preds}
    ids = sorted(gold)
    runs = {name: {str(r['sample_id']): r['prediction'] for r in scorer.read_jsonl(team[key])}
            for name, key in (('rag', 'rag'), ('team_product', 'product'))}
    runs['c1'] = {r['sample_id']: r['initial_answer'] for r in preds}
    runs['b'] = {r['sample_id']: r['prediction'] for r in preds}
    runs['b_bm25'] = {r['sample_id']: r['base'] for r in rows}
    runs['product'] = {sid: v[0] for sid, v in outcomes['chosen'].items()}

    def judged(sid, answer):
        value = ad.label_of(labels, gold, sid, answer)
        if value is None:
            raise ValueError(f'Unlabelled Test answer {sid}')
        return float(bool(value))

    arrays = {}
    for name, answers in runs.items():
        scores = [score_answer(answers[s], gold[s]) for s in ids]
        arrays[name] = {'judge': np.array([judged(s, answers[s]) for s in ids]), 'em': np.array([x['em'] for x in scores], float),
                        'normalized_accuracy': np.array([x['normalized_match'] for x in scores], float),
                        'f1': np.array([x['f1'] for x in scores], float)}
    metrics = {name: {m: 100 * float(v.mean()) for m, v in a.items()} for name, a in arrays.items()}
    paired = paired_table(arrays, PAIRS, ('judge', 'em', 'normalized_accuracy', 'f1'))
    policies = {name: {'judge': np.array([judged(s, result[s][0]) for s in ids])} for name, result in outcomes.items()}
    policies['b_bm25'] = {'judge': arrays['b_bm25']['judge']}
    gens = {name: float(np.mean([result[s][2] for s in ids])) for name, result in outcomes.items()} | {'b_bm25': 0.0}
    a, b = arrays['product'], arrays['b_bm25']
    up, down = a['judge'] > b['judge'], a['judge'] < b['judge']
    em_up, em_down = a['em'] > b['em'], a['em'] < b['em']
    return {'n': len(ids), 'metrics': metrics, 'paired': paired,
            'policies': {'extra_generations_per_item': gens,
                         'judge_accuracy': {k: 100 * float(v['judge'].mean()) for k, v in policies.items()},
                         'paired': paired_table(policies, POLICY_PAIRS, ('judge',))},
            'product_vs_b_bm25_judge_and_em': {
                'judge_fixes': int(up.sum()), 'judge_fixes_em_also_up': int((up & em_up).sum()),
                'judge_harms': int(down.sum()), 'judge_harms_em_also_down': int((down & em_down).sum()),
                'em_up_judge_not_fix': int((em_up & ~up).sum()), 'em_down_judge_not_harm': int((em_down & ~down).sum())}}


def sweep_routing(root, tmp):
    from scripts import recoverability_oracle as ro
    from scripts.score_quality_eval import prior_labels
    m = untar(root / ORACLE_RELEASE)
    items = jsonl(m['ladder_inputs.jsonl'])
    gold = {r['sample_id']: r['gold'] for r in jsonl(m['references.jsonl'])}
    prior = {(r['sample_id'], r['answer']): r['correct'] for r in jsonl(m['prior_labels.jsonl'])}
    label = ro.labeller(prior, prior_labels(batch_files(tmp, 'oracle', m, 'judge/outputs')), gold)
    if not all(it['route_1'] == (it['contexts'][P3] if it['routed'] else it['c1']) for it in items):
        raise ValueError('Route 1 is not the P3-20 context answer on routed items and C1 elsewhere')
    ok = lambda sid, answer: bool(label(sid, answer))
    return routing_ablation([ok(it['sample_id'], it['c1']) for it in items], [ok(it['sample_id'], it['contexts'][P3]) for it in items],
                            [it['routed'] for it in items], [it['p_fail'] for it in items],
                            shares=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))


def test_routing(p3_rows, labels, gold):
    from scripts import adaptive_selection as ad
    ok = {}
    for k in ('c1', 'p3', 'bm25_all', 'b_bm25'):
        values = [ad.label_of(labels, gold, r['sample_id'], r[k]) for r in p3_rows]
        if any(v is None for v in values):
            raise ValueError('Unlabelled Test routing answers: ' + k)
        ok[k] = [bool(v) for v in values]
    out = routing_ablation(ok['c1'], ok['p3'], [r['routed'] for r in p3_rows], [r['p_fail'] for r in p3_rows])
    out['bm25_rule'] = {'B+BM25 (routed items only)': float(np.mean(ok['b_bm25'])), 'on every item': float(np.mean(ok['bm25_all']))}
    return out


def unlabelled_pairs(p3_rows, labels, gold):
    from scripts import adaptive_selection as ad
    return sorted({(r['sample_id'], r[k]) for r in p3_rows for k in ('c1', 'p3', 'bm25_all', 'b_bm25')
                   if ad.label_of(labels, gold, r['sample_id'], r[k]) is None})


def report(root=ROOT):
    """The full report from the archived inputs of this release and the earlier ones."""
    from scripts import adaptive_selection as ad
    members = untar(root / RELEASE / 'results.tar.gz')
    policy = json.loads((root / ad.POLICY).read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as tmp:
        extra = batch_files(tmp, 'p60', members, 'judge/outputs')
        preds, team, labels, rel = load_test(root, tmp, extra)
        gold = {r['sample_id']: r['gold'] for r in preds}
        return {'schema_version': 'final_evaluation_v1', 'seed': SEED, 'bootstrap_resamples': N_BOOT,
                'test': test_section(preds, team, labels, rel, policy),
                'routing': {'sweep': sweep_routing(root, tmp), 'test': test_routing(jsonl(members['test_p3_rows.jsonl']), labels, gold)},
                'latency': latency_summary(jsonl(members['latency/latency.jsonl']))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('rows', help='P3-20 / BM25-rule answers of every Test item (raw candidates are local only)')
    r.add_argument('--results', type=Path, required=True, help='route-1 Test results.jsonl(.gz)')
    r.add_argument('--candidates', type=Path, nargs='+', required=True)
    r.add_argument('--generations', type=Path, nargs='+', required=True)
    r.add_argument('--output', type=Path, required=True)
    j = sub.add_parser('judge-inputs', help='the routing answers no earlier Test batch labelled')
    j.add_argument('--rows', type=Path, required=True)
    j.add_argument('--references', type=Path, required=True, help='sample_id/gold rows, local only')
    j.add_argument('--output-dir', type=Path, required=True)
    o = sub.add_parser('report')
    o.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    from scripts import adaptive_selection as ad
    if args.command == 'rows':
        results = list(ad.read_results(args.results))
        candidates = {row['sample_id']: row for path in args.candidates for row in ad.read_jsonl(path)}
        items = {row['sample_id']: row for row in jsonl(untar(ROOT / AD_RELEASE)['test/items.jsonl'])}
        rows = test_p3_rows(results, candidates, ad.stored_generations(results, args.generations), items)
        with args.output.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        return 0
    if args.command == 'judge-inputs':
        from baselines._lib.answer_scoring import score_answer
        from scripts.quality_sweep_offline import judge_id
        rows = ad.read_jsonl(args.rows)
        gold = {r['sample_id']: r['gold'] for r in ad.read_jsonl(args.references)}
        query = {r['sample_id']: r['query'] for r in jsonl(untar(ROOT / AD_RELEASE)['test/items.jsonl'])}
        with tempfile.TemporaryDirectory() as tmp:
            _, _, labels, _ = load_test(ROOT, tmp)
            pending = unlabelled_pairs(rows, labels, gold)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        shards = [[] for _ in range(4)]
        for sid, answer in pending:
            row = {'sample_id': judge_id(sid, answer), 'source_sample_id': sid, 'query': query[sid], 'prediction': answer,
                   'gold': gold[sid], 'normalized_match': bool(score_answer(answer, gold[sid])['normalized_match'])}
            shards[int(hashlib.sha256(row['sample_id'].encode()).hexdigest(), 16) % 4].append(row)
        for k, shard in enumerate(shards):
            (args.output_dir / f'shard_{k}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in shard),
                                                              encoding='utf-8')
        print(json.dumps({'rows': len(pending)}))
        return 0
    args.output.write_text(json.dumps(report(), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__': raise SystemExit(main())
