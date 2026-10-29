"""Offline side of the quality sweep: gold export, judge inputs and the registered grid choice.

Gold is read only here, on the local machine, never on the GPU host. Every configuration in
the registered grid is a choice among the precomputed outputs: recalibrated P60 thresholds
pick the failure-pool items, the action map picks which stored module answer they take,
and the correct gate optionally lets high-p_fail correct-labelled items take P3-20.
Correct means a normalized match or the team judge's CORRECT; UNCERTAIN is not correct.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from baselines._lib.answer_scoring import score_answer
from scripts.precompute_quality_sweep import gated_correct, load_registration

SELECTOR = 'configs/controller/p60_online_adapter.json'
FEATURES = ('p_fail', 'rec', 'p_corr_3c')
Z_KEYS = {'p_fail': 'p_fail', 'rec': 'rec', 'p_corr_3c': 'p_corr'}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def judge_id(sample_id, answer):
    return sample_id + '::' + hashlib.sha256(answer.encode()).hexdigest()[:10]


def export_references(delivery, ids, out):
    """sample_id and gold only, for the registered IDs, from the delivery's annotation references."""
    wanted, rows = set(ids), {}
    for path in sorted(Path(delivery).glob('nq_train_annotation_*.jsonl')):
        with path.open(encoding='utf-8') as handle:
            lines = list(handle)
        for line in lines:
            row = json.loads(line)
            if row['sample_id'] in wanted:
                gold = (row.get('annotation_reference') or {}).get('gold_answer')
                if gold in (None, '', []):
                    raise ValueError('Missing gold for ' + row['sample_id'])
                rows[row['sample_id']] = gold if isinstance(gold, list) else [gold]
    if set(rows) != wanted:
        raise ValueError(f'{len(wanted - set(rows))} registered IDs lack gold')
    with Path(out).open('x', encoding='utf-8') as handle:
        for sid in ids:
            handle.write(json.dumps({'sample_id': sid, 'gold': rows[sid]}, ensure_ascii=False) + '\n')


def read_records(path, ids, *, partial=False):
    records = read_jsonl(path)
    order = [r['sample_id'] for r in records]
    if order != ids[:len(order)] or (not partial and len(order) != len(ids)):
        raise ValueError('Sweep records are incomplete or out of registered order')
    return records


def in_pool(record):
    return not gated_correct(record['diagnosis'])


def z_stats(pool):
    out = {}
    for name in FEATURES:
        values = [r['frozen_selection'][name] for r in pool]
        out[Z_KEYS[name]] = {'mean': statistics.fmean(values), 'std': statistics.pstdev(values)}
    return out


def combo(record, z, alpha, lam):
    s = record['frozen_selection']
    zf = lambda name: (s[name] - z[Z_KEYS[name]]['mean']) / z[Z_KEYS[name]]['std']
    return alpha * zf('p_fail') + (1 - alpha) * zf('rec') - lam * zf('p_corr_3c')


def calibrate(pool, coverage, insuff_share, alpha, lam):
    """Thresholds that select `coverage` of the pool, `insuff_share` of them by p_insuff, the rest by combo.

    Mirrors the production rule: p_insuff >= t_insuff, else combo >= t_combo.
    """
    z = z_stats(pool)
    target = round(coverage * len(pool))
    by_insuff = sorted((r['frozen_selection']['p_insuff'] for r in pool), reverse=True)
    k = round(insuff_share * target)
    t_insuff = by_insuff[k - 1] if k else math.inf
    rest = [r for r in pool if r['frozen_selection']['p_insuff'] < t_insuff]
    need = target - (len(pool) - len(rest))
    combos = sorted((combo(r, z, alpha, lam) for r in rest), reverse=True)
    t_combo = combos[need - 1] if need > 0 else math.inf
    selected = {r['sample_id'] for r in pool
                if r['frozen_selection']['p_insuff'] >= t_insuff or combo(r, z, alpha, lam) >= t_combo}
    return {'z_stats': z, 't_insuff': t_insuff, 't_combo': t_combo, 'selected': selected}


def final_answer(record, selected, action_map, correct_gate, cut):
    c1 = record['initial_answer']
    if gated_correct(record['diagnosis']):
        open_gate = correct_gate != 'closed' and record['diagnosis']['stage1_failure_probability'] >= cut
        return record['actions']['retrieval_enhancement']['final_answer'] if open_gate else c1
    if record['sample_id'] not in selected:
        return c1
    action = action_map[record['diagnosis']['final_label']]
    return c1 if action == 'no_action' else record['actions'][action]['final_answer']


def grid(registration):
    g = registration['grid']
    for coverage, share, (name, action_map), gate in itertools.product(
            g['coverage_of_failure_pool'], g['insuff_lane_share'], g['action_maps'].items(), g['correct_gate']):
        yield {'coverage': coverage, 'insuff_share': share, 'action_map': name, 'correct_gate': gate}, action_map


def judge_pairs(records):
    """(sample, answer) pairs any configuration can need: C1 and every differing stored answer."""
    pairs = []
    for r in records:
        answers = {a['final_answer'] for a in r['actions'].values() if a['final_answer'] != r['initial_answer']}
        if answers:
            pairs.extend((r['sample_id'], a) for a in [r['initial_answer']] + sorted(answers))
    return pairs


def judge_rows(records, gold, exclude=()):
    queries = {r['sample_id']: r['query'] for r in records}
    rows, skip = [], set(exclude)
    for sid, answer in judge_pairs(records):
        jid = judge_id(sid, answer)
        if jid in skip:
            continue
        skip.add(jid)
        rows.append({'sample_id': jid, 'source_sample_id': sid, 'query': queries[sid], 'prediction': answer,
                     'gold': gold[sid], 'normalized_match': bool(score_answer(answer, gold[sid])['normalized_match'])})
    return rows


def load_judged(paths):
    judged = {}
    for path in paths:
        for row in read_jsonl(path):
            if row['judge_status'] in ('llm_ok', 'auto_correct'):
                judged[row['sample_id']] = bool(row['auto_correct']) or row['judge_label'] == 'CORRECT'
    return judged


def evaluate(records, gold, judged, selected, action_map, correct_gate, cut):
    counts = Counter()
    for r in records:
        after = final_answer(r, selected, action_map, correct_gate, cut)
        before = r['initial_answer']
        if after == before:
            continue
        counts['changed'] += 1
        nm = lambda a: bool(score_answer(a, gold[r['sample_id']])['normalized_match'])
        counts['nm_fixes'] += nm(after) and not nm(before)
        counts['nm_harms'] += nm(before) and not nm(after)
        jb, ja = judged.get(judge_id(r['sample_id'], before)), judged.get(judge_id(r['sample_id'], after))
        if jb is None or ja is None:
            counts['unjudged'] += 1
            continue
        counts['fixes'] += ja and not jb
        counts['harms'] += jb and not ja
    out = {k: counts[k] for k in ('changed', 'fixes', 'harms', 'nm_fixes', 'nm_harms', 'unjudged')}
    out.update(net=out['fixes'] - out['harms'], nm_net=out['nm_fixes'] - out['nm_harms'])
    return out


def choose(results, registration, metric='net'):
    """Registered rule: best net; within 3 items take lowest coverage, then the closed gate, then the earlier map.

    Remaining ties go to the higher net, then the smaller insuff lane share.
    """
    maps = list(registration['grid']['action_maps'])
    best = max(r[metric] for r in results)
    close = [r for r in results if r[metric] >= best - 3]
    return min(close, key=lambda r: (r['coverage'], r['correct_gate'] != 'closed', maps.index(r['action_map']),
                                     -r[metric], r['insuff_share']))


def score_grid(records, gold, judged, registration):
    pool = [r for r in records if in_pool(r)]
    d = json.loads((ROOT / SELECTOR).read_text(encoding='utf-8'))['d_combo']
    cut = registration['precompute']['correct_p_fail_cut']
    results, calibrations = [], {}
    for config, action_map in grid(registration):
        key = (config['coverage'], config['insuff_share'])
        if key not in calibrations:
            calibrations[key] = calibrate(pool, config['coverage'], config['insuff_share'], d['alpha'], d['lam'])
        cal = calibrations[key]
        row = {**config, 'selected_in_pool': len(cal['selected']),
               **evaluate(records, gold, judged, cal['selected'], action_map, config['correct_gate'], cut)}
        results.append(row)
    return pool, results, calibrations


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('references')
    r.add_argument('--delivery', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    j = sub.add_parser('judge-inputs')
    j.add_argument('--records', type=Path, required=True)
    j.add_argument('--references', type=Path, required=True)
    j.add_argument('--output-dir', type=Path, required=True)
    j.add_argument('--shards', type=int, default=4)
    j.add_argument('--exclude', type=Path, nargs='*', default=[], help='Earlier judge inputs already sent')
    j.add_argument('--partial', action='store_true', help='Records still being written')
    s = sub.add_parser('score')
    s.add_argument('--records', type=Path, required=True)
    s.add_argument('--references', type=Path, required=True)
    s.add_argument('--judged', type=Path, nargs='*', default=[])
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--partial', action='store_true')
    args = p.parse_args(argv)
    registration, ids = load_registration()
    if args.command == 'references':
        export_references(args.delivery, ids, args.output)
        return 0
    gold = {row['sample_id']: row['gold'] for row in read_jsonl(args.references)}
    records = read_records(args.records, ids, partial=args.partial)
    if args.command == 'judge-inputs':
        sent = {row['sample_id'] for path in args.exclude for row in read_jsonl(path)}
        rows = judge_rows(records, gold, sent)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        shards = [[] for _ in range(args.shards)]
        for row in rows:
            shards[int(hashlib.sha256(row['sample_id'].encode()).hexdigest(), 16) % args.shards].append(row)
        for i, shard in enumerate(shards):
            with (args.output_dir / f'shard_{i}.jsonl').open('x', encoding='utf-8') as handle:
                for row in shard:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'records': len(records), 'rows': len(rows),
                          'api_rows': sum(not row['normalized_match'] for row in rows),
                          'shards': [len(x) for x in shards]}))
        return 0
    pool, results, calibrations = score_grid(records, gold, load_judged(args.judged), registration)
    metric = 'net' if args.judged and not any(r['unjudged'] for r in results) else 'nm_net'
    chosen = choose(results, registration, metric)
    cal = calibrations[(chosen['coverage'], chosen['insuff_share'])]
    report = {'schema_version': 'quality_sweep_offline_v1', 'records': len(records), 'pool': len(pool),
              'partial': args.partial, 'metric': metric, 'chosen': chosen,
              'chosen_thresholds': {'z_stats': cal['z_stats'], 't_insuff': cal['t_insuff'], 't_combo': cal['t_combo']},
              'results': sorted(results, key=lambda r: -r[metric])}
    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('records', 'pool', 'metric', 'chosen')}, indent=2, default=str))
    return 0


if __name__ == '__main__': raise SystemExit(main())
