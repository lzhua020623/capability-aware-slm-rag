"""NQ recoverability ladder, oracle levels (configs/controller/recoverability_oracle_v1.json).

contexts: for every sweep item, the L3 context (gold passage first, then C1's Top-5 without it, first four)
          and the L4 context (gold passage alone) as l2_generate.py rows. Gold passage IDs come from the
          team delivery (local only); rows hold sample_id, query and passage_ids, with no marker of which
          passage is gold. A context equal to C1's Top-5 needs no generation (C1's answer).
judge-inputs / score: per item, the answers at C1, L1 (route 1), L2 (B+BM25), L2_any (any of C1 and the 18 L2
          contexts), L3 and L4 from ladder inputs (one row per item: answers, gold passage, its BGE rank and the
          C1 + 8-context agreement), with labels from a resolved prior table, the new judge batch and normalized
          match. L3 / L4 take the generated answer (primary) or the P3 rule (secondary: a rejected answer keeps C1).
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
LADDER = ('L1', 'L2', 'L2_any', 'L3', 'L4')
# Reported per level, outside the registered first-fix order: P3-20 on every item (the bge20_bm250_rrf context),
# the real-retrieval level comparable with FEVER, where P3 runs on every item.
EXTRA = ('P3_all',)
P3_ALL_CONTEXT = 'bge20_bm250_rrf'
POSITIONS = ('c1_top1', 'c1_top2_5', 'bge_6_20', 'bge_21_100', 'beyond_100')


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


def oracle_answers(items, generations):
    """sample_id -> {level: (primary, secondary)} for L3 and L4."""
    out = {}
    for it in items:
        c1, c1_top5 = it['c1'], tuple(it['c1_top5'])
        res = {}
        for level, ids in oracle_contexts(c1_top5, it['gold_passage_id']).items():
            if ids == c1_top5:
                res[level] = (c1, c1)
            else:
                answer, reason = generations[(it['sample_id'], ids)]
                res[level] = (answer.strip(), c1 if reason is not None else answer.strip())
        out[it['sample_id']] = res
    return out


def needed_strings(item, answers):
    return {item['c1'], item['route_1'], item['b_bm25'], *item['contexts'].values(),
            *(a for pair in answers.values() for a in pair)}


def labeller(prior, judged, gold):
    """Normalized match first, then the new judge batch, then the resolved prior labels; None if unjudged."""
    from baselines._lib.answer_scoring import score_answer

    def label(sid, answer):
        if score_answer(answer, gold[sid])['normalized_match']:
            return True
        if (sid, answer) in judged:
            return judged[(sid, answer)]
        return prior.get((sid, answer))
    return label


def any_correct(values):
    return True if any(v is True for v in values) else None if any(v is None for v in values) else False


def item_levels(item, answers, label, rule=0):
    sid = item['sample_id']
    return {'C1': label(sid, item['c1']), 'L1': label(sid, item['route_1']), 'L2': label(sid, item['b_bm25']),
            'L2_any': any_correct([label(sid, a) for a in [item['c1'], *item['contexts'].values()]]),
            'L3': label(sid, answers['L3'][rule]), 'L4': label(sid, answers['L4'][rule]),
            'P3_all': label(sid, item['contexts'][P3_ALL_CONTEXT])}


def first_fix(levels):
    for level in LADDER:
        if levels[level] is None:
            return None
        if levels[level]:
            return level
    return 'none'


def gold_position(item):
    top5, gold, rank = item['c1_top5'], item['gold_passage_id'], item['gold_bge_rank']
    if gold == top5[0]:
        return 'c1_top1'
    if gold in top5:
        return 'c1_top2_5'
    if rank is not None and rank <= 20:
        return 'bge_6_20'
    if rank is not None:
        return 'bge_21_100'
    return 'beyond_100'


def agreement_bucket(share):
    return 'all_agree' if share == 1 else 'two_thirds' if share >= 2 / 3 else 'one_third' if share >= 1 / 3 else 'below'


def auc(positive, negative):
    """Probability that a positive scores above a negative (ties count half)."""
    if not positive or not negative:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in positive for q in negative)
    return wins / (len(positive) * len(negative))


def ladder_report(items, answers, label):
    from collections import Counter
    report, unjudged = {}, 0
    for rule, name in ((0, 'primary'), (1, 'p3_rule')):
        failures, fixes, by_pos, by_agree, harm = [], Counter(), {}, {}, Counter()
        for it in items:
            levels = item_levels(it, answers[it['sample_id']], label, rule)
            if levels['C1'] is None or None in levels.values():
                unjudged += rule == 0
                continue
            if levels['C1']:
                harm['c1_correct'] += 1
                for level in (*LADDER, *EXTRA):
                    if level != 'L2_any':
                        harm[f'{level}_breaks'] += not levels[level]
                continue
            fix = first_fix(levels)
            failures.append((it, fix, levels))
            fixes[fix] += 1
            by_pos.setdefault(gold_position(it), Counter())[fix] += 1
            by_agree.setdefault(agreement_bucket(it['agreement']['share']), Counter())[fix] += 1
        boundary = [it for it, fix, _ in failures if fix == 'none']
        recovered = [it for it, fix, _ in failures if fix != 'none']
        l3_l4 = {}
        for it, _, lv in failures:
            row = l3_l4.setdefault(gold_position(it), Counter())
            row['n'] += 1
            row['L3'] += bool(lv['L3'])
            row['L4'] += bool(lv['L4'])
            row['L3_wrong_L4_right'] += not lv['L3'] and bool(lv['L4'])
            row['L3_right_L4_wrong'] += bool(lv['L3']) and not lv['L4']
        report[name] = {
            'c1_failures': len(failures), 'first_fix': {k: fixes[k] for k in (*LADDER, 'none')},
            # Two different boundaries: L4 still wrong (oracle-persistent) vs no tested condition right.
            'oracle_persistent': sum(not lv['L4'] for _, _, lv in failures),
            'unrecovered_under_all_tested': fixes['none'],
            'l3_l4_by_gold_position': {p: dict(l3_l4.get(p, {})) for p in POSITIONS},
            'standalone': {k: sum(bool(lv[k]) for _, _, lv in failures) for k in (*LADDER, *EXTRA)},
            'rfr': {k: sum(bool(lv[k]) for _, _, lv in failures) / max(1, len(failures)) for k in (*LADDER, *EXTRA)},
            'by_gold_position': {p: dict(by_pos.get(p, {})) for p in POSITIONS},
            'by_agreement': {b: dict(by_agree.get(b, {})) for b in ('all_agree', 'two_thirds', 'one_third', 'below')},
            'boundary_auc': {'p_fail': auc([i['p_fail'] for i in boundary], [i['p_fail'] for i in recovered]),
                             'disagreement': auc([1 - i['agreement']['share'] for i in boundary],
                                                 [1 - i['agreement']['share'] for i in recovered]),
                             'answer_groups': auc([i['agreement']['groups'] for i in boundary],
                                                  [i['agreement']['groups'] for i in recovered])},
            'c1_correct': dict(harm)}
    return {'items': len(items), 'unjudged_items': unjudged, 'complete': unjudged == 0, **report}


def read_generations(paths):
    out = {}
    for path in paths:
        for g in read_jsonl(path):
            out[(g['sample_id'], tuple(g['passage_ids']))] = (g['generated_answer'], g['reject_reason'])
    return out


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
    for cmd in ('judge-inputs', 'score'):
        s = sub.add_parser(cmd)
        s.add_argument('--inputs', type=Path, required=True, help='ladder inputs, one row per sweep item')
        s.add_argument('--prior', type=Path, required=True, help='resolved prior labels: sample_id, answer, correct')
        s.add_argument('--generations', type=Path, nargs='+', required=True)
        s.add_argument('--references', type=Path, required=True, help='sample_id/gold rows, local only')
        s.add_argument('--labels', type=Path, nargs='*', default=[], help='new judge batch outputs')
        if cmd == 'judge-inputs':
            s.add_argument('--output-dir', type=Path, required=True)
            s.add_argument('--shards', type=int, default=4)
        else:
            s.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command != 'contexts':
        from scripts.score_quality_eval import prior_labels
        items = read_jsonl(args.inputs)
        if [it['sample_id'] for it in items] != sweep_ids():
            raise ValueError('Ladder inputs do not follow the registered sweep order')
        answers = oracle_answers(items, read_generations(args.generations))
        gold = {r['sample_id']: r['gold'] for r in read_jsonl(args.references)}
        prior = {(r['sample_id'], r['answer']): r['correct'] for r in read_jsonl(args.prior)}
        label = labeller(prior, prior_labels(args.labels), gold)
        if args.command == 'score':
            report = ladder_report(items, answers, label)
            args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({k: report[k] for k in ('items', 'unjudged_items', 'complete')} | {'first_fix': report['primary']['first_fix']}))
            return 0
        from baselines._lib.answer_scoring import score_answer
        from scripts.quality_sweep_offline import judge_id
        import hashlib
        pending = []
        for it in items:
            sid = it['sample_id']
            for answer in sorted(needed_strings(it, answers[sid])):
                if label(sid, answer) is None:
                    pending.append({'sample_id': judge_id(sid, answer), 'source_sample_id': sid, 'query': it['query'],
                                    'prediction': answer, 'gold': gold[sid], 'normalized_match': False})
        args.output_dir.mkdir(parents=True, exist_ok=False)
        shards = [[] for _ in range(args.shards)]
        for row in pending:
            shards[int(hashlib.sha256(row['sample_id'].encode()).hexdigest(), 16) % args.shards].append(row)
        for k, shard in enumerate(shards):
            with (args.output_dir / f'shard_{k}.jsonl').open('x', encoding='utf-8') as handle:
                for row in shard:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'rows': len(pending), 'api_rows': len(pending)}))
        return 0
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
