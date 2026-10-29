"""Local scoring of a quality-round run: team predictions format, judge inputs and judge-assisted results.

`predictions` turns the GPU results (one file, or the shards of a sharded run) into the team's
predictions.jsonl (same fields and scoring as baselines/run_product_evaluation.py) plus metrics. `judge-inputs` writes the
(sample, answer) pairs that still need the team judge: every final prediction without a
normalized match, and the C1 answer of every changed row, minus strings already judged in
the given earlier judge results. `score` reports judge-assisted accuracy and the net change
against C1. Correct means a normalized match or CORRECT; UNCERTAIN is not correct.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from baselines._lib.answer_scoring import answer_in_context, score_answer
from baselines.run_product_evaluation import MODE, aggregate_product_metrics


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def judge_id(sample_id, answer):
    return sample_id + '::' + hashlib.sha256(answer.encode()).hexdigest()[:10]


def prediction_row(result, gold, prompt_style='context_fallback'):
    """Same fields as build_product_row, from a serialized AdaptiveRAGResult."""
    prediction = '' if result['final_answer'] is None else str(result['final_answer'])
    return {'sample_id': str(result['sample_id']), 'query': result['query'], 'prediction': prediction,
            'gold': gold, 'mode': MODE, **score_answer(prediction, gold),
            'initial_answer': result['initial_answer'],
            'diagnosis_label': result['diagnosis']['final_label'],
            'selected': bool(result['selection_decision']['needs_enhancement']),
            'controller_action': result['controller_decision']['action'],
            'enhancement_status': result['enhancement_result']['status'],
            'retrieved_passage_ids': [e['passage_id'] for e in result['initial_evidence']],
            'retrieved_ranks': [int(e['rank']) for e in result['initial_evidence']],
            'answer_in_context': answer_in_context(gold, result['initial_evidence']),
            'answer_in_context_final': answer_in_context(gold, result['final_evidence']),
            'rag_prompt_style': prompt_style,
            'failure_type_route': result['metadata'].get('failure_type_route')}


def merge_results(paths):
    """Rows of one or more results.jsonl; a manifest.json beside a shard must be completed, and all
    manifests must name the same registration."""
    rows, registrations = [], set()
    for path in paths:
        manifest = Path(path).parent / 'manifest.json'
        if manifest.exists():
            spec = json.loads(manifest.read_text(encoding='utf-8'))
            if spec.get('state') != 'completed':
                raise ValueError(f'{path} comes from a run that did not complete')
            registrations.add(spec['registration_sha256'])
        rows.extend(read_jsonl(path))
    if len(registrations) > 1:
        raise ValueError('Shards ran under different registrations')
    return rows


def prior_labels(paths):
    """(sample_id, prediction) -> correct, from earlier team judge results; failed rows are ignored."""
    labels = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get('judge_status') in ('llm_ok', 'auto_correct'):
                sid = str(row['sample_id']).split('::')[0]  # our judge IDs append '::' and an answer hash
                labels[(sid, row['prediction'])] = bool(row['auto_correct']) or row['judge_label'] == 'CORRECT'
    return labels


def needed(rows):
    """Every answer string whose correctness the report needs."""
    pairs = []
    for row in rows:
        pairs.append((row['sample_id'], row['prediction']))
        if row['prediction'] != row['initial_answer']:
            pairs.append((row['sample_id'], row['initial_answer']))
    return pairs


def judge_rows(rows, labels):
    by_id, out, seen = {r['sample_id']: r for r in rows}, [], set()
    for sid, answer in needed(rows):
        if (sid, answer) in labels or (sid, answer) in seen:
            continue
        seen.add((sid, answer))
        gold = by_id[sid]['gold']
        out.append({'sample_id': judge_id(sid, answer), 'source_sample_id': sid, 'query': by_id[sid]['query'],
                    'prediction': answer, 'gold': gold,
                    'normalized_match': bool(score_answer(answer, gold)['normalized_match'])})
    return out


def score(rows, labels):
    missing = [pair for pair in needed(rows) if pair not in labels]
    if missing:
        raise ValueError(f'{len(missing)} answer strings are not judged yet')
    correct = sum(labels[(r['sample_id'], r['prediction'])] for r in rows)
    change = Counter()
    for r in rows:
        if r['prediction'] == r['initial_answer']:
            continue
        before, after = labels[(r['sample_id'], r['initial_answer'])], labels[(r['sample_id'], r['prediction'])]
        change['changed'] += 1
        change['fixes'] += after and not before
        change['harms'] += before and not after
    by_action = Counter((r['controller_action'], r['prediction'] != r['initial_answer']) for r in rows)
    return {'n': len(rows), 'judge_assisted_correct': correct, 'judge_assisted_accuracy': correct / len(rows),
            'changed': change['changed'], 'fixes': change['fixes'], 'harms': change['harms'],
            'net_vs_c1': change['fixes'] - change['harms'],
            'actions': {f'{a}|changed={c}': n for (a, c), n in sorted(by_action.items())}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('predictions')
    a.add_argument('--results', type=Path, nargs='+', required=True)
    a.add_argument('--references', type=Path, required=True, help='sample_id/gold rows')
    a.add_argument('--output-dir', type=Path, required=True)
    j = sub.add_parser('judge-inputs')
    j.add_argument('--predictions', type=Path, required=True)
    j.add_argument('--prior', type=Path, nargs='*', default=[])
    j.add_argument('--output-dir', type=Path, required=True)
    j.add_argument('--shards', type=int, default=4)
    s = sub.add_parser('score')
    s.add_argument('--predictions', type=Path, required=True)
    s.add_argument('--judged', type=Path, nargs='+', required=True)
    s.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'predictions':
        gold = {r['sample_id']: r['gold'] for r in read_jsonl(args.references)}
        results = merge_results(args.results)
        if len({r['sample_id'] for r in results}) != len(results) or set(gold) != {r['sample_id'] for r in results}:
            raise ValueError('Results and references cover different IDs')
        rows = [prediction_row(r, gold[r['sample_id']]) for r in results]
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        metrics = aggregate_product_metrics(rows)
        (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps({k: metrics[k] for k in ('n', 'em', 'normalized_accuracy', 'selected_n')
                          if k in metrics}, indent=2))
        return 0
    rows = read_jsonl(args.predictions)
    if args.command == 'judge-inputs':
        todo = judge_rows(rows, prior_labels(args.prior))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        shards = [[] for _ in range(args.shards)]
        for row in todo:
            shards[int(hashlib.sha256(row['sample_id'].encode()).hexdigest(), 16) % args.shards].append(row)
        for i, shard in enumerate(shards):
            with (args.output_dir / f'shard_{i}.jsonl').open('x', encoding='utf-8') as handle:
                for row in shard:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'rows': len(todo), 'api_rows': sum(not r['normalized_match'] for r in todo),
                          'shards': [len(x) for x in shards]}))
        return 0
    report = score(rows, prior_labels(args.judged))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
