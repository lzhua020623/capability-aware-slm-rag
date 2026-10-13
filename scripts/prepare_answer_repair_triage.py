"""Order audited result pairs for human review without assigning semantic labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def priority(row):
    result = row['model_result']
    initial, final = result['initial_metrics']['correct'], result['final_metrics']['correct']
    if initial and not final:
        return 1, 'frozen_metric_regression'
    if not initial and final:
        return 2, 'frozen_metric_gain'
    if row['changed']:
        return 3, 'changed_without_frozen_correctness_gain'
    return 4, 'unchanged'


def write_new(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f'Refusing to overwrite an existing review file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.queue.read_bytes()
    rows = [json.loads(line) for line in source.splitlines()]
    seen = set()
    for row in rows:
        key = row['dataset'], row['cohort'], row['strategy'], row['sample_id']
        if key in seen:
            raise ValueError(f'Duplicate result pair: {key}')
        seen.add(key)
        digest = hashlib.sha256(json.dumps(row['model_result'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if digest != row['result_sha256']:
            raise ValueError(f'Result digest mismatch: {key}')
        if row['pair_review']['status'] != 'pending':
            raise ValueError('Create triage from the untouched audit queue; preserve edited human reviews separately')
        if any(row['pair_review'][k] is not None for k in
               ('initial_correct', 'final_correct', 'initial_supported', 'final_supported')):
            raise ValueError('Pending human labels must remain empty')
    rows.sort(key=lambda r: (priority(r)[0], r['dataset'] != 'fever',
                             r['cohort'] != 'reviewed', r['sample_id']))
    ordered = [{**r, 'triage_order': i, 'triage_priority': priority(r)[0],
                'triage_reason': priority(r)[1]} for i, r in enumerate(rows, 1)]
    write_new(args.out_dir / 'review_priority.jsonl',
              ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in ordered).encode())
    counts = Counter(r['triage_reason'] for r in ordered)
    report = {'source_queue': str(args.queue), 'source_sha256': hashlib.sha256(source).hexdigest(),
              'pairs': len(rows), 'unique_dataset_sample_ids': len({(r['dataset'], r['sample_id']) for r in rows}),
              'changed_pairs': sum(r['changed'] for r in rows), 'priority_counts': dict(counts),
              'semantic_labels_assigned': False, 'manual_review_status': 'deferred_by_user',
              'warning': 'Frozen-reference transitions are review priorities, not human semantic verdicts. Cohort overlaps are retained.'}
    write_new(args.out_dir / 'triage_summary.json', (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode())
    lines = ['# E 新输出人工审核顺序', '',
             f"排序仅依据冻结指标和是否改答案，不代表人工结论。{len(rows)} 条结果，{report['unique_dataset_sample_ids']} 个唯一 dataset/sample_id；各批分别保留。", '',
             f"优先审核前 {report['changed_pairs']} 条改动（{counts['frozen_metric_regression']} 条指标退步、{counts['frozen_metric_gain']} 条指标改善、{counts['changed_without_frozen_correctness_gain']} 条其他改动）；其余 {counts['unchanged']} 条保留答案的结果仍待完整证据支持审核。", '',
             '人工需要对同一条的初始答案与最终答案分别判定：正确性、证据支持，共四项；记录审核人和理由。',
             '请把确认写入 review_priority.jsonl 的 pair_review，完成时设 status=completed；四个布尔字段不能为空。',
             '原始 pending_semantic_pairs.jsonl 保留为不可改写的审计底稿。不要用冻结参考自动填人工字段。', '',
             '已有正确性判断也要按同一口径复查。引用字面来自原文，只能证明来源合法；返回原答案不代表答案已获证实。', '']
    for r in ordered:
        if not r['changed']:
            continue
        lines += [f"## {r['triage_order']}. {r['dataset'].upper()} / {r['cohort']} / {r['sample_id']}", '',
                  f"优先原因：{r['triage_reason']}；结果指纹：`{r['result_sha256']}`。", '',
                  f"问题：{r['query']}", '', f"初始：{r['initial_answer']}", '',
                  f"最终：{r['final_answer']}", '',
                  '人工状态：pending，正确性与支持性均未判定。', '']
    write_new(args.out_dir / 'review_priority.md', ('\n'.join(lines) + '\n').encode())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
