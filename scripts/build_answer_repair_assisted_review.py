"""Package explicit agent judgments; never infer judgments or fill human labels.

decisions.json is a hand-authored agent review keyed by dataset/sample_id.
The four-character code is initial/final correctness, initial/final C1 support:
1=true, 0=false, ?=unresolved. This script validates and renders those decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

FIELDS = ('initial_correct', 'final_correct', 'initial_supported', 'final_supported')
LABELS = {True: '是', False: '否', None: '待定'}
CITATIONS = {'supported': '支持', 'supported_in_context': '结合原段指代后支持',
             'unsupported': '不支持', 'partial': '仅部分支持', 'none': '无输出引用'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f'Refusing to overwrite a different review artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def transition(review):
    initial, final = review['initial_correct'], review['final_correct']
    if initial is None or final is None:
        return 'correctness_unresolved'
    if not initial and final:
        return 'suggested_fix'
    if initial and not final:
        return 'suggested_regression'
    return 'correct_preserved' if initial else 'unresolved_error_or_abstention'


def summarize(rows):
    review = [r['agent_review'] for r in rows]
    count = Counter(r['outcome'] for r in review)
    return {'records': len(rows), 'outcomes': dict(count),
            **{f: {'yes': sum(r[f] is True for r in review),
                   'no': sum(r[f] is False for r in review),
                   'unresolved': sum(r[f] is None for r in review)} for f in FIELDS},
            'citation_judgments': dict(Counter(r['final_citation_support'] for r in review)),
            'flags': dict(Counter(flag for r in review for flag in r['flags']))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.queue.read_bytes()
    decisions_raw = args.decisions.read_bytes()
    decisions = json.loads(decisions_raw)
    queue = [json.loads(line) for line in source.splitlines()]
    assert {(r['dataset'], r['sample_id']) for r in queue} == {
        (ds, sid) for ds, cases in decisions.items() for sid in cases}
    outputs, unique, material_hashes = [], {}, {}
    for row in queue:
        ds, sid = row['dataset'], row['sample_id']
        result = row['model_result']
        assert row['result_sha256'] == digest(json.dumps(result, sort_keys=True, ensure_ascii=False).encode())
        assert row['pair_review']['status'] == 'pending'
        assert all(row['pair_review'][field] is None for field in FIELDS)
        code, citation, confidence, reason, anchors, flags = decisions[ds][sid]
        assert len(code) == 4 and set(code) <= set('01?')
        assert citation in CITATIONS and confidence in ('high', 'medium', 'low') and reason and anchors
        values = dict(zip(FIELDS, ({'1': True, '0': False, '?': None}[c] for c in code)))
        if not row['changed']:
            assert values['initial_correct'] == values['final_correct']
            assert values['initial_supported'] == values['final_supported']
        assert (citation == 'none') == (not result['citations'])
        located = []
        for rank, quote in anchors:
            evidence = row['evidence'][rank - 1]
            start = evidence['text'].find(quote)
            assert start >= 0, (ds, sid, rank, quote)
            located.append({'rank': rank, 'passage_id': evidence['passage_id'],
                            'quote': quote, 'start': start, 'end': start + len(quote),
                            'evidence_sha256': digest(evidence['text'].encode())})
        # A traceable anchor is necessary, but this check is not an entailment judge.
        if citation in ('supported', 'supported_in_context'):
            assert any(a['quote'] in c['quote'] for a in located for c in result['citations']
                       if a['rank'] == c['rank'])
        material = {k: row[k] for k in ('query', 'initial_answer', 'final_answer', 'evidence', 'frozen_reference')}
        material['citations'] = result['citations']
        material_hash = digest(json.dumps(material, sort_keys=True, ensure_ascii=False).encode())
        key = ds, sid
        if key in material_hashes:
            assert material_hashes[key] == material_hash, f'Duplicate has different review material: {key}'
        material_hashes[key] = material_hash
        agent_review = {'status': 'assisted_complete_pending_human', 'reviewer_type': 'agent',
                        'reviewer': 'Codex', 'review_date': '2026-10-14', 'human_verified': False,
                        **values, 'final_citation_support': citation, 'confidence': confidence,
                        'reason_zh': reason, 'flags': flags, 'evidence_anchors': located,
                        'correctness_basis': ('frozen_reference_provisional' if 'reference_dependent_correctness' in flags
                                              else 'semantic_comparison_of_query_c1_and_frozen_reference'),
                        'reference_is_visible_to_reviewer': True}
        agent_review['outcome'] = transition(agent_review)
        output = {**row, 'review_number': row['triage_order'],
                  'shared_review_id': f'{ds}:{sid}',
                  'duplicate_of_review_number': unique[key]['review_number'] if key in unique else None,
                  'review_material_sha256': material_hash, 'agent_review': agent_review}
        outputs.append(output)
        unique.setdefault(key, output)
    by_run = defaultdict(list)
    for r in outputs:
        by_run[f"{r['dataset']}/{r['cohort']}/{r['strategy']}"].append(r)
    report = {'agent_assistance_only': True, 'human_verified': False, 'review_blinded': False,
              'source_queue_sha256': digest(source), 'decisions_sha256': digest(decisions_raw),
              'reviewed_result_pairs': len(outputs), 'unique_review_materials': len(unique),
              'duplicate_results': len(outputs) - len(unique), 'all_human_fields_unchanged_pending': True,
              'by_run': {name: summarize(rows) for name, rows in by_run.items()},
              'unique_samples': summarize(list(unique.values())),
              'unresolved_unique_review_numbers': [r['review_number'] for r in unique.values()
                                                   if any(r['agent_review'][f] is None for f in FIELDS)],
              'warning': 'Agent-assisted, reference-visible development review. Not human gold, independent evaluation, or held-out evidence.'}
    write_new(args.out_dir / 'assisted_pairs.jsonl',
              ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in outputs).encode())
    write_new(args.out_dir / 'summary.json', (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode())
    lines = ['# E v6r1：128 条结果辅助审核', '',
             '2026-10-14，Codex 代理辅助判断，全部待用户确认。审核使用冻结问题、初始/最终答案、参考标签/答案及 C1 Top-5，参考答案可见，属于非盲开发审核。', '',
             '按主题定位和审阅相关证据句、必要的完整段落与模型实际引用窗口；并非独立人工全文盲审。没有外部事实检索。引用片段均已在对应存档中逐字定位。', '',
             '“正确”判断是否回答问题的目标、关系、时间和完整性；允许同义短答、较长答案。“支持”判断 C1 是否足以支持该回答；合理拒答可以有支持但不算回答成功。事实短语真实但不回答本题，不算问题所需的支持。', '',
             'Top-5 支持与模型实际引用分开。缺少证据不等于反驳。FEVER 某些正确性仅依据冻结标签，已标 reference_dependent_correctness；这些不是独立事实确认。时间依赖问题采用冻结语料时点。', '',
             '128 条结果含 123 个唯一样本。5 条重复记录的审核材料完全相同，复用判断并保留各自结果 SHA-256。人工 pair_review 未改，agent_review 独立存放。待定项以 null 保存。', '',
             '| 批次 | 条数 | 建议语义修复 | 建议语义误修 | 正确性待定 |',
             '|---|---:|---:|---:|---:|']
    for name, group in report['by_run'].items():
        counts = group['outcomes']
        lines.append(f"| {name} | {group['records']} | {counts.get('suggested_fix', 0)} | {counts.get('suggested_regression', 0)} | {counts.get('correctness_unresolved', 0)} |")
    lines += ['', '以上是代理建议数；跨批次有重叠，不直接相加宣传效果。对“正确性待定”没有强行计入修复/误修。', '']
    for row in outputs:
        a = row['agent_review']
        lines += [f"## {row['review_number']}. {row['dataset'].upper()} / {row['cohort']} / {row['sample_id']}", '',
                  f"问题：{row['query']}", '', f"初始答案：{row['initial_answer']}", '',
                  f"最终答案：{row['final_answer']}", '',
                  f"建议：初始正确 **{LABELS[a['initial_correct']]}** → 最终正确 **{LABELS[a['final_correct']]}**；Top-5 支持 **{LABELS[a['initial_supported']]} → {LABELS[a['final_supported']]}**。", '',
                  f"实际输出引用：{CITATIONS[a['final_citation_support']]}。置信度：{a['confidence']}。", '',
                  a['reason_zh'], '',
                  '证据定位：' + '；'.join(f"R{x['rank']} `{x['passage_id']}`：{x['quote']}" for x in a['evidence_anchors']), '',
                  f"冻结参考（仅作对照）：{row['frozen_reference']}", '',
                  f"结果 SHA-256：`{row['result_sha256']}`；人工状态：pending。", '']
        if row['duplicate_of_review_number'] is not None:
            lines += [f"本条与第 {row['duplicate_of_review_number']} 条审核材料相同；独立保留本批结果指纹。", '']
    write_new(args.out_dir / 'assisted_review.md', ('\n'.join(lines) + '\n').encode())
    print(json.dumps({k: v for k, v in report.items() if k not in ('by_run', 'unique_samples')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
