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
LABELS = {True: 'Yes', False: 'No', None: 'Unresolved'}
CITATIONS = {'supported': 'Supported', 'supported_in_context': 'Supported after resolving a reference in the original passage',
             'unsupported': 'Unsupported', 'partial': 'Only partly supported', 'none': 'No output citation'}


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
    lines = ['# E v6r1: assisted review of 128 results', '',
             '2026-10-14. Codex agent-assisted judgments, all awaiting user confirmation. The review uses the frozen question, initial and final answers, reference label or answer, and C1 Top-5. The reference answer is visible, so this is a non-blind development review.', '',
             'Locate the topic and read the relevant evidence sentences, any needed full passages, and the model citation windows. This is not an independent full-text blind review. There is no external fact lookup. Citation spans were located verbatim in the matching archive.', '',
             '"Correct" asks whether the answer covers the question target, relation, time, and completeness. Synonymous short answers and longer answers are allowed. "Supported" asks whether C1 is enough to support that answer. A reasonable refusal can be supported without counting as a successful answer. A true fact phrase that does not answer this question is not the support the question needs.', '',
             'Top-5 support is separate from the model\'s actual citations. Missing evidence is not a contradiction. Some FEVER correctness rests only on the frozen label and is marked reference_dependent_correctness; those are not independent fact checks. Time-dependent questions use the frozen corpus time.', '',
             '128 results contain 123 unique samples. Five duplicate records have identical review material; the judgment is reused and each result SHA-256 is kept. Human pair_review is unchanged, and agent_review is stored separately. Unresolved items are stored as null.', '',
             '| Cohort | Items | Suggested semantic fix | Suggested semantic regression | Correctness unresolved |',
             '|---|---:|---:|---:|---:|']
    for name, group in report['by_run'].items():
        counts = group['outcomes']
        lines.append(f"| {name} | {group['records']} | {counts.get('suggested_fix', 0)} | {counts.get('suggested_regression', 0)} | {counts.get('correctness_unresolved', 0)} |")
    lines += ['', 'These are agent suggestion counts. Batches overlap, so do not add them together as an effect. Items marked correctness-unresolved are not forced into fix or regression.', '']
    for row in outputs:
        a = row['agent_review']
        lines += [f"## {row['review_number']}. {row['dataset'].upper()} / {row['cohort']} / {row['sample_id']}", '',
                  f"Question: {row['query']}", '', f"Initial answer: {row['initial_answer']}", '',
                  f"Final answer: {row['final_answer']}", '',
                  f"Suggestion: initial correct **{LABELS[a['initial_correct']]}** → final correct **{LABELS[a['final_correct']]}**; Top-5 support **{LABELS[a['initial_supported']]} → {LABELS[a['final_supported']]}**.", '',
                  f"Actual output citation: {CITATIONS[a['final_citation_support']]}. Confidence: {a['confidence']}.", '',
                  a['reason_zh'], '',
                  'Evidence location: ' + '; '.join(f"R{x['rank']} `{x['passage_id']}`: {x['quote']}" for x in a['evidence_anchors']), '',
                  f"Frozen reference (comparison only): {row['frozen_reference']}", '',
                  f"Result SHA-256: `{row['result_sha256']}`; human status: pending.", '']
        if row['duplicate_of_review_number'] is not None:
            lines += [f"This item has the same review material as item {row['duplicate_of_review_number']}; this batch's result fingerprint is kept separately.", '']
    write_new(args.out_dir / 'assisted_review.md', ('\n'.join(lines) + '\n').encode())
    print(json.dumps({k: v for k, v in report.items() if k not in ('by_run', 'unique_samples')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
