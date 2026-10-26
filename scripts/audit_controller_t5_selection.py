"""Audit one frozen selection export; score offline without another search.

The evaluator never reads final inputs, changes references, or promotes a
candidate on automatic QA scores alone. Original GPU bytes remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_controller_integration import summarize
from src.controller.evaluation import paired_statistics, replay_routes, transition, validate_predictions
from src.controller.v2a_policy import route_override
from src.evaluation.answer_repair import score_answer

RELEASE = 'artifacts/controller/t5_selection_v1/gpu_release_v2/execution_manifest.json'
POLICY = 'configs/controller/t5_selection_execution_v1.json'
CANDIDATE = 'artifacts/controller/t5_candidate_v1/candidate.json'
ACTION_MODES = {'no_action': 'c1', 'retrieval_enhancement': 'always_retrieve',
                'evidence_refinement': 'always_rerank', 'generation_repair': 'always_repair'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verified_members(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            raise ValueError('Duplicate or unsafe archive member')
        members = {n: archive.read(n) for n in names}
    checks = json.loads(members['output_checksums.json'])
    if set(checks) != set(members) - {'output_checksums.json'}:
        raise ValueError('Incomplete export checksum coverage')
    if any(sha(members[n]) != h for n, h in checks.items()):
        raise ValueError('Export checksum mismatch')
    return members


def verify_cache(rows):
    indexed = {(r['dataset'], r['sample_id'], r['mode']): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate cached prediction')
    checked = 0
    for row in rows:
        if row['mode'] == 'always_repair':
            if row['calls'] or row['final_answer'] != row['initial_answer'] or row['status'] != 'verification_unavailable':
                raise ValueError('Unavailable E performed a call or changed an answer')
        if row['mode'] != 'adaptive':
            continue
        fixed = indexed[row['dataset'], row['sample_id'], ACTION_MODES[row['routing']['decision']['action']]]
        if any(row[k] != fixed[k] for k in ('input_sha256', 'final_answer', 'final_evidence')):
            raise ValueError('Adaptive and fixed-action states differ')
        signature = lambda r: [(e['kind'], e['stage'], e['arguments']) for e in r['calls'] if e['stage'] != 'probe']
        if signature(row) != signature(fixed):
            raise ValueError('Adaptive and fixed-action call arguments differ')
        checked += 1
    return indexed, checked


def selection_decision(net_correct):
    return {'candidate_promoted': False,
            'status': 'rejected' if net_correct <= 0 else 'pending_semantic_review',
            'reason': 'selection_frozen_EM_net_not_positive' if net_correct <= 0 else 'automatic_scores_cannot_replace_semantic_review',
            'retain_original_route': True, 'alternate_thresholds_evaluated': 0,
            'final_evaluation_started': False}


def audit(path, root=ROOT):
    root = Path(root)
    members = verified_members(path)
    frozen = json.loads((root / RELEASE).read_text())
    policy = json.loads((root / POLICY).read_text())
    candidate = json.loads((root / CANDIDATE).read_text())
    if (candidate['threshold'] != .2 or policy['candidate_threshold'] != .2
            or policy['dataset_counts'] != {'nq': 200, 'fever': 200}
            or sha((root / CANDIDATE).read_bytes()) != frozen['offline_candidate_sha256']):
        raise ValueError('Frozen sole candidate or selection scope changed')
    seal = json.loads(members['execution_seal.json'])
    state = json.loads(members['manifest.json'])
    if (any(seal.get(k) != v for k, v in frozen.items())
            or json.loads(members['input_execution_manifest.json']) != frozen
            or state['state'] != 'completed' or state['n_predictions'] != 2000
            or state['failed_records'] != 0 or state['source_commit'] != frozen['source_commit']
            or state['E_mode'] != 'unavailable' or state['quality_scored'] is not False):
        raise ValueError('Incomplete run or execution seal mismatch')
    for name, key in [('engineering_receipt.json', 'engineering_receipt_sha256'),
                      ('environment.json', 'environment_sha256')]:
        if sha(members[name]) != seal[key]:
            raise ValueError('Execution receipt hash mismatch')
    engineering = json.loads(members['engineering_receipt.json'])
    active = engineering['active_long_checks']
    expected_paths = {(ds, mode) for ds in ('nq', 'fever') for mode in ('probe', 'always_retrieve', 'always_rerank')}
    if (not engineering['passed'] or len(active) != 6
            or {(c['dataset'], c['path']) for c in active} != expected_paths
            or any(not c['passed'] or c['minimum_actual_input_tokens'] < 14000 for c in active)
            or seal['E_engineering_accepted'] is not False):
        raise ValueError('Active-path engineering checks incomplete')
    inactive = engineering['inactive_E_checks']
    if len(inactive) != 2 or any(r['calls'] or r['final_answer'] != r['initial_answer'] for r in inactive):
        raise ValueError('Inactive E engineering assertion failed')
    lookup = json.loads(members['historical_lookup.json'])
    if len(lookup) != 2 or any(not r['same_ordered_top5'] for r in lookup):
        raise ValueError('Historical resource lookup mismatch')
    rows = [json.loads(line) for line in members['predictions.jsonl'].splitlines()]
    registration = {'datasets': frozen['datasets'], 'required_modes': frozen['modes']}
    acceptance = validate_predictions(rows, registration)
    if acceptance['execution_errors']:
        raise ValueError('Failed calls or budget blocks')
    indexed, cache_checks = verify_cache(rows)
    raw_summary = json.loads(members['summary.json'])
    if summarize(rows) != raw_summary:
        raise ValueError('Raw run summary does not replay')
    references, gold_hashes = {}, {}
    for ds, info in frozen['datasets'].items():
        raw = (root / f'artifacts/controller/data_v1/offline/{ds}/selection/gold.jsonl').read_bytes()
        if sha(raw) != info['offline_gold_sha256']:
            raise ValueError('Frozen selection reference hash changed')
        gold_hashes[ds] = sha(raw)
        refs = [json.loads(line) for line in raw.splitlines()]
        if (len(refs) != 200 or {r['sample_id'] for r in refs} != set(info['sample_ids'])
                or any(r['dataset'] != ds for r in refs)):
            raise ValueError('Selection references do not exactly cover inputs')
        references.update({(ds, r['sample_id']): r for r in refs})
    correct = lambda ds, sid, answer: score_answer(answer, references[ds, sid], ds)['correct']
    groups, comparisons = {}, {}
    for ds, info in frozen['datasets'].items():
        ids = info['sample_ids']
        base = [correct(ds, sid, indexed[ds, sid, 'c1']['final_answer']) for sid in ids]
        groups[ds] = {}
        for mode in frozen['modes']:
            batch = [indexed[ds, sid, mode] for sid in ids]
            values = [correct(ds, r['sample_id'], r['final_answer']) for r in batch]
            groups[ds][mode] = {'n': 200, 'correct': sum(values), 'accuracy': sum(values) / 200,
                'vs_C1': transition(base, values), 'paired_vs_C1': paired_statistics(values, base),
                'changed_answers': sum(r['answer_changed'] for r in batch),
                'measured_incremental_cost': raw_summary['groups'][ds + '/' + mode]}
    adaptive = [r for r in rows if r['mode'] == 'adaptive']
    routes = [{'dataset': r['dataset'], 'sample_id': r['sample_id'],
               'action': route_override(r, .2)['action']} for r in adaptive]
    replay = replay_routes(rows, routes)
    original_routes = [{'dataset': r['dataset'], 'sample_id': r['sample_id'],
                        'action': r['routing']['decision']['action']} for r in adaptive]
    original = replay_routes(rows, original_routes)
    old = {(r['dataset'], r['sample_id']): r for r in original['records']}
    changed, replay_records = [], replay['records']
    for ds in ('nq', 'fever'):
        after = [r for r in replay_records if r['dataset'] == ds]
        before = [old[ds, r['sample_id']] for r in after]
        if any(r['final_answer'] != indexed[ds, r['sample_id'], 'adaptive']['final_answer'] for r in before):
            raise ValueError('Original route replay differs from actual Adaptive')
        a = [correct(ds, r['sample_id'], r['final_answer']) for r in before]
        b = [correct(ds, r['sample_id'], r['final_answer']) for r in after]
        delta = {k: sum(r['counterfactual_cost_estimate'][k] for r in after)
                    - sum(r['counterfactual_cost_estimate'][k] for r in before)
                 for k in ('llm_calls', 'retrieval_calls', 'scoring_calls', 'provider_seconds_estimate')}
        overrides = sum(route_override(r, .2)['overridden'] for r in adaptive if r['dataset'] == ds)
        if ds == 'fever' and (after != before or overrides):
            raise ValueError('NQ-only candidate changed FEVER answers, routes or costs')
        pairs = 0
        for prev, nxt in zip(before, after):
            if prev['final_answer'] == nxt['final_answer']:
                continue
            pairs += 1
            actual = indexed[ds, nxt['sample_id'], 'adaptive']
            chosen = indexed[ds, nxt['sample_id'], nxt['source_mode']]
            changed.append({'review_id': f'S{len(changed) + 1:02}', 'dataset': ds,
                'sample_id': nxt['sample_id'], 'query': actual['input']['query'],
                'original_adaptive_answer': prev['final_answer'], 'candidate_answer': nxt['final_answer'],
                'gold': references[ds, nxt['sample_id']]['gold_answer_or_label'],
                'frozen_EM_before': correct(ds, nxt['sample_id'], prev['final_answer']),
                'frozen_EM_after': correct(ds, nxt['sample_id'], nxt['final_answer']),
                'initial_top5': actual['input']['evidence'], 'candidate_evidence': chosen['final_evidence'],
                'semantic_before': None, 'semantic_after': None, 'human_review': 'pending',
                'scope': 'Reference-visible explanatory review; not training data or another tuning round'})
        comparisons[ds] = {'n': 200, 'original_adaptive_correct': sum(a),
            'candidate_correct': sum(b), 'vs_original_adaptive': transition(a, b),
            'paired_vs_original_adaptive': paired_statistics(b, a),
            'overridden_routes': overrides, 'changed_answer_pairs': pairs,
            'semantic_review_pending': pairs, 'semantic_net_bounds': [-pairs, pairs],
            'incremental_cost_delta_estimate': delta}
    decision = selection_decision(comparisons['nq']['vs_original_adaptive']['net_correct'])
    report = {'schema_version': 'controller_t5_selection_audit_v1',
        'role': 'selection_descriptive_not_final_test', 'archive_sha256': sha(Path(path).read_bytes()),
        'archive_bytes': Path(path).stat().st_size, 'archive_checked_files': len(members) - 1,
        'source_commit': frozen['source_commit'], 'reference_sha256': gold_hashes,
        'acceptance': acceptance, 'active_long_checks_passed': 6, 'inactive_E_checks_passed': 2,
        'historical_queries_reproduced': 2, 'adaptive_fixed_cache_checks': cache_checks,
        'groups': groups, 'candidate_threshold': .2, 'candidate_comparison': comparisons,
        'decision': decision, 'new_GPU_calls': 0,
        'statistical_scope': 'Paired bootstrap/McNemar descriptive only; no final four-hypothesis Holm family without Full',
        'cost_scope': 'Measured costs exclude shared C1; candidate provider times are replay estimates, not deployment latency',
        'semantic_scope': 'NQ EM transitions are not human semantic verdicts; pending review cannot reverse the frozen EM rejection',
        'E_scope': 'Unavailable on both datasets; zero calls/changes do not demonstrate E quality or AC6 precision',
        'analysis_source_sha256': {p: sha((root / p).read_bytes()) for p in
            ('scripts/audit_controller_t5_selection.py', 'src/controller/evaluation.py',
             'src/controller/v2a_policy.py', 'src/evaluation/answer_repair.py',
             'src/evaluation/qa.py', 'src/evaluation/fever.py')}}
    return report, changed, replay_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Preserve previous selection audit')
    report, changed, replay = audit(args.archive)
    args.output_dir.mkdir(parents=True)
    shutil.copyfile(args.archive, args.output_dir / 'outputs.zip')
    for name, value in [('audit.json', report), ('decision.json', report['decision'])]:
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    for name, records in [('candidate_review15.jsonl', changed), ('candidate_replay.jsonl', replay)]:
        (args.output_dir / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    print(json.dumps({'acceptance': report['acceptance'], 'candidate': report['candidate_comparison'],
                      'decision': report['decision']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
