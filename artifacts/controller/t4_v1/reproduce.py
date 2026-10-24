"""Offline replay of the archived T4 run; no model loading, tuning or GPU calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_controller_integration import evaluate
from scripts.run_controller_integration import summarize
from scripts.run_controller_t4 import accept_engineering, sha
from src.controller.execution import record_action
from src.evaluation.answer_repair import score_answer

ARCHIVE_SHA = 'be3b926b8f808cd39b1c32b64564e5e5619518bebd2585af3fd8f0a6616bc7a5'
SINGLE = ('always_retrieve', 'always_rerank', 'always_repair')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def reproduce(output):
    if output.exists():
        raise FileExistsError('Choose a new directory; preserve prior replay and human fields')
    archive = HERE / 'controller_t4_development_v1_outputs.zip'
    assert sha(archive) == ARCHIVE_SHA
    with zipfile.ZipFile(archive) as handle:
        checks = json.loads(handle.read('output_checksums.json'))
        for rel, digest in checks.items():
            assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
            assert hashlib.sha256(handle.read(rel)).hexdigest() == digest, rel
        rows = [json.loads(line) for line in io.TextIOWrapper(handle.open('predictions.jsonl'), encoding='utf-8')]
        seal = json.loads(handle.read('execution_seal.json'))
        manifest = json.loads(handle.read('input_execution_manifest.json'))
        environment = json.loads(handle.read('environment.json'))
        assert hashlib.sha256(handle.read('environment.json')).hexdigest() == seal['environment_sha256']
        lookup = json.loads(handle.read('historical_lookup.json'))
        assert len(lookup) == 2 and all(r['same_ordered_top5'] for r in lookup)
    original = json.loads((ROOT / 'artifacts/controller/t3_v1/gpu_release_v1/execution_manifest.json').read_text())
    assert manifest == original
    for key in ('datasets', 'required_modes', 'source_commit', 'source_files', 'policy_sha256'):
        assert seal[key] == manifest[key], key
    engineering = ROOT / 'artifacts/controller/t3_v1/kaggle_engineering_v1/raw/engineering_receipt.json'
    assert sha(engineering) == seal['engineering_receipt_sha256']
    accept_engineering(json.loads(engineering.read_text()), seal, environment)
    refs = []
    for dataset, info in seal['datasets'].items():
        gold_path = ROOT / f'artifacts/controller/data_v1/offline/{dataset}/development/gold.jsonl'
        assert sha(gold_path) == info['offline_gold_sha256']
        refs.extend(json.loads(line) for line in gold_path.read_text().splitlines())
    # The registered evaluator validates every ID, arm, effective input hash,
    # prompt hash, attempted-call count and shared output budget before scoring.
    evaluation = evaluate(rows, refs, registration=seal)
    assert evaluation['registered_analysis']['acceptance']['complete']
    assert evaluation['registered_analysis']['acceptance']['execution_errors'] == 0
    gold = {(r['dataset'], str(r['sample_id'])): r for r in refs}
    indexed = {(r['dataset'], r['sample_id'], r['mode']): r for r in rows}
    lines = {(r['dataset'], r['sample_id'], r['mode']): n for n, r in enumerate(rows, 1)}

    def score(row, answer):
        return score_answer(answer, gold[(row['dataset'], row['sample_id'])], row['dataset'])

    def trace_ref(row):
        return {'archive': archive.name, 'member': 'predictions.jsonl',
                'line': lines[(row['dataset'], row['sample_id'], row['mode'])],
                'dataset': row['dataset'], 'sample_id': row['sample_id'], 'mode': row['mode']}

    opportunities = []
    for row in rows:
        if row['mode'] != 'adaptive':
            continue
        initial = score(row, row['initial_answer'])['correct']
        final = score(row, row['final_answer'])['correct']
        successful = [mode for mode in SINGLE if score(row,
                      indexed[(row['dataset'], row['sample_id'], mode)]['final_answer'])['correct']]
        harm = bool(initial and not final)
        missed = bool(not initial and not final and successful)
        if harm or missed:
            opportunities.append({'dataset': row['dataset'], 'sample_id': row['sample_id'],
                'query': row['input']['query'], 'initial_answer': row['initial_answer'],
                'adaptive_answer': row['final_answer'], 'chosen_action': record_action(row),
                'successful_fixed_actions': successful, 'avoidable_harm': harm,
                'missed_repair_opportunity': missed, 'routing': row['routing'],
                'metric_scope': 'NQ frozen EM / FEVER label; not human diagnosis or pure causal route loss',
                'trace_ref': trace_ref(row), 'human_review': {'status': 'pending', 'notes': ''}})
    opportunities.sort(key=lambda r: (r['dataset'], r['sample_id']))
    route_ids = {(r['dataset'], r['sample_id']) for r in opportunities}
    pairs = defaultdict(list)
    for row in rows:
        if row['dataset'] == 'nq' and row['answer_changed']:
            pairs[(row['sample_id'], row['initial_answer'], row['final_answer'])].append(row)
    ordered = sorted(pairs, key=lambda k: (
        0 if any(r['mode'] == 'adaptive' for r in pairs[k]) else
        1 if ('nq', k[0]) in route_ids else 2, k))
    changed = []
    for n, key in enumerate(ordered, 1):
        group = pairs[key]; row = group[0]
        changed.append({'review_id': f'T4N{n:03d}', 'dataset': 'nq', 'sample_id': key[0],
            'query': row['input']['query'], 'initial_answer': key[1], 'candidate_answer': key[2],
            'source_modes': [r['mode'] for r in group], 'initial_evidence': row['input']['evidence'],
            'final_evidence_by_mode': {r['mode']: r['final_evidence'] for r in group},
            'frozen_reference': gold[('nq', key[0])],
            'automatic_initial_metrics': score(row, key[1]),
            'automatic_candidate_metrics': score(row, key[2]),
            'routing_priority_query': ('nq', key[0]) in route_ids,
            'trace_refs': [trace_ref(r) for r in group],
            'human_review': {'status': 'pending', 'initial_correct': None,
                             'candidate_correct': None, 'reviewer': '', 'notes': ''}})
    flagged = []
    for row in rows:
        if row['mode'] == 'adaptive' and row['unresolved_failure']['flagged']:
            flagged.append({'review_id': f'T4F{len(flagged) + 1:03d}',
                'dataset': row['dataset'], 'sample_id': row['sample_id'],
                'frozen_reference': gold[(row['dataset'], row['sample_id'])], 'trace_ref': trace_ref(row),
                'complete_runtime_record': row,
                'review_instruction': 'Judge unresolved final-answer/evidence failure; an operational warning alone is not answer-error ground truth. AC6 and complete-trace review are separate judgments.',
                'human_review': {'status': 'pending', 'flag_is_true_failure': None,
                    'final_answer_correct': None, 'trace_review_completed': False,
                    'reviewer_independent_of_estimator_development': None, 'reviewer': '', 'notes': ''}})
    diagnostics = {'datasets': {}}
    for dataset in ('nq', 'fever'):
        full = [r for r in rows if r['dataset'] == dataset and r['mode'] == 'full_enhancement']
        stage_correct = [0] * 4; gains = [0] * 3; harms = [0] * 3
        for row in full:
            outcomes = [score(row, row['initial_answer'])['correct']] + [
                score(row, s['final_answer'])['correct'] for s in row['module_trace']['stages']]
            for n, value in enumerate(outcomes): stage_correct[n] += int(value)
            for n in range(3):
                gains[n] += int(not outcomes[n] and outcomes[n + 1])
                harms[n] += int(outcomes[n] and not outcomes[n + 1])
        invalid = [r for r in rows if r['dataset'] == dataset and r['mode'] == 'always_repair'
                   and r['status'] == 'invalid_output']
        diagnostics['datasets'][dataset] = {'full_stage_order': ['C1', 'retrieve', 'rerank', 'E'],
            'full_stage_correct': stage_correct, 'incremental_stage_repairs': gains,
            'incremental_stage_regressions': harms, 'always_E_invalid_output': len(invalid),
            'always_E_validation_errors': dict(Counter(r['module_trace']['validation_error'] for r in invalid)),
            'missed_opportunities_by_action': dict(Counter(r['chosen_action'] for r in opportunities
                if r['dataset'] == dataset and r['missed_repair_opportunity'])),
            'probe_calls': sum(e['kind'] == 'llm' and e['stage'] == 'probe' and e['attempted']
                for r in rows if r['dataset'] == dataset and r['mode'] == 'adaptive' for e in r['calls'])}
    review = {'NQ_changed_pairs': len(changed), 'NQ_unique_queries': len({r['sample_id'] for r in changed}),
        'NQ_adaptive_changed_pairs': sum('adaptive' in r['source_modes'] for r in changed),
        'adaptive_flagged_traces': len(flagged),
        'adaptive_flagged_by_dataset': dict(Counter(r['dataset'] for r in flagged)),
        'routing_priority_cases': len(opportunities), 'all_human_judgments_pending': True,
        'AC6': 'At least 50 human-verified flagged queries and observed precision >=90%; not yet met by human evidence.',
        'trace_criterion': 'At least 50 complete traces, reviewer independence recorded; may overlap AC6 but judged separately.'}
    assert len(changed) == len(pairs) and all(r['human_review']['status'] == 'pending' for r in changed + flagged)
    output.mkdir(parents=True)
    write(output / 'evaluation.json', evaluation)
    write(output / 'reconstructed_summary.json', summarize(rows))
    write(output / 'diagnostics.json', diagnostics)
    jsonl(output / 'nq_changed_answer_review.jsonl', changed)
    jsonl(output / 'adaptive_flagged_trace_review.jsonl', flagged)
    jsonl(output / 'routing_priority_review.jsonl', opportunities)
    write(output / 'review_manifest.json', review)
    write(output / 'reproduction_receipt.json', {'archive_sha256': ARCHIVE_SHA,
        'records': len(rows), 'validated_member_hashes': len(checks), 'new_GPU_calls': 0,
        'predictions_modified': False, 'all_human_judgments_pending': True,
        'files': {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    reproduce(parser.parse_args().output_dir)
