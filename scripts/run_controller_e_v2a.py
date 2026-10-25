"""One sealed E-only development run, after a mandatory real CUDA gate.

Full inputs are the cached pre-E states, not original C1. Real gold, selection
and final inputs are absent. Never score answers or pick deployment modes here.
r2 runs every sealed check, classifies failures and follows the registered branch.
"""
import argparse
from dataclasses import asdict, replace
import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.run_controller_integration import runtime_input
from src.controller.execution import CallLog, ExecutionConfig, IntegratedController, arm_config, decision_for, fingerprint
from src.controller.schema import ControllerAction
from src.diagnosis import DiagnosisInput, EvidenceItem
from src.evaluation.qa import normalize_answer

FIXTURES = 'configs/controller/e_v2a_r2_fixtures.json'
SEAL = 'configs/controller/e_v2a_r2_fixture_seal.json'
REGISTRATION = 'configs/controller/e_v2a_engineering_r2.json'
# r1 long construction; r2 reads the same values from its sealed fixture file.
LONG = {'padded_rank': 5, 'padding_unit': ' Background fictional catalogue', 'terminator': '.',
        'start_repetitions': 512, 'growth': 1.3, 'max_repetitions': 16384,
        'minimum_input_tokens': 14000, 'maximum_input_tokens': 32768 - 384, 'case_id_suffix': '-long'}
BRANCHES = {'passed': ('always_repair', 'full_E'), 'semantic_only': ('always_repair',), 'failed': ()}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def fictional_cases():
    cases = []
    for ds in ('nq', 'fever'):
        for kind in ('supported_keep', 'unsupported_replace', 'insufficient_unresolved'):
            missing = kind == 'insufficient_unresolved'
            evidence = [EvidenceItem(1, 'fictional-selen',
                'The fictional Selen Observatory was designed by Veyra Lorn. It is built from gray marble.', .9)]
            evidence += [EvidenceItem(i, f'fictional-other-{i}',
                f'Fictional archive {i} has a catalogue of unrelated maps and books.', .9 - i / 100)
                for i in range(2, 6)]
            query = ('When did the fictional Selen Observatory open?' if missing else 'Who designed the fictional Selen Observatory?') if ds == 'nq' else (
                'The fictional Selen Observatory opened in 2019.' if missing else
                'The fictional Selen Observatory was designed by Veyra Lorn.')
            initial = ('2019' if missing else 'Veyra Lorn' if kind == 'supported_keep' else 'Nalo Kest') if ds == 'nq' else (
                'REFUTES' if kind == 'unsupported_replace' else 'SUPPORTS')
            item = DiagnosisInput(f'v2a-r1-fiction-{ds}-{kind}', ds, query, evidence, initial)
            cases.append({'case_id': item.sample_id, 'synthetic': True, 'input': asdict(item),
                'expected': {'original_verification': 'supported' if kind == 'supported_keep' else
                             'unsupported' if kind == 'unsupported_replace' else 'insufficient',
                             'final_action': 'replace' if kind == 'unsupported_replace' else 'keep',
                             'unresolved': missing,
                             'final_answer': ('Veyra Lorn' if ds == 'nq' else 'SUPPORTS') if kind == 'unsupported_replace' else initial}})
    return cases


def check_expected(record, case):
    check = record['module_trace']['support_check']
    expected = case['expected']
    accepted = {normalize_answer(a) for a in expected.get('acceptable_final_answers', [expected['final_answer']])}
    return (check['contract_passed'] and not record['cost']['failed_calls'] and not record['cost']['budget_blocks']
            and all(check[k] == expected[k] for k in ('original_verification', 'final_action', 'unresolved'))
            and normalize_answer(record['final_answer']) in accepted)


def r2_fixtures(root=ROOT):
    root = Path(root)
    raw = (root / FIXTURES).read_bytes()
    seal = json.loads((root / SEAL).read_text())
    gate = json.loads((root / REGISTRATION).read_text())['gate']
    digest = hashlib.sha256(raw).hexdigest()
    if digest != seal['fixtures_sha256'] or digest != gate['fixtures_sha256'] or sha(root / SEAL) != gate['fixture_seal_sha256']:
        raise ValueError('Sealed r2 fixtures changed')
    return json.loads(raw)


def classify(record, case, long_minimum=None):
    """(None|execution|structure|semantic, reasons); the first failing class wins."""
    events = [e for e in record['calls'] if e['kind'] == 'llm' and e['attempted']]
    decoder = events[0].get('structured_decoder') or {} if events else {}
    execution = []
    if record.get('status') in ('execution_failed', 'budget_exceeded'):
        execution.append('status_' + record['status'])
    if record['cost']['failed_calls'] or record['cost']['budget_blocks']:
        execution.append('failed_or_blocked_call')
    if len(events) != 1 or events[0]['status'] != 'completed':
        execution.append('llm_call_not_completed')
    if decoder.get('hit_token_budget'):
        execution.append('token_budget_hit')
    if long_minimum is not None:
        if not events or ((events[0].get('output') or {}).get('input_tokens') or 0) < long_minimum:
            execution.append('long_input_below_minimum')
        if decoder.get('decoder') != 'xgrammar':
            execution.append('decoder_not_xgrammar')
        if decoder.get('prefill_chunk_size') != 256:
            execution.append('prefill_not_256')
    if execution:
        return 'execution', execution
    check = record['module_trace'].get('support_check') or {}
    if not check.get('contract_passed'):
        return 'structure', ['contract_failed:%s:%s' % (check.get('original_verification'), check.get('candidate_check'))]
    expected = case['expected']
    semantic = ['%s:%s!=%s' % (k, check.get(k), expected[k]) for k in ('original_verification', 'final_action', 'unresolved')
                if check.get(k) != expected[k]]
    if normalize_answer(record['final_answer']) not in {normalize_answer(a) for a in expected['acceptable_final_answers']}:
        semantic.append('final_answer')
    return ('semantic', semantic) if semantic else (None, [])


def key_evidence_cited(record, case):
    """Diagnostic only: did the selected verdict/replacement IDs include a registered key sentence?"""
    if not case['key_evidence_ids']:
        return None
    trace = record.get('module_trace') or {}
    cited = set((trace.get('support_check') or {}).get('verdict_evidence_ids') or [])
    try:
        cited |= set(json.loads(trace.get('raw_output') or '{}').get('replacement_evidence_ids') or [])
    except (ValueError, AttributeError, TypeError):
        pass
    return bool(cited & set(case['key_evidence_ids']))


def validate_plan(plan, manifest):
    if manifest['role'] != 'development' or manifest['contains_real_gold'] is not False:
        raise ValueError('Only gold-free development is allowed')
    expected = {(ds, sid, arm) for ds, ids in manifest['sample_ids'].items() for sid in ids
                for arm in ('always_repair', 'full_E')}
    seen = set()
    for entry in plan:
        item = runtime_input(entry['input'])
        key = (item.dataset.value, item.sample_id, entry['arm'])
        if key not in expected or key in seen or fingerprint(asdict(item)) != entry['input_sha256']:
            raise ValueError('Duplicate, unregistered or changed entering state')
        seen.add(key)
    if seen != expected or len(plan) != 800 or any(len(v) != 200 for v in manifest['sample_ids'].values()):
        raise ValueError('Require complete two-dataset, two-arm development 800')


def execute_one(pipeline, entry):
    item = runtime_input(entry['input'])
    calls = CallLog(arm_config(pipeline.config, 'always_repair'))
    started = perf_counter()
    stage = pipeline.execute(item, decision_for(ControllerAction.GENERATION_REPAIR, item.sample_id), calls=calls)
    return {'dataset': item.dataset.value, 'sample_id': item.sample_id, 'mode': entry['arm'],
            'input': asdict(item), 'input_sha256': fingerprint(asdict(item)),
            'source_T4': entry.get('source_T4'), **stage,
            'calls': calls.events, 'cost': calls.summary(), 'total_seconds': perf_counter() - started,
            'scope': 'new E invocation only; Full upstream state is historical cached output'}


def long_case(tokenizer, case, minimum=14000, spec=LONG):
    from src.compensation.evidence_v2a import prompt
    from src.compensation.evidence_id_repair import evidence_units
    from src.generation.qwen import apply_qwen_chat_template
    original = runtime_input(case['input'])
    if original.evidence[-1].rank != spec['padded_rank']: raise ValueError('Unregistered padded passage')
    padding = spec['start_repetitions']
    while padding <= spec['max_repetitions']:
        # One long unit prevents the ID enum from itself dominating the context.
        e = original.evidence[-1]
        tail = e.text + spec['padding_unit'] * padding + spec['terminator']
        item = replace(original, evidence=[*original.evidence[:-1], replace(e, text=tail)])
        n = len(tokenizer.encode(apply_qwen_chat_template(tokenizer, prompt(item, evidence_units(item)))))
        if n >= minimum:
            if n > spec['maximum_input_tokens']: raise ValueError('Synthetic gate exceeds fixed context')
            return {**case, 'case_id': case['case_id'] + spec['case_id_suffix'], 'input': asdict(item)}, n
        padding = int(padding * spec['growth'])
    raise RuntimeError('Unable to construct the registered long gate')


def gate(pipeline, fixtures, output):
    """Run all 18 + 2 sealed checks; never stop at the first failure."""
    checks = []
    long = fixtures['long']
    by_id = {c['case_id']: c for c in fixtures['cases']}
    planned = [(c, None) for c in fixtures['cases']] + [(by_id[i], long) for i in long['base_case_ids']]
    for base, spec in planned:
        case, n, record = base, None, None
        try:
            if spec is not None:
                case, n = long_case(pipeline.raw_generator.tokenizer, base, spec['minimum_input_tokens'], spec)
            record = execute_one(pipeline, {'input': case['input'], 'arm': 'engineering_long' if spec else 'engineering'})
            failure, reasons = classify(record, case, spec['minimum_input_tokens'] if spec else None)
        except Exception as exc:
            failure, reasons = 'execution', ['exception:%s:%s' % (type(exc).__name__, exc)]
        support = ((record or {}).get('module_trace') or {}).get('support_check') or {}
        check = {'case_id': base['case_id'] + long['case_id_suffix'] if spec else base['case_id'],
                 'kind': base['kind'], 'dataset': base['input']['dataset'], 'long': spec is not None,
                 'passed': failure is None, 'failure_class': failure, 'reasons': reasons,
                 'expected_input_tokens': n, 'synthetic': True, 'expected': case['expected'],
                 'diagnostic': {'key_evidence_cited': key_evidence_cited(record, case) if record else None,
                                'model_label': support.get('model_label'), 'type_warnings': support.get('type_warnings')},
                 'record': record}
        write(output / (check['case_id'] + '.json'), check); checks.append(check)
        print(json.dumps({'gate_check': check['case_id'], 'passed': check['passed'], 'failure_class': failure}), flush=True)
    classes = [c['failure_class'] for c in checks]
    outcome = ('failed' if {'execution', 'structure'} & set(classes) else
               'semantic_only' if 'semantic' in classes else 'passed')
    return {'outcome': outcome, 'passed': outcome == 'passed', 'real_CUDA': True, 'n_checks': len(checks),
            'failure_counts': {k: classes.count(k) for k in ('execution', 'structure', 'semantic')},
            'checks': [{k: v for k, v in c.items() if k != 'record'} for c in checks],
            'scope': '18 sealed fictional behavioral checks plus two long XGrammar/prefill256 paths; not benchmark gain'}


def development_entries(plan, outcome):
    return [e for e in plan if e['arm'] in BRANCHES[outcome]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args(argv)
    manifest = json.loads((ROOT / 'execution_manifest.json').read_text())
    for rel, digest in manifest['source_files'].items():
        if not (ROOT / rel).resolve().is_relative_to(ROOT.resolve()) or sha(ROOT / rel) != digest:
            raise ValueError('Frozen source mismatch: ' + rel)
    if sha(ROOT / 'inputs/development_plan.jsonl') != manifest['plan_sha256']:
        raise ValueError('Input plan mismatch')
    fixtures = r2_fixtures()
    if sha(ROOT / FIXTURES) != manifest['cases_sha256']:
        raise ValueError('Fictional cases changed')
    plan = [json.loads(l) for l in (ROOT / 'inputs/development_plan.jsonl').read_text().splitlines()]
    validate_plan(plan, manifest)
    if args.preflight:
        print(json.dumps({'passed': True, 'n': 800, 'gate_checks': len(fixtures['cases']) + 2,
                          'contains_real_gold': False})); return 0
    if args.output_dir.exists(): raise FileExistsError('Preserve previous outputs')
    args.output_dir.mkdir(parents=True)
    state = {'state': 'starting', 'engineering_outcome': None, 'engineering_passed': False,
             'development_branch': None, 'expected_predictions': None, 'n_predictions': 0,
             'quality_scored': False, 'role': 'development', 'source_commit': manifest['source_commit'],
             'engineering_revision': manifest['engineering_revision']}
    write(args.output_dir / 'manifest.json', state)
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('Real CUDA required; CPU cannot certify this gate')
        from src.compensation.qwen_repair import QwenRepairGenerator
        from src.compensation.qwen_v2a import QwenV2AGenerator
        from src.controller.runtime import configure_generation_runtime
        raw = QwenRepairGenerator(json_output=False)
        configure_generation_runtime(raw)
        structured = QwenV2AGenerator.from_shared(raw)
        config = ExecutionConfig(nq_repair_strategy='evidence_id_v2a', fever_repair_strategy='evidence_id_v2a')
        pipeline = IntegratedController(None, raw, {'nq': structured, 'fever': structured}, None, config=config)
        write(args.output_dir / 'environment.json', structured.runtime_environment)
        write(args.output_dir / 'execution_seal.json', {**manifest,
            'sealed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'real_benchmark_started': False, 'deployment_mode': 'allow_repair diagnostic only'})
        gates = args.output_dir / 'engineering'; gates.mkdir()
        receipt = gate(pipeline, fixtures, gates)
        receipt['manifest_sha256'] = sha(ROOT / 'execution_manifest.json')
        write(args.output_dir / 'engineering_receipt.json', receipt)
        entries = development_entries(plan, receipt['outcome'])
        state.update(engineering_outcome=receipt['outcome'], engineering_passed=receipt['passed'],
                     development_branch=receipt['outcome'], expected_predictions=len(entries))
        if receipt['outcome'] == 'failed':
            state['state'] = 'stopped_gate_failed'
            print(json.dumps({'engineering_outcome': 'failed', 'failure_counts': receipt['failure_counts'],
                              'E_mode': 'unavailable', 'development_predictions': 0}), flush=True)
            return 3
        if receipt['outcome'] == 'semantic_only':
            # Registered experimental verify-only branch: Always Repair only, no replacement.
            pipeline = IntegratedController(None, raw, {'nq': structured, 'fever': structured}, None,
                config=replace(config, nq_v2a_mode='verify_only', fever_v2a_mode='verify_only'))
        state['state'] = 'running_development'
        write(args.output_dir / 'manifest.json', state)
        with (args.output_dir / 'predictions.jsonl').open('x') as handle:
            for entry in entries:
                r = execute_one(pipeline, entry)
                handle.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'); handle.flush()
                state['n_predictions'] += 1
                if state['n_predictions'] % 20 == 0:
                    write(args.output_dir / 'manifest.json', state)
                    print(json.dumps({'n_predictions': state['n_predictions'], 'mode': entry['arm'],
                                      'dataset': r['dataset'], 'status': r['status']}), flush=True)
        state['state'] = 'completed'
    except Exception as exc:
        state.update(state='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        write(args.output_dir / 'manifest.json', state)
    return 0


if __name__ == '__main__': raise SystemExit(main())
