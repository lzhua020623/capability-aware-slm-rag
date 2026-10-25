"""One sealed E-only development run, after a mandatory real CUDA gate.

Full inputs are the cached pre-E states, not original C1. Real gold, selection
and final inputs are absent. Never score answers or pick deployment modes here.
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
            evidence = [EvidenceItem(1, 'fictional-velora',
                'The fictional Velora Archive was designed by Nerin Sol. It is built from green basalt.', .9)]
            evidence += [EvidenceItem(i, f'fictional-other-{i}',
                f'Fictional archive {i} has a catalogue of unrelated maps and books.', .9 - i / 100)
                for i in range(2, 6)]
            query = ('When did the fictional Velora Archive open?' if missing else 'Who designed the fictional Velora Archive?') if ds == 'nq' else (
                'The fictional Velora Archive opened in 2017.' if missing else
                'The fictional Velora Archive was designed by Nerin Sol.')
            initial = ('2017' if missing else 'Nerin Sol' if kind == 'supported_keep' else 'Erven Dale') if ds == 'nq' else (
                'REFUTES' if kind == 'unsupported_replace' else 'SUPPORTS')
            item = DiagnosisInput(f'v2a-fiction-{ds}-{kind}', ds, query, evidence, initial)
            cases.append({'case_id': item.sample_id, 'synthetic': True, 'input': asdict(item),
                'expected': {'original_verification': 'supported' if kind == 'supported_keep' else
                             'unsupported' if kind == 'unsupported_replace' else 'insufficient',
                             'final_action': 'replace' if kind == 'unsupported_replace' else 'keep',
                             'unresolved': missing,
                             'final_answer': ('Nerin Sol' if ds == 'nq' else 'SUPPORTS') if kind == 'unsupported_replace' else initial}})
    return cases


def check_expected(record, case):
    check = record['module_trace']['support_check']
    expected = case['expected']
    return (check['contract_passed'] and not record['cost']['failed_calls'] and not record['cost']['budget_blocks']
            and all(check[k] == expected[k] for k in ('original_verification', 'final_action', 'unresolved'))
            and normalize_answer(record['final_answer']) == normalize_answer(expected['final_answer']))


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


def long_case(tokenizer, case, minimum=14000):
    from src.compensation.evidence_v2a import prompt
    from src.compensation.evidence_id_repair import evidence_units
    from src.generation.qwen import apply_qwen_chat_template
    original = runtime_input(case['input'])
    padding = 512
    while padding <= 16384:
        # One long unit prevents the ID enum from itself dominating the context.
        e = original.evidence[-1]
        tail = e.text + ' Background fictional catalogue' * padding + '.'
        item = replace(original, evidence=[*original.evidence[:-1], replace(e, text=tail)])
        n = len(tokenizer.encode(apply_qwen_chat_template(tokenizer, prompt(item, evidence_units(item)))))
        if n >= minimum:
            if n + 384 > 32768: raise ValueError('Synthetic gate exceeds fixed context')
            return {**case, 'case_id': case['case_id'] + '-long', 'input': asdict(item)}, n
        padding = int(padding * 1.3)
    raise RuntimeError('Unable to construct the registered long gate')


def gate(pipeline, cases, output):
    checks = []
    for case in cases:
        record = execute_one(pipeline, {'input': case['input'], 'arm': 'engineering'})
        check = {'case_id': case['case_id'], 'passed': bool(check_expected(record, case)),
                 'synthetic': True, 'expected': case['expected'], 'record': record}
        write(output / (case['case_id'] + '.json'), check); checks.append(check)
        if not check['passed']: raise RuntimeError('Fictional behavioral gate failed: ' + case['case_id'])
    for ds in ('nq', 'fever'):
        base = next(c for c in cases if c['input']['dataset'] == ds and c['expected']['final_action'] == 'replace')
        case, n = long_case(pipeline.raw_generator.tokenizer, base)
        record = execute_one(pipeline, {'input': case['input'], 'arm': 'engineering_long'})
        events = [e for e in record['calls'] if e['kind'] == 'llm' and e['attempted']]
        passed = (check_expected(record, case) and len(events) == 1
                  and events[0]['status'] == 'completed'
                  and events[0]['output']['input_tokens'] >= 14000
                  and events[0]['structured_decoder'].get('decoder') == 'xgrammar'
                  and events[0]['structured_decoder'].get('prefill_chunk_size') == 256)
        check = {'case_id': case['case_id'], 'passed': bool(passed), 'expected_input_tokens': n,
                 'synthetic': True, 'expected': case['expected'], 'record': record}
        write(output / (case['case_id'] + '.json'), check); checks.append(check)
        if not passed: raise RuntimeError('XGrammar/prefill256 long gate failed: ' + ds)
    return {'passed': True, 'real_CUDA': True, 'checks': checks,
            'scope': 'six fresh fictional behavioral cases plus two combined long-input paths; not benchmark gain'}


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
    if sha(ROOT / 'engineering/cases.json') != manifest['cases_sha256']:
        raise ValueError('Fictional cases changed')
    plan = [json.loads(l) for l in (ROOT / 'inputs/development_plan.jsonl').read_text().splitlines()]
    cases = json.loads((ROOT / 'engineering/cases.json').read_text())
    validate_plan(plan, manifest)
    if cases != fictional_cases(): raise ValueError('Predeclared expected fixtures changed')
    if args.preflight:
        print(json.dumps({'passed': True, 'n': 800, 'contains_real_gold': False})); return 0
    if args.output_dir.exists(): raise FileExistsError('Preserve previous outputs')
    args.output_dir.mkdir(parents=True)
    state = {'state': 'starting', 'engineering_passed': False, 'n_predictions': 0,
             'quality_scored': False, 'role': 'development', 'source_commit': manifest['source_commit']}
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
        receipt = gate(pipeline, cases, gates)
        receipt['manifest_sha256'] = sha(ROOT / 'execution_manifest.json')
        write(args.output_dir / 'engineering_receipt.json', receipt)
        state.update(state='running_development', engineering_passed=True)
        write(args.output_dir / 'manifest.json', state)
        with (args.output_dir / 'predictions.jsonl').open('x') as handle:
            for entry in plan:
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
