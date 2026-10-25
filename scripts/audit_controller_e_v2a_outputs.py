"""Validate a preserved v2a export against its sealed inputs, without scoring gold.

An intact failed export is an audit success, not engineering acceptance. This
does not change expected cases, deployment policy, prompts or runtime outputs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_controller_e_v2a import FIXTURES, check_expected, classify, development_entries
from src.controller.execution import fingerprint

V1_NAME = 'controller_e_v2a_development_v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_outputs(archive, expected_raw):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute()
                or '..' in PurePosixPath(n).parts or n.endswith('/') for n in names):
            raise ValueError('Ambiguous or unsafe output member')
        files = {n: z.read(n) for n in names}
    checks = json.loads(files['output_checksums.json'])
    if set(checks) != set(files) - {'output_checksums.json'}:
        raise ValueError('Output checksum coverage mismatch')
    for rel, sha in checks.items():
        if digest(files[rel]) != sha:
            raise ValueError('Output checksum mismatch: ' + rel)
    if files['input_execution_manifest.json'] != expected_raw:
        raise ValueError('Outputs belong to different sealed inputs')
    return files, checks


def audit_r2(archive, expected_raw, expected, fixtures_raw, plan_raw):
    if digest(fixtures_raw) != expected['cases_sha256'] or digest(plan_raw) != expected['plan_sha256']:
        raise ValueError('Sealed cases or plan mismatch')
    fixtures = json.loads(fixtures_raw)
    long = fixtures['long']
    cases = {c['case_id']: (c, None) for c in fixtures['cases']}
    for base_id in long['base_case_ids']:
        cases[base_id + long['case_id_suffix']] = (cases[base_id][0], long['minimum_input_tokens'])
    plan_rows = [json.loads(l) for l in plan_raw.splitlines()]
    files, checks = read_outputs(archive, expected_raw)
    state = json.loads(files['manifest.json'])
    if state['source_commit'] != expected['source_commit'] or state['role'] != 'development' or state['quality_scored']:
        raise ValueError('Source, role or scoring scope mismatch')
    gate_checks = []
    for rel in sorted(n for n in files if n.startswith('engineering/')):
        c = json.loads(files[rel])
        cid = c['case_id']
        if cid not in cases or rel != 'engineering/' + cid + '.json' or c['expected'] != cases[cid][0]['expected']:
            raise ValueError('Unregistered gate or changed expectation')
        base, minimum = cases[cid]
        record = c['record']
        if record is None:
            failure = 'execution'
            if c['failure_class'] != 'execution':
                raise ValueError('Missing record must be an execution failure')
        else:
            if minimum is None and record['input'] != base['input']:
                raise ValueError('Fictional input changed')
            if minimum is not None and record['input']['evidence'][:-1] != base['input']['evidence'][:-1]:
                raise ValueError('Long check changed registered evidence')
            failure, _ = classify(record, base, minimum)
        if c['failure_class'] != failure or c['passed'] is not (failure is None):
            raise ValueError('Saved engineering pass disagrees with observed record')
        check = ((record or {}).get('module_trace') or {}).get('support_check') or {}
        gate_checks.append({'case_id': cid, 'kind': base['kind'], 'dataset': base['input']['dataset'],
                            'long': minimum is not None, 'passed': failure is None, 'failure_class': failure,
                            'contract_passed': check.get('contract_passed'), 'expected': c['expected'],
                            'observed': {k: check.get(k) for k in ('original_verification', 'model_label', 'final_action', 'unresolved')},
                            'observed_final_answer': (record or {}).get('final_answer'),
                            'diagnostic': c.get('diagnostic')})
    complete = {c['case_id'] for c in gate_checks} == set(cases)
    classes = [c['failure_class'] for c in gate_checks]
    outcome = None
    if complete:
        outcome = ('failed' if {'execution', 'structure'} & set(classes) else
                   'semantic_only' if 'semantic' in classes else 'passed')
    if state.get('engineering_outcome') is not None:
        receipt = json.loads(files['engineering_receipt.json'])
        if state['engineering_outcome'] != outcome or receipt['outcome'] != outcome:
            raise ValueError('Saved gate outcome disagrees with recomputed checks')
    entries = development_entries(plan_rows, outcome) if outcome else []
    allowed = {(e['input']['dataset'], e['input']['sample_id'], e['arm']): e for e in entries}
    records = [json.loads(l) for l in files.get('predictions.jsonl', b'').splitlines()]
    keys = set()
    for record in records:
        key = (record['dataset'], record['sample_id'], record['mode'])
        if key in keys or key not in allowed:
            raise ValueError('Development output outside the registered branch')
        keys.add(key)
        entering = allowed[key]
        if record['input'] != entering['input'] or record['input_sha256'] != entering['input_sha256'] or fingerprint(record['input']) != entering['input_sha256']:
            raise ValueError('Development entering state changed')
        mode = ((record.get('module_trace') or {}).get('support_check') or {}).get('deployment_mode')
        if outcome == 'semantic_only' and (mode != 'verify_only' or record['final_answer'] != record['input']['initial_answer']):
            raise ValueError('Semantic-only branch replaced an answer')
    if len(records) != state['n_predictions']:
        raise ValueError('Prediction count disagrees with manifest')
    if state['state'] == 'completed' and (not outcome or outcome == 'failed' or keys != set(allowed)):
        raise ValueError('Incomplete run cannot be marked completed')
    return {'audit_passed': True, 'engineering_revision': expected['engineering_revision'],
            'output_zip_sha256': digest(Path(archive).read_bytes()),
            'checksums_verified': len(checks), 'source_commit': state['source_commit'], 'state': state['state'],
            'engineering_outcome': outcome, 'engineering_passed': outcome == 'passed',
            'gate_checks': gate_checks, 'gate_checks_passed': sum(c['passed'] for c in gate_checks),
            'failure_counts': {k: classes.count(k) for k in ('execution', 'structure', 'semantic')},
            'short_checks_completed': sum(not c['long'] for c in gate_checks),
            'long_checks_completed': sum(c['long'] for c in gate_checks),
            'gate_datasets': dict(Counter(c['dataset'] for c in gate_checks)),
            'development_branch_expected': len(entries), 'n_development_predictions': len(records),
            'error': state.get('error'), 'quality_scored': False, 'promotion': False,
            'scope': 'export integrity and registered r2 gate classes only; no benchmark or semantic gain claim'}


def audit(archive, release_dir):
    archive, release_dir = Path(archive), Path(release_dir)
    expected_raw = (release_dir / 'execution_manifest.json').read_bytes()
    expected = json.loads(expected_raw)
    name = json.loads((release_dir / 'bundle_receipt.json').read_text()).get('bundle_name', V1_NAME)
    if expected.get('engineering_revision', '').startswith('evidence_id_v2a_v3'):
        with tarfile.open(release_dir / (name + '.tar.gz')) as t:
            prefix = name + '/'
            if t.extractfile(prefix + 'execution_manifest.json').read() != expected_raw:
                raise ValueError('External and bundled execution manifests differ')
            fixtures_raw = t.extractfile(prefix + FIXTURES).read()
            plan_raw = t.extractfile(prefix + 'inputs/development_plan.jsonl').read()
        return audit_r2(archive, expected_raw, expected, fixtures_raw, plan_raw)
    with tarfile.open(release_dir / (name + '.tar.gz')) as t:
        prefix = name + '/'
        if t.extractfile(prefix + 'execution_manifest.json').read() != expected_raw:
            raise ValueError('External and bundled execution manifests differ')
        cases_raw = t.extractfile(prefix + 'engineering/cases.json').read()
        plan_raw = t.extractfile(prefix + 'inputs/development_plan.jsonl').read()
    if digest(cases_raw) != expected['cases_sha256'] or digest(plan_raw) != expected['plan_sha256']:
        raise ValueError('Sealed cases or plan mismatch')
    cases = {c['case_id']: c for c in json.loads(cases_raw)}
    for ds in ('nq', 'fever'):
        base = next(c for c in cases.values() if c['input']['dataset'] == ds
                    and c['expected']['final_action'] == 'replace')
        cases[base['case_id'] + '-long'] = {**base, 'case_id': base['case_id'] + '-long'}
    plan = {(r['input']['dataset'], r['input']['sample_id'], r['arm']): r
            for r in map(json.loads, plan_raw.splitlines())}
    files, checks = read_outputs(archive, expected_raw)
    state = json.loads(files['manifest.json'])
    if state['source_commit'] != expected['source_commit'] or state['role'] != 'development' or state['quality_scored']:
        raise ValueError('Source, role or scoring scope mismatch')
    gate_checks, gate_ids = [], set()
    for rel in sorted(n for n in files if n.startswith('engineering/')):
        c = json.loads(files[rel])
        cid = c['case_id']
        if cid not in cases or cid in gate_ids or rel != 'engineering/' + cid + '.json' or c['expected'] != cases[cid]['expected']:
            raise ValueError('Unregistered gate or changed expectation')
        gate_ids.add(cid)
        record = c['record']
        if not cid.endswith('-long') and record['input'] != cases[cid]['input']:
            raise ValueError('Fictional input changed')
        passed = bool(check_expected(record, cases[cid]))
        if cid.endswith('-long'):
            events = [e for e in record['calls'] if e['kind'] == 'llm' and e['attempted']]
            passed = bool(passed and len(events) == 1 and events[0]['status'] == 'completed'
                          and events[0]['output']['input_tokens'] >= 14000
                          and events[0]['structured_decoder'].get('decoder') == 'xgrammar'
                          and events[0]['structured_decoder'].get('prefill_chunk_size') == 256)
        if c['passed'] is not passed:
            raise ValueError('Saved engineering pass disagrees with observed record')
        check = record['module_trace']['support_check']
        gate_checks.append({'case_id': cid, 'dataset': record['dataset'], 'passed': passed,
                            'contract_passed': check['contract_passed'],
                            'expected': c['expected'],
                            'observed': {k: check[k] for k in ('original_verification', 'final_action', 'unresolved')},
                            'observed_final_answer': record['final_answer']})
    records = [json.loads(l) for l in files.get('predictions.jsonl', b'').splitlines()]
    keys = set()
    for record in records:
        key = (record['dataset'], record['sample_id'], record['mode'])
        if key in keys or key not in plan:
            raise ValueError('Duplicate or unregistered development output')
        keys.add(key)
        entering = plan[key]
        if record['input'] != entering['input'] or record['input_sha256'] != entering['input_sha256'] or fingerprint(record['input']) != entering['input_sha256']:
            raise ValueError('Development entering state changed')
    if len(records) != state['n_predictions']:
        raise ValueError('Prediction count disagrees with manifest')
    if state['engineering_passed']:
        receipt = json.loads(files['engineering_receipt.json'])
        if gate_ids != set(cases) or not all(c['passed'] for c in gate_checks) or not receipt['passed']:
            raise ValueError('Engineering acceptance without all eight passing checks')
    elif records:
        raise ValueError('Development started before engineering acceptance')
    if state['state'] == 'completed' and (len(keys) != len(plan) or not state['engineering_passed']):
        raise ValueError('Incomplete run cannot be marked completed')
    return {'audit_passed': True, 'output_zip_sha256': digest(archive.read_bytes()),
            'checksums_verified': len(checks), 'source_commit': state['source_commit'],
            'state': state['state'], 'engineering_passed': state['engineering_passed'],
            'gate_checks': gate_checks, 'gate_checks_passed': sum(c['passed'] for c in gate_checks),
            'short_checks_completed': sum(not c['case_id'].endswith('-long') for c in gate_checks),
            'long_checks_completed': sum(c['case_id'].endswith('-long') for c in gate_checks),
            'gate_datasets': dict(Counter(c['dataset'] for c in gate_checks)),
            'n_development_predictions': len(records), 'error': state.get('error'),
            'quality_scored': False, 'promotion': False,
            'scope': 'export integrity and registered gates only; no benchmark or semantic gain claim'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--release-dir', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.archive, args.release_dir), ensure_ascii=False, indent=2))
