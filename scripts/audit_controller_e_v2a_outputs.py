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
from scripts.run_controller_e_v2a import check_expected
from src.controller.execution import fingerprint


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(archive, release_dir):
    archive, release_dir = Path(archive), Path(release_dir)
    expected_raw = (release_dir / 'execution_manifest.json').read_bytes()
    expected = json.loads(expected_raw)
    with tarfile.open(release_dir / 'controller_e_v2a_development_v1.tar.gz') as t:
        prefix = 'controller_e_v2a_development_v1/'
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
