"""Freeze a gold-free, self-contained E development notebook from T4 states."""
import argparse
import base64
from dataclasses import asdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.run_controller_e_v2a import fictional_cases, sha, validate_plan
from scripts.run_controller_integration import runtime_input
from src.controller.execution import fingerprint

ARCHIVE = 'artifacts/controller/t4_v1/controller_t4_development_v1_outputs.zip'
POLICY = 'configs/controller/e_v2a_amendment_v1.json'
NAME = 'controller_e_v2a_development_v1'


def frozen_plan(root=ROOT):
    root = Path(root)
    policy = json.loads((root / POLICY).read_text())
    archive_sha = sha(root / ARCHIVE)
    if archive_sha != policy['historical_files_unchanged'][ARCHIVE]:
        raise ValueError('T4 archive fingerprint changed')
    with zipfile.ZipFile(root / ARCHIVE) as z:
        with z.open('predictions.jsonl') as f:
            rows = [json.loads(l) for l in f]
        original = json.loads(z.read('input_execution_manifest.json'))
    ids = {ds: original['datasets'][ds]['sample_ids'] for ds in ('nq', 'fever')}
    indexed = {(r['dataset'], r['sample_id'], r['mode']): r for r in rows}
    if len(indexed) != 2800 or len(rows) != 2800: raise ValueError('Incomplete original T4')
    entries = []
    for arm in ('always_repair', 'full_E'):
        for ds in ('nq', 'fever'):
            for sid in ids[ds]:
                r = indexed[(ds, sid, 'always_repair' if arm == 'always_repair' else 'full_enhancement')]
                entering = dict(r['input'])
                if arm == 'full_E':
                    stages = r['module_trace']['stages']
                    if [s['action'] for s in stages] != ['retrieval_enhancement', 'evidence_refinement', 'generation_repair']:
                        raise ValueError('Unexpected Full order')
                    entering.update(initial_answer=stages[1]['final_answer'], evidence=stages[1]['final_evidence'])
                    expected = stages[2]['input_sha256']
                else:
                    expected = r['input_sha256']
                entering = asdict(runtime_input(entering))
                if fingerprint(entering) != expected: raise ValueError('Changed actual pre-E state')
                entries.append({'input': entering, 'input_sha256': expected, 'arm': arm,
                    'source_T4': {'archive_sha256': archive_sha, 'mode': r['mode'],
                                  'original_input_sha256': r['input_sha256'], 'sample_id': sid}})
    validate_plan(entries, {'role': 'development', 'contains_real_gold': False, 'sample_ids': ids})
    routed = {ds: [r['sample_id'] for r in rows if r['dataset'] == ds and r['mode'] == 'adaptive'
                   and r['routing']['decision']['action'] == 'generation_repair'] for ds in ('nq', 'fever')}
    if {ds: len(v) for ds, v in routed.items()} != policy['deployment_mode']['reference_n']:
        raise ValueError('Routed reference subset changed')
    return entries, ids, routed


def notebook(raw, digest):
    code = f'''from pathlib import Path
import base64,hashlib,io,json,os,subprocess,sys,tarfile,zipfile
raw=base64.b64decode({base64.b64encode(raw).decode()!r},validate=True)
assert hashlib.sha256(raw).hexdigest()=={digest!r}
WORK=Path('/kaggle/working')
BASE=WORK/('e_v2a_'+{digest[:12]!r}); BASE.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
    for m in t.getmembers():
        assert m.isfile() and (BASE/m.name).resolve().is_relative_to(BASE.resolve())
    t.extractall(BASE,filter='data')
ROOT=BASE/'{NAME}'
for rel,h in json.loads((ROOT/'bundle_checksums.json').read_text()).items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
OUTPUT=WORK/'controller_e_v2a_development_v1_outputs'
LOG=WORK/'controller_e_v2a_run.log'
os.environ['OMP_NUM_THREADS']='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.chdir(ROOT)
def run(command):
    with LOG.open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        if p.wait():raise RuntimeError('Frozen command failed: '+repr(command))
try:
    run([sys.executable,'-m','pip','install','-r','requirements-controller-e-v2a.txt'])
    run([sys.executable,'-m','unittest','tests.test_evidence_v2a','tests.test_controller_v2a_policy','-q'])
    run([sys.executable,'scripts/run_controller_e_v2a.py','--preflight','--output-dir',str(OUTPUT)])
    run([sys.executable,'-u','scripts/run_controller_e_v2a.py','--output-dir',str(OUTPUT)])
finally:
    OUTPUT.mkdir(exist_ok=True)
    if LOG.exists():(OUTPUT/'run.log').write_bytes(LOG.read_bytes())
    (OUTPUT/'input_execution_manifest.json').write_bytes((ROOT/'execution_manifest.json').read_bytes())
    checks={{str(p.relative_to(OUTPUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.rglob('*') if p.is_file()}}
    (OUTPUT/'output_checksums.json').write_text(json.dumps(checks,indent=2))
    archive=WORK/'controller_e_v2a_development_v1_outputs.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(OUTPUT)))
    print('Preserved output:',archive,'SHA256:',hashlib.sha256(archive.read_bytes()).hexdigest())
'''
    compile(code, 'E_v2a_notebook', 'exec')
    return {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True,
                                'isInternetEnabled': True, 'language': 'python', 'sourceType': 'notebook'}},
        'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Controller E v2a development frozen 20261025\n',
            'Keep private. Same Qwen/NF4, one call/384 tokens. Six fresh fictional behavior checks and two '
            '14k+ token XGrammar/prefill256 checks MUST pass before 800 development E invocations. '
            'Full uses cached actual pre-E states. No index download, real gold, selection or final data; '
            'no runtime quality scoring, tuning, retries or promotion. Partial outputs always exported.']},
            {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)}]}


def prepare(output, root=ROOT):
    root, output = Path(root), Path(output)
    if output.exists(): raise FileExistsError('Preserve existing bundle')
    paths = ['src', 'scripts', 'tests', 'configs', 'requirements-controller-e-v2a.txt']
    subprocess.run(['git', 'diff', 'HEAD', '--exit-code', '--', *paths], cwd=root, check=True, stdout=subprocess.PIPE)
    if subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', *paths], cwd=root, text=True).strip():
        raise ValueError('Commit implementation before freezing GPU code')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    plan, ids, routed = frozen_plan(root)
    sources = [*sorted((root / 'src').rglob('*.py'))]
    sources += [root / n for n in ('scripts/run_controller_e_v2a.py', 'scripts/run_controller_integration.py',
        'configs/base.yaml', POLICY, 'requirements-controller-e-v2a.txt', 'requirements-controller-t3.txt',
        'requirements-answer-repair-v8.txt', 'models/diagnosis/diagnosis_v2_expanded_hierarchical.json',
        'tests/__init__.py', 'tests/test_evidence_v2a.py', 'tests/test_controller_v2a_policy.py', 'tests/test_controller_integration.py')]
    payload = {str(p.relative_to(root)): p.read_bytes() for p in sources}
    plan_raw = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in plan).encode()
    cases_raw = json.dumps(fictional_cases(), ensure_ascii=False, indent=2).encode()
    manifest = {'schema_version': 'controller_e_v2a_execution_v1', 'role': 'development',
        'source_commit': commit, 'source_files': {k: hashlib.sha256(v).hexdigest() for k, v in payload.items()},
        'sample_ids': ids, 'fixed_T4_E_routed_ids': routed, 'n_predictions': 800,
        'plan_sha256': hashlib.sha256(plan_raw).hexdigest(), 'cases_sha256': hashlib.sha256(cases_raw).hexdigest(),
        'source_T4_archive_sha256': sha(root / ARCHIVE), 'contains_real_gold': False,
        'contains_selection_or_final_inputs': False, 'GPU_engineering_passed': False}
    payload.update({'inputs/development_plan.jsonl': plan_raw, 'engineering/cases.json': cases_raw,
                    'execution_manifest.json': (json.dumps(manifest, indent=2) + '\n').encode()})
    payload['bundle_checksums.json'] = json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in payload.items()}, indent=2).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as t:
            for rel, data in sorted(payload.items()):
                info = tarfile.TarInfo(NAME + '/' + rel); info.size = len(data); info.mode = 0o644
                t.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue(); digest = hashlib.sha256(raw).hexdigest()
    output.mkdir(parents=True)
    (output / (NAME + '.tar.gz')).write_bytes(raw)
    nb = (json.dumps(notebook(raw, digest), ensure_ascii=False, indent=1) + '\n').encode()
    (output / (NAME + '.ipynb')).write_bytes(nb)
    (output / 'execution_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    receipt = {'archive_sha256': digest, 'archive_bytes': len(raw),
               'notebook_sha256': hashlib.sha256(nb).hexdigest(), 'source_commit': commit,
               'n_predictions': 800, 'contains_real_gold': False, 'GPU_started': False}
    (output / 'bundle_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir', type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().output_dir), indent=2))
