"""Freeze development-only T3/T4 inputs and code; never package QA references."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_controller_integration import read_inputs
from src.controller.execution import fingerprint
from src.controller.runtime import canonical_input
from dataclasses import asdict

NAME = 'controller_t3_gpu_v1'
RUNNERS = ('scripts/run_controller_integration.py', 'scripts/evaluate_controller_integration.py',
           'scripts/run_controller_t4.py', 'scripts/prepare_controller_t3_gpu.py')


def notebook(receipt, *, engineering_only=True, resource_receipt=None, engineering_sha=None):
    if not engineering_only and (not resource_receipt or not engineering_sha):
        raise ValueError('T4 notebook requires registered resource archive and CUDA receipt hashes')
    loader = f'''import os, sys, json, hashlib, shutil, subprocess, tarfile, zipfile
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Enable CUDA GPU and Internet'
WORK=Path('/kaggle/working')
BUNDLE=Path('/tmp')/('controller_'+{receipt['checksum_index_sha256'][:12]!r})/{NAME!r}
OUTPUT=WORK/{('controller_t3_engineering_v1_outputs' if engineering_only else 'controller_t4_development_v1_outputs')!r}
assert not OUTPUT.exists(), 'Preserve previous outputs'
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
if not BUNDLE.exists():
    archives=list(Path('/kaggle/input').rglob({(NAME+'.tar.gz')!r}))
    indexes=[p for p in Path('/kaggle/input').rglob('bundle_checksums.json') if file_sha(p)=={receipt['checksum_index_sha256']!r}]
    BUNDLE.parent.mkdir(parents=True,exist_ok=True)
    if indexes:
        assert len(indexes)==1
        shutil.copytree(indexes[0].parent,BUNDLE)
    else:
        assert len(archives)==1 and file_sha(archives[0])=={receipt['archive_sha256']!r}, 'Attach the private code dataset'
        with tarfile.open(archives[0]) as t:t.extractall(BUNDLE.parent,filter='data')
index=BUNDLE/'bundle_checksums.json'
assert file_sha(index)=={receipt['checksum_index_sha256']!r}
for rel,digest in json.loads(index.read_text()).items():
    assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
    assert file_sha(BUNDLE/rel)==digest,rel
print('Frozen code and development inputs verified; no gold/selection/final input files')
'''
    install = '''subprocess.run([sys.executable,'-m','pip','install','-r',str(BUNDLE/'requirements-controller-t3.txt')],check=True)
subprocess.run([sys.executable,'-m','unittest','tests.test_controller_t3_execution','tests.test_controller_t3_evaluation',
                'tests.test_controller_integration','tests.test_controller_evaluation','tests.test_controller_t3_gate','-q'],cwd=BUNDLE,check=True)
subprocess.run([sys.executable,'scripts/run_controller_t4.py','--preflight','--output-dir',str(OUTPUT)],cwd=BUNDLE,check=True)
'''
    resources = ''
    if not engineering_only:
        resources = f'''# Bind immutable assets outside /kaggle/working so output exports do not duplicate GB resources.
markers=[p for p in Path('/kaggle/input').rglob('resource_bundle_v1.json') if json.loads(p.read_text()).get('schema_version')=='controller_t4_resource_bundle_v1']
if markers:
    assert len(markers)==1
    RESOURCES=markers[0].parent
else:
    archives=list(Path('/kaggle/input').rglob('controller_t4_resource_bundle_v1.tar.gz'))
    assert len(archives)==1 and file_sha(archives[0])=={resource_receipt['sha256']!r}
    TEMP=Path('/tmp/controller_resource_unpack'); TEMP.mkdir(exist_ok=False)
    with tarfile.open(archives[0]) as t:t.extractall(TEMP,filter='data')
    RESOURCES=TEMP/'controller_t4_resources_v1'
manifest=json.loads((RESOURCES/'resource_bundle_v1.json').read_text())
expected={resource_receipt['files']!r}
assert manifest['files']==expected and manifest['contains_gold'] is False
for rel,info in expected.items():
    source=RESOURCES/rel
    assert source.stat().st_size==info['bytes'] and file_sha(source)==info['sha256'],rel
    target=BUNDLE/rel;target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists()
    target.symlink_to(source)
receipts=[p for p in Path('/kaggle/input').rglob('engineering_receipt.json') if file_sha(p)=={engineering_sha!r}]
assert len(receipts)==1, 'Attach the accepted CUDA receipt'
shutil.copyfile(receipts[0],BUNDLE/'engineering_receipt.json')
print('Original retrieval assets and accepted CUDA receipt verified')
'''
    run = f'''command=[sys.executable,'-u','scripts/run_controller_t4.py','--output-dir',str(OUTPUT)]
command+={(['--engineering-only'] if engineering_only else ['--engineering-receipt','engineering_receipt.json'])!r}
log=WORK/'controller_t3_t4_run.log'
with log.open('x') as handle:
    p=subprocess.Popen(command,cwd=BUNDLE,env={{**os.environ,'PYTHONUNBUFFERED':'1'}},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout:
        print(line,end='',flush=True);handle.write(line);handle.flush()
    returncode=p.wait()
OUTPUT.mkdir(exist_ok=True)
shutil.copyfile(log,OUTPUT/'run.log')
shutil.copyfile(BUNDLE/'execution_manifest.json',OUTPUT/'input_execution_manifest.json')
print('Process return code:',returncode,'; exporting all partial or completed outputs')
'''
    export = '''assert OUTPUT.exists()
checks={str(p.relative_to(OUTPUT)):file_sha(p) for p in OUTPUT.rglob('*') if p.is_file() and p.name!='output_checksums.json'}
(OUTPUT/'output_checksums.json').write_text(json.dumps(checks,indent=2))
archive=WORK/(OUTPUT.name+'.zip')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
    for p in sorted(OUTPUT.rglob('*')):
        if p.is_file():z.write(p,str(p.relative_to(OUTPUT)))
print('Exported:',archive,'SHA256:',file_sha(archive))
assert returncode==0, 'Preserve failed run and partial outputs; no semantic retry'
'''
    title = '# Controller T3 real CUDA engineering gates' if engineering_only else '# Controller T4 frozen development comparison'
    intro = title + '\n\nPrivate dataset + T4 GPU + Internet. Frozen Qwen/NF4, CPU FP32 reranker, prefill256. No QA gold, selection or final inputs.\n'
    intro += ('Eight synthetic long-input paths (>=14k token), true CPU scoring timing and short Full checks; no benchmark inference.'
              if engineering_only else 'One run of development 200 per dataset, seven paired groups. C1 cached. No tuning or training. Gold scored offline.')
    cells = [{'cell_type':'markdown','metadata':{},'source':intro.splitlines(keepends=True)}]
    for source in (loader, install, resources, run, export):
        if not source:
            continue
        compile(source, '<controller-notebook>', 'exec')
        cells.append({'cell_type':'code','metadata':{},'source':source.splitlines(keepends=True),'outputs':[],'execution_count':None})
    return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                    'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_source_commit(root):
    # No freezing uncommitted implementation or silently including unrelated work.
    paths = ('src', 'scripts', 'configs', 'tests', 'models/diagnosis',
             'requirements-controller-t3.txt', 'requirements-answer-repair-v8.txt')
    subprocess.run(['git', 'diff', 'HEAD', '--exit-code', '--', *paths],
                   cwd=root, check=True, stdout=subprocess.PIPE)
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--',
                                        *paths],
                                       cwd=root, text=True).strip()
    if untracked:
        raise ValueError('Commit implementation before sealing GPU code')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def prepare(output, *, root=ROOT):
    output, root = Path(output), Path(root)
    if output.exists():
        raise FileExistsError('Preserve existing frozen bundle')
    commit = read_source_commit(root)
    policy_path = 'configs/controller/t3_execution_policy_v1.json'
    policy = json.loads((root / policy_path).read_text())
    if policy['role'] != 'development' or policy['counts_per_dataset'] != 200 or policy['T4_ready'] is not False:
        raise ValueError('Only predeclared development 200 is supported')
    ids = json.loads((root / 'artifacts/controller/data_v1/selection_ids.json').read_text())
    c1 = json.loads((root / 'configs/controller/fever_c1_completed_v1.json').read_text())
    datasets = {}; payload = {}
    from src.diagnosis.adapter import c1_record_to_input
    old_nq = read_inputs(root / 'artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch06_runtime.jsonl')[0]
    with (root / 'results/preliminary/fever_c1_7b.jsonl').open() as handle:
        old_fever = c1_record_to_input(json.loads(next(handle)))
    historical = [asdict(i) for i in (old_nq, old_fever)]
    payload['engineering/historical_inputs.jsonl'] = ''.join(json.dumps(i, ensure_ascii=False) + '\n' for i in historical).encode()
    for ds in ('nq', 'fever'):
        rel = (c1['runtime_files']['development']['path'] if ds == 'fever' else
               'artifacts/controller/data_v1/nq/development/runtime.jsonl')
        items = read_inputs(root / rel)
        if [i.sample_id for i in items] != ids[ds]['development'] or len(items) != 200:
            raise ValueError('Development input ID/order mismatch')
        target = f'inputs/{ds}_development_runtime.jsonl'
        payload[target] = (root / rel).read_bytes()
        gold = root / f'artifacts/controller/data_v1/offline/{ds}/development/gold.jsonl'
        datasets[ds] = {'input': {'path': target, 'sha256': sha(root / rel)}, 'source_path': rel,
                        'sample_ids': [i.sample_id for i in items], 'n': len(items),
                        'effective_input_hashes': {i.sample_id: fingerprint(asdict(canonical_input(i))) for i in items},
                        'offline_gold_sha256': sha(gold), 'offline_gold_packaged': False}
    sources = sorted((root / 'src').rglob('*.py'))
    sources += [root / p for p in RUNNERS]
    sources += [root / p for p in ('configs/base.yaml', 'configs/controller/integration_t3_v1.json',
                policy_path, 'configs/controller/t5_selection_supplement_v1.json',
                'configs/controller/experiment_protocol_v1.json',
                'configs/controller/nq_train_resources_v1.json', 'configs/controller/fever_resources_v1.json',
                'models/diagnosis/diagnosis_v2_expanded_hierarchical.json',
                'requirements-controller-t3.txt', 'requirements-answer-repair-v8.txt')]
    # Portable integration checks only; full historical artifact tests stay local.
    sources += [root / ('tests/' + n) for n in ('__init__.py', 'test_controller_integration.py',
                'test_controller_evaluation.py', 'test_controller_t3_execution.py', 'test_controller_t3_evaluation.py',
                'test_controller_t3_gate.py')]
    hashes = {}
    for path in sources:
        rel = str(path.relative_to(root)); payload[rel] = path.read_bytes(); hashes[rel] = sha(path)
    manifest = {'schema_version': 'controller_t3_execution_manifest_v1', 'source_commit': commit,
                'role': 'development', 'datasets': datasets, 'required_modes': policy['required_modes'],
                'source_files': hashes, 'policy_sha256': sha(root / policy_path),
                'parent_T1_sha256': sha(root / policy['parent_design']),
                'data_manifest_sha256': sha(root / 'artifacts/controller/data_v1/manifest.json'),
                'historical_checks': {'path': 'engineering/historical_inputs.jsonl',
                    'sha256': hashlib.sha256(payload['engineering/historical_inputs.jsonl']).hexdigest(),
                    'scope': 'two previously exposed queries, matching Train/FEVER indexes; lookup engineering only'},
                'contains_gold': False, 'contains_selection_or_final_inputs': False,
                'GPU_engineering_passed': False, 'T4_started': False}
    payload['execution_manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    checksums = {rel: hashlib.sha256(raw).hexdigest() for rel, raw in sorted(payload.items())}
    payload['bundle_checksums.json'] = (json.dumps(checksums, indent=2, sort_keys=True) + '\n').encode()
    output.mkdir(parents=True)
    archive = output / (NAME + '.tar.gz')
    with tarfile.open(archive, 'x:gz') as handle:
        for rel, raw in sorted(payload.items()):
            info = tarfile.TarInfo(NAME + '/' + rel); info.size = len(raw); info.mode = 0o644
            handle.addfile(info, io.BytesIO(raw))
    (output / 'execution_manifest.json').write_bytes(payload['execution_manifest.json'])
    receipt = {'archive': str(archive.relative_to(root)) if archive.is_relative_to(root) else str(archive),
               'archive_sha256': sha(archive), 'bytes': archive.stat().st_size,
               'checksum_index_sha256': hashlib.sha256(payload['bundle_checksums.json']).hexdigest(),
               'execution_manifest_sha256': hashlib.sha256(payload['execution_manifest.json']).hexdigest(),
               'n_files': len(payload), 'source_commit': commit, 'GPU_engineering_passed': False, 'T4_started': False}
    (output / 'bundle_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (output / 'controller_t3_engineering_v1.ipynb').write_text(json.dumps(notebook(receipt), indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir), indent=2))


if __name__ == '__main__':
    main()
