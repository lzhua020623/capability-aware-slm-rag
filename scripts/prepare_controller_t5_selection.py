"""Package one selection run after the single candidate has been committed."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.prepare_controller_t3_gpu import read_source_commit
from scripts.run_controller_integration import read_inputs,file_sha
from scripts.run_controller_t5_selection import MODES,POLICY,verify
from src.controller.execution import fingerprint
from src.controller.runtime import canonical_input

NAME='controller_t5_selection_v1'


def notebook(archive_sha,resources):
    code='''from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,sys,tarfile,zipfile
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
matches=list(Path('/kaggle/input').rglob('NAME.tar.gz.bin'))
assert len(matches)==1 and sha(matches[0])=='ARCHIVE_SHA', 'Attach the one frozen private code dataset'
BASE=Path('/tmp/controller_t5_ARCHIVE_PREFIX');BASE.mkdir(exist_ok=False)
with tarfile.open(matches[0]) as t:
    assert all(m.isfile() and (BASE/m.name).resolve().is_relative_to(BASE.resolve()) for m in t.getmembers())
    t.extractall(BASE,filter='data')
ROOT=BASE/'NAME'
for rel,h in json.loads((ROOT/'bundle_checksums.json').read_text()).items():assert sha(ROOT/rel)==h,rel
OUTPUT=Path('/kaggle/working/NAME_outputs');assert not OUTPUT.exists()
LOG=Path('/kaggle/working/NAME_run.log')
os.environ['OMP_NUM_THREADS']='4';os.environ['TOKENIZERS_PARALLELISM']='false'
os.chdir(ROOT)
def run(command):
    with LOG.open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        if p.wait():raise RuntimeError('Frozen selection command failed: '+repr(command))
try:
    run([sys.executable,'-m','pip','install','-r','requirements-controller-e-v2a.txt'])
    run([sys.executable,'-m','unittest','tests.test_controller_integration','tests.test_evidence_v2a','-q'])
    run([sys.executable,'scripts/run_controller_t5_selection.py','--preflight','--output-dir',str(OUTPUT)])
    markers=[p for p in Path('/kaggle/input').rglob('resource_bundle_v1.json') if json.loads(p.read_text()).get('schema_version')=='controller_t4_resource_bundle_v1']
    assert len(markers)==1, 'Attach existing private Controller T4 retrieval resources dataset'
    resources=markers[0].parent;registry=json.loads(markers[0].read_text())
    expected=RESOURCE_FILES
    assert registry['files']==expected and registry['contains_gold'] is False
    for rel,info in expected.items():
        src=resources/rel;assert src.stat().st_size==info['bytes'] and sha(src)==info['sha256'],rel
        dst=ROOT/rel;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();dst.symlink_to(src)
    print('Resources verified; selection only; E unavailable; candidate is NOT promoted',flush=True)
    run([sys.executable,'-u','scripts/run_controller_t5_selection.py','--output-dir',str(OUTPUT)])
finally:
    OUTPUT.mkdir(exist_ok=True)
    if LOG.exists():(OUTPUT/'run.log').write_bytes(LOG.read_bytes())
    (OUTPUT/'input_execution_manifest.json').write_bytes((ROOT/'execution_manifest.json').read_bytes())
    checks={str(p.relative_to(OUTPUT)):sha(p) for p in OUTPUT.rglob('*') if p.is_file()}
    (OUTPUT/'output_checksums.json').write_text(json.dumps(checks,indent=2))
    archive=Path('/kaggle/working/NAME_outputs.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(OUTPUT)))
    print('Preserved output:',archive,'SHA256:',sha(archive),flush=True)
'''
    code=code.replace('RESOURCE_FILES',repr(resources['files'])).replace('ARCHIVE_PREFIX',archive_sha[:12]).replace('ARCHIVE_SHA',archive_sha).replace('NAME',NAME)
    compile(code,'<selection-notebook>','exec')
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'kaggle':{'accelerator':'nvidiaTeslaT4','isGpuEnabled':True,'isInternetEnabled':True,'language':'python','sourceType':'notebook'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':['# Controller T5 single selection · frozen 20261026\n',
            'Private. 200 NQ + 200 FEVER, four arms + cached C1 (2000 records). Six active long-path engineering checks first; E unavailable, zero E model calls. No gold, development/final inputs, tuning, retry, scoring or promotion. Attach frozen code .bin and existing private T4 retrieval resources.']},
            {'cell_type':'code','metadata':{},'source':code.splitlines(keepends=True),'execution_count':None,'outputs':[]}]}


def prepare(output,root=ROOT):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError('Preserve prior selection release')
    commit=read_source_commit(root)
    frozen=json.loads((root/'artifacts/controller/e_v2a_interface_r2/gpu_release_v2/execution_manifest.json').read_text())
    for rel,digest in frozen['source_files'].items():
        if file_sha(root/rel)!=digest:raise ValueError('Frozen core source changed before selection: '+rel)
    policy=json.loads((root/POLICY).read_text());candidate_path=root/policy['candidate_path']
    if file_sha(candidate_path)!=policy['candidate_sha256']:raise ValueError('Unregistered candidate hash')
    candidate=json.loads(candidate_path.read_text())
    if (candidate['stage']!='frozen_for_single_selection_not_promoted' or candidate['threshold']!=.2
            or candidate['selection_candidates']!=1 or candidate['promoted']):raise ValueError('Candidate not frozen')
    ids=json.loads((root/'artifacts/controller/data_v1/selection_ids.json').read_text())
    fever=json.loads((root/'configs/controller/fever_c1_completed_v1.json').read_text())
    payload={};datasets={}
    for ds in ('nq','fever'):
        rel=(fever['runtime_files']['selection']['path'] if ds=='fever' else
            'artifacts/controller/data_v1/nq/selection/runtime.jsonl')
        if ds=='fever' and file_sha(root/rel)!=fever['runtime_files']['selection']['sha256']:
            raise ValueError('Unaccepted FEVER C1 inputs')
        items=read_inputs(root/rel)
        if len(items)!=200 or [i.sample_id for i in items]!=ids[ds]['selection']:raise ValueError('Wrong selection cohort')
        if set(ids[ds]['selection']) & (set(ids[ds]['development']) | set(ids[ds]['final_candidate'])):
            raise ValueError('Selection overlaps development/final')
        target='inputs/'+ds+'_selection_runtime.jsonl';payload[target]=(root/rel).read_bytes()
        datasets[ds]={'input_path':target,'input_sha256':file_sha(root/rel),'sample_ids':[i.sample_id for i in items],
            'effective_input_hashes':{i.sample_id:fingerprint(asdict(canonical_input(i))) for i in items},
            'offline_gold_sha256':file_sha(root/f'artifacts/controller/data_v1/offline/{ds}/selection/gold.jsonl'),
            'offline_gold_packaged':False}
    sources=sorted((root/'src').rglob('*.py'))+[root/p for p in (
        'scripts/run_controller_integration.py','scripts/run_controller_t5_selection.py','scripts/run_controller_e_v2a.py',
        'scripts/prepare_controller_t5_selection.py','scripts/prepare_controller_t3_gpu.py',
        'configs/base.yaml',POLICY,'configs/controller/integration_t3_v1.json','configs/controller/nq_train_resources_v1.json',
        'configs/controller/fever_resources_v1.json','models/diagnosis/diagnosis_v2_expanded_hierarchical.json',
        'requirements-controller-e-v2a.txt','requirements-controller-t3.txt','requirements-answer-repair-v8.txt',
        'tests/__init__.py','tests/test_controller_integration.py','tests/test_evidence_v2a.py')]
    hashes={}
    for path in sources:
        rel=str(path.relative_to(root));payload[rel]=path.read_bytes();hashes[rel]=file_sha(path)
    runtime_candidate={k:candidate[k] for k in ('dataset','threshold','eligible_action','new_action','existing_supported_probe_veto')}
    payload['inputs/candidate_policy.json']=(json.dumps(runtime_candidate,indent=2)+'\n').encode()
    old=root/'artifacts/controller/t3_v1/gpu_release_v1'
    old_receipt=json.loads((old/'bundle_receipt.json').read_text())
    archive=old/'controller_t3_gpu_v1.tar.gz'
    if file_sha(archive)!=old_receipt['archive_sha256']:raise ValueError('Historical lookup fixtures changed')
    with tarfile.open(archive) as t:
        payload['engineering/historical_inputs.jsonl']=t.extractfile('controller_t3_gpu_v1/engineering/historical_inputs.jsonl').read()
    hashes['engineering/historical_inputs.jsonl']=hashlib.sha256(payload['engineering/historical_inputs.jsonl']).hexdigest()
    manifest={'schema_version':'controller_t5_selection_manifest_v1','source_commit':commit,'role':'selection',
        'contains_gold':False,'contains_development_or_final_inputs':False,'datasets':datasets,'modes':list(MODES),
        'source_files':hashes,'candidate_sha256':hashlib.sha256(payload['inputs/candidate_policy.json']).hexdigest(),
        'offline_candidate_sha256':policy['candidate_sha256'],'expected_predictions':2000,
        'E_mode':'unavailable','E_engineering_accepted':False,'selection_started':False,'quality_scored':False}
    payload['execution_manifest.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    payload['bundle_checksums.json']=(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in sorted(payload.items())},indent=2)+'\n').encode()
    output.mkdir(parents=True)
    archive=output/(NAME+'.tar.gz')
    with tarfile.open(archive,'x:gz') as t:
        for rel,raw in sorted(payload.items()):
            member=tarfile.TarInfo(NAME+'/'+rel);member.size=len(raw);member.mode=0o644;t.addfile(member,io.BytesIO(raw))
    digest=file_sha(archive)
    (output/(NAME+'.tar.gz.bin')).write_bytes(archive.read_bytes())
    resources=json.loads((root/'data/processed/controller_resources/controller_t4_resource_bundle_v1.tar.receipt.json').read_text())
    nb=notebook(digest,resources);nbpath=output/(NAME+'_dataset.ipynb');nbpath.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
    (output/'execution_manifest.json').write_bytes(payload['execution_manifest.json'])
    receipt={'source_commit':commit,'archive_sha256':digest,'bytes':archive.stat().st_size,'files':len(payload),
        'notebook_sha256':file_sha(nbpath),'resource_dataset_url':'https://www.kaggle.com/datasets/yearo111/controller-t4-retrieval-resources-frozen-20261024',
        'resource_archive_sha256':resources['sha256'],'GPU_started':False,'quality_scored':False}
    (output/'bundle_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    print(json.dumps(prepare(p.parse_args().output_dir),indent=2))
