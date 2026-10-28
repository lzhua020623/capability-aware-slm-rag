"""Frozen 5000-request extension and one-off audited timeout quarantine."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from src.data.team_nq import BoundaryError,TeamNQBoundary,read_json,sha
from src.distillation.e_prep import ROOT,digest,teacher_task
from src.distillation.expansion_inventory import build_inventory
from src.distillation.expansion_prep import bound,selected_rows,preflight as prior_preflight
from src.distillation.expansion_collect import nano
from src.distillation.teacher_pilot import request_body,reservation,usage_cost,audit_response
from src.distillation.teacher_cost import response_price
from src.distillation.teacher_rate import load_profile

PLAN='configs/distillation/e_expansion_5000_v1.json'
ASSETS=('src/distillation/expansion_5000_prep.py','src/distillation/expansion_5000_collect.py',
        'scripts/run_e_teacher_5000.py')


def load_plan(root=ROOT):
    root=Path(root);spec=read_json(root/PLAN)
    expected={'schema_version':'e_expansion_5000_v1','total_request_cap_including_pilot':5000,
        'reused_pilot_n':500,'max_post_pilot_requests':4500,'cumulative_budget_nanousd':40000000000,
        'timeout_seconds':600,'automatic_retries':0,'max_inflight':1,
        'automatic_further_expansion':False,'training_enabled':False,'semantic_approval_automatic':False}
    if any(type(spec.get(k)) is not type(v) or spec.get(k)!=v for k,v in expected.items()):
        raise BoundaryError('Only the authorised 5000 / USD40 extension is registered')
    bound(root,spec['prior_plan'])
    acceptance=bound(root,spec['user_acceptance'])
    if (acceptance.get('effective_user_instruction')!='可以直接上5000'
            or acceptance.get('post_pilot_request_cap')!=4500 or acceptance.get('cumulative_budget_usd')!=40
            or acceptance.get('automatic_expansion_above_5000') is not False
            or acceptance.get('training_enabled_by_this_record') is not False):
        raise BoundaryError('5000-candidate budget acceptance changed')
    return spec


def paid_audit(raw,task,parent):
    usage_cost(raw,reservation(request_body(task,parent),parent),parent)
    amount=nano(response_price(raw)[0]);rid=raw.get('id')
    if not isinstance(rid,str) or not rid:raise BoundaryError('Missing provider response identity')
    try:audit=audit_response(raw,task,parent)
    except Exception:audit=None
    return amount,rid,audit


def prior_state(root,spec,prior,parent,sample,tasks):
    """Certify all 54 paid responses; retain ordinal55's entire unknown charge."""
    folder=root/prior['run_directory'];journal=root/spec['prior_journal']['path']
    if sha(journal)!=spec['prior_journal']['sha256']:raise BoundaryError('Prior journal changed; do not resend')
    archived=bound(root,spec['prior_archive_manifest']);audit=bound(root,spec['prior_timeout_audit'])
    archive=(root/spec['prior_archive_manifest']['path']).parent/archived['archive']
    if sha(archive)!=archived['sha256']:raise BoundaryError('Prior raw archive changed')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=55 or len({m.name for m in members})!=55:
            raise BoundaryError('Prior archive member inventory changed')
        for member in members:
            if not member.isfile() or member.name not in archived['members']:
                raise BoundaryError('Unexpected prior archive member')
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=archived['members'][member.name]:
                raise BoundaryError('Prior archived bytes changed')
    if set(archived['members'])!={'journal.jsonl'}|{p.name for p in folder.glob('*.response.json')}:
        raise BoundaryError('Prior raw response inventory differs from archive')
    if any(sha(folder/name)!=fingerprint for name,fingerprint in archived['members'].items()):
        raise BoundaryError('Prior local response differs from archive')
    events=[json.loads(line) for line in journal.read_text().splitlines()]
    expected={'kind':'seal','schema_version':'e_expansion_2000_journal_v1',
        'plan_sha256':spec['prior_plan']['sha256'],'execution_receipt_sha256':sha(root/prior['execution_receipt']),
        'ordered_new_ids':sample['new_ids'],'reused_pilot_n':500,
        'pilot_price_estimate_nanousd':prior['pilot_price_estimate_nanousd'],
        'budget_nanousd':prior['cumulative_budget_nanousd']}
    if len(events)!=111 or events[0]!=expected:raise BoundaryError('Prior request history incomplete or changed')
    state={'attempted':55,'completed':0,'valid':0,'spent_nanousd':prior['pilot_price_estimate_nanousd'],
           'provider_ids':[],'unavailable_ids':[]}
    for i in range(55):
        reserve,settle=events[1+2*i:3+2*i];task=tasks[i];sid=task['sample_id']
        expected_reserve=reservation(request_body(task,parent),parent)
        if reserve!={'kind':'reserve','sample_id':sid,'reserved_nanousd':expected_reserve['microusd']*1000,
                     'input_tokens_bound':expected_reserve['input_tokens_bound']}:
            raise BoundaryError('Prior request order or reservation changed')
        if settle.get('kind')!='settle' or settle.get('sample_id')!=sid:
            raise BoundaryError('Prior settlement identity changed')
        if state['spent_nanousd']+reserve['reserved_nanousd']>prior['cumulative_budget_nanousd']:
            raise BoundaryError('Prior budget reservation exceeded')
        if i==54:
            skip=spec['audited_skip']
            if (skip!={'ordinal':55,'sample_id':sid,'error_type':'TimeoutError',
                       'retain_nanousd':reserve['reserved_nanousd'],'retry':False,'replace_sample':False}
                    or settle.get('request_completed') is not False or settle.get('error_type')!='TimeoutError'
                    or settle.get('cost_nanousd')!=skip['retain_nanousd']
                    or (folder/(sid+'.response.json')).exists()):
                raise BoundaryError('Only the frozen no-response timeout may be skipped')
            state['spent_nanousd']+=skip['retain_nanousd'];state['unavailable_ids'].append(sid)
            continue
        path=folder/(sid+'.response.json');raw=json.loads(path.read_text())
        amount,rid,checked=paid_audit(raw,task,parent)
        if (settle.get('request_completed') is not True or settle.get('response_sha256')!=sha(path)
                or settle.get('cost_nanousd')!=amount or amount>reserve['reserved_nanousd']
                or settle.get('provider_response_id')!=rid or rid in state['provider_ids']
                or settle.get('audit')!=checked or checked is None
                or settle.get('target_structural_passed') is not True
                or settle.get('semantic_verified') is not False or settle.get('approved_for_SFT') is not False):
            raise BoundaryError('Prior successful response audit changed')
        state['spent_nanousd']+=amount;state['completed']+=1;state['valid']+=1;state['provider_ids'].append(rid)
    if (spec['prior_import']!={'attempts':55,'completed':54,'unknown':1,
            'cost_with_pilot_and_unknown_nanousd':state['spent_nanousd']}
            or audit['source_journal_sha256']!=spec['prior_journal']['sha256']
            or audit['cumulative_including_unknown_full_reservation_nanousd']!=state['spent_nanousd']):
        raise BoundaryError('Prior timeout accounting differs from audit')
    return state


def manifest(boundary,spec,old_sample):
    inventory=build_inventory(boundary);ids=inventory['new_order_max5000'][:4500]
    if ids[:1500]!=old_sample['new_ids']:raise BoundaryError('5000 extension does not preserve prior1500 prefix')
    boundary.require(ids,'train')
    return {'schema_version':'e_expansion_5000_samples_v1','plan_sha256':sha(boundary.root/PLAN),
        'boundary':boundary.receipt(),'inventory_sha256':digest(inventory),
        'reused_pilot_ids':inventory['reused_pilot_ids'],'post_pilot_ids':ids,
        'post_pilot_AI_strata':dict(Counter(boundary.labels[sid] for sid in ids)),
        'sampling_runtime_or_gold_read':False,'request_cap_including_pilot':5000,
        'audited_unavailable_ids':[spec['audited_skip']['sample_id']],
        'maximum_complete_candidates_after_skip':4999,'ready_for_SFT':False}


def prepare(root=ROOT):
    root=Path(root);spec=load_plan(root);prior,parent,old_sample,old_tasks,_=prior_preflight(root)
    state=prior_state(root,spec,prior,parent,old_sample,old_tasks)
    boundary=TeamNQBoundary(root);sample=manifest(boundary,spec,old_sample)
    output=root/spec['inputs_directory'];sample_path=root/spec['sample_manifest'];receipt_path=root/spec['execution_receipt']
    if output.exists() or sample_path.exists() or receipt_path.exists():raise FileExistsError('Preserve the frozen 5000 export')
    rows={};refs={};profile=load_profile(root,parent)
    source=root/boundary.policy['source_delivery']['default_local_directory']
    for row in selected_rows(boundary,source,sample['post_pilot_ids']):
        sid=row['sample_id'];task=teacher_task(boundary,row)
        reserve=reservation(request_body(task,parent),parent)
        if reserve['input_tokens_bound']+4096>profile['token_reservation_cap']:raise BoundaryError('Rate envelope exceeded')
        gold=(row.get('annotation_reference') or {}).get('gold_answer')
        if not isinstance(gold,str) or not gold.strip():raise BoundaryError('Missing offline reference')
        rows[sid]=task;refs[sid]={'sample_id':sid,'gold_answer':gold,'usage':'offline only; never teacher input'}
    if set(rows)!=set(sample['post_pilot_ids']):raise BoundaryError('Missing frozen source IDs')
    if [rows[t['sample_id']] for t in old_tasks]!=old_tasks:raise BoundaryError('Prior teacher tasks changed')
    output.parent.mkdir(parents=True,exist_ok=True);sample_path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='e-5000-',dir=output.parent) as tmp:
        stage=Path(tmp)
        for name,data in [('teacher_tasks.jsonl',rows),('offline_references.jsonl',refs)]:
            with (stage/name).open('w',encoding='utf-8') as handle:
                for sid in sample['post_pilot_ids']:handle.write(json.dumps(data[sid],ensure_ascii=False,allow_nan=False)+'\n')
        sample_path.write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
        receipt={'schema_version':'e_expansion_5000_execution_v1','plan_sha256':sha(root/PLAN),
            'sample_manifest_sha256':sha(sample_path),'n_post_pilot_tasks':4500,'prior_state':state,
            'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in stage.iterdir()},
            'assets':{p:sha(root/p) for p in ASSETS},'ready_for_SFT':False,'new_API_calls':0,
            'official_test_runtime_read':False,'original_1500_tasks_unchanged':True}
        shutil.move(str(stage),str(output))
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt


def preflight(root=ROOT):
    root=Path(root);spec=load_plan(root);prior,parent,old_sample,old_tasks,_=prior_preflight(root)
    state=prior_state(root,spec,prior,parent,old_sample,old_tasks)
    boundary=TeamNQBoundary(root);sample=read_json(root/spec['sample_manifest'])
    if sample!=manifest(boundary,spec,old_sample):raise BoundaryError('5000 sample scope changed')
    receipt=read_json(root/spec['execution_receipt'])
    if (receipt.get('schema_version')!='e_expansion_5000_execution_v1'
            or receipt.get('plan_sha256')!=sha(root/PLAN)
            or receipt.get('sample_manifest_sha256')!=sha(root/spec['sample_manifest'])
            or receipt.get('n_post_pilot_tasks')!=4500 or receipt.get('prior_state')!=state
            or receipt.get('ready_for_SFT') is not False or receipt.get('official_test_runtime_read') is not False
            or receipt.get('assets')!={p:sha(root/p) for p in ASSETS}):
        raise BoundaryError('5000 execution seal changed')
    path=root/spec['inputs_directory']/'teacher_tasks.jsonl'
    if sha(path)!=receipt['files']['teacher_tasks.jsonl']['sha256']:raise BoundaryError('5000 task file changed')
    tasks=[json.loads(line) for line in path.read_text().splitlines()]
    if [t['sample_id'] for t in tasks]!=sample['post_pilot_ids']:raise BoundaryError('5000 task order changed')
    profile=load_profile(root,parent)
    for task in tasks:
        if task!=teacher_task(boundary,task['runtime']):raise BoundaryError('Teacher contract or runtime changed')
        reserve=reservation(request_body(task,parent),parent)
        if reserve['input_tokens_bound']+4096>profile['token_reservation_cap']:raise BoundaryError('Rate envelope exceeded')
    return spec,parent,sample,tasks,state
