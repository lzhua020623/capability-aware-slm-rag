"""Prepare the authorised 500+1500 NQ collection; no teacher API calls."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import shutil

from src.data.team_nq import BoundaryError,TeamNQBoundary,ID_FIELD,read_json,sha
from src.distillation.e_prep import ROOT,digest,load_protocol,teacher_task
from src.distillation.expansion_inventory import batch_release,build_inventory
from src.distillation.teacher_cost import reconcile_archive
from src.distillation.teacher_pilot import registration,request_body,reservation
from src.distillation.teacher_rate import load_profile

PLAN='configs/distillation/e_expansion_wave1_v1.json'
ASSETS=('src/distillation/expansion_prep.py','src/distillation/expansion_collect.py',
    'scripts/run_e_teacher_expansion.py','src/distillation/teacher_cost.py',
    'src/distillation/e_prep.py','src/distillation/teacher_pilot.py','src/distillation/teacher_rate.py',
    'src/distillation/expansion_inventory.py','configs/distillation/e_teacher_transport_v2.json',
    'configs/distillation/e_prep_v1.json')


def bound(root,item):
    path=(root/item['path']).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path)!=item['sha256']:
        raise BoundaryError('Missing/changed expansion provenance: '+item['path'])
    return read_json(path)


def load_plan(root=ROOT):
    root=Path(root);plan=read_json(root/PLAN)
    expected={'schema_version':'e_teacher_expansion_wave1_v1','budget_selected':2000,
        'first_wave_total_candidates':2000,'primary_new_requests':1500,'max_new_requests':1500,
        'reused_pilot_records':500,'cumulative_budget_usd':20,'cumulative_budget_nanousd':20000000000,
        'pilot_price_estimate_nanousd':3427050000,'candidate_collection_enabled':True,
        'automatic_further_expansion':False,'training_enabled':False,'semantic_approval_automatic':False,
        'transport_profile':'token_window_v2','automatic_retries':0,'max_inflight':1}
    if any(type(plan.get(k)) is not type(v) or plan.get(k)!=v for k,v in expected.items()):
        raise BoundaryError('Only the latest 2000-candidate / USD20 collection is authorised')
    parent=registration(root)
    if bound(root,plan['teacher_registration'])!=parent:
        raise BoundaryError('Teacher registration changed')
    release=bound(root,plan['user_release'])
    if release!=batch_release(root):raise BoundaryError('Batch release changed')
    bound(root,plan['pilot_sample_manifest']);summary=bound(root,plan['pilot_result_summary'])
    if summary['n']!=500 or not summary['complete'] or summary['journal_sha256']!=plan['pilot_journal_sha256']:
        raise BoundaryError('Pilot reuse is incomplete or changed')
    acceptance=bound(root,plan['budget_acceptance'])
    if (acceptance.get('effective_user_instruction')!='先2000条 有效再扩量'
            or acceptance.get('selected_total_candidates')!=2000
            or acceptance.get('reuse_pilot')!=500 or acceptance.get('new_requests')!=1500
            or acceptance.get('cumulative_budget_usd')!=20
            or acceptance.get('further_paid_expansion_automatic') is not False):
        raise BoundaryError('Latest budget acceptance missing')
    quote=reconcile_archive(root/'artifacts/controller/e_teacher_pilot_20261028')
    if Decimal(quote['published_price_estimate_usd'])*1000000000!=plan['pilot_price_estimate_nanousd']:
        raise BoundaryError('Pilot published-price reconciliation changed')
    load_protocol(root);load_profile(root,parent)
    return plan,parent


def manifest(boundary,plan):
    inventory=build_inventory(boundary);new=inventory['new_order_max5000'][:1500]
    boundary.require(new,'train')
    return {'schema_version':'e_expansion_2000_manifest_v1','plan_sha256':sha(boundary.root/PLAN),
        'boundary':boundary.receipt(),'inventory_sha256':digest(inventory),
        'reused_pilot_ids':inventory['reused_pilot_ids'],'new_ids':new,
        'new_AI_strata':dict(Counter(boundary.labels[sid] for sid in new)),
        'sampling_runtime_or_gold_read':False,'total_candidates':2000,'new_request_cap':1500,
        'automatic_further_expansion':False,'training_enabled':False}


def selected_rows(boundary,source,wanted):
    """Skip every nonselected ID before decoding question/evidence/gold."""
    wanted=set(wanted);seen=set()
    for line in boundary.source_lines(source):
        if not line.strip():continue
        ids=ID_FIELD.findall(line)
        if len(ids)!=1:raise BoundaryError('Source needs one exact string sample_id')
        sid=ids[0].decode('ascii')
        if sid in seen or sid not in boundary.membership:raise BoundaryError('Duplicate/unknown source ID')
        seen.add(sid)
        if sid not in wanted:continue
        boundary.require([sid],'train')
        row=json.loads(line);boundary.require_record(row,'train')
        yield row


def prepare(root=ROOT):
    root=Path(root);plan,parent=load_plan(root);boundary=TeamNQBoundary(root)
    sample=manifest(boundary,plan);sample_path=root/plan['sample_manifest']
    sample_path.parent.mkdir(parents=True,exist_ok=True)
    if sample_path.exists():
        if read_json(sample_path)!=sample:raise BoundaryError('Frozen expansion sample manifest changed')
    else:sample_path.write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
    output=root/plan['inputs_directory'];receipt_path=root/plan['execution_receipt']
    if output.exists() or receipt_path.exists():raise FileExistsError('Preserve prepared expansion; use offline verification')
    source=root/boundary.policy['source_delivery']['default_local_directory']
    if not source.is_dir():raise BoundaryError('Original checksum-bound C1 delivery directory required')
    tasks={};refs={};profile=load_profile(root,parent)
    for row in selected_rows(boundary,source,sample['new_ids']):
        sid=row['sample_id'];task=teacher_task(boundary,row)
        envelope=reservation(request_body(task,parent),parent)
        if envelope['input_tokens_bound']+parent['max_completion_tokens_including_reasoning']>profile['token_reservation_cap']:
            raise BoundaryError('Selected task exceeds registered transport envelope')
        gold=(row.get('annotation_reference') or {}).get('gold_answer')
        if not isinstance(gold,str) or not gold.strip():raise BoundaryError('Selected offline reference missing')
        tasks[sid]=task;refs[sid]={'sample_id':sid,'gold_answer':gold,
            'primary_label_AI':boundary.labels[sid],'usage':'offline only; never teacher input'}
    if set(tasks)!=set(sample['new_ids']):raise BoundaryError('Selected source rows incomplete')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='e-expansion-',dir=output.parent) as temporary:
        stage=Path(temporary)
        for name,rows in [('teacher_tasks.jsonl',tasks),('offline_references.jsonl',refs)]:
            with (stage/name).open('w',encoding='utf-8') as handle:
                for sid in sample['new_ids']:handle.write(json.dumps(rows[sid],ensure_ascii=False,allow_nan=False)+'\n')
        receipt={'schema_version':'e_expansion_2000_execution_v1','plan_sha256':sha(root/PLAN),
            'sample_manifest_sha256':sha(sample_path),'n_new_tasks':1500,'reused_pilot_n':500,
            'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in stage.iterdir()},
            'assets':{name:sha(root/name) for name in ASSETS},
            'source_delivery':{s['file']:s['sha256'] for s in boundary.delivery['shards']},
            'only_selected_runtime_rows_decoded':1500,'official_test_runtime_read':False,
            'API_calls':0,'training_runs':0,'ready_for_candidate_API':True,'ready_for_SFT':False}
        shutil.move(str(stage),str(output))
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt


def preflight(root=ROOT):
    root=Path(root);plan,parent=load_plan(root);boundary=TeamNQBoundary(root)
    sample=read_json(root/plan['sample_manifest'])
    if sample!=manifest(boundary,plan):raise BoundaryError('Expansion sample IDs or roles changed')
    receipt=read_json(root/plan['execution_receipt'])
    if (receipt.get('schema_version')!='e_expansion_2000_execution_v1'
            or receipt.get('plan_sha256')!=sha(root/PLAN)
            or receipt.get('sample_manifest_sha256')!=sha(root/plan['sample_manifest'])
            or receipt.get('n_new_tasks')!=1500 or receipt.get('reused_pilot_n')!=500
            or receipt.get('ready_for_candidate_API') is not True or receipt.get('ready_for_SFT') is not False
            or receipt.get('official_test_runtime_read') is not False):
        raise BoundaryError('Expansion execution seal missing or changed')
    for name,expected in receipt['assets'].items():
        if name not in ASSETS or sha(root/name)!=expected:raise BoundaryError('Expansion source changed: '+name)
    if set(receipt['assets'])!=set(ASSETS):raise BoundaryError('Expansion asset seal incomplete')
    path=root/plan['inputs_directory']/'teacher_tasks.jsonl'
    if sha(path)!=receipt['files']['teacher_tasks.jsonl']['sha256']:raise BoundaryError('Expansion teacher tasks changed')
    tasks=[json.loads(line) for line in path.read_text().splitlines()]
    if [t['sample_id'] for t in tasks]!=sample['new_ids']:raise BoundaryError('Expansion task cohort changed')
    profile=load_profile(root,parent)
    for task in tasks:
        if task!=teacher_task(boundary,task['runtime']):raise BoundaryError('Expansion prompt/runtime/schema changed')
        reserve=reservation(request_body(task,parent),parent)
        if reserve['input_tokens_bound']+parent['max_completion_tokens_including_reasoning']>profile['token_reservation_cap']:
            raise BoundaryError('Expansion rate envelope invalid')
    # No offline reference file is opened by this live-client preflight.
    return plan,parent,sample,tasks,receipt
