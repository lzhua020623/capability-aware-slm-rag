"""Offline teacher-pilot replay and frozen semantic-review selection.

No network, training or automatic semantic approval. Reference answers are
offline diagnostics, never evidence-verification labels or teacher inputs.
"""
from collections import Counter
import fcntl
import hashlib
import html
import json
from pathlib import Path
import statistics

from src.data.team_nq import BoundaryError, read_json, sha
from src.distillation.e_prep import ROOT, digest, to_input
from src.compensation.evidence_id_repair import evidence_units
from src.distillation.teacher_pilot import (preflight, request_body, reservation,
    usage_cost, audit_response)
from src.distillation.teacher_rate import PROFILE_PATH
from src.evaluation.qa import exact_match, token_f1


def replay(spec, tasks, events, responses, count_target, references=None):
    """Replay journal state, exact provider outputs and costs independently of run()."""
    expected={'kind':'seal','registration':digest(spec),'tasks':spec['tasks_sha256'],
              'ordered_ids':[t['sample_id'] for t in tasks]}
    if not events or events[0]!=expected:
        raise BoundaryError('Pilot seal/cohort/registration mismatch')
    pending=None;position=0;settled={};profile_positions=[]
    for event in events[1:]:
        kind=event.get('kind')
        if kind=='transport_profile':
            if pending is not None:raise BoundaryError('Profile changed during a request')
            profile_positions.append({'before_ordinal':position+1,**event})
        elif kind=='reserve':
            if pending is not None or position>=len(tasks):raise BoundaryError('Duplicate/in-flight/extra request')
            task=tasks[position]
            amount=reservation(request_body(task,spec),spec)
            if event.get('sample_id')!=task['sample_id'] or any(event.get(k)!=v for k,v in amount.items()):
                raise BoundaryError('Request order or cost reservation mismatch')
            pending=event
        elif kind=='settle':
            if pending is None or event.get('sample_id')!=pending['sample_id'] or not event.get('accepted'):
                raise BoundaryError('Missing/duplicate/failed settlement')
            if event['sample_id'] in settled:raise BoundaryError('Repeated sample settlement')
            settled[event['sample_id']]=event;position+=1;pending=None
        else:raise BoundaryError('Unknown/duplicate journal event')
    if pending is not None or position!=len(tasks):raise BoundaryError('Pilot incomplete or pending')
    if set(responses)!=set(settled):raise BoundaryError('Raw response IDs differ from the complete pilot')
    if references is not None and set(references)!=set(settled):raise BoundaryError('Offline reference cohort mismatch')
    records=[];labels=Counter();actions=Counter();tokens=Counter();spent=0;unresolved=0
    for ordinal,task in enumerate(tasks,1):
        sid=task['sample_id'];raw=responses[sid];audit=audit_response(raw,task,spec)
        event=settled[sid];reserve=reservation(request_body(task,spec),spec)
        if spent+reserve['microusd']>spec['pilot_budget_microusd']:
            raise BoundaryError('Request would exceed its frozen cost reservation budget')
        cost=usage_cost(raw,reserve,spec)
        if cost!=event['microusd'] or audit!=event['audit'] or raw.get('id')!=event.get('provider_response_id'):
            raise BoundaryError('Provider output/cost differs from journal')
        spent+=cost
        text=raw['choices'][0]['message']['content'];value=json.loads(text)
        n=count_target(text)
        if type(n) is not int or n<=0:raise BoundaryError('Invalid target token count')
        label=audit['model_label'];action=audit['candidate_action'];labels[label]+=1;actions[action]+=1
        unresolved+=int(audit['candidate_unresolved'])
        usage=raw['usage'];tokens['input']+=usage['prompt_tokens'];tokens['completion_including_reasoning']+=usage['completion_tokens']
        tokens['reasoning_subset']+=usage.get('completion_tokens_details',{}).get('reasoning_tokens',0)
        tokens['cache_write_subset']+=usage.get('prompt_tokens_details',{}).get('cache_write_tokens',0)
        tokens['cached_input_subset']+=usage.get('prompt_tokens_details',{}).get('cached_tokens',0)
        group='replace' if action=='replace' else {'supported':'supported_keep','insufficient':'insufficient_keep','contradicted':'contradicted_no_candidate'}[label]
        row={'ordinal':ordinal,'sample_id':sid,'runtime':task['runtime'],'teacher_output':value,
            'teacher_visible_output':text,'audit':audit,'review_group':group,
            'target_tokens_with_EOS':n,'target_fits_384':n<=spec['student_output_tokens'],
            'accounting_microusd':cost,'provider_response_sha256':digest(raw),
            'semantic_verified':False,'approved_for_SFT':False}
        if references is not None:
            ref=references[sid]['gold_answer'];original=task['runtime']['initial_answer'];final=audit['candidate_final_answer']
            row['offline_diagnostic']={'reference':ref,'C1_EM':exact_match(original,ref),
                'teacher_final_EM':exact_match(final,ref),'C1_token_F1':token_f1(original,ref),
                'teacher_final_token_F1':token_f1(final,ref),'is_semantic_judgment':False}
        records.append(row)
    lengths=[r['target_tokens_with_EOS'] for r in records]
    summary={'n':len(tasks),'complete':True,'structural_provenance_passed':len(records),
        'structural_provenance_pass_rate':1.0,'verification_labels':dict(labels),'actions':dict(actions),
        'unresolved':unresolved,'target_tokens_with_EOS':{'min':min(lengths),'median':statistics.median(lengths),
        'max':max(lengths),'over_384':sum(n>spec['student_output_tokens'] for n in lengths)},
        'provider_tokens':dict(tokens),'accounting_usd_upper_estimate':spent/1e6,
        'invoice_verified':False,'transport_profiles':profile_positions,
        'semantic_targets_approved':0,'student_training_runs':0,'population_representative':False,
        'scope':'Balanced Train teacher pilot; not a trained-student or final-test evaluation'}
    if references is not None:
        pairs=[r['offline_diagnostic'] for r in records]
        summary['offline_EM_diagnostic']={'C1_correct':sum(r['C1_EM'] for r in pairs),
            'teacher_final_correct':sum(r['teacher_final_EM'] for r in pairs),
            'fixes':sum(not r['C1_EM'] and r['teacher_final_EM'] for r in pairs),
            'harms':sum(r['C1_EM'] and not r['teacher_final_EM'] for r in pairs),
            'semantic_review_done':False,'is_population_or_student_gain':False}
    return records,summary


def review_selection(records, spec):
    """Direct SHA256(sample_id), no selectable seed or gold/score-based screening."""
    groups={name:[] for name in ('replace','supported_keep','insufficient_keep','contradicted_no_candidate')}
    for row in records:groups[row['review_group']].append(row)
    for pool in groups.values():pool.sort(key=lambda r:(hashlib.sha256(r['sample_id'].encode()).hexdigest(),r['sample_id']))
    selected=groups['replace'][:50];keeps=[groups[name][:] for name in ('supported_keep','insufficient_keep','contradicted_no_candidate')]
    while len(selected)<spec['quality_gate']['semantic_review_n'] and any(keeps):
        for pool in keeps:
            if pool and len(selected)<spec['quality_gate']['semantic_review_n']:selected.append(pool.pop(0))
    return [{**r,'review_id':f'T{i:03}','human_review':{'semantic_correct':None,
            'harmful_replacement':None,'note':''}} for i,r in enumerate(selected,1)]


def assess_gate(summary, selected, spec, human_reviews=None):
    policy=spec['quality_gate'];reviews={} if human_reviews is None else human_reviews
    ids={r['sample_id'] for r in selected}
    if set(reviews)-ids:raise BoundaryError('Review contains unregistered sample IDs')
    labels=Counter(r['audit']['model_label'] for r in selected)
    replacements=[r for r in selected if r['audit']['candidate_action']=='replace']
    structural=(summary['structural_provenance_pass_rate']>=policy['structural_provenance_pass_rate_min'])
    coverage=(len(selected)==policy['semantic_review_n'] and len(replacements)>=25
        and all(labels[k]>=policy['minimum_count_each_verification_label_in_review'] for k in ('supported','insufficient','contradicted')))
    complete=len(selected)==policy['semantic_review_n'] and all(type(reviews.get(r['sample_id'],{}).get('semantic_correct')) is bool
        and (r['audit']['candidate_action']!='replace' or type(reviews.get(r['sample_id'],{}).get('harmful_replacement')) is bool) for r in selected)
    correct=sum(reviews.get(r['sample_id'],{}).get('semantic_correct') is True for r in selected)
    harms=sum(reviews.get(r['sample_id'],{}).get('harmful_replacement') is True for r in replacements)
    passed=structural and coverage and complete and correct>=policy['semantic_correct_min_of_100'] and harms<=policy['harmful_replacements_max_in_review']
    return {'structural_gate_passed':structural,'review_coverage_sufficient':coverage,
        'selected_n':len(selected),'selected_replacements':len(replacements),'selected_labels':dict(labels),
        'human_review_complete':complete,'semantic_correct':correct if complete else None,
        'harmful_replacements':harms if complete else None,'semantic_gate_passed':passed,
        'expansion_enabled':False,'training_enabled':False,'approved_for_SFT':0,
        'pending':['human semantic confirmation'] if not complete else ['separate expansion/training registration']}


def review_page(selected):
    esc=lambda x:html.escape(str(x))
    cards=[]
    for row in selected:
        runtime=row['runtime'];value=row['teacher_output'];rid=row['review_id']
        evidence=''.join('<details><summary>R'+str(e['rank'])+' · '+esc(e['passage_id'])+'</summary><p>'+esc(e['text'])+'</p></details>' for e in runtime['evidence'])
        cited=set(value.get('verdict_evidence_ids',[]))|set(value.get('replacement_evidence_ids',[]))
        quotes=''.join('<li><code>'+esc(u.id)+'</code> '+esc(u.text)+'</li>'
            for u in evidence_units(to_input(runtime)) if u.id in cited)
        field=lambda key:('<select data-key="'+key+'"><option value="">未审</option><option value="true">是</option><option value="false">否</option><option value="unknown">待定</option></select>')
        harm=('<p>是否把原本正确的答案改错？ '+field('harmful_replacement')+'</p>') if row['audit']['candidate_action']=='replace' else ''
        offline=row.get('offline_diagnostic')
        ref=('<details><summary>离线参考（辅助理解题意，不直接决定证据核验）</summary><p>'+esc(offline['reference'])+'</p></details>') if offline else ''
        cards.append('<article data-id="'+rid+'"><h2>'+rid+' · '+esc(runtime['query'])+'</h2><small>'+esc(row['sample_id'])+' · '+esc(row['review_group'])+'</small><h3>C1 原答</h3><pre>'+esc(runtime['initial_answer'])+'</pre><h3>教师核验</h3><pre>'+esc(json.dumps(value,ensure_ascii=False,indent=2))+'</pre><p>最终动作：'+esc(row['audit']['candidate_action'])+'；最终答：'+esc(row['audit']['candidate_final_answer'])+'</p><h3>已引用证据句</h3><ul>'+quotes+'</ul><h3>原始 Top-5</h3>'+evidence+ref+'<p>教师整条监督（核验标签、候选、理由）语义是否正确？ '+field('semantic_correct')+'</p>'+harm+'<p>备注：<textarea data-key="note"></textarea></p></article>')
    cohort=digest([[r['sample_id'],r.get('provider_response_sha256','')] for r in selected])
    ids=json.dumps([{'review_id':r['review_id'],'sample_id':r['sample_id']} for r in selected]).replace('<','\\u003c')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>E 教师试产语义审核</title><style>body{max-width:1100px;margin:24px auto;padding:0 20px;background:#f4f6f8;font:16px/1.7 system-ui;color:#17212b}header{position:sticky;top:0;background:#f4f6f8;padding:12px}article{padding:24px;background:white;margin:20px 0;border:1px solid #ddd;border-radius:12px}pre{white-space:pre-wrap;background:#eef2f6;padding:12px}details{margin:8px 0}summary{cursor:pointer}textarea{width:95%;min-height:60px}select,button{font-size:16px;padding:6px}</style><header><b>E 教师语义审核</b> <button id="export">导出审核 JSON</button> <span id="count"></span></header><p>默认全部未审。supported 必须有证据支持原答；contradicted 必须明确证明原答错误，不能只因缺证；insufficient 指缺少所问事实或证据冲突。核对实体、时间、关系和限定词。引用/EM 合法不自动等于语义正确。所有人工选择保存在当前浏览器，导出后还需回填仓库；网页不会调用模型或自动批准训练。</p>'''+''.join(cards)+'''<script>
const ids='''+ids+''';const cohort='''+json.dumps(cohort)+''';const key='e-teacher-sol-pilot-semantic-v1:'+cohort;let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){};
const count=()=>document.getElementById('count').textContent='已填写 '+ids.filter(r=>saved[r.review_id]?.semantic_correct).length+'/'+ids.length;
for(const a of document.querySelectorAll('article'))for(const x of a.querySelectorAll('[data-key]')){x.value=(saved[a.dataset.id]||{})[x.dataset.key]||'';x.addEventListener('change',()=>{saved[a.dataset.id]=saved[a.dataset.id]||{};saved[a.dataset.id][x.dataset.key]=x.value;localStorage.setItem(key,JSON.stringify(saved));count()});}count();
document.getElementById('export').onclick=()=>{const rows=ids.map(r=>({...r,cohort_sha256:cohort,source:'user_browser_input',human_review:saved[r.review_id]||{}}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='e_teacher_pilot_human_review.json';a.click();URL.revokeObjectURL(a.href);};</script></html>'''


def audit_local(output, root=ROOT):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Preserve previous audit/review work')
    spec,tasks=preflight(root);folder=root/spec['run_directory']
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _audit_locked(output,root,spec,tasks,folder)


def _audit_locked(output,root,spec,tasks,folder):
    events=[json.loads(line) for line in (folder/'journal.jsonl').read_text().splitlines()]
    files={p.name.removesuffix('.response.json'):p for p in folder.glob('*.response.json')}
    raw={sid:read_json(path) for sid,path in files.items()}
    # Offline references are physically distinct and hash-bound to the original export.
    info=read_json(root/'artifacts/controller/e_distillation_v1/pilot_inputs_manifest.json')
    ref_path=(root/spec['tasks']).with_name('offline_references.jsonl')
    if sha(ref_path)!=info['files']['offline_references.jsonl']['sha256']:
        raise BoundaryError('Original offline reference fingerprint changed')
    refs={r['sample_id']:r for r in map(json.loads,ref_path.read_text().splitlines())}
    from transformers import AutoTokenizer
    revision='a09a35458c702b33eeacc393d103063234e8bc28'
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct',revision=revision,local_files_only=True)
    count=lambda text:len(tokenizer.encode(text+tokenizer.eos_token,add_special_tokens=False))
    records,summary=replay(spec,tasks,events,raw,count,refs)
    for event in summary['transport_profiles']:
        if (event.get('profile')!='token_window_v2' or event.get('profile_sha256')!=sha(root/PROFILE_PATH)
                or event.get('client_sha256')!=sha(root/'src/distillation/teacher_pilot.py')
                or event.get('scheduler_sha256')!=sha(root/'src/distillation/teacher_rate.py')):
            raise BoundaryError('Executed transport source/profile fingerprint changed')
    first_v2=min((e['before_ordinal'] for e in summary['transport_profiles']),default=len(tasks)+1)
    created=[raw[t['sample_id']].get('created') for t in tasks[first_v2-1:]]
    if created and all(type(v) in (int,float) for v in created):
        summary['v2_provider_created_span_minutes']=(max(created)-min(created))/60
    summary.update(journal_sha256=sha(folder/'journal.jsonl'),tasks_sha256=spec['tasks_sha256'],
        registration_sha256=sha(root/'configs/distillation/e_teacher_sol_pilot_v1.json'),
        tokenizer={'model':'Qwen/Qwen2.5-7B-Instruct','revision':revision,'local_files_only':True,'counts_include_EOS':True})
    selected=review_selection(records,spec);gate=assess_gate(summary,selected,spec)
    output.mkdir(parents=True,exist_ok=False)
    def write(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    write('audit_summary.json',summary);write('gate_status.json',gate)
    for name,rows in [('audited_records.jsonl',records),('semantic_review100.jsonl',selected)]:
        (output/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    (output/'semantic_review100.html').write_text(review_page(selected))
    write('review_manifest.json',{'sampling':'SHA256(sample_id) ascending, no seed; replacements first then fixed round-robin keeps',
        'n':len(selected),'groups':dict(Counter(r['review_group'] for r in selected)),
        'semantic_human_confirmed':0,'approved_for_SFT':0,'selection_inputs':'behavior group and sample_id only, no reference/EM',
        'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
    return summary,gate
