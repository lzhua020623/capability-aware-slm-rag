"""Continue at ordinal56; one imported timeout stays charged and is never resent."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import time

from src.data.team_nq import BoundaryError,sha,read_json
from src.distillation.e_prep import ROOT,digest
from src.distillation.expansion_5000_prep import PLAN,preflight,paid_audit
from src.distillation.teacher_pilot import endpoint,request_body,reservation,send
from src.distillation.teacher_rate import load_profile,TokenWindow


def run(root=ROOT,*,limit_post_pilot=4500,environment=os.environ,transport=send,
        clock=time.monotonic,sleep=time.sleep,progress=None):
    root=Path(root);spec,parent,sample,tasks,base=preflight(root)
    if type(limit_post_pilot) is not int or not 1<=limit_post_pilot<=4500:
        raise BoundaryError('At most 4500 post-pilot requests including the55 prior attempts')
    url=endpoint(parent,environment);key=environment.get(parent['key_environment'])
    if not isinstance(key,str) or not key.strip():raise BoundaryError('Azure API key missing; use local hidden input')
    prior=read_json(root/spec['prior_plan']['path']);folder=root/spec['run_directory']
    folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (root/prior['run_directory']/'run.lock').open('a') as prior_lock, (folder/'run.lock').open('a') as lock:
        fcntl.flock(prior_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sha(root/spec['prior_journal']['path'])!=spec['prior_journal']['sha256']:
            raise BoundaryError('Prior journal changed since preflight')
        journal=folder/'journal.jsonl';events=[json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        seal={'kind':'seal','schema_version':'e_expansion_5000_journal_v1','plan_sha256':sha(root/PLAN),
            'execution_receipt_sha256':sha(root/spec['execution_receipt']),
            'post_pilot_ids_sha256':digest(sample['post_pilot_ids']),'imported_state':base,
            'timeout_seconds':spec['timeout_seconds'],'budget_nanousd':spec['cumulative_budget_nanousd']}
        if events and events[0]!=seal:raise BoundaryError('5000 journal belongs to another frozen run')
        def append(event):
            with journal.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            events.append(event)
        if not events:append(seal)
        state={k:(v.copy() if isinstance(v,list) else v) for k,v in base.items()};pending=None
        for event in events[1:]:
            if state['attempted']>=len(tasks):raise BoundaryError('Journal exceeds registered request count')
            task=tasks[state['attempted']];sid=task['sample_id'];reserved=reservation(request_body(task,parent),parent)
            if event.get('kind')=='reserve':
                if (pending is not None or event.get('sample_id')!=sid
                        or event.get('ordinal')!=state['attempted']+1
                        or event.get('reserved_nanousd')!=reserved['microusd']*1000
                        or event.get('input_tokens_bound')!=reserved['input_tokens_bound']
                        or state['spent_nanousd']+event['reserved_nanousd']>spec['cumulative_budget_nanousd']):
                    raise BoundaryError('5000 request order or budget reservation changed')
                pending=event
            elif event.get('kind')=='settle':
                if pending is None or event.get('sample_id')!=sid or event.get('request_completed') is not True:
                    raise BoundaryError('New unknown-charge request requires a separate audit; no automatic retry or skip')
                path=folder/(sid+'.response.json')
                if sha(path)!=event.get('response_sha256'):raise BoundaryError('Stored response changed')
                amount,rid,audit=paid_audit(json.loads(path.read_text()),task,parent)
                if (event.get('cost_nanousd')!=amount or amount>pending['reserved_nanousd']
                        or event.get('provider_response_id')!=rid or rid in state['provider_ids']
                        or event.get('audit')!=audit or event.get('target_structural_passed')!=(audit is not None)
                        or event.get('semantic_verified') is not False or event.get('approved_for_SFT') is not False):
                    raise BoundaryError('Stored response identity, cost or target audit changed')
                state['attempted']+=1;state['completed']+=1;state['valid']+=audit is not None
                state['spent_nanousd']+=amount;state['provider_ids'].append(rid);pending=None
            else:raise BoundaryError('Unexpected 5000 journal event')
        if pending is not None:raise BoundaryError('Interrupted request has unknown charge; no automatic retry')
        notify=lambda e:progress(e) if progress else None
        def summary(status):
            return {'status':status,'post_pilot_attempted':state['attempted'],
                'post_pilot_completed':state['completed'],'total_complete_candidates':500+state['completed'],
                'audited_unknown_requests_skipped':len(state['unavailable_ids']),
                'price_estimate_including_unknown_reservations_usd':state['spent_nanousd']/1e9,
                'budget_usd':40,'max_total_requests_including_pilot':5000,'maximum_complete_candidates':4999}
        notify(summary('resume'))
        rate=TokenWindow(load_profile(root,parent),resumed=bool(state['attempted']),clock=clock,sleep=sleep,progress=progress)
        stop='requested_limit_reached'
        for task in tasks[state['attempted']:limit_post_pilot]:
            body=request_body(task,parent);reserved=reservation(body,parent);held=reserved['microusd']*1000
            if state['spent_nanousd']+held>spec['cumulative_budget_nanousd']:stop='budget_reservation_limit';break
            sid=task['sample_id'];path=folder/(sid+'.response.json')
            if path.exists():raise BoundaryError('Unjournalled response exists; audit before paid calls')
            rate.acquire(reserved['input_tokens_bound']+body['max_completion_tokens'])
            ordinal=state['attempted']+1
            append({'kind':'reserve','sample_id':sid,'ordinal':ordinal,'reserved_nanousd':held,
                    'input_tokens_bound':reserved['input_tokens_bound'],'started_UTC':datetime.now(timezone.utc).isoformat()})
            started=clock();phase='transport'
            notify({'status':'request_started','post_pilot_ordinal':ordinal,'timeout_seconds':spec['timeout_seconds']})
            try:
                raw=transport(url,key,body,spec['timeout_seconds']);phase='persist_response'
                with path.open('x',encoding='utf-8') as handle:
                    json.dump(raw,handle,ensure_ascii=False,allow_nan=False);handle.flush();os.fsync(handle.fileno())
                phase='usage_and_identity';amount,rid,audit=paid_audit(raw,task,parent)
                if amount>held or rid in state['provider_ids']:raise BoundaryError('Invalid response cost or repeated identity')
            except Exception as error:
                status=getattr(error,'code',None)
                context={'phase':phase,'error_type':type(error).__name__,
                    'elapsed_seconds':round(max(0,clock()-started),3),
                    'timeout_seconds':spec['timeout_seconds'],'http_status':status if type(status) is int else None}
                append({'kind':'settle','sample_id':sid,'request_completed':False,
                    'cost_nanousd':held,'error_context':context,'semantic_verified':False,'approved_for_SFT':False})
                notify({'status':'stopped_unknown_charge','post_pilot_ordinal':ordinal,**context})
                raise BoundaryError('Request stopped at ordinal '+str(ordinal)+': '+context['error_type']+
                    ' during '+phase+'; full reservation retained; no automatic retry') from None
            append({'kind':'settle','sample_id':sid,'request_completed':True,
                'target_structural_passed':audit is not None,'audit':audit,'cost_nanousd':amount,
                'provider_response_id':rid,'response_sha256':sha(path),'elapsed_seconds':round(max(0,clock()-started),3),
                'semantic_verified':False,'approved_for_SFT':False})
            state['attempted']+=1;state['completed']+=1;state['valid']+=audit is not None
            state['spent_nanousd']+=amount;state['provider_ids'].append(rid)
            notify({**summary('completed'),'target_structural_passed':audit is not None,'quarantined':audit is None})
        return {**summary('finished'),'stop_reason':stop,'post_pilot_structural_passed':state['valid'],
                'invoice_verified':False,'student_training_runs':0,'automatic_further_expansion':False}
