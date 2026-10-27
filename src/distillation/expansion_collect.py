"""Serial, resumable 1500-new-task collector under the authorised USD20 estimate."""
import fcntl
import json
import os
from pathlib import Path
import time

from src.data.team_nq import BoundaryError,sha
from src.distillation.e_prep import ROOT
from src.distillation.expansion_prep import PLAN,preflight
from src.distillation.teacher_pilot import endpoint,request_body,reservation,usage_cost,audit_response,send
from src.distillation.teacher_cost import response_price
from src.distillation.teacher_rate import load_profile,TokenWindow


def nano(amount):
    value=amount*1000000000
    if value!=value.to_integral_value():raise BoundaryError('Currency precision exceeds nanodollar accounting')
    return int(value)


def run(root=ROOT,*,limit_new=1500,environment=os.environ,transport=send,
        clock=time.monotonic,sleep=time.sleep,progress=None):
    root=Path(root);plan,parent,sample,tasks,receipt=preflight(root)
    if type(limit_new) is not int or not 1<=limit_new<=1500:
        raise BoundaryError('This wave authorises at most 1500 new requests, 2000 including reuse')
    url=endpoint(parent,environment);key=environment.get(parent['key_environment'])
    if not isinstance(key,str) or not key.strip():raise BoundaryError('Azure API key missing; use local hidden input')
    folder=root/plan['run_directory'];folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=folder/'journal.jsonl'
        events=[json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        seal={'kind':'seal','schema_version':'e_expansion_2000_journal_v1','plan_sha256':sha(root/PLAN),
            'execution_receipt_sha256':sha(root/plan['execution_receipt']),
            'ordered_new_ids':sample['new_ids'],'reused_pilot_n':500,
            'pilot_price_estimate_nanousd':plan['pilot_price_estimate_nanousd'],
            'budget_nanousd':plan['cumulative_budget_nanousd']}
        if events and events[0]!=seal:raise BoundaryError('Expansion journal belongs to another frozen wave')
        def append(event):
            with journal.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            events.append(event)
        if not events:append(seal)
        completed=[];pending=None;spent=plan['pilot_price_estimate_nanousd'];valid=0;provider_ids=set()
        for event in events[1:]:
            if event['kind']=='reserve':
                i=len(completed)
                if pending is not None or i>=len(tasks) or event['sample_id']!=tasks[i]['sample_id']:
                    raise BoundaryError('Repeated or out-of-order expansion request')
                expected=reservation(request_body(tasks[i],parent),parent)
                if (type(event.get('reserved_nanousd')) is not int
                        or event['reserved_nanousd']!=expected['microusd']*1000
                        or event.get('input_tokens_bound')!=expected['input_tokens_bound']):
                    raise BoundaryError('Expansion cost reservation changed')
                pending=event
            elif event['kind']=='settle':
                if pending is None or event['sample_id']!=pending['sample_id'] or not event.get('request_completed'):
                    raise BoundaryError('Incomplete/unknown-charge request; manual audit required, no retry')
                task=tasks[len(completed)];path=folder/(event['sample_id']+'.response.json')
                if sha(path)!=event.get('response_sha256'):raise BoundaryError('Stored raw response changed')
                raw=json.loads(path.read_text())
                usage_cost(raw,reservation(request_body(task,parent),parent),parent)
                actual=nano(response_price(raw)[0])
                if (type(event.get('cost_nanousd')) is not int or actual!=event['cost_nanousd']
                        or actual>pending['reserved_nanousd']):
                    raise BoundaryError('Stored response charge differs from journal')
                rid=raw.get('id')
                if (not isinstance(rid,str) or not rid or rid in provider_ids
                        or event.get('provider_response_id')!=rid
                        or event.get('semantic_verified') is not False or event.get('approved_for_SFT') is not False):
                    raise BoundaryError('Response identity/approval changed')
                try:audit=audit_response(raw,task,parent)
                except Exception:audit=None
                if bool(audit)!=event['target_structural_passed'] or audit!=event.get('audit'):
                    raise BoundaryError('Stored target audit differs from raw response')
                if spent+pending['reserved_nanousd']>plan['cumulative_budget_nanousd']:
                    raise BoundaryError('Prior request exceeded cumulative budget reservation')
                spent+=actual;valid+=bool(audit);provider_ids.add(rid);completed.append(event);pending=None
            else:raise BoundaryError('Unexpected expansion journal event')
        if pending is not None:raise BoundaryError('Interrupted unknown-charge request; no automatic retry')
        notify=lambda event:progress(event) if progress else None
        notify({'status':'resume','reused_pilot':500,'new_completed':len(completed),
            'total_candidates_completed':500+len(completed),'new_request_cap':1500,
            'published_price_estimate_total_usd':spent/1e9,'budget_usd':20})
        rate=TokenWindow(load_profile(root,parent),resumed=bool(completed),clock=clock,sleep=sleep,progress=progress)
        stopped='requested_limit_reached'
        for task in tasks[len(completed):limit_new]:
            body=request_body(task,parent);reserve=reservation(body,parent)
            held=reserve['microusd']*1000
            if spent+held>plan['cumulative_budget_nanousd']:
                stopped='budget_reservation_limit';break
            sid=task['sample_id'];path=folder/(sid+'.response.json')
            if path.exists():raise BoundaryError('Unjournalled response exists; audit before any paid request')
            rate.acquire(reserve['input_tokens_bound']+body['max_completion_tokens'])
            append({'kind':'reserve','sample_id':sid,
                'reserved_nanousd':held,'input_tokens_bound':reserve['input_tokens_bound']})
            try:
                response=transport(url,key,body,parent['timeout_seconds'])
                with path.open('x',encoding='utf-8') as handle:
                    json.dump(response,handle,ensure_ascii=False,allow_nan=False);handle.flush();os.fsync(handle.fileno())
                usage_cost(response,reserve,parent)
                actual=nano(response_price(response)[0])
                if actual>held:raise BoundaryError('Published charge exceeded conservative reservation')
                rid=response.get('id')
                if not isinstance(rid,str) or not rid or rid in provider_ids:
                    raise BoundaryError('Missing or repeated provider response ID')
            except Exception as error:
                append({'kind':'settle','sample_id':sid,'request_completed':False,
                    'target_structural_passed':False,'cost_nanousd':held,
                    'error_type':type(error).__name__,'approved_for_SFT':False})
                raise BoundaryError('Request failed/unknown charge; reservation retained; no automatic retry') from None
            try:
                audit=audit_response(response,task,parent);error_type=None
            except Exception as error:
                audit=None;error_type=type(error).__name__
            event={'kind':'settle','sample_id':sid,'request_completed':True,
                'target_structural_passed':audit is not None,'audit':audit,'cost_nanousd':actual,
                'provider_response_id':rid,'response_sha256':sha(path),'quarantine_error_type':error_type,
                'semantic_verified':False,'approved_for_SFT':False}
            append(event);provider_ids.add(rid);completed.append(event);spent+=actual;valid+=audit is not None
            notify({'status':'completed','new_completed':len(completed),
                'total_candidates_completed':500+len(completed),'target_total':2000,
                'target_structural_passed':audit is not None,'quarantined':audit is None,
                'published_price_estimate_total_usd':spent/1e9,'budget_usd':20})
        return {'new_requests_completed':len(completed),'reused_pilot':500,
            'total_candidates_completed':500+len(completed),'new_structural_passed':valid,
            'new_quarantined':len(completed)-valid,'published_price_estimate_total_usd':spent/1e9,
            'budget_usd':20,'stop_reason':stopped,'invoice_verified':False,
            'student_training_runs':0,'further_expansion_automatic':False}
