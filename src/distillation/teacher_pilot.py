"""Budgeted, train-only Azure teacher pilot. No training or automatic retries.

The journal is written before network transmission. Unknown charges remain fully
reserved. The accounting envelope is conservative, not an Azure billing export.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit

from src.data.team_nq import BoundaryError, TeamNQBoundary, read_json, sha
from src.distillation.e_prep import (ROOT, load_protocol, teacher_task,
    validate_sample_manifest, audit_teacher_output, digest)
from src.distillation.teacher_rate import PROFILE_PATH, TokenWindow, load_profile

REGISTRATION = 'configs/distillation/e_teacher_sol_pilot_v1.json'


def registration(root=ROOT):
    spec = read_json(Path(root) / REGISTRATION)
    if (spec['schema_version'] != 'e_teacher_sol_pilot_v1'
            or spec['model'] != 'gpt-6.1-sol' or spec['version'] != '2026-09-29'
            or spec['response_model'] != 'gpt-6.1-sol-2026-09-29'
            or spec['deployment_type'] != 'GlobalStandard'
            or spec['pilot_budget_microusd'] != 15_000_000
            or spec['max_requests'] != 500 or spec['automatic_retries'] != 0
            or spec['max_completion_tokens_including_reasoning'] != 4096
            or spec['accounting_input_usd_per_million'] != 5
            or spec['accounting_output_usd_per_million'] != 15
            or spec['training_enabled'] is not False or spec['expansion_enabled'] is not False):
        raise BoundaryError('Unregistered teacher execution configuration')
    return spec


def preflight(root=ROOT):
    """Check ALL tasks before network; never open offline references or test text."""
    root = Path(root)
    spec, boundary, protocol = registration(root), TeamNQBoundary(root), load_protocol(root)
    manifest_path = root / spec['sample_manifest']
    if sha(manifest_path) != spec['sample_manifest_sha256']:
        raise BoundaryError('Teacher sample manifest fingerprint changed')
    manifest = validate_sample_manifest(boundary, protocol, read_json(manifest_path))
    path = root / spec['tasks']
    if sha(path) != spec['tasks_sha256']:
        raise BoundaryError('Teacher task file fingerprint changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    wanted = manifest['sample_ids']['teacher_pilot']
    if [row['sample_id'] for row in rows] != wanted:
        raise BoundaryError('Teacher tasks differ from frozen train pilot')
    for row in rows:
        if row != teacher_task(boundary, row['runtime']):
            raise BoundaryError('Teacher runtime, prompt, schema or fingerprint changed')
    # Same IDs as preparation; interleave so a budget stop is not one AI stratum.
    strata = protocol['sampling']['strata']
    pools = [[row for row in rows if boundary.labels[row['sample_id']] == label] for label in strata]
    ordered = [row for group in zip(*pools, strict=True) for row in group]
    return spec, ordered


def endpoint(spec, environment=os.environ):
    value = environment.get(spec['endpoint_environment'], '').rstrip('/') + '/'
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port
            or parsed.query or parsed.fragment or parsed.path != '/openai/v1/'
            or hashlib.sha256(value.encode()).hexdigest() != spec['endpoint_sha256']):
        raise BoundaryError('Azure endpoint missing or differs from the observed teacher resource')
    return value


def request_body(task, spec):
    return {'model': spec['deployment'], 'messages': task['teacher_messages'],
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'e_teacher_v1', 'strict': True, 'schema': task['teacher_output_schema']}},
        'reasoning_effort': spec['reasoning_effort'],
        'max_completion_tokens': spec['max_completion_tokens_including_reasoning'],
        'store': False, 'stream': False}


def reservation(body, spec):
    input_bound = len(json.dumps(body, ensure_ascii=False).encode()) + 4096
    if input_bound > 90000:
        raise BoundaryError('Teacher request exceeds registered text/token envelope')
    return {'input_tokens_bound': input_bound,
        'microusd': input_bound * spec['accounting_input_usd_per_million']
        + body['max_completion_tokens'] * spec['accounting_output_usd_per_million']}


def usage_cost(response, reserved, spec):
    usage = response['usage']
    prompt, completion = usage['prompt_tokens'], usage['completion_tokens']
    if (type(prompt) is not int or type(completion) is not int
            or not 0 <= prompt <= reserved['input_tokens_bound']
            or not 0 <= completion <= spec['max_completion_tokens_including_reasoning']):
        raise BoundaryError('Provider usage exceeds reservation or is invalid')
    # completion_tokens already includes reasoning: do not count it twice.
    return prompt * spec['accounting_input_usd_per_million'] + completion * spec['accounting_output_usd_per_million']


def audit_response(response, task, spec):
    if response.get('model') != spec['response_model']:
        raise BoundaryError('Teacher model version mismatch; stop before accepting targets')
    choices = response.get('choices', [])
    if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
        raise BoundaryError('Teacher output truncated, filtered or incomplete')
    message = choices[0]['message']
    if message.get('refusal') or message.get('tool_calls'):
        raise BoundaryError('Teacher refused or attempted tools')
    text = message.get('content')
    if not isinstance(text, str):
        raise BoundaryError('Teacher has no visible JSON output')
    return audit_teacher_output(task['runtime'], text)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BoundaryError('Azure redirect refused; no credentials forwarded')


def send(base_url, key, body, timeout):
    req = urllib.request.Request(base_url + 'chat/completions',
        data=json.dumps(body, ensure_ascii=False).encode(), method='POST',
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    # No SDK retries, redirects, tools or implicit environment proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def run(root=ROOT, *, total_requests=3, environment=os.environ, transport=send,
        clock=time.monotonic, sleep=time.sleep, transport_profile='legacy', progress=None):
    spec, tasks = preflight(root)
    if type(total_requests) is not int or not 1 <= total_requests <= spec['max_requests']:
        raise BoundaryError('Only 1-500 cumulative pilot requests are registered')
    if transport_profile not in ('legacy','token_window_v2'):
        raise BoundaryError('Unknown teacher transport profile')
    rate_spec = load_profile(root,spec) if transport_profile == 'token_window_v2' else None
    # Check the complete cohort's rate envelope before any credentials/network.
    if rate_spec:
        for task in tasks:
            bound=reservation(request_body(task,spec),spec)['input_tokens_bound']
            if bound+spec['max_completion_tokens_including_reasoning'] > rate_spec['token_reservation_cap']:
                raise BoundaryError('Frozen pilot exceeds the transport token envelope')
    base_url = endpoint(spec, environment)
    key = environment.get(spec['key_environment'])
    if not isinstance(key, str) or not key.strip():
        raise BoundaryError('AZURE_OPENAI_API_KEY missing; never paste a key into chat')
    folder = Path(root) / spec['run_directory']
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (folder / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = folder / 'journal.jsonl'
        events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        seal = {'kind': 'seal', 'registration': digest(spec),
            'tasks': spec['tasks_sha256'], 'ordered_ids': [t['sample_id'] for t in tasks]}
        if events and events[0] != seal:
            raise BoundaryError('Pilot journal belongs to another registration/cohort')

        def append(event):
            with journal.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush(); os.fsync(handle.fileno())
            events.append(event)

        if not events:
            append(seal)
        reserved_events = [event for event in events if event['kind'] == 'reserve']
        settlements = {event['sample_id']: event for event in events if event['kind'] == 'settle'}
        for event in reserved_events:
            if event['sample_id'] not in settlements or not settlements[event['sample_id']]['accepted']:
                raise BoundaryError('Earlier incomplete/failed request: stop; no automatic retry')
        if len(reserved_events) != len(settlements) or len(settlements) != len(set(e['sample_id'] for e in reserved_events)):
            raise BoundaryError('Pilot journal has inconsistent request accounting')
        expected = [task['sample_id'] for task in tasks[:len(reserved_events)]]
        if [event['sample_id'] for event in reserved_events] != expected:
            raise BoundaryError('Pilot journal order differs from frozen task order')
        spent = sum(event['microusd'] for event in settlements.values())
        if progress:
            progress({'status':'resume','requests':len(reserved_events),'target_requests':total_requests,
                'transport_profile':transport_profile,'accounting_usd_upper_estimate':spent/1_000_000})
        rate = None
        if rate_spec:
            append({'kind':'transport_profile','profile':transport_profile,
                'profile_sha256':sha(Path(root)/PROFILE_PATH),
                'client_sha256':sha(Path(__file__)),
                'scheduler_sha256':sha(Path(__file__).with_name('teacher_rate.py')),
                'request_body_unchanged':True})
            rate = TokenWindow(rate_spec,resumed=bool(reserved_events),clock=clock,
                               sleep=sleep,progress=progress)
        # A resumed process conservatively starts a fresh minute before sending.
        last_start = clock() if reserved_events else None
        for task in tasks[len(reserved_events):total_requests]:
            body = request_body(task, spec)
            reserve = reservation(body, spec)
            if spent + reserve['microusd'] > spec['pilot_budget_microusd']:
                break
            if rate is not None:
                rate.acquire(reserve['input_tokens_bound']+body['max_completion_tokens'])
            elif last_start is not None:
                remaining = 61 - (clock() - last_start)
                while remaining > 0:
                    sleep(min(30, remaining))  # deliberately <= one request / minute
                    remaining = 61 - (clock() - last_start)
            sid = task['sample_id']
            append({'kind': 'reserve', 'sample_id': sid, **reserve})
            last_start = clock()
            response = None
            try:
                response = transport(base_url, key, body, spec['timeout_seconds'])
                # Persist full provider response locally before parsing; never headers/key.
                with (folder / (sid + '.response.json')).open('x', encoding='utf-8') as handle:
                    json.dump(response, handle, ensure_ascii=False, allow_nan=False)
                actual = usage_cost(response, reserve, spec)
                audit = audit_response(response, task, spec)
            except Exception as error:
                # We cannot certify the provider charge: reservation stays spent.
                append({'kind': 'settle', 'sample_id': sid, 'microusd': reserve['microusd'],
                    'accepted': False, 'error_type': type(error).__name__,
                    'semantic_verified': False, 'approved_for_SFT': False})
                raise BoundaryError('Teacher request failed; full reservation retained; no retry') from None
            append({'kind': 'settle', 'sample_id': sid, 'microusd': actual,
                'accepted': True, 'provider_response_id': response.get('id'), 'audit': audit})
            spent += actual
            if progress:
                progress({'status':'completed','requests':sum(e['kind']=='reserve' for e in events),
                    'target_requests':total_requests,'verification':audit['model_label'],
                    'action':audit['candidate_action'],'accounting_usd_upper_estimate':spent/1_000_000,
                    'semantic_targets_approved':0})
        return {'requests': sum(e['kind'] == 'reserve' for e in events),
            'accounting_usd_upper_estimate': spent / 1_000_000,
            'budget_usd': spec['pilot_budget_microusd'] / 1_000_000,
            'semantic_targets_approved': 0, 'training_runs': 0}
