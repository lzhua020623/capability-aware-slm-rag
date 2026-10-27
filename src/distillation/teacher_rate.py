"""Serial token-window pacing; independent of prompts, cost and target acceptance."""
from collections import deque
import math
from pathlib import Path
import time

from src.data.team_nq import BoundaryError, read_json, sha

PROFILE_PATH = 'configs/distillation/e_teacher_transport_v2.json'


def load_profile(root, registration):
    root = Path(root)
    value = read_json(root / PROFILE_PATH)
    expected = {'schema_version':'e_teacher_transport_v2', 'profile':'token_window_v2',
        'parent_registration':'configs/distillation/e_teacher_sol_pilot_v1.json',
        'parent_registration_sha256':'209e5298371262070a4226fee7b7e16c71573bbddfc8a3ca1bf74e4d9b8235e7',
        'rolling_window_seconds':61, 'token_reservation_cap':95000, 'request_cap':60,
        'minimum_start_interval_seconds':5, 'max_inflight':1, 'resume_cooldown_seconds':61,
        'unused_output_reservation_returned_to_rate_window':False, 'automatic_retries':0,
        'new_journal':False, 'new_sample_ids':False, 'request_body_changes':False,
        'model_prompt_budget_or_quality_gate_changes':False,
        'record_profile_and_source_hash_before_network':True}
    if any(value.get(k) != v for k,v in expected.items()):
        raise BoundaryError('Unregistered teacher transport profile')
    if sha(root/value['parent_registration']) != value['parent_registration_sha256']:
        raise BoundaryError('Teacher parent registration fingerprint changed')
    if value['token_reservation_cap'] > registration['configured_TPM']:
        raise BoundaryError('Transport exceeds the observed deployment quota')
    return value


class TokenWindow:
    """Reserve maximum input+completion in a rolling window, not billed usage.

    The caller sends serially and keeps all existing cost and failure guards.
    A fresh process waits a full window on resume because old journal entries
    do not contain trustworthy monotonic send times.
    """
    def __init__(self, profile, *, resumed=False, clock=time.monotonic,
                 sleep=time.sleep, progress=None):
        self.profile, self.clock, self.sleep, self.progress = profile, clock, sleep, progress
        self.starts = deque()
        self.not_before = clock() + (profile['resume_cooldown_seconds'] if resumed else 0)
        self.last = None

    def acquire(self, tokens):
        spec = self.profile
        if type(tokens) is not int or not 0 < tokens <= spec['token_reservation_cap']:
            raise BoundaryError('Request exceeds the registered rolling token window')
        reported = False
        while True:
            now = self.clock()
            if not math.isfinite(now):
                raise BoundaryError('Non-finite rate clock')
            while self.starts and self.starts[0][0] <= now-spec['rolling_window_seconds']:
                self.starts.popleft()
            ready = self.not_before
            if self.last is not None:
                ready = max(ready, self.last+spec['minimum_start_interval_seconds'])
            if (len(self.starts) >= spec['request_cap']
                    or sum(n for _,n in self.starts)+tokens > spec['token_reservation_cap']):
                ready = max(ready,self.starts[0][0]+spec['rolling_window_seconds'])
            if now >= ready:
                self.starts.append((now,tokens));self.last=now
                return now
            if self.progress and not reported:
                self.progress({'status':'rate_wait','wait_seconds':round(ready-now,2)})
                reported=True
            self.sleep(min(30,ready-now))
