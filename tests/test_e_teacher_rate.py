"""Pacing, same-journal resume and progress checks; no network or paid calls."""
import copy
import fcntl
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock,patch

from src.data.team_nq import BoundaryError,TeamNQBoundary
from src.distillation.e_prep import ROOT,teacher_task
from src.distillation import teacher_pilot as pilot
from src.distillation.teacher_rate import PROFILE_PATH,TokenWindow,load_profile
from tests.test_team_nq_boundary import fictional
from tests.test_e_distillation_prep import target


class Clock:
    def __init__(self):self.now=0.;self.sleeps=[]
    def __call__(self):return self.now
    def sleep(self,n):
        self.sleeps.append(n);self.now+=n


class TeacherRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=pilot.registration(ROOT)
        cls.profile=load_profile(ROOT,cls.spec)
        boundary=TeamNQBoundary(ROOT)
        ids=json.loads((ROOT/cls.spec['sample_manifest']).read_text())['sample_ids']['teacher_pilot'][:5]
        cls.tasks=[teacher_task(boundary,fictional(sid)) for sid in ids]

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.clock=Clock()
        for name in (pilot.REGISTRATION,PROFILE_PATH):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,path)
        self.env={'AZURE_OPENAI_ENDPOINT':'https://qzhu0160-4829-resource.services.ai.azure.com/openai/v1/',
                  'AZURE_OPENAI_API_KEY':'FICTIONAL_SECRET_MUST_NEVER_APPEAR'}

    def response(self):
        return {'model':self.spec['response_model'],'id':'fictional',
            'usage':{'prompt_tokens':1000,'completion_tokens':800},
            'choices':[{'finish_reason':'stop','message':{'content':target()}}]}

    def run_fake(self,transport,n=5,profile='token_window_v2',progress=None):
        with patch.object(pilot,'preflight',return_value=(self.spec,self.tasks)):
            return pilot.run(self.root,total_requests=n,environment=self.env,transport=transport,
                transport_profile=profile,clock=self.clock,sleep=self.clock.sleep,progress=progress)

    def test_registration_preserves_original_budget_and_cohort(self):
        self.assertEqual(self.profile['max_inflight'],1)
        self.assertFalse(self.profile['new_journal'])
        self.assertFalse(self.profile['request_body_changes'])
        self.assertEqual(self.profile['token_reservation_cap'],95000)
        self.assertEqual(self.spec['max_requests'],500)
        self.assertEqual(self.spec['pilot_budget_microusd'],15000000)

    def test_profile_or_parent_fingerprint_change_is_rejected(self):
        for key,value in [('token_reservation_cap',100001),('max_inflight',2),
                          ('automatic_retries',1),('request_body_changes',True),
                          ('parent_registration_sha256','0'*64)]:
            changed=copy.deepcopy(self.profile);changed[key]=value
            with patch('src.distillation.teacher_rate.read_json',return_value=changed):
                with self.assertRaises(BoundaryError):load_profile(ROOT,self.spec)
        changed=copy.deepcopy(self.spec);changed['configured_TPM']=90000
        with self.assertRaises(BoundaryError):load_profile(ROOT,changed)
        (self.root/pilot.REGISTRATION).write_text('{}')
        with self.assertRaises(BoundaryError):load_profile(self.root,self.spec)

    def test_every_rolling_window_obeys_token_and_request_caps(self):
        clock=Clock();rate=TokenWindow(self.profile,clock=clock,sleep=clock.sleep)
        sends=[]
        sizes=[15705+(i*7919)%7218 for i in range(500)]
        for tokens in sizes:
            stamp=rate.acquire(tokens);sends.append((stamp,tokens));clock.now+=4
        for i,(stamp,_) in enumerate(sends):
            window=[n for t,n in sends[:i+1] if t>stamp-61]
            self.assertLessEqual(sum(window),95000)
            self.assertLessEqual(len(window),60)
            if i:self.assertGreaterEqual(stamp-sends[i-1][0],5)
        self.assertLess(clock.now,500*61)
        self.assertLessEqual(max(clock.sleeps),30)

    def test_partial_window_does_not_release_unused_completion_tokens(self):
        clock=Clock();rate=TokenWindow(self.profile,clock=clock,sleep=clock.sleep)
        self.assertEqual(rate.acquire(40000),0)
        self.assertEqual(rate.acquire(40000),5)
        self.assertEqual(rate.acquire(40000),61)
        # The request still reserves 40k even if the service generates one token.
        self.assertEqual(rate.acquire(40000),66)

    def test_request_cap_spacing_and_boundary_expiry(self):
        spec=copy.deepcopy(self.profile);spec['request_cap']=2
        rate=TokenWindow(spec,clock=self.clock,sleep=self.clock.sleep)
        stamps=[rate.acquire(10) for _ in range(6)]
        self.assertEqual(stamps,[0,5,61,66,122,127])

    def test_resumed_process_waits_one_window_then_uses_new_pacing(self):
        rate=TokenWindow(self.profile,resumed=True,clock=self.clock,sleep=self.clock.sleep)
        self.assertEqual(rate.acquire(20000),61)
        self.assertEqual(rate.acquire(20000),66)
        self.assertEqual(rate.acquire(20000),71)
        self.assertLessEqual(max(self.clock.sleeps),30)

    def test_invalid_size_and_clock_rejected(self):
        rate=TokenWindow(self.profile,clock=self.clock,sleep=self.clock.sleep)
        for value in (0,-1,True,95001,1.5,float('nan')):
            with self.assertRaises(BoundaryError):rate.acquire(value)
        rate=TokenWindow(self.profile,clock=lambda:float('nan'),sleep=self.clock.sleep)
        with self.assertRaises(BoundaryError):rate.acquire(100)

    def test_legacy_completed_requests_are_reused_without_duplicate_billing(self):
        transport=Mock(side_effect=lambda *args:self.response())
        first=self.run_fake(transport,n=3,profile='legacy')
        self.assertEqual(first['requests'],3)
        before=self.clock.now
        result=self.run_fake(transport,n=5)
        self.assertEqual(result['requests'],5)
        self.assertEqual(transport.call_count,5)
        self.assertGreaterEqual(self.clock.now-before,66)
        self.assertLess(self.clock.now-before,122)
        self.assertEqual(result['accounting_usd_upper_estimate'],0.085)
        events=[json.loads(s) for s in (self.root/self.spec['run_directory']/'journal.jsonl').read_text().splitlines()]
        self.assertEqual([e['sample_id'] for e in events if e['kind']=='reserve'],[t['sample_id'] for t in self.tasks])
        profiles=[e for e in events if e['kind']=='transport_profile']
        self.assertEqual(len(profiles),1)
        self.assertEqual(profiles[0]['profile'],'token_window_v2')

    def test_v2_requests_use_exact_original_body_and_show_safe_progress(self):
        seen=[]
        def transport(endpoint,key,body,timeout):
            self.assertEqual(body,pilot.request_body(self.tasks[len(seen)],self.spec))
            seen.append(body);return self.response()
        progress=[]
        result=self.run_fake(transport,progress=progress.append)
        self.assertEqual(result['requests'],5)
        completed=[e for e in progress if e['status']=='completed']
        self.assertEqual([e['requests'] for e in completed],[1,2,3,4,5])
        self.assertNotIn('FICTIONAL_SECRET',json.dumps(progress))
        self.assertFalse(any(set(e)&{'messages','query','initial_answer','gold','content'} for e in progress))
        self.assertTrue(all(e['semantic_targets_approved']==0 for e in completed))

    def test_unknown_profile_failed_call_and_concurrent_lock_never_retry(self):
        transport=Mock(side_effect=TimeoutError())
        with self.assertRaises(BoundaryError):self.run_fake(transport,profile='unknown')
        transport.assert_not_called()
        with self.assertRaisesRegex(BoundaryError,'full reservation'):self.run_fake(transport)
        self.assertEqual(transport.call_count,1)
        with self.assertRaisesRegex(BoundaryError,'no automatic retry'):self.run_fake(transport)
        self.assertEqual(transport.call_count,1)

    def test_existing_process_lock_prevents_second_writer(self):
        folder=self.root/self.spec['run_directory'];folder.mkdir(parents=True)
        transport=Mock()
        with (folder/'run.lock').open('a') as handle:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):self.run_fake(transport)
        transport.assert_not_called()
        self.assertFalse((folder/'journal.jsonl').exists())


if __name__=='__main__':unittest.main()
