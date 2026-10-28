"""5000-cap extension, one audited timeout skip, and no paid retry or secret logs."""
import copy
from contextlib import redirect_stdout
import fcntl
import gzip
import io
import json
from pathlib import Path
import shutil
import tempfile
import tarfile
import unittest
from unittest.mock import Mock,patch
from urllib.error import HTTPError

from src.data.team_nq import BoundaryError,TeamNQBoundary,read_json,sha
from src.distillation.e_prep import ROOT,teacher_task
from src.distillation import expansion_5000_prep as prep,expansion_5000_collect as collect
from src.distillation.teacher_pilot import REGISTRATION,registration,request_body,reservation
from src.distillation.teacher_rate import PROFILE_PATH
from tests.test_team_nq_boundary import fictional
from tests.test_e_distillation_prep import target
from tests.test_e_teacher_rate import Clock


class Expansion5000PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=prep.load_plan(ROOT)
        cls.prior=read_json(ROOT/cls.spec['prior_plan']['path']);cls.parent=registration(ROOT)
        cls.old_sample=read_json(ROOT/cls.prior['sample_manifest'])
        archived_tasks=ROOT/'artifacts/controller/e_teacher_expansion_2000_v1/teacher_tasks.jsonl.gz'
        cls.old_tasks=[json.loads(s) for s in gzip.decompress(archived_tasks.read_bytes()).splitlines()]
        cls.boundary=TeamNQBoundary(ROOT)
        cls.sample=prep.manifest(cls.boundary,cls.spec,cls.old_sample)
        # Replay from committed archives, without requiring ignored live runtime files.
        cls.tmp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.tmp.cleanup);cls.prior_root=Path(cls.tmp.name)
        for item in ('prior_archive_manifest','prior_timeout_audit'):
            rel=cls.spec[item]['path'];p=cls.prior_root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
        rel=cls.prior['execution_receipt'];p=cls.prior_root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
        m=read_json(ROOT/cls.spec['prior_archive_manifest']['path'])
        rel=Path(cls.spec['prior_archive_manifest']['path']).parent/m['archive']
        shutil.copyfile(ROOT/rel,cls.prior_root/rel)
        source=cls.prior_root/cls.prior['run_directory'];source.mkdir(parents=True)
        with tarfile.open(ROOT/rel,'r:gz') as tar:
            for member in tar.getmembers():
                if Path(member.name).name!=member.name or not member.isfile():raise ValueError('Unsafe test archive')
                (source/member.name).write_bytes(tar.extractfile(member).read())

    def test_latest_authorisation_caps_all_requests_at5000_and_total_cost40(self):
        self.assertEqual(self.spec['max_post_pilot_requests'],4500)
        self.assertEqual(self.spec['cumulative_budget_nanousd'],40_000_000_000)
        self.assertEqual(self.spec['timeout_seconds'],600)
        self.assertEqual(self.spec['automatic_retries'],0)
        self.assertFalse(self.spec['training_enabled']);self.assertFalse(self.spec['automatic_further_expansion'])
        accepted=prep.bound(ROOT,self.spec['user_acceptance'])
        self.assertEqual(accepted['effective_user_instruction'],'可以直接上5000')
        self.assertEqual(accepted['cumulative_budget_usd'],40)

    def test_budget_retry_timeout_and_training_mutations_are_rejected(self):
        for key,value in [('max_post_pilot_requests',4501),('cumulative_budget_nanousd',41000000000),
                          ('automatic_retries',1),('training_enabled',True),('timeout_seconds',180)]:
            spec=copy.deepcopy(self.spec);spec[key]=value
            with patch.object(prep,'read_json',side_effect=lambda p:spec if Path(p)==ROOT/prep.PLAN else read_json(p)):
                with self.assertRaises(BoundaryError):prep.load_plan(ROOT)

    def test_4500_ids_preserve_1500_prefix_and_are_train_only_not_test_validation_or_pilot(self):
        ids=self.sample['post_pilot_ids'];self.assertEqual(len(ids),4500)
        self.assertEqual(len(set(ids)),4500);self.assertEqual(ids[:1500],self.old_sample['new_ids'])
        self.assertEqual(set(self.sample['post_pilot_AI_strata'].values()),{1125})
        self.boundary.require(ids,'train')
        for blocked in (self.sample['reused_pilot_ids'],self.boundary.splits['validation'],
                        self.boundary.splits['test'],self.boundary.legacy):self.assertFalse(set(ids)&set(blocked))
        self.assertEqual(self.sample['maximum_complete_candidates_after_skip'],4999)
        self.assertFalse(self.sample['sampling_runtime_or_gold_read']);self.assertFalse(self.sample['ready_for_SFT'])

    def test_real_54_responses_replayed_and_unknown_full_charge_not_refunded(self):
        state=prep.prior_state(self.prior_root,self.spec,self.prior,self.parent,self.old_sample,self.old_tasks)
        self.assertEqual(state['attempted'],55);self.assertEqual(state['completed'],54)
        self.assertEqual(state['valid'],54);self.assertEqual(state['spent_nanousd'],3916639000)
        self.assertEqual(state['unavailable_ids'],['-5176122245177684084'])
        self.assertEqual(len(set(state['provider_ids'])),54)

    def test_skip_cannot_retry_refund_replace_or_change_unknown_ordinal(self):
        for key,value in [('retry',True),('retain_nanousd',0),('replace_sample',True),('ordinal',54)]:
            spec=copy.deepcopy(self.spec);spec['audited_skip'][key]=value
            with self.assertRaises(BoundaryError):
                prep.prior_state(self.prior_root,spec,self.prior,self.parent,self.old_sample,self.old_tasks)

    def test_changed_prior_journal_is_detected_before_import(self):
        spec=copy.deepcopy(self.spec);spec['prior_journal']['sha256']='0'*64
        with self.assertRaises(BoundaryError):
            prep.prior_state(self.prior_root,spec,self.prior,self.parent,self.old_sample,self.old_tasks)

    def test_frozen_real_5000_export_and_client_assets_match(self):
        sample=read_json(ROOT/self.spec['sample_manifest'])
        self.assertEqual(sample,self.sample)
        receipt=read_json(ROOT/self.spec['execution_receipt'])
        self.assertEqual(receipt['n_post_pilot_tasks'],4500)
        self.assertEqual(receipt['plan_sha256'],sha(ROOT/prep.PLAN))
        self.assertEqual(receipt['sample_manifest_sha256'],sha(ROOT/self.spec['sample_manifest']))
        self.assertEqual(receipt['assets'],{p:sha(ROOT/p) for p in prep.ASSETS})
        self.assertFalse(receipt['ready_for_SFT']);self.assertFalse(receipt['official_test_runtime_read'])
        self.assertEqual(receipt['new_API_calls'],0)


class Expansion5000CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=prep.load_plan(ROOT);cls.parent=registration(ROOT)
        b=TeamNQBoundary(ROOT);ids=b.allowed_ids('train')[:4]
        cls.tasks=[teacher_task(b,fictional(s)) for s in ids]

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.spec=copy.deepcopy(type(self).spec);self.clock=Clock();self.serial=0
        for name in (prep.PLAN,REGISTRATION,PROFILE_PATH,self.spec['prior_plan']['path']):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
        p=self.root/self.spec['execution_receipt'];p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        self.prior=read_json(self.root/self.spec['prior_plan']['path'])
        self.old_folder=self.root/self.prior['run_directory'];self.old_folder.mkdir(parents=True)
        journal=self.root/self.spec['prior_journal']['path'];journal.write_text('fictional original journal')
        self.spec['prior_journal']['sha256']=sha(journal)
        self.base={'attempted':1,'completed':0,'valid':0,'spent_nanousd':3916639000,
                   'provider_ids':['old-response'],'unavailable_ids':[self.tasks[0]['sample_id']]}
        self.env={'AZURE_OPENAI_ENDPOINT':'https://qzhu0160-4829-resource.services.ai.azure.com/openai/v1/',
                  'AZURE_OPENAI_API_KEY':'FICTIONAL_SECRET_NEVER_LOG'}

    def response(self,*args):
        self.serial+=1
        return {'model':self.parent['response_model'],'service_tier':'default','id':'fiction-'+str(self.serial),
            'usage':{'prompt_tokens':1000,'completion_tokens':100},
            'choices':[{'finish_reason':'stop','message':{'content':target()}}]}

    def run_fake(self,transport=None,n=4,progress=None):
        with patch.object(collect,'preflight',return_value=(self.spec,self.parent,
                {'post_pilot_ids':[t['sample_id'] for t in self.tasks]},self.tasks,self.base)):
            return collect.run(self.root,limit_post_pilot=n,environment=self.env,transport=transport or self.response,
                clock=self.clock,sleep=self.clock.sleep,progress=progress)

    def journal(self):return self.root/self.spec['run_directory']/'journal.jsonl'
    def events(self):return [json.loads(s) for s in self.journal().read_text().splitlines()]

    def test_recovery_does_not_resend_skipped_ID_and_preserves_budget_base(self):
        network=Mock(side_effect=self.response);progress=[]
        first=self.run_fake(network,n=2,progress=progress.append)
        self.assertEqual(network.call_count,1)
        self.assertEqual(network.call_args.args[2],request_body(self.tasks[1],self.parent))
        self.assertEqual(network.call_args.args[3],600)
        self.assertEqual(first['post_pilot_attempted'],2)
        self.assertEqual(first['total_complete_candidates'],501)
        self.assertAlmostEqual(first['price_estimate_including_unknown_reservations_usd'],3.916639+.003)
        result=self.run_fake(network,n=4,progress=progress.append)
        self.assertEqual(network.call_count,3);self.assertEqual(result['post_pilot_attempted'],4)
        for call,task in zip(network.call_args_list,self.tasks[1:],strict=True):
            self.assertEqual(call.args[2],request_body(task,self.parent))
        again=Mock();self.run_fake(again,n=4);again.assert_not_called()
        self.assertNotIn(self.env['AZURE_OPENAI_API_KEY'],json.dumps(self.events()+progress))
        self.assertFalse(result['automatic_further_expansion']);self.assertEqual(result['student_training_runs'],0)

    def test_future_timeout_stops_and_remains_blocked_instead_of_automatically_skipping(self):
        network=Mock(side_effect=TimeoutError(self.env['AZURE_OPENAI_API_KEY']));progress=[]
        with self.assertRaisesRegex(BoundaryError,'TimeoutError during transport'):self.run_fake(network,progress=progress.append)
        event=self.events()[-1]
        self.assertFalse(event['request_completed']);self.assertEqual(event['cost_nanousd'],self.events()[-2]['reserved_nanousd'])
        self.assertEqual(event['error_context']['timeout_seconds'],600)
        self.assertEqual(event['error_context']['phase'],'transport')
        self.assertNotIn(self.env['AZURE_OPENAI_API_KEY'],json.dumps(self.events()+progress))
        retry=Mock()
        with self.assertRaises(BoundaryError):self.run_fake(retry)
        retry.assert_not_called()

    def test_http_error_records_status_without_response_body_exception_text_or_key(self):
        secret=self.env['AZURE_OPENAI_API_KEY']
        network=Mock(side_effect=HTTPError('https://fiction.invalid',429,secret,None,io.BytesIO(secret.encode())))
        with self.assertRaises(BoundaryError):self.run_fake(network)
        self.assertEqual(self.events()[-1]['error_context']['http_status'],429)
        self.assertNotIn(secret,self.journal().read_text())

    def test_known_paid_invalid_target_is_quarantined_without_extra_sampling(self):
        def network(*args):
            raw=self.response();raw['choices'][0]['message']['content']='not JSON';return raw
        result=self.run_fake(network,n=2)
        self.assertEqual(result['post_pilot_completed'],1);self.assertEqual(result['post_pilot_structural_passed'],0)
        self.assertTrue(self.events()[-1]['request_completed']);self.assertFalse(self.events()[-1]['approved_for_SFT'])

    def test_wrong_model_is_not_accepted_as_a_billed_valid_target(self):
        def network(*args):
            raw=self.response();raw['model']='wrong-version';return raw
        with self.assertRaises(BoundaryError):self.run_fake(network)
        self.assertEqual(self.events()[-1]['error_context']['phase'],'usage_and_identity')
        retry=Mock()
        with self.assertRaises(BoundaryError):self.run_fake(retry)
        retry.assert_not_called()

    def test_budget40_includes_prior_unknown_and_reserves_before_network(self):
        held=reservation(request_body(self.tasks[1],self.parent),self.parent)['microusd']*1000
        self.base['spent_nanousd']=40_000_000_000-held+1;network=Mock()
        result=self.run_fake(network);network.assert_not_called()
        self.assertEqual(result['stop_reason'],'budget_reservation_limit')
        self.assertEqual(result['post_pilot_attempted'],1)

    def test_limits_cannot_replace_unknown_or_exceed_4500(self):
        for n in (0,4501,5000,True,1.0):
            with self.assertRaises(BoundaryError):self.run_fake(n=n)

    def test_prior_and_current_writer_locks_prevent_concurrent_calls(self):
        for folder in (self.old_folder,self.root/self.spec['run_directory']):
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/'run.lock').open('a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);network=Mock()
                with self.assertRaises(BlockingIOError):self.run_fake(network)
                network.assert_not_called()

    def test_new_unknown_crash_and_modified_prior_journal_are_not_retried(self):
        self.run_fake(n=2)
        events=self.events();self.journal().write_text(''.join(json.dumps(e)+'\n' for e in events[:-1]))
        network=Mock()
        with self.assertRaises(BoundaryError):self.run_fake(network)
        network.assert_not_called()
        self.journal().write_text(''.join(json.dumps(e)+'\n' for e in events))
        (self.root/self.spec['prior_journal']['path']).write_text('changed')
        with self.assertRaises(BoundaryError):self.run_fake(network)
        network.assert_not_called()

    def test_raw_and_cost_or_identity_tampering_stop_before_more_calls(self):
        self.run_fake(n=2);original=self.events()
        for key,value in [('cost_nanousd',0),('provider_response_id','wrong'),('approved_for_SFT',True)]:
            events=copy.deepcopy(original);events[-1][key]=value
            self.journal().write_text(''.join(json.dumps(e)+'\n' for e in events));network=Mock()
            with self.assertRaises(BoundaryError):self.run_fake(network)
            network.assert_not_called()
        self.journal().write_text(''.join(json.dumps(e)+'\n' for e in original))
        raw=self.root/self.spec['run_directory']/(self.tasks[1]['sample_id']+'.response.json');raw.write_text(raw.read_text()+' ')
        with self.assertRaises(BoundaryError):self.run_fake()


class Expansion5000CommandTests(unittest.TestCase):
    def test_default_is_offline_and_hidden_key_needs_a_terminal(self):
        from scripts import run_e_teacher_5000 as cli
        fake=({}, {}, {}, [1]*4500, {'completed':54,'attempted':55})
        with patch.object(cli,'preflight',return_value=fake),patch.object(cli,'run') as run,redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main([]),0);run.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['new_API_calls'],0)
        with patch.object(cli,'preflight',return_value=fake),patch.object(cli.sys.stdin,'isatty',return_value=False), \
             patch.object(cli.getpass,'getpass') as getpass,patch.object(cli,'run') as run,redirect_stdout(io.StringIO()):
            with self.assertRaises(BoundaryError):cli.main(['--execute','--prompt-key'])
            getpass.assert_not_called();run.assert_not_called()


if __name__=='__main__':unittest.main()
