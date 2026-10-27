"""2000-candidate scope, role isolation and paid-call crash safety; no API calls."""
from collections import Counter
from contextlib import redirect_stdout
import copy
from decimal import Decimal
import fcntl
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock,patch

from src.data.team_nq import BoundaryError,TeamNQBoundary,read_json,sha
from src.distillation.e_prep import ROOT,teacher_task
from src.distillation import expansion_prep as prep,expansion_collect as collect
from src.distillation.teacher_pilot import REGISTRATION,request_body,reservation
from src.distillation.teacher_rate import PROFILE_PATH
from tests.test_team_nq_boundary import fictional
from tests.test_e_distillation_prep import target
from tests.test_e_teacher_rate import Clock


class ExpansionPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan,cls.parent=prep.load_plan(ROOT)
        cls.boundary=TeamNQBoundary(ROOT);cls.sample=prep.manifest(cls.boundary,cls.plan)

    def test_latest_instruction_authorises_only_1500_new_under_20_total(self):
        self.assertEqual(self.plan['first_wave_total_candidates'],2000)
        self.assertEqual(self.plan['max_new_requests'],1500)
        self.assertEqual(self.plan['cumulative_budget_nanousd'],20_000_000_000)
        self.assertFalse(self.plan['automatic_further_expansion'])
        acceptance=prep.bound(ROOT,self.plan['budget_acceptance'])
        self.assertEqual(acceptance['effective_user_instruction'],'先2000条 有效再扩量')
        self.assertFalse(acceptance['training_enabled_by_this_record'])

    def test_budget_model_release_and_training_changes_fail_before_API(self):
        for key,value in [('max_new_requests',4500),('first_wave_total_candidates',5000),
                          ('cumulative_budget_nanousd',40_000_000_000),('cumulative_budget_usd',40),
                          ('training_enabled',True),('automatic_further_expansion',True),
                          ('candidate_collection_enabled',1)]:
            plan=copy.deepcopy(self.plan);plan[key]=value
            def changed(path):return plan if Path(path)==ROOT/prep.PLAN else read_json(path)
            with patch.object(prep,'read_json',side_effect=changed):
                with self.assertRaises(BoundaryError):prep.load_plan(ROOT)
        plan=copy.deepcopy(self.plan);plan['teacher_registration']['sha256']='0'*64
        with patch.object(prep,'read_json',side_effect=lambda p:plan if Path(p)==ROOT/prep.PLAN else read_json(p)):
            with self.assertRaises(BoundaryError):prep.load_plan(ROOT)

    def test_manifest_is_balanced_train_prefix_with_pilot_and_evaluation_excluded(self):
        sample=self.sample;new=sample['new_ids']
        self.assertEqual(len(new),1500);self.assertEqual(len(set(new)),1500)
        self.assertEqual(set(Counter(self.boundary.labels[s] for s in new).values()),{375})
        self.assertEqual(len(sample['reused_pilot_ids']),500)
        self.assertFalse(set(new)&set(sample['reused_pilot_ids']))
        self.boundary.require(new,'train')
        for blocked in (self.boundary.splits['validation'],self.boundary.splits['test'],self.boundary.legacy):
            self.assertFalse(set(new)&set(blocked))
        self.assertFalse(sample['sampling_runtime_or_gold_read']);self.assertFalse(sample['training_enabled'])

    def test_frozen_real_export_receipt_matches_cohort_sources_and_live_assets(self):
        self.assertEqual(read_json(ROOT/self.plan['sample_manifest']),self.sample)
        receipt=read_json(ROOT/self.plan['execution_receipt'])
        self.assertEqual(receipt['plan_sha256'],sha(ROOT/prep.PLAN))
        self.assertEqual(receipt['sample_manifest_sha256'],sha(ROOT/self.plan['sample_manifest']))
        self.assertEqual(receipt['n_new_tasks'],1500);self.assertEqual(receipt['reused_pilot_n'],500)
        self.assertEqual(receipt['assets'],{name:sha(ROOT/name) for name in prep.ASSETS})
        self.assertEqual(receipt['source_delivery'],{s['file']:s['sha256'] for s in self.boundary.delivery['shards']})
        self.assertFalse(receipt['official_test_runtime_read']);self.assertFalse(receipt['ready_for_SFT'])
        self.assertTrue(receipt['ready_for_candidate_API']);self.assertEqual(receipt['API_calls'],0)

    def test_nonselected_train_validation_and_test_text_is_skipped_before_JSON_decode(self):
        selected=self.sample['new_ids'][0]
        skipped=[self.sample['reused_pilot_ids'][0],self.boundary.allowed_ids('validation')[0],
                 self.boundary.splits['test'][0]]
        lines=[('{"sample_id":"'+s+'", invalid JSON and supervision not decoded').encode() for s in skipped]
        lines.append(json.dumps(fictional(selected)).encode())
        with patch.object(self.boundary,'source_lines',return_value=iter(lines)):
            rows=list(prep.selected_rows(self.boundary,'unused',[selected]))
        self.assertEqual([r['sample_id'] for r in rows],[selected])

    def test_selected_test_and_duplicate_source_ID_are_rejected(self):
        test=self.boundary.splits['test'][0];sid=self.sample['new_ids'][0]
        with patch.object(self.boundary,'source_lines',return_value=[json.dumps(fictional(test)).encode()]):
            with self.assertRaises(BoundaryError):list(prep.selected_rows(self.boundary,'unused',[test]))
        line=json.dumps(fictional(sid)).encode()
        with patch.object(self.boundary,'source_lines',return_value=[line,line]):
            with self.assertRaises(BoundaryError):list(prep.selected_rows(self.boundary,'unused',[sid]))

    def test_preparation_separates_gold_and_live_preflight_does_not_open_references(self):
        # Small fictional source exercises I/O; the real 1500-ID seal is checked separately.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sample=copy.deepcopy(self.sample);sample['new_ids']=sample['new_ids'][:2]
            for name in (prep.PLAN,*prep.ASSETS):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
            source=root/self.boundary.policy['source_delivery']['default_local_directory'];source.mkdir(parents=True)
            plan=copy.deepcopy(self.plan)
            def rows(*args):
                for sid in sample['new_ids']:
                    row=fictional(sid);row['annotation_reference']['gold_answer']='OFFLINE_REFERENCE_SENTINEL'
                    yield row
            with patch.object(prep,'load_plan',return_value=(plan,self.parent)), \
                 patch.object(prep,'TeamNQBoundary',return_value=self.boundary), \
                 patch.object(prep,'manifest',return_value=sample), \
                 patch.object(prep,'selected_rows',side_effect=rows), \
                 patch.object(prep,'load_profile',return_value={'token_reservation_cap':95000}):
                receipt=prep.prepare(root)
                inputs=root/plan['inputs_directory'];tasks=inputs/'teacher_tasks.jsonl'
                self.assertNotIn('OFFLINE_REFERENCE_SENTINEL',tasks.read_text())
                self.assertIn('OFFLINE_REFERENCE_SENTINEL',(inputs/'offline_references.jsonl').read_text())
                self.assertFalse(receipt['ready_for_SFT'])
                (inputs/'offline_references.jsonl').unlink()
                self.assertEqual(len(prep.preflight(root)[3]),2)
                with self.assertRaises(FileExistsError):prep.prepare(root)
                original=tasks.read_bytes();tasks.write_bytes(original+b'\n')
                with self.assertRaises(BoundaryError):prep.preflight(root)
                tasks.write_bytes(original)
                (root/prep.ASSETS[0]).write_text('changed source')
                with self.assertRaises(BoundaryError):prep.preflight(root)


class ExpansionCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan,cls.parent=prep.load_plan(ROOT);boundary=TeamNQBoundary(ROOT)
        ids=prep.manifest(boundary,cls.plan)['new_ids'][:3]
        cls.tasks=[teacher_task(boundary,fictional(sid)) for sid in ids]

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.clock=Clock();self.plan=copy.deepcopy(type(self).plan)
        for name in (prep.PLAN,REGISTRATION,PROFILE_PATH):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
        p=self.root/self.plan['execution_receipt'];p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        self.env={'AZURE_OPENAI_ENDPOINT':'https://qzhu0160-4829-resource.services.ai.azure.com/openai/v1/',
                  'AZURE_OPENAI_API_KEY':'FICTIONAL_TEST_SECRET_NEVER_PRINT'}
        self.serial=0

    def response(self,*args):
        self.serial+=1
        return {'model':self.parent['response_model'],'service_tier':'default','id':'fiction-'+str(self.serial),
            'usage':{'prompt_tokens':1000,'completion_tokens':100,
                     'completion_tokens_details':{'reasoning_tokens':20}},
            'choices':[{'finish_reason':'stop','message':{'content':target()}}]}

    def run_fake(self,transport=None,n=3,progress=None):
        with patch.object(collect,'preflight',return_value=(self.plan,self.parent,
                {'new_ids':[t['sample_id'] for t in self.tasks]},self.tasks,{})):
            return collect.run(self.root,limit_new=n,environment=self.env,
                transport=transport or self.response,clock=self.clock,sleep=self.clock.sleep,progress=progress)

    def events(self):
        return [json.loads(s) for s in (self.root/self.plan['run_directory']/'journal.jsonl').read_text().splitlines()]

    def write_events(self,events):
        (self.root/self.plan['run_directory']/'journal.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))

    def test_reuses_500_once_and_resumes_without_resending_or_rebilling(self):
        network=Mock(side_effect=self.response);progress=[]
        first=self.run_fake(network,n=1,progress=progress.append)
        self.assertEqual(first['total_candidates_completed'],501)
        result=self.run_fake(network,n=3,progress=progress.append)
        self.assertEqual(network.call_count,3)
        self.assertEqual(result['total_candidates_completed'],503)
        self.assertAlmostEqual(result['published_price_estimate_total_usd'],3.42705+3*.003)
        self.assertFalse(result['further_expansion_automatic']);self.assertEqual(result['student_training_runs'],0)
        self.assertIn(30,self.clock.sleeps)  # resume waits a full rolling window, in bounded sleeps
        again=Mock();self.run_fake(again,n=3);again.assert_not_called()
        for call,task in zip(network.call_args_list,self.tasks):
            self.assertEqual(call.args[2],request_body(task,self.parent))
        self.assertNotIn(self.env['AZURE_OPENAI_API_KEY'],json.dumps(self.events()+progress))

    def test_20_dollar_budget_is_reserved_before_network_and_keeps_pilot_spend(self):
        held=reservation(request_body(self.tasks[0],self.parent),self.parent)['microusd']*1000
        self.plan['pilot_price_estimate_nanousd']=20_000_000_000-held+1
        network=Mock();result=self.run_fake(network)
        network.assert_not_called();self.assertEqual(result['new_requests_completed'],0)
        self.assertEqual(result['stop_reason'],'budget_reservation_limit')
        self.assertEqual(len(self.events()),1)

    def test_request_limits_cannot_expand_to_5000(self):
        for n in (0,1501,4500,True,1.0):
            with self.assertRaises(BoundaryError):self.run_fake(n=n)

    def test_invalid_but_billed_target_is_quarantined_and_next_ID_continues_without_retry(self):
        def network(*args):
            result=self.response()
            if self.serial==1:result['choices'][0]['message']['content']='invalid JSON'
            return result
        result=self.run_fake(network)
        self.assertEqual(self.serial,3);self.assertEqual(result['new_quarantined'],1)
        self.assertEqual(result['new_structural_passed'],2)
        again=Mock();self.run_fake(again);again.assert_not_called()
        settlements=[e for e in self.events() if e['kind']=='settle']
        self.assertTrue(all(e['approved_for_SFT'] is False for e in settlements))

    def test_unknown_charge_retains_full_reservation_and_blocks_retry(self):
        network=Mock(side_effect=TimeoutError('fictional failure'))
        with self.assertRaises(BoundaryError):self.run_fake(network)
        network.assert_called_once();event=self.events()[-1]
        self.assertFalse(event['request_completed'])
        self.assertEqual(event['cost_nanousd'],self.events()[-2]['reserved_nanousd'])
        retry=Mock()
        with self.assertRaises(BoundaryError):self.run_fake(retry)
        retry.assert_not_called()

    def test_wrong_version_tier_and_excess_usage_stop_not_quarantine_and_continue(self):
        for mutation in (lambda r:r.update(model='wrong'),lambda r:r.update(service_tier='priority'),
                         lambda r:r['usage'].update(completion_tokens=4097)):
            with self.subTest(mutation=mutation):
                folder=self.root/self.plan['run_directory'];shutil.rmtree(folder,ignore_errors=True)
                def network(*args):
                    r=self.response();mutation(r);return r
                with self.assertRaises(BoundaryError):self.run_fake(network)
                self.assertEqual(len(self.events()),3)
                self.assertFalse(self.events()[-1]['request_completed'])

    def test_raw_response_cost_order_and_identity_corruption_fail_before_new_paid_request(self):
        self.run_fake(n=1);original=self.events()
        for key,value in [('cost_nanousd',0),('provider_response_id','changed'),
                          ('approved_for_SFT',True),('sample_id',self.tasks[1]['sample_id'])]:
            events=copy.deepcopy(original);events[-1][key]=value;self.write_events(events)
            network=Mock()
            with self.assertRaises(BoundaryError):self.run_fake(network)
            network.assert_not_called()
        self.write_events(original)
        path=self.root/self.plan['run_directory']/(self.tasks[0]['sample_id']+'.response.json')
        path.write_text(path.read_text()+' ')
        with self.assertRaises(BoundaryError):self.run_fake()

    def test_pending_request_and_unjournalled_response_do_not_resend(self):
        self.run_fake(n=1);events=self.events();self.write_events(events[:-1]);network=Mock()
        with self.assertRaises(BoundaryError):self.run_fake(network)
        network.assert_not_called()
        self.write_events(events[:1])
        with self.assertRaises(BoundaryError):self.run_fake(network)
        network.assert_not_called()

    def test_concurrent_writer_lock_prevents_calls(self):
        folder=self.root/self.plan['run_directory'];folder.mkdir(parents=True)
        with (folder/'run.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);network=Mock()
            with self.assertRaises(BlockingIOError):self.run_fake(network)
            network.assert_not_called()

    def test_nanodollars_preserve_fractional_microdollar_cache_price(self):
        self.assertEqual(collect.nano(Decimal('0.0000001')),100)
        with self.assertRaises(BoundaryError):collect.nano(Decimal('0.0000000001'))


class ExpansionCommandTests(unittest.TestCase):
    def test_default_command_only_checks_and_never_collects(self):
        from scripts import run_e_teacher_expansion as cli
        with patch.object(cli,'preflight',return_value=({}, {}, {}, [1]*1500, {})), \
             patch.object(cli,'run') as run,redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main([]),0);run.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['API_calls'],0)

    def test_hidden_input_rejects_nonterminal_without_reading_secret_or_network(self):
        from scripts import run_e_teacher_expansion as cli
        with patch.object(cli.sys.stdin,'isatty',return_value=False), \
             patch.object(cli.getpass,'getpass') as getpass,patch.object(cli,'run') as run, \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(BoundaryError):cli.main(['--execute','--prompt-key'])
            getpass.assert_not_called();run.assert_not_called()


if __name__=='__main__':unittest.main()
