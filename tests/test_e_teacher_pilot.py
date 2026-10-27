"""Fictional runtime and mocked network; no credentials or paid requests."""
import copy
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.data.team_nq import BoundaryError, TeamNQBoundary
from src.distillation.e_prep import ROOT, teacher_task
from src.distillation import teacher_pilot as pilot
from tests.test_team_nq_boundary import fictional
from tests.test_e_distillation_prep import target


class TeacherPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundary = TeamNQBoundary(ROOT)
        cls.spec = pilot.registration(ROOT)
        ids = json.loads((ROOT/cls.spec['sample_manifest']).read_text())['sample_ids']['teacher_pilot'][:3]
        cls.tasks = [teacher_task(cls.boundary, fictional(sid)) for sid in ids]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.env = {'AZURE_OPENAI_ENDPOINT': 'https://qzhu0160-4829-resource.services.ai.azure.com/openai/v1/',
                    'AZURE_OPENAI_API_KEY': 'FICTIONAL_TEST_SECRET'}
        self.now = 0

    def response(self):
        return {'model': self.spec['response_model'], 'id': 'fictional',
            'usage': {'prompt_tokens': 1000, 'completion_tokens': 800,
                      'completion_tokens_details': {'reasoning_tokens': 600}},
            'choices': [{'finish_reason': 'stop', 'message': {'content': target(), 'refusal': None}}]}

    def pause(self, seconds):
        self.now += seconds

    def run_fake(self, transport, n=3, env=None):
        with patch.object(pilot, 'preflight', return_value=(self.spec, self.tasks)):
            return pilot.run(self.folder, total_requests=n, environment=env or self.env,
                transport=transport, clock=lambda:self.now, sleep=self.pause)

    def test_registration_has_version_and_separate_teacher_student_budgets(self):
        self.assertEqual(self.spec['response_model'], 'gpt-6.1-sol-2026-09-29')
        self.assertEqual(self.spec['max_completion_tokens_including_reasoning'],4096)
        self.assertEqual(self.spec['student_output_tokens'],384)
        self.assertEqual(self.spec['upgrade_policy_observed'],'OnceNewDefaultVersionAvailable')
        self.assertFalse(self.spec['automatic_upgrade_disabled'])
        self.assertFalse(self.spec['training_enabled'])

    def test_changed_budget_retry_model_or_promotion_is_rejected(self):
        for key, value in [('model','another'),('version','latest'),('max_requests',501),
                           ('pilot_budget_microusd',16000000),('automatic_retries',1),
                           ('training_enabled',True),('expansion_enabled',True),
                           ('accounting_input_usd_per_million',2)]:
            changed=copy.deepcopy(self.spec);changed[key]=value
            with patch.object(pilot,'read_json',return_value=changed):
                with self.assertRaises(BoundaryError):pilot.registration()

    def test_endpoint_exact_hash_no_query_credentials_or_other_resource(self):
        self.assertTrue(pilot.endpoint(self.spec,self.env).startswith('https://'))
        for value in ['', 'http://qzhu0160-4829-resource.services.ai.azure.com/openai/v1/',
                      self.env['AZURE_OPENAI_ENDPOINT']+'?key=secret',
                      'https://evil.services.ai.azure.com/openai/v1/',
                      'https://user:pass@qzhu0160-4829-resource.services.ai.azure.com/openai/v1/']:
            with self.assertRaises(BoundaryError):
                pilot.endpoint(self.spec,{'AZURE_OPENAI_ENDPOINT':value})

    def test_body_has_no_gold_tools_temperature_or_student_384_teacher_cap(self):
        body=pilot.request_body(self.tasks[0],self.spec)
        self.assertEqual(body['model'],self.spec['deployment'])
        self.assertEqual(body['messages'],self.tasks[0]['teacher_messages'])
        self.assertTrue(body['response_format']['json_schema']['strict'])
        self.assertEqual(body['max_completion_tokens'],4096)
        self.assertFalse(body['store']);self.assertFalse(body['stream'])
        self.assertFalse(set(body)&{'tools','temperature','gold','runtime','annotation_reference'})

    def test_reservation_includes_schema_bytes_and_total_completion(self):
        body=pilot.request_body(self.tasks[0],self.spec)
        reserve=pilot.reservation(body,self.spec)
        self.assertGreater(reserve['input_tokens_bound'],len(json.dumps(body).encode()))
        self.assertEqual(reserve['microusd'],reserve['input_tokens_bound']*5+4096*15)
        body['messages']=[{'role':'user','content':'x'*90000}]
        with self.assertRaises(BoundaryError):pilot.reservation(body,self.spec)

    def test_reasoning_tokens_are_counted_once_in_cost(self):
        reserve=pilot.reservation(pilot.request_body(self.tasks[0],self.spec),self.spec)
        self.assertEqual(pilot.usage_cost(self.response(),reserve,self.spec),1000*5+800*15)

    def test_missing_negative_boolean_or_above_envelope_usage_rejected(self):
        reserve=pilot.reservation(pilot.request_body(self.tasks[0],self.spec),self.spec)
        for mutate in [lambda r:r.pop('usage'),lambda r:r['usage'].update(prompt_tokens=-1),
                       lambda r:r['usage'].update(prompt_tokens=True),
                       lambda r:r['usage'].update(completion_tokens=4097),
                       lambda r:r['usage'].update(prompt_tokens=reserve['input_tokens_bound']+1)]:
            response=self.response();mutate(response)
            with self.assertRaises((BoundaryError,KeyError)):pilot.usage_cost(response,reserve,self.spec)

    def test_visible_target_still_not_semantically_or_SFT_approved(self):
        audit=pilot.audit_response(self.response(),self.tasks[0],self.spec)
        self.assertTrue(audit['structural_provenance_passed'])
        self.assertFalse(audit['semantic_verified']);self.assertFalse(audit['approved_for_SFT'])

    def test_alias_version_truncation_refusal_tools_empty_and_illegal_span_rejected(self):
        for mutate in [lambda r:r.update(model='gpt-6.1-sol'),
                       lambda r:r['choices'][0].update(finish_reason='length'),
                       lambda r:r['choices'][0]['message'].update(refusal='no'),
                       lambda r:r['choices'][0]['message'].update(tool_calls=[{}]),
                       lambda r:r.update(choices=[]),
                       lambda r:r['choices'][0]['message'].update(content=None),
                       lambda r:r['choices'][0]['message'].update(content=target(answer='Invented'))]:
            response=self.response();mutate(response)
            with self.assertRaises(BoundaryError):pilot.audit_response(response,self.tasks[0],self.spec)

    def test_redirect_never_forwards_credentials(self):
        with self.assertRaises(BoundaryError):
            pilot.NoRedirect().redirect_request(None,None,302,'Found',{},'https://evil.invalid')

    def test_success_persists_usage_and_can_resume_without_duplicate_calls(self):
        transport=Mock(side_effect=lambda *a:self.response())
        first=self.run_fake(transport,1)
        self.assertEqual(first['requests'],1)
        second=self.run_fake(transport,3)
        self.assertEqual(second['requests'],3)
        self.assertEqual(transport.call_count,3)
        self.assertEqual(second['accounting_usd_upper_estimate'],0.051)
        self.assertEqual(second['semantic_targets_approved'],0)
        self.assertGreaterEqual(self.now,122)
        journal=(self.folder/self.spec['run_directory']/'journal.jsonl').read_text()
        self.assertNotIn('FICTIONAL_TEST_SECRET',journal)
        self.run_fake(transport,3)
        self.assertEqual(transport.call_count,3)

    def test_timeout_reserves_full_cost_stops_and_cannot_retry(self):
        transport=Mock(side_effect=TimeoutError('private detail must not be logged'))
        with self.assertRaisesRegex(BoundaryError,'full reservation retained'):self.run_fake(transport)
        self.assertEqual(transport.call_count,1)
        journal=(self.folder/self.spec['run_directory']/'journal.jsonl').read_text()
        self.assertNotIn('private detail',journal)
        events=[json.loads(line) for line in journal.splitlines()]
        self.assertEqual(events[-1]['microusd'],events[-2]['microusd'])
        with self.assertRaisesRegex(BoundaryError,'no automatic retry'):self.run_fake(transport)
        self.assertEqual(transport.call_count,1)

    def test_paid_bad_output_is_saved_and_stops_before_second_call(self):
        response=self.response();response['model']='wrong-version'
        transport=Mock(return_value=response)
        with self.assertRaises(BoundaryError):self.run_fake(transport)
        self.assertEqual(transport.call_count,1)
        saved=list((self.folder/self.spec['run_directory']).glob('*.response.json'))
        self.assertEqual(len(saved),1)
        self.assertEqual(json.loads(saved[0].read_text())['model'],'wrong-version')

    def test_inflight_reservation_after_crash_cannot_be_replayed(self):
        transport=Mock(side_effect=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):self.run_fake(transport)
        with self.assertRaisesRegex(BoundaryError,'no automatic retry'):self.run_fake(transport)
        self.assertEqual(transport.call_count,1)

    def test_budget_reserves_before_send_and_stops_without_overrun(self):
        spec=copy.deepcopy(self.spec)
        reserve=pilot.reservation(pilot.request_body(self.tasks[0],spec),spec)
        spec['pilot_budget_microusd']=reserve['microusd']-1
        with patch.object(pilot,'preflight',return_value=(spec,self.tasks)):
            transport=Mock()
            report=pilot.run(self.folder,total_requests=3,environment=self.env,transport=transport)
        transport.assert_not_called();self.assertEqual(report['requests'],0)

    def test_missing_credentials_or_invalid_count_never_calls_network(self):
        transport=Mock()
        for n,env in [(0,self.env),(501,self.env),(True,self.env),(3,{'AZURE_OPENAI_ENDPOINT':self.env['AZURE_OPENAI_ENDPOINT']})]:
            with self.assertRaises(BoundaryError):self.run_fake(transport,n,env)
        transport.assert_not_called()

    def test_manifest_or_tasks_fingerprint_failure_before_network(self):
        with patch.object(pilot,'sha',return_value='tampered'):
            with self.assertRaisesRegex(BoundaryError,'manifest fingerprint'):pilot.preflight(ROOT)
        original_sha=pilot.sha
        def wrong_tasks(path):
            return 'tampered' if Path(path).name=='teacher_tasks.jsonl' else original_sha(path)
        with patch.object(pilot,'sha',side_effect=wrong_tasks):
            with self.assertRaisesRegex(BoundaryError,'task file fingerprint'):pilot.preflight(ROOT)

    def test_actual_preflight_rebuilds_all_500_tasks_and_interleaves(self):
        if not (ROOT/self.spec['tasks']).exists():self.skipTest('Local ignored pilot content not present')
        spec,tasks=pilot.preflight(ROOT)
        self.assertEqual(len(tasks),500)
        self.assertEqual(len({t['sample_id'] for t in tasks}),500)
        labels=[self.boundary.labels[t['sample_id']] for t in tasks]
        for i in range(0,500,4):self.assertEqual(len(set(labels[i:i+4])),4)
        self.boundary.require([t['sample_id'] for t in tasks],'train')

    def test_cli_default_is_offline_and_prompt_key_requires_execute(self):
        from scripts import run_e_teacher_pilot as cli
        with patch.object(cli,'preflight',return_value=(self.spec,self.tasks)),patch.object(cli,'run') as network:
            with redirect_stdout(io.StringIO()) as capture:self.assertEqual(cli.main([]),0)
            self.assertEqual(json.loads(capture.getvalue())['API_calls'],0)
            network.assert_not_called()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):cli.main(['--prompt-key'])


if __name__=='__main__':unittest.main()
