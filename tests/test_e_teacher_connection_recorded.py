"""Replay captured Azure outputs without network or credentials."""
import hashlib
import json
from pathlib import Path
import unittest

from src.data.team_nq import TeamNQBoundary
from src.distillation.e_prep import ROOT, digest, teacher_task
from src.distillation.teacher_pilot import (registration, request_body, reservation,
    usage_cost, audit_response)


class RecordedTeacherConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=ROOT/'artifacts/controller/e_teacher_sol_connection_v1'
        cls.spec=registration(ROOT)
        cls.audit=json.loads((cls.folder/'connection_audit.json').read_text())
        cls.tasks=[json.loads(line) for line in (cls.folder/'connection_tasks.jsonl').read_text().splitlines()]
        cls.events=[json.loads(line) for line in (cls.folder/'journal_prefix.jsonl').read_text().splitlines()]

    def test_recorded_tasks_rebuild_and_are_original_train_only_prefix(self):
        boundary=TeamNQBoundary(ROOT)
        for task in self.tasks:
            self.assertEqual(task,teacher_task(boundary,task['runtime']))
        boundary.require([t['sample_id'] for t in self.tasks],'train')
        self.assertEqual([t['sample_id'] for t in self.tasks],self.events[0]['ordered_ids'][:3])
        self.assertEqual(self.events[0]['registration'],digest(self.spec))
        self.assertFalse(any({'gold_answer','annotation_reference','primary_label'}&set(t['runtime']) for t in self.tasks))

    def test_real_raw_responses_replay_contract_and_cost_exactly(self):
        reports={r['sample_id']:r for r in self.audit['records']}
        reservations={e['sample_id']:e for e in self.events if e['kind']=='reserve'}
        settlements={e['sample_id']:e for e in self.events if e['kind']=='settle'}
        for task in self.tasks:
            sid=task['sample_id'];path=self.folder/'responses'/(sid+'.response.json')
            raw=json.loads(path.read_text());record=reports[sid]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['raw_response_sha256'])
            self.assertEqual(audit_response(raw,task,self.spec),record['audit'])
            expected=reservation(request_body(task,self.spec),self.spec)
            self.assertEqual({k:reservations[sid][k] for k in expected},expected)
            cost=usage_cost(raw,expected,self.spec)
            self.assertEqual(cost,record['accounting_microusd'])
            self.assertEqual(cost,settlements[sid]['microusd'])
            self.assertEqual(settlements[sid]['audit'],record['audit'])
            self.assertTrue(settlements[sid]['accepted'])

    def test_connection_is_not_semantic_success_or_a_repair_gain(self):
        self.assertEqual(self.audit['n_real_API_calls'],3)
        self.assertEqual(self.audit['actual_model'],'gpt-6.1-sol-2026-09-29')
        self.assertEqual(self.audit['semantic_approved_for_SFT'],0)
        self.assertEqual(self.audit['training_runs'],0)
        self.assertFalse(self.audit['invoice_cost_verified'])
        self.assertEqual([r['model_label'] for r in self.audit['records']],['insufficient','supported','supported'])
        for record in self.audit['records']:
            self.assertEqual(record['action'],'keep')
            self.assertFalse(record['audit']['semantic_verified'])
            self.assertFalse(record['audit']['approved_for_SFT'])
            self.assertLessEqual(record['target_tokens_with_EOS'],384)
        self.assertAlmostEqual(self.audit['accounting_upper_estimate_usd'],0.04104)

    def test_registered_files_have_complete_unchanged_fingerprints(self):
        expected=json.loads((self.folder/'checksums.json').read_text())
        actual={str(p.relative_to(self.folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.folder.rglob('*') if p.is_file() and p.name!='checksums.json'}
        self.assertEqual(actual,expected)
        self.assertEqual(self.audit['execution_registration_sha256'],
                         hashlib.sha256((ROOT/'configs/distillation/e_teacher_sol_pilot_v1.json').read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
