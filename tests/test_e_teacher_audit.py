"""Offline journal corruption, immutable selection and semantic-gate checks."""
import copy
import json
import unittest
import tempfile
import fcntl
from pathlib import Path
from unittest.mock import patch

from src.data.team_nq import BoundaryError,TeamNQBoundary
from src.distillation.e_prep import ROOT,digest,teacher_task
from src.distillation.teacher_pilot import registration,request_body,reservation,audit_response,usage_cost
from src.distillation.teacher_audit import replay,review_selection,assess_gate,review_page,audit_local
from tests.test_team_nq_boundary import fictional
from tests.test_e_distillation_prep import target


class TeacherAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=registration(ROOT);cls.boundary=TeamNQBoundary(ROOT)
        cls.ids=json.loads((ROOT/cls.spec['sample_manifest']).read_text())['sample_ids']['teacher_pilot']
        cls.tasks=[teacher_task(cls.boundary,fictional(sid)) for sid in cls.ids[:5]]

    def fixture(self):
        events=[{'kind':'seal','registration':digest(self.spec),'tasks':self.spec['tasks_sha256'],
                 'ordered_ids':[t['sample_id'] for t in self.tasks]}];responses={}
        for i,task in enumerate(self.tasks):
            sid=task['sample_id']
            text=target('supported','') if i%2 else target()
            raw={'model':self.spec['response_model'],'id':'fixture-'+sid,
                'usage':{'prompt_tokens':1000,'completion_tokens':800,
                         'completion_tokens_details':{'reasoning_tokens':600}},
                'choices':[{'finish_reason':'stop','message':{'content':text}}]}
            reserve=reservation(request_body(task,self.spec),self.spec)
            events.append({'kind':'reserve','sample_id':sid,**reserve})
            events.append({'kind':'settle','sample_id':sid,'accepted':True,
                'provider_response_id':raw['id'],'microusd':usage_cost(raw,reserve,self.spec),
                'audit':audit_response(raw,task,self.spec)})
            responses[sid]=raw
        return events,responses

    def rows(self):
        rows=[]
        kinds=['replace']*49+['supported_keep']*40+['insufficient_keep']*22+['contradicted_no_candidate']*9
        for i,group in enumerate(kinds):
            label={'replace':'contradicted','supported_keep':'supported','insufficient_keep':'insufficient',
                   'contradicted_no_candidate':'contradicted'}[group]
            action='replace' if group=='replace' else 'keep'
            rows.append({'sample_id':self.ids[i],'review_group':group,'runtime':self.boundary.teacher_input(fictional(self.ids[i])),
                'teacher_output':{'original_verification':label,'reason':'Fictional judgment'},
                'audit':{'model_label':label,'candidate_action':action,'candidate_final_answer':'Aster'},
                'offline_diagnostic':{'reference':'OFFLINE_REFERENCE_SENTINEL','C1_EM':0,'teacher_final_EM':1}})
        return rows

    def test_complete_replay_matches_exact_outputs_cost_and_keeps_semantics_pending(self):
        events,responses=self.fixture()
        refs={t['sample_id']:{'gold_answer':'Aster'} for t in self.tasks}
        rows,report=replay(self.spec,self.tasks,events,responses,lambda text:123,refs)
        self.assertEqual(report['n'],5);self.assertTrue(report['complete'])
        self.assertAlmostEqual(report['accounting_usd_upper_estimate'],0.085)
        self.assertEqual(report['provider_tokens']['reasoning_subset'],3000)
        self.assertEqual(report['provider_tokens']['completion_including_reasoning'],4000)
        self.assertEqual(report['actions'],{'replace':3,'keep':2})
        self.assertEqual(report['offline_EM_diagnostic']['fixes'],3)
        self.assertFalse(report['offline_EM_diagnostic']['is_population_or_student_gain'])
        self.assertTrue(all(not r['approved_for_SFT'] and not r['semantic_verified'] for r in rows))

    def test_incomplete_duplicate_wrong_order_extra_or_failed_journal_rejected(self):
        for mutate in [lambda e:e.pop(),lambda e:e.append(copy.deepcopy(e[-1])),
                       lambda e:e[1].update(sample_id=self.ids[4]),
                       lambda e:e[2].update(accepted=False),
                       lambda e:e.insert(2,copy.deepcopy(e[1])),
                       lambda e:e.append({'kind':'unknown'}),
                       lambda e:e[0].update(registration='changed')]:
            events,responses=self.fixture();mutate(events)
            with self.assertRaises(BoundaryError):replay(self.spec,self.tasks,events,responses,lambda text:100)

    def test_raw_response_cost_audit_or_model_cannot_be_silently_changed(self):
        for mutate in [lambda e,r:e[1].update(microusd=1),
                       lambda e,r:e[2].update(microusd=1),
                       lambda e,r:e[2]['audit'].update(approved_for_SFT=True),
                       lambda e,r:r[self.ids[0]].update(model='another'),
                       lambda e,r:r[self.ids[0]].update(id='different'),
                       lambda e,r:r.pop(self.ids[0]),
                       lambda e,r:r.update(extra=copy.deepcopy(r[self.ids[0]]))]:
            events,responses=self.fixture();mutate(events,responses)
            with self.assertRaises(BoundaryError):replay(self.spec,self.tasks,events,responses,lambda text:100)

    def test_profile_during_inflight_request_and_unknown_reference_cohort_rejected(self):
        events,responses=self.fixture();events.insert(2,{'kind':'transport_profile'})
        with self.assertRaises(BoundaryError):replay(self.spec,self.tasks,events,responses,lambda text:100)
        events,responses=self.fixture()
        with self.assertRaises(BoundaryError):replay(self.spec,self.tasks,events,responses,lambda text:100,{})

    def test_long_target_is_flagged_without_truncation_or_semantic_approval(self):
        events,responses=self.fixture()
        rows,report=replay(self.spec,self.tasks,events,responses,lambda text:385)
        self.assertEqual(report['target_tokens_with_EOS']['over_384'],5)
        self.assertTrue(all(not r['target_fits_384'] and not r['approved_for_SFT'] for r in rows))
        for bad in [0,-1,True,'100']:
            with self.assertRaises(BoundaryError):replay(self.spec,self.tasks,events,responses,lambda text:bad)

    def test_frozen_hash_selection_covers_all_49_repairs_and_9_no_candidate(self):
        selected=review_selection(self.rows(),self.spec)
        self.assertEqual(len(selected),100)
        self.assertEqual([r['review_id'] for r in selected],[f'T{i:03}' for i in range(1,101)])
        from collections import Counter
        self.assertEqual(Counter(r['review_group'] for r in selected),
            {'replace':49,'supported_keep':21,'insufficient_keep':21,'contradicted_no_candidate':9})
        self.assertTrue(all(r['human_review']['semantic_correct'] is None for r in selected))

    def test_selection_ignores_input_order_gold_and_EM(self):
        a=self.rows();b=list(reversed(copy.deepcopy(a)))
        for row in b:row['offline_diagnostic']={'reference':'CHANGED','C1_EM':1,'teacher_final_EM':0}
        ids=lambda rows:[r['sample_id'] for r in review_selection(rows,self.spec)]
        self.assertEqual(ids(a),ids(b))

    def test_review_gate_pending_by_default_and_never_enables_training(self):
        selected=review_selection(self.rows(),self.spec)
        summary={'structural_provenance_pass_rate':1.0}
        report=assess_gate(summary,selected,self.spec)
        self.assertTrue(report['review_coverage_sufficient'])
        self.assertFalse(report['human_review_complete']);self.assertFalse(report['semantic_gate_passed'])
        reviews={r['sample_id']:{'semantic_correct':True,'harmful_replacement':False} for r in selected}
        passed=assess_gate(summary,selected,self.spec,reviews)
        self.assertTrue(passed['semantic_gate_passed'])
        self.assertFalse(passed['training_enabled']);self.assertFalse(passed['expansion_enabled'])
        self.assertEqual(passed['approved_for_SFT'],0)

    def test_semantic_threshold_harm_uncertainty_and_low_coverage_veto(self):
        selected=review_selection(self.rows(),self.spec);summary={'structural_provenance_pass_rate':1.0}
        base={r['sample_id']:{'semantic_correct':True,'harmful_replacement':False} for r in selected}
        for n,expected in [(5,True),(6,False)]:
            reviews=copy.deepcopy(base)
            for row in selected[:n]:reviews[row['sample_id']]['semantic_correct']=False
            self.assertEqual(assess_gate(summary,selected,self.spec,reviews)['semantic_gate_passed'],expected)
        reviews=copy.deepcopy(base);reviews[selected[0]['sample_id']]['harmful_replacement']=True
        self.assertFalse(assess_gate(summary,selected,self.spec,reviews)['semantic_gate_passed'])
        reviews=copy.deepcopy(base);reviews[selected[0]['sample_id']]['semantic_correct']='unknown'
        self.assertFalse(assess_gate(summary,selected,self.spec,reviews)['semantic_gate_passed'])
        self.assertFalse(assess_gate({'structural_provenance_pass_rate':.94},selected,self.spec,base)['semantic_gate_passed'])
        self.assertFalse(assess_gate(summary,selected[:20],self.spec)['review_coverage_sufficient'])
        with self.assertRaises(BoundaryError):assess_gate(summary,selected,self.spec,{'not-selected':{}})

    def test_review_HTML_escapes_untrusted_text_and_defaults_to_unreviewed(self):
        rows=review_selection(self.rows(),self.spec)[:2]
        rows[0]['runtime']['query']='</script><img src=x onerror=alert(1)>'
        rows[0]['teacher_output']['verdict_evidence_ids']=['R1S1']
        page=review_page(rows)
        self.assertNotIn('<img src=x',page)
        self.assertIn('&lt;img src=x',page)
        self.assertIn('<option value="">未审</option>',page)
        self.assertIn('原始 Top-5',page)
        self.assertIn('<code>R1S1</code>',page)
        self.assertIn('OFFLINE_REFERENCE_SENTINEL',page)
        self.assertIn('user_browser_input',page)
        self.assertNotIn('selected="selected"',page)
        changed=copy.deepcopy(rows);changed[0]['provider_response_sha256']='different-output'
        self.assertNotEqual(review_page(changed),page)
        self.assertIn('cohort_sha256',page)

    def test_audit_refuses_active_writer_before_loading_models_or_creating_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/self.spec['run_directory'];folder.mkdir(parents=True)
            output=root/'review'
            with (folder/'run.lock').open('a') as handle:
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch('src.distillation.teacher_audit.preflight',return_value=(self.spec,self.tasks)):
                    with self.assertRaises(BlockingIOError):audit_local(output,root)
            self.assertFalse(output.exists())

    def test_audit_never_overwrites_existing_human_work(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'review';output.mkdir();(output/'human.json').write_text('preserve')
            with patch('src.distillation.teacher_audit.preflight') as preflight:
                with self.assertRaises(FileExistsError):audit_local(output,ROOT)
                preflight.assert_not_called()
            self.assertEqual((output/'human.json').read_text(),'preserve')


if __name__=='__main__':unittest.main()
