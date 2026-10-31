"""Cost accounting, paired uncertainty and complete frozen-policy replay."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import analyze_accuracy_cost as ac


class AccuracyCostUnitTests(unittest.TestCase):
    def test_unjudged_changed_answers_fail_closed(self):
        self.assertEqual(ac.paired(['same'],['same'],[None],[None]).tolist(),[0])
        with self.assertRaisesRegex(ValueError,'Unjudged'):
            ac.paired(['a'],['b'],[None],[True])
        with self.assertRaisesRegex(ValueError,'lengths'):
            ac.paired(['a'],[],[True],[])

    def test_pairing_not_independent_accuracy_intervals(self):
        d=ac.paired(['a','b','c','d'],['x','y','c','z'],[False,True,True,False],[True,False,True,True])
        self.assertEqual(d.tolist(),[1,-1,0,1])
        s=ac.paired_stats(d,5)
        self.assertEqual((s['fixes'],s['harms'],s['net'],s['delta_pp']),(2,1,1,25))
        same=ac.paired_stats(np.zeros(100),5)
        self.assertEqual(same['ci95_pp'],[0.,0.])
        self.assertEqual(same['mcnemar_p'],1.)

    def test_holm_uses_whole_comparison_family(self):
        self.assertEqual(ac.holm({'a':.01,'b':.03,'c':.8}),{'a':.03,'b':.06,'c':.8})

    def test_frontier_compares_total_cost_and_does_not_promote(self):
        def row(c,q):return {'calls_mean':c,'gain_vs_b_bm25':{'delta_pp':q}}
        points={'base':row(1,0),'better':row(2,1),'worse':row(3,.5),'tradeoff':row(4,2)}
        self.assertEqual(ac.pareto(points),['base','better','tradeoff'])

    def test_prefix_reuses_contexts_and_retains_overhead(self):
        r={'stage':8,'selector_seconds':4.,'selector_cost':{'llm_calls':3},
           'contexts':{n:{'source':v} for n,v in zip('abcdefgh',['c1','generated','reused','generated','reused','generated','reused','reused'])},
           'llm_events':[{'seconds':x,'input_tokens':int(x*100)} for x in [.2,.3,.5]]}
        self.assertEqual(ac.selector_prefix(r,3,list('abcdefgh')),{'calls':1,'seconds':3.2,'input_tokens':20})
        self.assertEqual(ac.selector_prefix(r,0,list('abcdefgh')),{'calls':0,'seconds':0.,'input_tokens':0})
        with self.assertRaisesRegex(ValueError,'unmeasured'):
            ac.selector_prefix({**r,'stage':3},5,list('abcdefgh'))
        with self.assertRaisesRegex(ValueError,'Nested'):
            ac.selector_prefix({**r,'selector_seconds':.1},3,list('abcdefgh'))

    def test_only_source_hashes_normalize_line_endings(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'lf.py',Path(d)/'crlf.py'
            a.write_bytes(b'x = 1\n');b.write_bytes(b'x = 1\r\n')
            self.assertEqual(ac.source_digest(a),ac.source_digest(b))
            self.assertNotEqual(ac.digest(a),ac.digest(b))

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            ac.indexed([{'sample_id':'a'},{'sample_id':'a'}])

    def test_hash_seal_catches_tampering_before_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'input_hashes.json').write_text((ac.RELEASE/'input_hashes.json').read_text())
            with patch.object(ac,'digest',return_value='changed'):
                with self.assertRaisesRegex(ValueError,'Input changed'):
                    ac.load_inputs(release=p)


class AccuracyCostReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary,cls.records,cls.timing=ac.analyze()

    def test_entire_summary_replays(self):
        checks=json.loads((ac.RELEASE/'checksums.json').read_text())
        actual={p.relative_to(ac.RELEASE).as_posix() for p in ac.RELEASE.iterdir() if p.is_file() and p.name!='checksums.json'}
        self.assertEqual(set(checks),actual)
        for name, expected in checks.items():
            self.assertEqual(ac.digest(ac.RELEASE/name),expected,name)
        published=json.loads((ac.RELEASE/'summary.json').read_text())
        self.assertEqual(json.loads(json.dumps(self.summary)),published)
        with gzip.open(ac.RELEASE/'paired_records.jsonl.gz','rt') as f:
            self.assertEqual([json.loads(l) for l in f],self.records)

    def test_total_cost_includes_base_and_rejected_generations(self):
        t=self.summary['roles']['test']['strategies']
        self.assertEqual((t['c1']['correct'],t['b']['correct'],t['p3_all']['correct'],t['b_bm25']['correct'],t['chosen']['correct']),
                         (6400,6632,6624,6737,6975))
        self.assertEqual(t['c1']['calls_total'],11114)
        self.assertEqual(t['chosen']['calls_total'],t['b_bm25']['calls_total']+round(t['chosen']['extra_calls_mean']*11114))
        self.assertGreater(t['b_bm25']['calls_mean'],1.6)
        self.assertLess(t['b_bm25']['calls_mean'],1.8)
        self.assertTrue(all(v['known_labels']==11114 for v in t.values()))
        self.assertEqual(t['fixed3_vote']['calls_total'],t['fixed3_selector']['calls_total'])
        self.assertEqual(t['chosen']['calls_total'],t['theta0.0_tau0.8_selector']['calls_total'])

    def test_all_policies_preserved_no_test_selection(self):
        self.assertEqual(self.summary['policy_selection'],'none')
        self.assertEqual(len(self.summary['roles']['test']['strategies']),23)
        self.assertEqual(len(self.summary['roles']['confirm']['strategies']),21)
        self.assertEqual(self.summary['validation']['confirm_replay_vs_actual_generation_delta'],2)
        self.assertEqual(self.summary['validation']['timing_answer_matches'],300)

    def test_new_final_evaluation_archive_agrees(self):
        old=ac.read_archive(ac.ROOT/'artifacts/controller/final_evaluation_v1/results.tar.gz')
        ours=ac.read_archive(ac.RELEASE/'inputs.tar.gz')
        self.assertEqual(old['latency/latency.jsonl'],ours['timing/latency.jsonl'])
        p3=ac.indexed(ac.jsonl(old['test_p3_rows.jsonl']))
        for r in ac.jsonl(ours['test/cost_and_p3.jsonl']):
            q=p3[r['sample_id']]
            self.assertEqual((r['c1'],r['p3_all'],r['base'],r['routed']),
                             (q['c1'],q['p3'],q['b_bm25'],q['routed']))

    def test_measured_and_estimated_latency_kept_separate(self):
        t=self.summary['timing'];v=self.summary['roles']
        self.assertEqual((t['chosen']['n'],v['test']['n'],v['confirm']['n']),(300,11114,2440))
        self.assertEqual(t['chosen']['kind'],'measured_total')
        self.assertEqual(t['c1']['kind'],'measured_component_prefix')
        self.assertEqual(t['theta0.39_tau0.6_selector']['kind'],'two_run_prefix_estimate')
        self.assertNotIn('fixed8_selector',t)
        self.assertAlmostEqual(t['chosen']['seconds_mean'],2.6877742828475313)
        self.assertLess(abs(t['chosen_prefix_estimate']['seconds_mean']-t['chosen']['seconds_mean']),.02)
        self.assertAlmostEqual(t['chosen']['calls_mean'],4.683333333333334)
        self.assertNotEqual(v['test']['strategies']['chosen']['calls_mean'],t['chosen']['calls_mean'])


if __name__=='__main__':unittest.main()
