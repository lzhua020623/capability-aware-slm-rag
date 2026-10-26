from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.prepare_controller_t5_candidate import (ROOT, ARCHIVE, analyze, prepare,
    semantic_delta_bounds, verify_cache)

FOLDER = ROOT / 'artifacts/controller/t5_candidate_v1'


class T5CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with zipfile.ZipFile(ROOT / ARCHIVE) as z:
            cls.rows = [json.loads(l) for l in z.read('predictions.jsonl').splitlines()]

    def test_unknown_semantics_remain_bounds_including_partial_judgments(self):
        self.assertEqual(semantic_delta_bounds(None,None), [-1,1])
        self.assertEqual(semantic_delta_bounds(False,None), [0,1])
        self.assertEqual(semantic_delta_bounds(None,True), [0,1])
        self.assertEqual(semantic_delta_bounds(True,None), [-1,0])
        self.assertEqual(semantic_delta_bounds(None,None,exact_same=True), [0,0])
        with self.assertRaises(ValueError): semantic_delta_bounds(1,True)

    def test_actual_frozen_grid_uses_one_shared_unavailable_baseline_and_no_promotion(self):
        result=json.loads((FOLDER/'analysis.json').read_text())
        self.assertEqual(result['baseline_correct'], {'nq':78,'fever':174})
        self.assertEqual([(t['candidate_EM_correct'],t['fixed'],t['harmed']) for t in result['trials']],
                         [(81,5,2),(79,2,1),(78,0,0)])
        self.assertEqual([t['semantic_net_bounds'] for t in result['trials']], [[-11,14],[-9,10],[-4,4]])
        self.assertEqual([t['fully_adjudicated_changed_pairs'] for t in result['trials']], [4,1,0])
        self.assertTrue(all(t['FEVER_answers_and_costs_unchanged'] for t in result['trials']))
        candidate=json.loads((FOLDER/'candidate.json').read_text())
        self.assertEqual((candidate['threshold'],candidate['selection_candidates']),(.2,1))
        self.assertEqual(candidate['stage'],'frozen_for_single_selection_not_promoted')
        self.assertFalse(candidate['promoted'])
        self.assertFalse(candidate['selection_scored'])
        self.assertEqual(result['new_GPU_calls'],0)

    def test_replay_rejects_changed_answers_states_and_module_arguments(self):
        for field in ('final_answer','input_sha256','calls'):
            rows=deepcopy(self.rows)
            adaptive=next(r for r in rows if r['mode']=='adaptive' and r['routing']['decision']['action']=='retrieval_enhancement')
            if field=='calls':
                event=next(e for e in adaptive['calls'] if e['stage']!='probe')
                event['arguments']['unregistered']='change'
            else: adaptive[field]='changed'
            with self.assertRaises(ValueError): verify_cache(rows)

    def test_policy_projection_keeps_original_answer_and_only_probe_cost_when_E_unavailable(self):
        rows=[json.loads(l) for l in (FOLDER/'replay_records.jsonl').read_text().splitlines()]
        original={(r['dataset'],r['sample_id']):r for r in self.rows if r['mode']=='adaptive'}
        self.assertEqual(len(rows),1600)
        e=[r for r in rows if r['action']=='generation_repair']
        self.assertEqual(len(e),64)  # fixed NQ 16 across baseline and three candidates
        for row in e:
            old=original[row['dataset'],row['sample_id']]
            self.assertEqual(row['final_answer'],old['initial_answer'])
            self.assertEqual(row['source_mode'],'registered_unavailable_keep')
            probe=[c for c in old['calls'] if c['stage']=='probe' and c['attempted']]
            self.assertEqual(row['cost_estimate']['llm_calls'],len(probe))
            self.assertEqual(row['semantic_verdict'],'incomplete')

    def test_snapshot_rebuild_reads_development_only_and_does_not_overwrite(self):
        read_bytes=Path.read_bytes
        def guarded(path):
            if any(part in ('selection','final_candidate') for part in path.parts):
                raise AssertionError('Selection/final data opened before candidate freeze')
            return read_bytes(path)
        with tempfile.TemporaryDirectory() as td, patch.object(Path,'read_bytes',guarded):
            output=Path(td)/'candidate'
            candidate=prepare(output)
            self.assertEqual(candidate,json.loads((FOLDER/'candidate.json').read_text()))
            for name in ('analysis.json','replay_records.jsonl'):
                self.assertEqual((output/name).read_bytes(),(FOLDER/name).read_bytes())
            with self.assertRaises(FileExistsError): prepare(output)


if __name__=='__main__':unittest.main()
