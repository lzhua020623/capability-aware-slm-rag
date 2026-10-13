"""Safety of candidate gating, cost/resume accounting and reviewed-cohort isolation."""
from __future__ import annotations

import copy
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_reviewed_answer_repair import build_cohort
from scripts.prepare_answer_repair_experiment import sha256
from scripts.summarize_reviewed_answer_repair import result_digest, summarize_semantic_pairs
from scripts.run_answer_repair import main, runtime_input
from src.compensation.answer_repair import Generation, repair_answer
from src.compensation.support_check import CHECKS
from src.evaluation.answer_repair import summarize_repairs
from tests.test_answer_repair import c1_row, request


def candidate(answer="Paris", quote="The capital of France is Paris."):
    return json.dumps({"answer": answer, "citations": [{"rank": 1, "quote": quote}]})


def checks(**overrides):
    return json.dumps({**{k: "yes" for k in CHECKS}, **overrides})


class SequenceGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.runtime_environment = {"model_revision": "test-only"}

    def generate(self, prompt, max_new_tokens):
        self.calls.append((prompt, max_new_tokens))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return Generation(value, 0.25, 100, 20, raw_text=value)


class GuardTests(unittest.TestCase):
    def test_accept_records_both_calls_without_claiming_independence(self):
        gen = SequenceGenerator([candidate(), checks()])
        result = repair_answer(request(), gen, strategy="guarded_reanswer")
        self.assertEqual((result.final_answer, result.status, result.extra_calls), ("Paris", "repaired", 2))
        self.assertEqual((result.latency, result.input_tokens, result.output_tokens_estimate), (0.5, 200, 40))
        self.assertEqual([c[1] for c in gen.calls], [384, 192])
        self.assertEqual(result.independent_support_status, "not_checked")
        self.assertEqual(result.support_check["status"], "accepted")
        row = {**asdict(result), "changed": True, "initial_metrics": {"correct": False},
               "final_metrics": {"correct": True}}
        self.assertEqual(summarize_repairs([row])["same_model_support_check_counts"], {"accepted": 1})

    def test_any_negative_or_uncertain_check_preserves_original(self):
        for key in CHECKS:
            for value in ("no", "uncertain"):
                with self.subTest(key=key, value=value):
                    result = repair_answer(request(), SequenceGenerator([candidate(), checks(**{key: value})]),
                                           strategy="guarded_reanswer")
                    self.assertEqual((result.final_answer, result.status), ("London", "support_check_rejected"))
                    self.assertEqual(result.citations, [])
                    self.assertEqual(result.candidate_answer, "Paris")
                    self.assertTrue(result.support_check["candidate_citations"])

    def test_bad_guard_output_cannot_promote_candidate(self):
        for output in ('{}', checks() + ' explanation', checks(query_match=True),
                       checks().replace('"coverage": "yes"', '"coverage": "yes", "coverage": "no"')):
            result = repair_answer(request(), SequenceGenerator([candidate(), output]), strategy="guarded_reanswer")
            self.assertEqual((result.status, result.final_answer, result.extra_calls),
                             ("support_check_invalid", "London", 2))

    def test_invalid_or_abstaining_candidate_skips_guard(self):
        for output in ('not JSON', candidate(quote="Made up."), '{"answer":"","citations":[]}'):
            gen = SequenceGenerator([output])
            result = repair_answer(request(), gen, strategy="guarded_reanswer")
            self.assertEqual(result.extra_calls, 1)
            self.assertIsNone(result.support_check)

    def test_kept_candidate_is_checked_too(self):
        result = repair_answer(request(answer="Paris"), SequenceGenerator([candidate(), checks(entity_match="no")]),
                               strategy="guarded_reanswer")
        self.assertEqual((result.final_answer, result.status, result.candidate_status),
                         ("Paris", "support_check_rejected", "kept"))
        self.assertFalse(result.changed)

    def test_second_call_backend_failure_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "GPU failed"):
            repair_answer(request(), SequenceGenerator([candidate(), RuntimeError("GPU failed")]),
                          strategy="guarded_reanswer")

    def test_v5_span_spacing_fix_preserves_tokens_and_old_behavior(self):
        item = request(text="The interval is 1922 - 1991 .")
        text = candidate("1922-1991", "The interval is 1922 - 1991 .")
        self.assertEqual(repair_answer(item, SequenceGenerator([text]), strategy="query_grounded").status, "repaired")
        self.assertEqual(repair_answer(item, SequenceGenerator([text]), strategy="grounded_reanswer").status, "invalid_output")
        for bad in ("1922 1991", "922-1991", "1922-1992"):
            result = repair_answer(item, SequenceGenerator([candidate(bad, "The interval is 1922 - 1991 .")]),
                                   strategy="query_grounded")
            self.assertEqual(result.status, "invalid_output")

    def test_both_prompts_exclude_original_and_all_offline_annotations(self):
        row = c1_row(prediction="INITIAL_SECRET")
        for field in ("gold_answer_or_label", "human_review", "agent_assessment", "correct", "gold_evidence_text"):
            row[field] = "OFFLINE_SECRET"
        gen = SequenceGenerator([candidate(), checks()])
        repair_answer(runtime_input(row, "nq"), gen, strategy="guarded_reanswer")
        for prompt, _ in gen.calls:
            self.assertNotIn("OFFLINE_SECRET", prompt)
            self.assertNotIn("INITIAL_SECRET", prompt)

    def test_two_call_resume_and_environment_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, out = root / 'c1.jsonl', root / 'out.jsonl'
            source.write_text(''.join(json.dumps(c1_row(str(n))) + '\n' for n in (1, 2)))
            argv = ['--dataset', 'nq', '--c1', str(source), '--all', '--strategy', 'guarded_reanswer', '--out', str(out)]
            first = SequenceGenerator([candidate(), checks(), RuntimeError('interrupted')])
            with contextlib.redirect_stdout(io.StringIO()), patch('src.compensation.qwen_repair.QwenRepairGenerator', return_value=first):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    main(argv)
            self.assertEqual(json.loads(out.read_text())['extra_calls'], 2)
            changed = SequenceGenerator([])
            changed.runtime_environment = {'model_revision': 'different'}
            with contextlib.redirect_stdout(io.StringIO()), patch('src.compensation.qwen_repair.QwenRepairGenerator', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'Runtime environment'):
                    main(argv)
            self.assertEqual(changed.calls, [])
            second = SequenceGenerator([candidate(), checks()])
            with contextlib.redirect_stdout(io.StringIO()), patch('src.compensation.qwen_repair.QwenRepairGenerator', return_value=second):
                main(argv)
            self.assertEqual(len(second.calls), 2)
            self.assertEqual(json.loads(out.with_suffix('.summary.json').read_text())['extra_calls'], 4)


def fixture():
    packet = {'packet_id': 'test', 'source_commits': {}, 'records': []}
    reviews, sources, splits = [], {}, [{'splits': {}}]
    for ds in ('fever', 'nq'):
        rows = []
        for n, use in enumerate(('include_e_core', 'correct_control', 'boundary_case'), 1):
            sid = str(n)
            row = c1_row(sid, 'Paris' if n == 2 else 'London')
            if ds == 'fever':
                row.update(dataset='fever', prediction='SUPPORTS' if n == 2 else 'REFUTES', gold_answer_or_label='SUPPORTS')
            rows.append(row)
            packet['records'].append({'review_id': f'{ds}:{sid}', 'sample_id': sid, 'dataset': ds,
                'previous_E_dev50': False, 'query': row['question_or_claim'], 'initial_answer': row['prediction'],
                'evidence': [{'rank': i+1, **e} for i,e in enumerate(row['evidence'])],
                'offline_reference': {'gold_answer_or_label': row['gold_answer_or_label']}})
            reviews.append({'review_id': f'{ds}:{sid}', 'sample_id': sid, 'dataset': ds,
                'packet_id': 'test', 'review_number': len(reviews)+1, 'confirmation': {'user_text': 'yes'},
                'human_review': {'status': 'completed', 'reviewer': 'tester', 'reason': 'fixture', 'use': use,
                    'initial_answer_status': 'correct' if n == 2 else 'incorrect',
                    'evidence_status': 'sufficient_clear', 'primary_label': 'correct' if n == 2 else 'evidence_utilisation'}})
        sources[ds] = ''.join(json.dumps(r)+'\n' for r in rows).encode()
        splits[0]['splits'][ds] = {'train': ['1','2','3'], 'validation': ['4'], 'test': ['5']}
    packet['packet_sha256'] = sha256(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode())
    for row in reviews:
        row['packet_sha256'] = packet['packet_sha256']
    return packet, reviews, splits, sources


class ReviewedCohortTests(unittest.TestCase):
    def test_excludes_deferred_and_keeps_frozen_source_rows(self):
        packet, reviews, splits, sources = fixture()
        manifest, inputs = build_cohort(packet, reviews, splits, sources)
        self.assertEqual(len(manifest['included']), 4)
        self.assertEqual(len(manifest['deferred']), 2)
        self.assertEqual([json.loads(l) for l in inputs['inputs/nq_reviewed_c1.jsonl'].splitlines()],
                         [json.loads(l) for l in sources['nq'].splitlines()][:2])

    def test_rejects_duplicates_pending_tampering_and_heldout(self):
        for alteration in ('duplicate', 'pending', 'hash', 'heldout', 'source', 'core_label'):
            p,h,s,c = fixture()
            if alteration == 'duplicate': h.append(copy.deepcopy(h[0]))
            if alteration == 'pending': h[0]['human_review']['status'] = 'pending'
            if alteration == 'hash': p['records'][0]['query'] = 'tampered'
            if alteration == 'heldout':
                s[0]['splits']['nq']['train'].remove('1'); s[0]['splits']['nq']['validation'].append('1')
            if alteration == 'source': c['nq'] = c['nq'].replace(b'The capital', b'Tampered capital')
            if alteration == 'core_label': h[0]['human_review']['primary_label'] = 'correct'
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                build_cohort(p,h,s,c)

    def test_zero_human_output_reviews_do_not_become_zero_error_rate(self):
        results = [{'sample_id': '1', 'final_answer': 'x'}]
        report = summarize_semantic_pairs(results, [])
        self.assertIsNone(report['fixed'])
        self.assertIsNone(report['regression_rate_reviewed_subset'])
        self.assertEqual(report['pending_pairs'], 1)

    def test_semantic_metrics_require_paired_review_of_exact_result(self):
        results = [{'sample_id': '1', 'final_answer': 'x'}]
        review = {'sample_id': '1', 'result_sha256': result_digest(results[0]),
                  'pair_review': {'status': 'completed', 'reviewer': 'person', 'reason': 'checked both',
                                 'initial_correct': False, 'final_correct': True,
                                 'initial_supported': False, 'final_supported': True}}
        self.assertEqual(summarize_semantic_pairs(results, [review])['grounded_fixed'], 1)
        review['pair_review']['initial_correct'] = None
        with self.assertRaises(ValueError): summarize_semantic_pairs(results, [review])
        review['pair_review']['initial_correct'] = False
        review['result_sha256'] = 'wrong'
        with self.assertRaises(ValueError): summarize_semantic_pairs(results, [review])


if __name__ == '__main__':
    unittest.main()
