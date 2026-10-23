"""Hand-calculated paired metrics; labels are supplied only to offline scoring."""

import unittest
from copy import deepcopy

from src.compensation.retrieval_enhancement.evaluation import evaluate_nq


def hits(ids):
    return [{"passage_id": pid, "text": f"Evidence {pid}", "score": 0.8} for pid in ids]


def record(sid, selected, candidates="ABCDEF", *, extra_calls=1):
    return {"sample_id": sid, "base_evidence": hits("ABCDE"),
            "enhanced_evidence": hits(selected), "candidates": hits(candidates),
            "base_latency": 0.2, "trace": {"retrieval_calls": extra_calls, "llm_calls": 0,
            "latency": 0.3, "status": "completed" if extra_calls else "no_rewrite"}}


class NQEvaluationTests(unittest.TestCase):
    def setUp(self):
        # One selected recovery, one candidate-only recovery, one loss, one
        # retained hit and one miss: recovery can coexist with zero net gain.
        self.records = [record("1", "AFBCD"), record("2", "ABCDE"),
                        record("3", "AFBCD"), record("4", "ABCDE", extra_calls=0),
                        record("5", "ABCDE", extra_calls=0)]
        self.annotations = [{"id": sid, "gold_passage_id": gold}
                            for sid, gold in zip("12345", "FFEAZ")]

    def test_paired_recovery_loss_and_recall_match_manual_counts(self):
        result = evaluate_nq(self.records, self.annotations)
        m = result["metrics"]["strict_id"]
        self.assertEqual(result["metrics"]["n"], 5)
        self.assertEqual((m["base_hit"], m["enhanced_hit"], m["base_misses"]), (2, 2, 3))
        self.assertEqual((m["candidate_recovered"], m["selected_recovered"], m["evidence_lost"]), (2, 1, 1))
        self.assertAlmostEqual(m["candidate_recovery_rate"], 2 / 3)
        self.assertAlmostEqual(m["selected_recovery_rate"], 1 / 3)
        self.assertEqual(m["evidence_loss_rate"], 0.5)
        self.assertEqual(m["base_recall_at5"], 0.4)
        self.assertEqual(m["enhanced_recall_at5"], 0.4)
        self.assertEqual(m["recall_at5_delta"], 0.0)
        self.assertTrue(result["per_sample"][1]["strict_id"]["candidate_recovered"])
        self.assertFalse(result["per_sample"][1]["strict_id"]["selected_recovered"])

    def test_cost_and_statuses_include_all_queries_even_no_rewrite(self):
        m = evaluate_nq(self.records, self.annotations)["metrics"]
        self.assertEqual(m["cost"]["base_retrieval_calls"], 5)
        self.assertEqual(m["cost"]["additional_retrieval_calls"], 3)
        self.assertEqual(m["cost"]["total_retrieval_calls"], 8)
        self.assertEqual(m["cost"]["llm_calls"], 0)
        self.assertEqual(m["cost"]["mean_additional_retrieval_calls"], 0.6)
        self.assertAlmostEqual(m["cost"]["mean_total_latency_seconds"], 0.5)
        self.assertEqual(m["statuses"], {"completed": 3, "no_rewrite": 2})

    def test_alias_of_base_text_is_not_falsely_counted_as_recovery(self):
        run = record("1", "ABCDE", extra_calls=0)
        run["candidates"] = hits("ABCDE")
        run["candidates"][0]["alias_passage_ids"] = ["F"]
        metrics = evaluate_nq([run], [{"id": "1", "gold_passage_id": "F"}])["metrics"]
        self.assertEqual(metrics["strict_id"]["enhanced_recall_at5"], 0.0)
        self.assertEqual(metrics["alias_aware"]["base_recall_at5"], 1.0)
        self.assertEqual(metrics["alias_aware"]["enhanced_recall_at5"], 1.0)
        self.assertEqual(metrics["alias_aware"]["selected_recovered"], 0)

    def test_empty_and_zero_denominators_are_null(self):
        metrics = evaluate_nq([], [])["metrics"]
        self.assertEqual(metrics["n"], 0)
        self.assertIsNone(metrics["strict_id"]["base_recall_at5"])
        self.assertIsNone(metrics["cost"]["mean_total_latency_seconds"])
        hit = evaluate_nq([record("1", "ABCDE")], [{"id": "1", "gold_passage_id": "A"}])["metrics"]["strict_id"]
        self.assertIsNone(hit["selected_recovery_rate"])
        miss = evaluate_nq([record("1", "ABCDE")], [{"id": "1", "gold_passage_id": "Z"}])["metrics"]["strict_id"]
        self.assertIsNone(miss["evidence_loss_rate"])

    def test_id_join_is_order_independent_and_inputs_are_not_modified(self):
        before = deepcopy((self.records, self.annotations))
        a = evaluate_nq(self.records, self.annotations)
        b = evaluate_nq(self.records, list(reversed(self.annotations)))
        self.assertEqual(a, b)
        self.assertEqual(before, (self.records, self.annotations))

    def test_reference_answers_and_gold_text_are_not_read(self):
        class Label(dict):
            def get(self, key, *args):
                if key != "gold_passage_id":
                    raise AssertionError(f"unexpected label access: {key}")
                return super().get(key, *args)
        labels = [Label(item, short_answer="ignored", gold_evidence_text="ignored") for item in self.annotations]
        self.assertEqual(evaluate_nq(self.records, labels), evaluate_nq(self.records, self.annotations))

    def test_duplicate_missing_or_invalid_labels_fail_explicitly(self):
        for labels in ([], [self.annotations[0]] * 2,
                       [{"id": "1", "gold_passage_id": None}],
                       [{"id": "1", "gold_passage_id": ""}]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_nq(self.records[:1], labels)
        with self.assertRaisesRegex(ValueError, "duplicate runtime"):
            evaluate_nq(self.records[:1] * 2, self.annotations)

    def test_invalid_runs_fail_instead_of_polluting_metrics(self):
        changes = [lambda r: r.update(base_evidence=hits("ABCDEF")),
                   lambda r: r.update(dataset="fever"),
                   lambda r: r.update(enhanced_evidence=hits("Z")),
                   lambda r: r.update(base_latency=float("nan")),
                   lambda r: r["trace"].update(retrieval_calls=-1),
                   lambda r: r["trace"].update(llm_calls=True),
                   lambda r: r["trace"].update(latency=float("inf")),
                   lambda r: r["trace"].update(status=""),
                   lambda r: r["candidates"][0].update(alias_passage_ids="F")]
        for change in changes:
            run = deepcopy(self.records[0])
            change(run)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_nq([run], self.annotations)


if __name__ == "__main__":
    unittest.main()
