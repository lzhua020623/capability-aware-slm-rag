"""Human uncertainty and different outputs for one sample must survive confirmation."""
import unittest

from scripts.apply_answer_repair_final_confirmations import apply


FIELDS = ("initial_correct", "final_correct", "initial_supported", "final_supported")


def row(number, values, strategy="short_answer_quote", sample_id="same-sample"):
    return {"review_number": number, "review_set_id": "final", "source_alias": "N96", "source_run": "final_nq75",
            "dataset": "nq", "sample_id": sample_id, "cohort": "nq75", "strategy": strategy,
            "result_sha256": f"result-{number}", "review_material_sha256": f"material-{number}",
            "query": "Question", "initial_answer": "initial", "final_answer": f"final-{number}", "changed": True,
            "agent_review": {**dict(zip(FIELDS, values)), "reason_zh": "Explicit assisted opinion"}}


def confirmation(accepted, overrides=None, numbers=None):
    overrides = overrides or {}
    return {"review_set_id": "final", "reviewer": "user", "confirmed_at": "2026-10-21",
            "timezone": "Australia/Sydney", "user_text": "Explicit decisions", "mode": "chat_assisted",
            "source_sha256": {}, "accept_agent": accepted, "override": overrides,
            "all_review_numbers_addressed": numbers or [*accepted, *map(int, overrides)]}


class FinalConfirmationTests(unittest.TestCase):
    def test_agreeing_to_uncertainty_does_not_resolve_it(self):
        outputs, report = apply([row(4, (False, None, False, True))], confirmation([4]))
        self.assertIsNone(outputs[0]["human_review"]["final_correct"])
        self.assertEqual(report["correctness_resolved_pairs"], 0)
        group = next(iter(report["by_run"].values()))
        self.assertIsNone(group["all_changed_pairs_net_change"])
        self.assertEqual(group["all_changed_pairs_net_change_bounds"], [0, 1])

    def test_same_sample_different_materials_are_not_copied(self):
        outputs, report = apply([row(13, (False, True, False, True)),
                                 row(15, (False, False, False, False), "conservative_repair")], confirmation([13, 15]))
        self.assertEqual([x["human_review"]["outcome"] for x in outputs], ["fixed", "wrong_preserved"])
        self.assertEqual(report["human_reviewed_pairs"], 2)
        self.assertEqual(report["unique_dataset_sample_ids_reviewed"], 1)
        self.assertEqual(len(report["by_run"]), 2)

    def test_preference_is_separate_from_binary_regression(self):
        c = confirmation([], {"12": {"initial_correct": True, "final_correct": None,
                                      "initial_supported": None, "final_supported": None,
                                      "preference": "initial_better", "reason": "Prefer full answer, keep correctness open",
                                      "support_fields_confirmed": False}})
        outputs, report = apply([row(12, (True, True, True, True))], c)
        self.assertEqual(outputs[0]["human_review"]["outcome"], "correctness_unresolved")
        self.assertEqual(report["initial_answer_preferred_numbers"], [12])
        group = next(iter(report["by_run"].values()))
        self.assertEqual(group["all_changed_pairs_net_change_bounds"], [-1, 0])
        self.assertNotIn("regressed", group["outcomes"])

    def test_explicit_both_wrong_is_not_an_assisted_regression(self):
        c = confirmation([], {"37": {"initial_correct": False, "final_correct": False,
                                      "initial_supported": None, "final_supported": None, "preference": None,
                                      "reason": "Both need the specific recipient", "support_fields_confirmed": False}})
        outputs, report = apply([row(37, (True, False, True, False))], c)
        self.assertEqual(outputs[0]["human_review"]["outcome"], "wrong_preserved")
        self.assertEqual(report["correctness_resolved_pairs"], 1)
        self.assertEqual(report["support_unresolved_numbers"], [37])

    def test_missing_or_conflicting_coverage_is_rejected(self):
        pairs = [row(1, (False, True, False, True))]
        for c in [confirmation([]), confirmation([1], {"1": {}}), confirmation([99])]:
            with self.subTest(c=c), self.assertRaises(ValueError):
                apply(pairs, c)

    def test_prior_initial_human_labels_need_explicit_separate_reconsideration(self):
        pair = row(1, (False, True, False, True))
        pair["initial_human_review"] = {"initial_correct": True}
        with self.assertRaisesRegex(ValueError, "prior C1 labels"):
            apply([pair], confirmation([1]))

    def test_closed_preference_keeps_unscored_label_and_net_bounds(self):
        c = confirmation([], {"12": {"initial_correct": True, "final_correct": None,
                                      "initial_supported": None, "final_supported": None,
                                      "preference": "initial_better", "reason": "Keep C1 without a binary verdict",
                                      "support_fields_confirmed": False}})
        c["workflow_decisions"] = {"12": {"status": "closed_keep_initial", "reason": "Already handled"}}
        outputs, report = apply([row(12, (True, True, True, True))], c)
        self.assertIsNone(outputs[0]["human_review"]["final_correct"])
        self.assertEqual(report["workflow_pending_numbers"], [])
        self.assertEqual(report["closed_without_binary_score_numbers"], [12])
        self.assertEqual(next(iter(report["by_run"].values()))["all_changed_pairs_net_change_bounds"], [-1, 0])

    def test_external_check_does_not_become_human_gold(self):
        pair = row(19, (True, None, True, True))
        c = confirmation([19])
        c["workflow_decisions"] = {"19": {"status": "closed_external_check", "reason": "Fact check completed",
                                           "external_assessment": {"basis": "assistant_external_fact_check",
                                                                   "initial_correct": True, "final_correct": True,
                                                                   "sources": [{"url": "https://example.test/record", "title": "Record"}]}}}
        outputs, report = apply([pair], c)
        self.assertIsNone(outputs[0]["human_review"]["final_correct"])
        self.assertEqual(report["workflow_closed_pairs"], 1)
        self.assertEqual(report["correctness_resolved_pairs"], 0)
        c["workflow_decisions"]["19"]["external_assessment"]["sources"] = []
        with self.assertRaises(ValueError):
            apply([pair], c)


if __name__ == "__main__":
    unittest.main()
