"""Confirmations are applied by review number, copied to duplicates, never inferred."""
import unittest

from scripts.apply_answer_repair_confirmations import apply_confirmations

FIELDS = ("initial_correct", "final_correct", "initial_supported", "final_supported")


def pair(number, sample_id, cohort, values, changed=True):
    return {"review_number": number, "shared_review_id": f"nq:{sample_id}", "dataset": "nq", "cohort": cohort,
            "strategy": "extractive_reanswer", "sample_id": sample_id, "result_sha256": f"sha{number}",
            "review_material_sha256": "m", "query": "q", "initial_answer": "a", "final_answer": "b" if changed else "a",
            "changed": changed, "agent_review": {**dict(zip(FIELDS, values)), "reason_zh": "理由"}}


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.pairs = [pair(1, "x", "reviewed", (False, True, False, True)),
                      pair(2, "x", "dev50", (False, True, False, True)),
                      pair(3, "y", "dev50", (True, False, True, False)),
                      pair(4, "z", "dev50", (None, True, True, True)),
                      pair(5, "w", "dev50", (True, True, True, True), changed=False)]
        self.base = {"reviewer": "Qihan Zhu", "confirmed_at": "2026-10-14", "user_text": "同意"}

    def test_accept_copies_to_duplicates_and_counts_unique_fixes(self):
        rows, report = apply_confirmations(self.pairs, {**self.base, "accept_agent": [1, 3]})
        status = {r["review_number"]: r["pair_review"]["status"] for r in rows}
        self.assertEqual(status, {1: "completed", 2: "completed", 3: "completed", 4: "pending", 5: "pending"})
        self.assertEqual((report["unique_fixed"], report["unique_regressed"], report["unique_pending"]), (1, 1, 2))
        self.assertEqual(report["by_run"]["nq/dev50/extractive_reanswer"]["fixed"], 1)
        self.assertEqual(rows[0]["result_sha256"], "sha1")

    def test_override_resolves_unresolved_and_requires_booleans(self):
        rows, report = apply_confirmations(self.pairs, {**self.base, "override": {"4": {"initial_correct": True, "reason": "口径 2"}}})
        self.assertEqual(rows[3]["pair_review"]["initial_correct"], True)
        self.assertIn("用户改判", rows[3]["pair_review"]["reason"])
        self.assertEqual(report["unique_fixed"], 0)
        with self.assertRaises(ValueError):
            apply_confirmations(self.pairs, {**self.base, "override": {"4": {"initial_correct": "yes"}}})
        with self.assertRaises(ValueError):
            apply_confirmations(self.pairs, {**self.base, "accept_agent": [4]})   # still unresolved

    def test_conflicts_unknown_numbers_and_missing_reviewer_are_rejected(self):
        for bad in ({"accept_agent": [1], "skip": [1]}, {"accept_agent": [99]},
                    {"accept_agent": [1], "override": {"2": {"final_correct": False}}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                apply_confirmations(self.pairs, {**self.base, **bad})
        with self.assertRaises(ValueError):
            apply_confirmations(self.pairs, {"reviewer": " ", "confirmed_at": "d", "user_text": "x", "accept_agent": [1]})


if __name__ == "__main__":
    unittest.main()
