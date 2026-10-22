"""Strata never read answer metrics; roles/exposure and runtime isolation hold."""
import json
from pathlib import Path
import unittest

from scripts.prepare_nq_label_audit import LABELS, select


class LabelAuditSelectionTests(unittest.TestCase):
    def fixture(self):
        rows, roles = [], {}
        for label in LABELS:
            for n in range(12):
                sid = f"{label}-{n}"
                role = "development" if n < 10 else "reserved_evaluation"
                rows.append({"sample_id": sid, "data_role": role, "label": label,
                             "em": n % 2, "gold": "unused"})
                roles[sid] = role
        return rows, roles

    def test_stable_balanced_selection_excludes_roles_exposures_and_ignores_metrics(self):
        rows, roles = self.fixture()
        excluded = {rows[0]["sample_id"]}
        first = select(rows, roles, excluded, count=3)
        changed = [dict(r, em=999, gold="changed", prediction="changed") for r in reversed(rows)]
        self.assertEqual(first, select(changed, roles, excluded, count=3))
        ids, strata, _ = first
        self.assertEqual(len(ids), 12)
        self.assertTrue(all(roles[sid] == "development" and sid not in excluded for sid in ids))
        self.assertEqual({key: list(strata.values()).count(key) for key in LABELS}, {key: 3 for key in LABELS})

    def test_incomplete_duplicate_or_changed_partition_is_rejected(self):
        rows, roles = self.fixture()
        for changed in (rows[:-1], rows + [rows[0]], [dict(rows[0], data_role="calibration"), *rows[1:]]):
            with self.assertRaises(ValueError):
                select(changed, roles, set(), count=3)

    def test_real_packet_complete_pending_and_separate_from_runtime(self):
        root = Path(__file__).resolve().parents[1] / "artifacts/answer_repair/nq_label_audit_200_v1"
        if not (root / "manifest.json").exists():
            self.skipTest("optional archived packet not present")
        manifest = json.loads((root / "manifest.json").read_text())
        packet = json.loads((root / "review_packet.json").read_text())
        self.assertEqual(len(packet["records"]), 200)
        self.assertFalse(set(manifest["sample_ids"]) & set(manifest["excluded_known_exposed_ids"]))
        runtime = [json.loads(line) for line in (root / "runtime.jsonl").read_text().splitlines()]
        for row, pure in zip(packet["records"], runtime):
            self.assertFalse(row["human_verified"])
            self.assertEqual(row["human_review"]["status"], "pending")
            self.assertIsNone(row["human_review"]["answer_supported_by_top5"])
            self.assertEqual(set(pure), {"sample_id", "dataset", "query", "initial_answer", "evidence"})
            self.assertEqual(pure["sample_id"], row["sample_id"])


if __name__ == "__main__":
    unittest.main()
