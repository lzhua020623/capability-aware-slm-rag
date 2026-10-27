"""Recount must validate all joins and stop before exporting mismatched weights."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.recompute_phase5_7_routing import export, recompute


class RoutingExportTests(unittest.TestCase):
    def sources(self):
        def run(command, cwd):
            path = command[-1].split(":", 1)[1]
            if "/controller/" in path:
                return json.dumps({"decisions": [{"sample_id": "x", "dataset": "nq", "action": "retrieval_enhancement"}]}).encode()
            if "/diagnosis/" in path:
                return json.dumps({"predictions": [{"sample_id": "x", "dataset": "nq", "target_label": "correct"}]}).encode()
            return json.dumps({"samples": [{"sample_id": "x", "dataset": "nq", "offline_evaluation": {
                "target_label": "correct", "prompt_development_sample": False}}]}).encode()
        return run

    def test_different_counts_stop_before_export(self):
        with tempfile.TemporaryDirectory() as t, patch("scripts.recompute_phase5_7_routing.subprocess.check_output", side_effect=self.sources()):
            target = Path(t) / "output"
            with self.assertRaisesRegex(ValueError, "STOP:.*no weights computed"):
                export(t, target)
            self.assertFalse(target.exists())

    def test_disagreeing_target_is_rejected_before_count_gate(self):
        source = self.sources()
        def disagree(command, cwd):
            raw = source(command, cwd)
            if "/probe/" in command[-1]:
                data = json.loads(raw)
                data["samples"][0]["offline_evaluation"]["target_label"] = "evidence_insufficiency"
                return json.dumps(data).encode()
            return raw
        with patch("scripts.recompute_phase5_7_routing.subprocess.check_output", side_effect=disagree):
            with self.assertRaisesRegex(ValueError, "target_label mismatch"):
                recompute("unused")


if __name__ == "__main__":
    unittest.main()
