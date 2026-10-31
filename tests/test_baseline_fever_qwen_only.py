"""FEVER Qwen-only baseline: scoring, loader, CLI smoke (dry-run), NQ isolation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from baselines._lib.fever_scoring import (  # noqa: E402
    aggregate_fever_metrics,
    build_fever_output_row,
    normalize_fever_prediction,
)
from baselines._lib.prompts import build_fever_closed_book_prompt  # noqa: E402
from baselines._lib.records_io import (  # noqa: E402
    load_fever_dev_records,
    sample_fever_smoke_records,
)
from baselines.run_baseline_evaluation import (  # noqa: E402
    STATUS_READY_FEVER,
    fever_readiness,
    load_mode_config,
    main,
    parse_args,
)


class FeverNormalizeTests(unittest.TestCase):
    def test_exact_and_case(self):
        self.assertEqual(normalize_fever_prediction("SUPPORTS"), "SUPPORTS")
        self.assertEqual(normalize_fever_prediction(" refutes "), "REFUTES")
        self.assertEqual(normalize_fever_prediction('"Supports"'), "SUPPORTS")

    def test_invalid_never_defaults(self):
        self.assertEqual(normalize_fever_prediction(""), "INVALID")
        self.assertEqual(normalize_fever_prediction(None), "INVALID")
        self.assertEqual(normalize_fever_prediction("NOT ENOUGH INFO"), "INVALID")
        self.assertEqual(normalize_fever_prediction("yes"), "INVALID")
        self.assertEqual(normalize_fever_prediction("SUPPORTS or REFUTES"), "INVALID")

    def test_single_token_in_noise(self):
        self.assertEqual(normalize_fever_prediction("Answer: SUPPORTS."), "SUPPORTS")
        self.assertEqual(normalize_fever_prediction("The claim REFUTES."), "REFUTES")


class FeverMetricsTests(unittest.TestCase):
    def test_aggregate_keys_and_invalid(self):
        rows = [
            build_fever_output_row(
                sample_id="1", claim="c1", prediction="SUPPORTS", gold="SUPPORTS"
            ),
            build_fever_output_row(
                sample_id="2", claim="c2", prediction="REFUTES", gold="REFUTES"
            ),
            build_fever_output_row(
                sample_id="3", claim="c3", prediction="huh", gold="SUPPORTS"
            ),
        ]
        m = aggregate_fever_metrics(rows, mode="fever_qwen_only")
        expected = {
            "n",
            "accuracy",
            "macro_f1",
            "supports_precision",
            "supports_recall",
            "supports_f1",
            "refutes_precision",
            "refutes_recall",
            "refutes_f1",
            "invalid_n",
            "mode",
            "dataset",
        }
        self.assertEqual(set(m), expected)
        self.assertEqual(m["n"], 3)
        self.assertEqual(m["invalid_n"], 1)
        self.assertAlmostEqual(m["accuracy"], 2 / 3)
        self.assertEqual(m["dataset"], "fever")
        self.assertEqual(m["mode"], "fever_qwen_only")
        self.assertNotIn("em", m)
        self.assertNotIn("f1", m)

    def test_output_row_schema(self):
        row = build_fever_output_row(
            sample_id="42",
            claim="Fox released Soul Food.",
            prediction="supports",
            gold="SUPPORTS",
        )
        self.assertEqual(row["dataset"], "fever")
        self.assertEqual(row["mode"], "fever_qwen_only")
        self.assertEqual(row["claim"], "Fox released Soul Food.")
        self.assertEqual(row["query"], row["claim"])
        self.assertEqual(row["normalized_prediction"], "SUPPORTS")
        self.assertTrue(row["correct"])


class FeverPromptTests(unittest.TestCase):
    def test_closed_book_no_evidence(self):
        text = build_fever_closed_book_prompt("The sky is green.")
        self.assertIn("Claim: The sky is green.", text)
        self.assertNotIn("Evidence:", text)
        self.assertNotIn("Context:", text)
        self.assertIn("SUPPORTS", text)
        self.assertIn("REFUTES", text)
        self.assertIn("Do not provide explanations", text)


class FeverLoaderTests(unittest.TestCase):
    def test_load_and_balance_from_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fever_dev.jsonl"
            rows_in = []
            for i in range(10):
                label = "SUPPORTS" if i < 5 else "REFUTES"
                rows_in.append(
                    {"id": 1000 + i, "claim": f"claim {i}", "label": label}
                )
            path.write_text(
                "\n".join(json.dumps(r) for r in rows_in) + "\n", encoding="utf-8"
            )
            loaded = load_fever_dev_records(path, expected_n=10)
            self.assertEqual(len(loaded), 10)
            self.assertEqual(len({r["sample_id"] for r in loaded}), 10)
            smoke = sample_fever_smoke_records(loaded, n=4, seed=7, balanced=True)
            self.assertEqual(len(smoke), 4)
            labels = [r["gold"] for r in smoke]
            self.assertEqual(labels.count("SUPPORTS"), 2)
            self.assertEqual(labels.count("REFUTES"), 2)

    def test_frozen_fever_eval_records_if_present(self):
        path = ROOT / "baselines" / "data" / "fever_dev_13332.jsonl"
        if not path.is_file():
            self.skipTest("fever_dev_13332.jsonl not present")
        rows = load_fever_dev_records(
            path,
            id_field="sample_id",
            claim_field="claim",
            gold_field="gold",
            expected_n=13332,
        )
        self.assertEqual(len(rows), 13332)
        self.assertEqual(len({r["sample_id"] for r in rows}), 13332)
        labels = {r["gold"] for r in rows}
        self.assertEqual(labels, {"SUPPORTS", "REFUTES"})
        from collections import Counter

        counts = Counter(r["gold"] for r in rows)
        self.assertEqual(counts["SUPPORTS"], 6666)
        self.assertEqual(counts["REFUTES"], 6666)
        # Compact schema only — no corpus/evidence blobs.
        sample = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(
            set(sample),
            {"sample_id", "claim", "query", "gold", "dataset"},
        )


class FeverConfigCLITests(unittest.TestCase):
    def test_config_independent_of_nq(self):
        fever = load_mode_config("fever_qwen_only")
        qwen = load_mode_config("qwen_only")
        self.assertEqual(fever["mode"], "fever_qwen_only")
        self.assertEqual(fever["generator"]["model_name"], qwen["generator"]["model_name"])
        self.assertEqual(fever["generator"]["quantization"], "nf4")
        self.assertIsNone(fever["retrieval"])
        self.assertEqual(fever["dataset"]["expected_n"], 13332)
        self.assertIn("fever_dev_13332.jsonl", fever["dataset"]["records_path"])
        self.assertIn("fever_dev_13332_split.json", fever["split"]["default_path"])
        self.assertEqual(fever["output"]["default_dir"], "baselines/outputs/fever_qwen_only")
        self.assertNotIn("nq_train", fever["dataset"]["records_path"])
        self.assertNotEqual(
            Path(fever["_config_path"]).name, Path(qwen["_config_path"]).name
        )
        # NQ config paths unchanged.
        self.assertIn("nq_train_74097", qwen["dataset"]["records_path"])
        self.assertEqual(qwen["split"]["expected_test_n"], 11114)

    def test_parse_fever_mode(self):
        ns = parse_args(["--mode", "fever_qwen_only", "--smoke", "--dry-run", "--limit", "4"])
        self.assertEqual(ns.mode, "fever_qwen_only")
        self.assertTrue(ns.smoke)
        self.assertEqual(ns.limit, 4)

    def test_validate_readiness(self):
        cfg = load_mode_config("fever_qwen_only")
        path = ROOT / cfg["dataset"]["records_path"]
        if not path.is_file():
            self.skipTest("fever_dev_13332.jsonl not present")
        status, details = fever_readiness(cfg)
        self.assertEqual(status, STATUS_READY_FEVER)
        self.assertEqual(details["n"], 13332)
        self.assertEqual(details["label_counts"]["SUPPORTS"], 6666)
        self.assertEqual(details["label_counts"]["REFUTES"], 6666)

    def test_validate_split_only_cli_no_inference(self):
        cfg = load_mode_config("fever_qwen_only")
        if not (ROOT / cfg["dataset"]["records_path"]).is_file():
            self.skipTest("fever bundle not present")
        rc = main(["--mode", "fever_qwen_only", "--validate-split-only"])
        self.assertEqual(rc, 0)

    def test_resources_manifest_readable(self):
        path = ROOT / "baselines" / "resources" / "fever" / "fever_resources_v1.json"
        if not path.is_file():
            self.skipTest("fever_resources_v1.json not present")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["dataset"], "fever")
        names = {r["logical_name"] for r in data["resources"]}
        self.assertIn("fever_corpus", names)
        self.assertIn("fever_bge_ivfpq_index", names)

    def test_dry_run_smoke_writes_fever_outputs_only(self):
        cfg = load_mode_config("fever_qwen_only")
        path = ROOT / cfg["dataset"]["records_path"]
        if not path.is_file():
            self.skipTest("fever_dev_13332.jsonl not present")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fever_qwen_only_smoke"
            # Ensure NQ formal dir is untouched even if present.
            nq_pred = ROOT / "baselines" / "outputs" / "qwen_only" / "predictions.jsonl"
            nq_before = nq_pred.read_bytes() if nq_pred.is_file() else None
            rc = main(
                [
                    "--mode",
                    "fever_qwen_only",
                    "--smoke",
                    "--dry-run",
                    "--limit",
                    "4",
                    "--output-dir",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["n"], 4)
            self.assertEqual(metrics["dataset"], "fever")
            self.assertEqual(metrics["mode"], "fever_qwen_only")
            self.assertIn("macro_f1", metrics)
            self.assertNotIn("em", metrics)
            preds = (out / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(preds), 4)
            row = json.loads(preds[0])
            self.assertEqual(row["dataset"], "fever")
            self.assertIn(row["gold"], ("SUPPORTS", "REFUTES"))
            if nq_before is not None:
                self.assertEqual(nq_pred.read_bytes(), nq_before)

    def test_nq_modes_unchanged_in_choices(self):
        # NQ modes still accepted; fever is additive.
        ns = parse_args(["--mode", "qwen_only", "--smoke", "--dry-run"])
        self.assertEqual(ns.mode, "qwen_only")


if __name__ == "__main__":
    unittest.main()
