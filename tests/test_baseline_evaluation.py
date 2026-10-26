"""Smoke tests for baselines/ (synthetic only; no frozen Test inference)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

from baselines._lib.metrics_io import (  # noqa: E402
    aggregate_metrics,
    append_prediction,
    load_done_ids,
    score_row,
)
from baselines._lib.prompts import build_closed_book_prompt  # noqa: E402
from baselines._lib.records_io import synthetic_smoke_records  # noqa: E402
from baselines._lib.split_io import SplitError, extract_test_ids, validate_split_file  # noqa: E402
from baselines.run_baseline_evaluation import (  # noqa: E402
    STATUS_FROZEN_SPLIT_MISSING,
    build_output_row,
    load_mode_config,
    main,
    parse_args,
)
from src.evaluation.qa import best_em_f1  # noqa: E402
from src.generation.qwen import build_rag_prompt  # noqa: E402


class ConfigAndCLITests(unittest.TestCase):
    def test_mode_configs_exist(self):
        for mode in ("qwen_only", "rag"):
            cfg = load_mode_config(mode)
            self.assertEqual(cfg["mode"], mode)
            self.assertEqual(cfg["generator"]["model_name"], "Qwen/Qwen2.5-7B-Instruct")
            self.assertEqual(cfg["split"]["expected_test_n"], 11114)

    def test_parse_mode_required(self):
        with self.assertRaises(SystemExit):
            parse_args([])
        ns = parse_args(["--mode", "rag", "--smoke", "--dry-run"])
        self.assertEqual(ns.mode, "rag")
        self.assertTrue(ns.smoke)


class PromptAndSchemaTests(unittest.TestCase):
    def test_closed_book_has_no_context_block(self):
        text = build_closed_book_prompt("What is 2+2?")
        self.assertIn("Question: What is 2+2?", text)
        self.assertNotIn("Context:", text)
        self.assertNotIn("using only the provided context", text)
        self.assertIn("shortest answer span", text)

    def test_closed_book_shares_format_with_rag_instruction(self):
        from baselines._lib.prompts import CLOSED_BOOK_INSTRUCTION
        from src.generation.qwen import PROMPT_INSTRUCTION

        self.assertEqual(
            PROMPT_INSTRUCTION.replace(" using only the provided context", ""),
            CLOSED_BOOK_INSTRUCTION,
        )

    def test_rag_prompt_from_main(self):
        prompt = build_rag_prompt("Q?", [{"text": "passage A"}])
        self.assertIn("Context:", prompt)
        self.assertIn("[1] passage A", prompt)

    def test_output_schema_qwen_only(self):
        row = build_output_row(
            mode="qwen_only",
            sample_id="s1",
            query="q",
            prediction="blue",
            gold=["blue"],
        )
        self.assertEqual(
            set(row),
            {"sample_id", "query", "prediction", "gold", "mode", "em", "f1"},
        )
        self.assertNotIn("retrieved_passage_ids", row)

    def test_output_schema_rag(self):
        row = build_output_row(
            mode="rag",
            sample_id="s1",
            query="q",
            prediction="blue",
            gold=["blue"],
            hits=[{"passage_id": "p1", "text": "t", "rank": 1}],
        )
        self.assertEqual(row["retrieved_passage_ids"], ["p1"])
        self.assertEqual(row["retrieved_ranks"], [1])

    def test_evaluator_is_main_qa(self):
        em, f1 = score_row("Blue", ["blue"])
        em2, f1_2 = best_em_f1("Blue", ["blue"])
        self.assertEqual((em, f1), (em2, f1_2))
        self.assertEqual(em, 1)


class SplitValidationTests(unittest.TestCase):
    def test_rejects_wrong_count_and_dupes(self):
        split = {
            "version": "diagnosis_nq_train_74097_rep08",
            "splits": {"nq": {"test": ["a", "b", "a"]}},
        }
        with self.assertRaises(SplitError):
            extract_test_ids(
                split,
                test_key=["splits", "nq", "test"],
                expected_test_n=3,
                version_prefix="diagnosis_nq_train_74097",
            )

    def test_validate_split_file_ok(self):
        ids = [f"id_{i}" for i in range(11114)]
        payload = {
            "version": "diagnosis_nq_train_74097_rep08",
            "splits": {"nq": {"train": [], "validation": [], "test": ids}},
            "summary": {"test": 11114},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            summary = validate_split_file(
                path,
                test_key=["splits", "nq", "test"],
                expected_test_n=11114,
                version_prefix="diagnosis_nq_train_74097",
            )
            self.assertTrue(summary["ok"])
            self.assertEqual(summary["test_n"], 11114)


class ResumeAndSmokeCLITests(unittest.TestCase):
    def test_resume_skips_done_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "predictions.jsonl"
            append_prediction(
                pred,
                build_output_row(
                    mode="qwen_only",
                    sample_id="smoke_001",
                    query="q",
                    prediction="blue",
                    gold=["blue"],
                ),
            )
            done = load_done_ids(pred)
            self.assertEqual(done, {"smoke_001"})
            remaining = [r for r in synthetic_smoke_records(3) if r["sample_id"] not in done]
            self.assertEqual([r["sample_id"] for r in remaining], ["smoke_002", "smoke_003"])

    def test_aggregate_metrics(self):
        rows = [
            build_output_row(
                mode="qwen_only", sample_id="a", query="q", prediction="x", gold=["x"]
            ),
            build_output_row(
                mode="qwen_only", sample_id="b", query="q", prediction="y", gold=["z"]
            ),
        ]
        m = aggregate_metrics(rows, mode="qwen_only")
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["em"], 0.5)

    def test_cli_smoke_dry_run_both_modes(self):
        for mode in ("qwen_only", "rag"):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / mode
                code = main(
                    [
                        "--mode",
                        mode,
                        "--smoke",
                        "--dry-run",
                        "--limit",
                        "2",
                        "--output-dir",
                        str(out),
                    ]
                )
                self.assertEqual(code, 0)
                pred = out / "predictions.jsonl"
                metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
                run_cfg = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
                self.assertTrue(pred.is_file())
                self.assertEqual(metrics["n"], 2)
                self.assertEqual(run_cfg["mode"], mode)
                lines = [json.loads(x) for x in pred.read_text(encoding="utf-8").splitlines() if x]
                self.assertEqual(len(lines), 2)
                if mode == "rag":
                    self.assertIn("retrieved_passage_ids", lines[0])
                else:
                    self.assertNotIn("retrieved_passage_ids", lines[0])

    def test_cli_resume_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "qwen_only"
            self.assertEqual(
                main(
                    [
                        "--mode",
                        "qwen_only",
                        "--smoke",
                        "--dry-run",
                        "--limit",
                        "2",
                        "--output-dir",
                        str(out),
                    ]
                ),
                0,
            )
            # second pass with resume + larger limit should append only new ids
            self.assertEqual(
                main(
                    [
                        "--mode",
                        "qwen_only",
                        "--smoke",
                        "--dry-run",
                        "--limit",
                        "3",
                        "--resume",
                        "--output-dir",
                        str(out),
                    ]
                ),
                0,
            )
            lines = [
                json.loads(x)
                for x in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if x
            ]
            ids = [r["sample_id"] for r in lines]
            self.assertEqual(ids, ["smoke_001", "smoke_002", "smoke_003"])

    def test_missing_split_reports_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no_such_split.json"
            out = Path(tmp) / "out"
            code = main(
                [
                    "--mode",
                    "qwen_only",
                    "--split-file",
                    str(missing),
                    "--output-dir",
                    str(out),
                ]
            )
            self.assertEqual(code, 2)
            readiness = json.loads((out / "readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["status"], STATUS_FROZEN_SPLIT_MISSING)

    def test_no_controller_imports_in_runner(self):
        text = (ROOT / "baselines" / "run_baseline_evaluation.py").read_text(encoding="utf-8")
        for banned in (
            "src.controller",
            "src.compensation",
            "src.diagnosis",
            "src.probe",
            "run_final_test_evaluation",
        ):
            self.assertNotIn(banned, text)


if __name__ == "__main__":
    unittest.main()
