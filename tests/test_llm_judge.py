"""Unit tests for baselines LLM Judge (mocked OpenAI; no network)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

from baselines._lib.judge_io import (  # noqa: E402
    PILOT_GROUP_ORDER,
    build_judge_payload,
    build_judge_prompt,
    is_normalized_correct,
    load_judge_config,
    parse_judge_response_text,
    pilot_stratum,
    select_stratified_pilot,
    summarize_results,
)
from baselines.judge import run_llm_judge  # noqa: E402


# Calibration cases for rubric policy (not hardcoded into production judge logic).
RUBRIC_CALIBRATION_CASES = [
    {
        "name": "age_concise",
        "question": "What is the minimum age to serve?",
        "prediction": "18",
        "gold": ["18 years or older"],
        "expected": "CORRECT",
    },
    {
        "name": "year_when",
        "question": "When did the event happen?",
        "prediction": "2010",
        "gold": ["15 July 2010"],
        "expected": "CORRECT",
    },
    {
        "name": "year_when_1916",
        "question": "When was it founded?",
        "prediction": "1916",
        "gold": ["August 1, 1916"],
        "expected": "CORRECT",
    },
    {
        "name": "province_where",
        "question": "Where is it located?",
        "prediction": "Gansu Province",
        "gold": ["a county in Gansu Province, China"],
        "expected": "CORRECT",
    },
    {
        "name": "partial_language_set",
        "question": "Which languages are official?",
        "prediction": "French",
        "gold": ["Spanish and French"],
        "expected": "INCORRECT",
    },
    {
        "name": "wrong_number",
        "question": "What was the budget?",
        "prediction": "$261b",
        "gold": ["$313.5b"],
        "expected": "INCORRECT",
    },
]


def _pilot_pool_rows() -> list[dict]:
    """Deterministic pool covering all strata + normalized_match=true (excluded)."""
    rows: list[dict] = []
    for i in range(12):
        rows.append(
            {
                "sample_id": f"z{i:02d}",
                "query": f"q_z{i}",
                "prediction": "wrong",
                "gold": ["gold"],
                "em": 0,
                "f1": 0.0,
                "normalized_match": False,
            }
        )
    for i in range(10):
        rows.append(
            {
                "sample_id": f"h{i:02d}",
                "query": f"q_h{i}",
                "prediction": "partial high",
                "gold": ["partial high answer"],
                "em": 0,
                "f1": 0.6 + (i % 3) * 0.1,
                "normalized_match": 0,
            }
        )
    for i in range(11):
        rows.append(
            {
                "sample_id": f"m{i:02d}",
                "query": f"q_m{i}",
                "prediction": "mid",
                "gold": ["mid answer"],
                "em": 0,
                "f1": 0.1 + (i % 4) * 0.1,
                "normalized_match": False,
            }
        )
    for i in range(7):
        rows.append(
            {
                "sample_id": f"r{i:02d}",
                "query": f"q_r{i}",
                "prediction": "rem",
                "gold": ["rem"],
                "em": 1,
                "f1": 1.0,
                "normalized_match": False,
            }
        )
    rows.append(
        {
            "sample_id": "auto_ok",
            "query": "skip me",
            "prediction": "Paris",
            "gold": ["Paris"],
            "em": 1,
            "f1": 1.0,
            "normalized_match": True,
        }
    )
    return rows


class JudgeIoTests(unittest.TestCase):
    def test_normalized_gate(self):
        self.assertTrue(is_normalized_correct({"normalized_match": 1}))
        self.assertTrue(is_normalized_correct({"normalized_match": True}))
        self.assertFalse(is_normalized_correct({"normalized_match": 0}))
        self.assertFalse(is_normalized_correct({"normalized_match": False}))

    def test_payload_excludes_scores(self):
        row = {
            "sample_id": "s1",
            "query": "Who?",
            "prediction": "Ada",
            "gold": ["Ada Lovelace"],
            "em": 0,
            "f1": 0.5,
            "normalized_match": 0,
        }
        payload = build_judge_payload(row)
        self.assertEqual(set(payload), {"question", "gold", "prediction"})
        self.assertNotIn("em", payload)
        self.assertNotIn("f1", payload)
        self.assertNotIn("normalized_match", payload)

    def test_parse_first_line_label_stable(self):
        label, reason = parse_judge_response_text(
            "CORRECT\nsame person",
            max_reason_chars=80,
        )
        self.assertEqual(label, "CORRECT")
        self.assertEqual(reason, "same person")
        label2, reason2 = parse_judge_response_text("INCORRECT\n", max_reason_chars=80)
        self.assertEqual(label2, "INCORRECT")
        self.assertEqual(reason2, "")
        label3, _ = parse_judge_response_text("UNCERTAIN\nmaybe", max_reason_chars=80)
        self.assertEqual(label3, "UNCERTAIN")

    def test_parse_does_not_guess_label_from_reason(self):
        with self.assertRaises(ValueError):
            parse_judge_response_text(
                "The answer looks CORRECT overall\nmore text",
                max_reason_chars=80,
            )
        with self.assertRaises(ValueError):
            parse_judge_response_text(
                '{"label":"CORRECT","reason":"same"}',
                max_reason_chars=80,
            )

    def test_summarize_uncertain_not_correct(self):
        rows = [
            {"auto_correct": True, "judge_label": "CORRECT", "judge_status": "auto_correct"},
            {"auto_correct": False, "judge_label": "CORRECT", "judge_status": "llm_ok"},
            {"auto_correct": False, "judge_label": "UNCERTAIN", "judge_status": "llm_ok"},
            {"auto_correct": False, "judge_label": "INCORRECT", "judge_status": "llm_ok"},
        ]
        summary = summarize_results(rows)
        self.assertEqual(summary["auto_correct_n"], 1)
        self.assertEqual(summary["llm_correct_n"], 1)
        self.assertEqual(summary["llm_uncertain_n"], 1)
        self.assertEqual(summary["judge_assisted_correct_n"], 2)
        self.assertAlmostEqual(summary["judge_assisted_accuracy"], 0.5)

    def test_summarize_api_attempted_includes_failures_and_retries(self):
        rows = [
            {
                "auto_correct": False,
                "judge_label": "INCORRECT",
                "judge_status": "llm_ok",
                "api_attempts": 1,
                "retry_n": 0,
            },
            {
                "auto_correct": False,
                "judge_label": None,
                "judge_status": "llm_failed",
                "api_attempts": 2,
                "retry_n": 1,
            },
            {
                "auto_correct": False,
                "judge_label": None,
                "judge_status": "llm_failed",
                "api_attempts": 1,
                "retry_n": 0,
            },
        ]
        summary = summarize_results(rows)
        self.assertEqual(summary["api_called_n"], 1)
        self.assertEqual(summary["api_succeeded_n"], 1)
        self.assertEqual(summary["failed_n"], 2)
        self.assertEqual(summary["api_attempted_n"], 4)
        self.assertEqual(summary["retry_n"], 1)

    def test_gpt61_sol_responses_payload_omits_sampling(self):
        kwargs = run_llm_judge.build_responses_create_kwargs(
            deployment="gpt-6.1-sol",
            prompt="judge me",
            max_output_tokens=256,
            reasoning_effort="low",
        )
        self.assertEqual(kwargs["model"], "gpt-6.1-sol")
        self.assertEqual(kwargs["input"], "judge me")
        self.assertEqual(kwargs["reasoning"], {"effort": "low"})
        self.assertEqual(kwargs["max_output_tokens"], 256)
        for banned in ("temperature", "top_p", "logprobs", "top_logprobs"):
            self.assertNotIn(banned, kwargs)

    def test_rubric_allows_concise_and_rejects_false_positives(self):
        cfg = load_judge_config()
        self.assertEqual(int(cfg["max_output_tokens"]), 256)
        rubric = cfg["rubric"]
        correct = rubric["CORRECT"].lower()
        incorrect = rubric["INCORRECT"].lower()
        self.assertIn("concise", correct)
        self.assertIn("sufficient", correct)
        self.assertTrue("equivalent" in incorrect or "wrong" in incorrect)

        prompt = build_judge_prompt(
            {"question": "q", "gold": ["g"], "prediction": "p"},
            cfg,
        )
        self.assertIn("Line 1", prompt)
        self.assertIn("CORRECT", prompt)
        self.assertNotIn("JSON object", prompt)

        # Document expected labels for calibration cases (policy tests, not baked into runner).
        by_name = {c["name"]: c for c in RUBRIC_CALIBRATION_CASES}
        self.assertEqual(by_name["age_concise"]["expected"], "CORRECT")
        self.assertEqual(by_name["year_when"]["expected"], "CORRECT")
        self.assertEqual(by_name["year_when_1916"]["expected"], "CORRECT")
        self.assertEqual(by_name["province_where"]["expected"], "CORRECT")
        self.assertEqual(by_name["partial_language_set"]["expected"], "INCORRECT")
        self.assertEqual(by_name["wrong_number"]["expected"], "INCORRECT")

        # Prompt principles must support these classes of decisions.
        blob = (correct + " " + incorrect + " " + prompt).lower()
        self.assertIn("question", blob)
        self.assertTrue("grain" in blob or "detail" in blob or "concise" in blob)
        for case in RUBRIC_CALIBRATION_CASES:
            case_prompt = build_judge_prompt(
                {
                    "question": case["question"],
                    "gold": case["gold"],
                    "prediction": case["prediction"],
                },
                cfg,
            )
            self.assertIn(case["question"], case_prompt)
            self.assertIn(case["prediction"], case_prompt)
            self.assertIn(case["expected"], ("CORRECT", "INCORRECT", "UNCERTAIN"))

    def test_pilot_stratum_labels(self):
        self.assertEqual(pilot_stratum({"em": 0, "f1": 0.0}), "em0_f1_0")
        self.assertEqual(pilot_stratum({"em": 0, "f1": 0.5}), "em0_f1_ge_0_5")
        self.assertEqual(pilot_stratum({"em": 0, "f1": 0.25}), "em0_f1_mid")
        self.assertEqual(pilot_stratum({"em": 1, "f1": 0.0}), "remainder")

    def test_select_stratified_pilot_quotas_and_reproducible(self):
        rows = _pilot_pool_rows()
        selected_a, manifest_a, counts_a = select_stratified_pilot(
            rows, pilot_sample=30, seed=42
        )
        selected_b, manifest_b, counts_b = select_stratified_pilot(
            rows, pilot_sample=30, seed=42
        )
        self.assertEqual(counts_a, {"em0_f1_0": 8, "em0_f1_ge_0_5": 8, "em0_f1_mid": 8, "remainder": 6})
        self.assertEqual(counts_a, counts_b)
        self.assertEqual([r["sample_id"] for r in selected_a], [r["sample_id"] for r in selected_b])
        self.assertEqual(len(selected_a), 30)
        self.assertEqual(len({r["sample_id"] for r in selected_a}), 30)
        self.assertNotIn("auto_ok", {str(r["sample_id"]) for r in selected_a})
        for m in manifest_a:
            self.assertEqual(
                set(m.keys()),
                {
                    "sample_id",
                    "query",
                    "prediction",
                    "gold",
                    "em",
                    "f1",
                    "normalized_match",
                    "pilot_group",
                },
            )
            self.assertFalse(is_normalized_correct(m))
            self.assertIn(m["pilot_group"], PILOT_GROUP_ORDER)
        self.assertEqual(manifest_a, manifest_b)
        selected_c, _, _ = select_stratified_pilot(rows, pilot_sample=30, seed=7)
        self.assertEqual(len(selected_c), 30)

    def test_select_stratified_pilot_shortfall_backfill(self):
        rows = []
        for i in range(3):
            rows.append(
                {
                    "sample_id": f"z{i}",
                    "query": "qz",
                    "prediction": "p",
                    "gold": ["g"],
                    "em": 0,
                    "f1": 0.0,
                    "normalized_match": False,
                }
            )
        for i in range(40):
            rows.append(
                {
                    "sample_id": f"m{i:02d}",
                    "query": "qm",
                    "prediction": "p",
                    "gold": ["g"],
                    "em": 0,
                    "f1": 0.2,
                    "normalized_match": False,
                }
            )
        selected, _, counts = select_stratified_pilot(rows, pilot_sample=30, seed=42)
        self.assertEqual(sum(counts.values()), 30)
        self.assertEqual(counts["em0_f1_0"], 8)
        labeled = [r for r in selected if r["pilot_group"] == "em0_f1_0"]
        self.assertEqual(len(labeled), 8)
        true_zeros = {f"z{i}" for i in range(3)}
        labeled_ids = {str(r["sample_id"]) for r in labeled}
        self.assertTrue(true_zeros.issubset(labeled_ids))
        self.assertEqual(len({str(r["sample_id"]) for r in selected}), 30)


class JudgeRunnerTests(unittest.TestCase):
    def _write_preds(self, path: Path) -> None:
        rows = [
            {
                "sample_id": "auto1",
                "query": "capital of france",
                "prediction": "Paris",
                "gold": ["Paris"],
                "em": 1,
                "f1": 1.0,
                "normalized_match": 1,
            },
            {
                "sample_id": "need1",
                "query": "capital of france",
                "prediction": "City of Light",
                "gold": ["Paris"],
                "em": 0,
                "f1": 0.0,
                "normalized_match": 0,
            },
            {
                "sample_id": "need2",
                "query": "2+2",
                "prediction": "five",
                "gold": ["4", "four"],
                "em": 0,
                "f1": 0.0,
                "normalized_match": 0,
            },
        ]
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    def test_dry_run_default_no_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preds = tmp_path / "predictions.jsonl"
            out = tmp_path / "judge_out"
            self._write_preds(preds)
            with mock.patch.object(run_llm_judge, "create_openai_client") as create_client:
                rc = run_llm_judge.run(
                    [
                        "--input",
                        str(preds),
                        "--output-dir",
                        str(out),
                        "--limit",
                        "3",
                        "--dry-run",
                    ]
                )
                self.assertEqual(rc, 0)
                create_client.assert_not_called()
            results = [
                json.loads(line)
                for line in (out / "judge_results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(results), 3)
            by_id = {r["sample_id"]: r for r in results}
            self.assertTrue(by_id["auto1"]["auto_correct"])
            self.assertEqual(by_id["auto1"]["judge_status"], "auto_correct")
            self.assertEqual(by_id["need1"]["judge_status"], "dry_run")
            summary = json.loads((out / "judge_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["api_called_n"], 0)
            self.assertTrue(summary["dry_run"])

    def test_live_uses_mock_client_and_resume_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preds = tmp_path / "predictions.jsonl"
            out = tmp_path / "judge_out"
            self._write_preds(preds)

            class FakeResp:
                output_text = "INCORRECT\nnot equivalent"

            class FakeClient:
                def __init__(self):
                    self.calls = 0
                    self.last_kwargs: dict | None = None
                    self.responses = self

                def create(self, **kwargs):
                    self.calls += 1
                    self.last_kwargs = dict(kwargs)
                    prompt = kwargs.get("input") or ""
                    self.assert_prompt_clean(prompt)
                    return FakeResp()

                def assert_prompt_clean(self, prompt: str):
                    low = prompt.lower()
                    assert "normalized_match" not in low
                    assert '"em"' not in low and " em=" not in low
                    assert "token f1" not in low and "normalized accuracy" not in low

            fake = FakeClient()

            env = {
                "AZURE_OPENAI_API_KEY": "test-key-should-not-appear",
                "AZURE_OPENAI_BASE_URL": "https://example.invalid/openai/v1",
                "AZURE_OPENAI_DEPLOYMENT": "gpt-6.1-sol",
            }
            with mock.patch.dict("os.environ", env, clear=False):
                with mock.patch.object(run_llm_judge, "create_openai_client", return_value=fake):
                    rc = run_llm_judge.run(
                        [
                            "--input",
                            str(preds),
                            "--output-dir",
                            str(out),
                            "--live",
                        ]
                    )
                    self.assertEqual(rc, 0)
                    self.assertEqual(fake.calls, 2)
                    assert fake.last_kwargs is not None
                    self.assertEqual(fake.last_kwargs["model"], "gpt-6.1-sol")
                    self.assertIn("input", fake.last_kwargs)
                    self.assertEqual(fake.last_kwargs.get("reasoning"), {"effort": "low"})
                    self.assertEqual(fake.last_kwargs.get("max_output_tokens"), 256)
                    for banned in ("temperature", "top_p", "logprobs", "top_logprobs"):
                        self.assertNotIn(banned, fake.last_kwargs)

                summary = json.loads((out / "judge_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["api_called_n"], 2)
                self.assertEqual(summary["api_succeeded_n"], 2)
                self.assertEqual(summary["api_attempted_n"], 2)
                self.assertEqual(summary["this_run_api_attempted"], 2)

                fake.calls = 0
                with mock.patch.object(run_llm_judge, "create_openai_client", return_value=fake):
                    rc2 = run_llm_judge.run(
                        [
                            "--input",
                            str(preds),
                            "--output-dir",
                            str(out),
                            "--live",
                            "--resume",
                        ]
                    )
                    self.assertEqual(rc2, 0)
                    self.assertEqual(fake.calls, 0)

            results_text = (out / "judge_results.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("test-key-should-not-appear", results_text)
            run_cfg = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
            self.assertNotIn("test-key-should-not-appear", json.dumps(run_cfg))
            self.assertFalse(run_cfg.get("temperature_sent"))

    def test_live_failed_counts_attempted_not_succeeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preds = tmp_path / "predictions.jsonl"
            out = tmp_path / "judge_out"
            self._write_preds(preds)

            class FakeClient:
                def __init__(self):
                    self.responses = self

                def create(self, **kwargs):
                    raise RuntimeError("BadRequestError 400: Unsupported parameter 'temperature'")

            fake = FakeClient()
            env = {
                "AZURE_OPENAI_API_KEY": "test-key-should-not-appear",
                "AZURE_OPENAI_BASE_URL": "https://example.invalid/openai/v1",
                "AZURE_OPENAI_DEPLOYMENT": "gpt-6.1-sol",
            }
            with mock.patch.dict("os.environ", env, clear=False):
                with mock.patch.object(run_llm_judge, "create_openai_client", return_value=fake):
                    rc = run_llm_judge.run(
                        [
                            "--input",
                            str(preds),
                            "--output-dir",
                            str(out),
                            "--live",
                        ]
                    )
                    self.assertEqual(rc, 0)
            summary = json.loads((out / "judge_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["api_called_n"], 0)
            self.assertEqual(summary["api_succeeded_n"], 0)
            self.assertEqual(summary["failed_n"], 2)
            self.assertEqual(summary["api_attempted_n"], 2)
            self.assertEqual(summary["this_run_api_attempted"], 2)
            self.assertEqual(summary["this_run_api_called"], 0)
            results = [
                json.loads(line)
                for line in (out / "judge_results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            failed_rows = [r for r in results if r["judge_status"] == "llm_failed"]
            self.assertEqual(len(failed_rows), 2)
            self.assertTrue(all(r["judge_label"] is None for r in failed_rows))

    def test_malformed_first_response_triggers_one_format_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preds = tmp_path / "predictions.jsonl"
            out = tmp_path / "judge_out"
            # Only one non-auto row so call accounting is clear.
            preds.write_text(
                json.dumps(
                    {
                        "sample_id": "need1",
                        "query": "capital of france",
                        "prediction": "City of Light",
                        "gold": ["Paris"],
                        "em": 0,
                        "f1": 0.0,
                        "normalized_match": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            class FakeResp:
                def __init__(self, text: str):
                    self.output_text = text

            class FakeClient:
                def __init__(self):
                    self.calls = 0
                    self.prompts: list[str] = []
                    self.responses = self

                def create(self, **kwargs):
                    self.calls += 1
                    prompt = str(kwargs.get("input") or "")
                    self.prompts.append(prompt)
                    if self.calls == 1:
                        return FakeResp("sure, that seems fine")
                    return FakeResp("INCORRECT\nnot equivalent")

            fake = FakeClient()
            env = {
                "AZURE_OPENAI_API_KEY": "test-key-should-not-appear",
                "AZURE_OPENAI_BASE_URL": "https://example.invalid/openai/v1",
                "AZURE_OPENAI_DEPLOYMENT": "gpt-6.1-sol",
            }
            with mock.patch.dict("os.environ", env, clear=False):
                with mock.patch.object(run_llm_judge, "create_openai_client", return_value=fake):
                    rc = run_llm_judge.run(
                        [
                            "--input",
                            str(preds),
                            "--output-dir",
                            str(out),
                            "--live",
                        ]
                    )
                    self.assertEqual(rc, 0)
            self.assertEqual(fake.calls, 2)
            self.assertIn("FORMAT RETRY", fake.prompts[1])
            summary = json.loads((out / "judge_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["api_succeeded_n"], 1)
            self.assertEqual(summary["api_attempted_n"], 2)
            self.assertEqual(summary["retry_n"], 1)
            self.assertEqual(summary["failed_n"], 0)
            result = json.loads((out / "judge_results.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(result["judge_status"], "llm_ok")
            self.assertEqual(result["judge_label"], "INCORRECT")
            self.assertEqual(result["retry_n"], 1)

    def test_parse_failure_after_format_retry_is_failed_not_uncertain(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preds = tmp_path / "predictions.jsonl"
            out = tmp_path / "judge_out"
            preds.write_text(
                json.dumps(
                    {
                        "sample_id": "need1",
                        "query": "q",
                        "prediction": "p",
                        "gold": ["g"],
                        "em": 0,
                        "f1": 0.0,
                        "normalized_match": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            class FakeResp:
                output_text = "not a label"

            class FakeClient:
                def __init__(self):
                    self.calls = 0
                    self.responses = self

                def create(self, **kwargs):
                    self.calls += 1
                    return FakeResp()

            fake = FakeClient()
            env = {
                "AZURE_OPENAI_API_KEY": "k",
                "AZURE_OPENAI_BASE_URL": "https://example.invalid/openai/v1",
                "AZURE_OPENAI_DEPLOYMENT": "gpt-6.1-sol",
            }
            with mock.patch.dict("os.environ", env, clear=False):
                with mock.patch.object(run_llm_judge, "create_openai_client", return_value=fake):
                    rc = run_llm_judge.run(
                        ["--input", str(preds), "--output-dir", str(out), "--live"]
                    )
                    self.assertEqual(rc, 0)
            self.assertEqual(fake.calls, 2)
            result = json.loads((out / "judge_results.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(result["judge_status"], "llm_failed")
            self.assertIsNone(result["judge_label"])
            summary = json.loads((out / "judge_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["failed_n"], 1)
            self.assertEqual(summary["api_succeeded_n"], 0)

    def test_pilot_sample_dry_run_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preds = tmp_path / "predictions.jsonl"
            out = tmp_path / "judge_out"
            preds.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in _pilot_pool_rows()) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(run_llm_judge, "create_openai_client") as create_client:
                rc = run_llm_judge.run(
                    [
                        "--input",
                        str(preds),
                        "--output-dir",
                        str(out),
                        "--pilot-sample",
                        "30",
                        "--seed",
                        "42",
                        "--dry-run",
                    ]
                )
                self.assertEqual(rc, 0)
                create_client.assert_not_called()
            manifest_path = out / "pilot_manifest.jsonl"
            self.assertTrue(manifest_path.is_file())
            manifest = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(manifest), 30)
            run_cfg = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_cfg["pilot_sample"], 30)
            self.assertEqual(run_cfg["seed"], 42)
            self.assertEqual(
                run_cfg["pilot_group_counts"],
                {"em0_f1_0": 8, "em0_f1_ge_0_5": 8, "em0_f1_mid": 8, "remainder": 6},
            )

    def test_pilot_sample_refuses_with_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preds = tmp_path / "predictions.jsonl"
            out = tmp_path / "judge_out"
            self._write_preds(preds)
            with self.assertRaises(SystemExit) as ctx:
                run_llm_judge.run(
                    [
                        "--input",
                        str(preds),
                        "--output-dir",
                        str(out),
                        "--pilot-sample",
                        "2",
                        "--limit",
                        "2",
                        "--dry-run",
                    ]
                )
            self.assertIn("pilot-sample", str(ctx.exception).lower())

    def test_non_retryable_4xx_does_not_loop(self):
        class Exc400(Exception):
            status_code = 400

        self.assertFalse(run_llm_judge.is_retryable_api_error(Exc400("bad")))

        class Exc429(Exception):
            status_code = 429

        self.assertTrue(run_llm_judge.is_retryable_api_error(Exc429("rate")))

        class Exc503(Exception):
            status_code = 503

        self.assertTrue(run_llm_judge.is_retryable_api_error(Exc503("down")))


if __name__ == "__main__":
    unittest.main()
