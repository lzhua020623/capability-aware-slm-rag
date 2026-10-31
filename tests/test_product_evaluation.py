"""Product evaluation harness: scoring parity, sealed Test gate, AIC@5, Judge preset."""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]

from baselines._lib.answer_scoring import answer_in_context, score_answer
from baselines._lib.judge_io import load_judge_config, resolve_predictions_path
from baselines.run_product_evaluation import (
    MODE,
    aggregate_product_metrics,
    assert_safe_output_dir,
    build_product_row,
    enforce_sealed_gate,
    formal_readiness,
    load_eval_ids,
    load_product_config,
    main as product_main,
)
from src.data.team_nq import BoundaryError, TeamNQBoundary


class Evidence:
    def __init__(self, passage_id, text, rank, score=0.9):
        self.passage_id = passage_id
        self.text = text
        self.rank = rank
        self.score = score


def _fake_result(*, final="Paris", initial="Paris", selected=False, action="no_action"):
    diagnosis = SimpleNamespace(final_label=SimpleNamespace(value="correct"))
    selection = SimpleNamespace(needs_enhancement=selected)
    controller = SimpleNamespace(action=SimpleNamespace(value=action))
    enhancement = SimpleNamespace(status=SimpleNamespace(value="no_action" if action == "no_action" else "success"))
    initial_ev = (
        Evidence("p1", "The capital of France is Paris.", 1),
        Evidence("p2", "Lyon is another city.", 2),
    )
    final_ev = (
        Evidence("p9", "Unrelated text without the gold string.", 1),
        Evidence("p8", "Still unrelated.", 2),
    )
    return SimpleNamespace(
        sample_id="123",
        query="What is the capital of France?",
        final_answer=final,
        initial_answer=initial,
        initial_evidence=initial_ev,
        final_evidence=final_ev,
        diagnosis=diagnosis,
        selection_decision=selection,
        controller_decision=controller,
        enhancement_result=enhancement,
    )


class ProductEvaluationTests(unittest.TestCase):
    def test_config_mode_and_split_defaults(self):
        cfg = load_product_config()
        self.assertEqual(cfg["mode"], MODE)
        self.assertEqual(cfg["split"]["expected_n"]["test"], 11114)
        self.assertEqual(cfg["scoring"]["aic_evidence"], "initial_evidence")

    def test_frozen_split_matches_baseline_blob_and_count(self):
        split_path = ROOT / "baselines/splits/diagnosis_nq_train_74097_split.json"
        self.assertTrue(split_path.is_file())
        cfg = load_product_config()
        ids = load_eval_ids(cfg, split_path, "test")
        self.assertEqual(len(ids), 11114)
        self.assertEqual(len(set(ids)), 11114)
        # Byte identity with feature/baseline-evaluation checkout of the same path.
        from src.data.team_nq import TEAM_SPLIT_SHA256

        digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
        self.assertEqual(digest, TEAM_SPLIT_SHA256)

    def test_scoring_parity_with_baseline_function(self):
        scored = score_answer("Paris", ["paris"])
        self.assertIn("em", scored)
        self.assertIn("f1", scored)
        self.assertIn("normalized_match", scored)
        self.assertIn("normalized_match_type", scored)

    def test_aic_uses_initial_evidence_not_final(self):
        result = _fake_result()
        row = build_product_row(result=result, gold=["Paris"], prompt_style="context_fallback")
        self.assertEqual(row["mode"], MODE)
        self.assertEqual(row["prediction"], "Paris")
        self.assertEqual(row["answer_in_context"], 1)
        self.assertEqual(row["answer_in_context_final"], 0)
        self.assertEqual(row["retrieved_passage_ids"], ["p1", "p2"])
        metrics = aggregate_product_metrics([row])
        self.assertAlmostEqual(metrics["answer_in_context_at_5"], 1.0)
        self.assertAlmostEqual(metrics["answer_in_context_at_5_final"], 0.0)
        # Sanity: direct helper agrees with initial hits only.
        initial_hits = [
            {"passage_id": "p1", "text": "The capital of France is Paris.", "rank": 1},
        ]
        final_hits = [{"passage_id": "p9", "text": "Unrelated text without the gold string.", "rank": 1}]
        self.assertEqual(answer_in_context(["Paris"], initial_hits), 1)
        self.assertEqual(answer_in_context(["Paris"], final_hits), 0)

    def test_product_trace_fields_present(self):
        row = build_product_row(
            result=_fake_result(selected=True, action="retrieval_enhancement", final="Paris"),
            gold=["Paris"],
            prompt_style="context_fallback",
        )
        for key in (
            "initial_answer",
            "diagnosis_label",
            "selected",
            "controller_action",
            "enhancement_status",
            "rag_prompt_style",
        ):
            self.assertIn(key, row)
        self.assertTrue(row["selected"])
        self.assertEqual(row["controller_action"], "retrieval_enhancement")

    def test_refuse_overwrite_baseline_outputs(self):
        with self.assertRaises(SystemExit):
            assert_safe_output_dir(ROOT / "baselines/outputs/qwen_only")
        with self.assertRaises(SystemExit):
            assert_safe_output_dir(ROOT / "baselines/outputs/rag_matched_test_context_fallback")
        assert_safe_output_dir(ROOT / "baselines/outputs/adaptive_product_test")

    def test_sealed_test_locked_by_default(self):
        with self.assertRaises(SystemExit):
            enforce_sealed_gate("test", False, smoke=False)
        enforce_sealed_gate("test", True, smoke=False)
        with self.assertRaises(SystemExit):
            enforce_sealed_gate("test", True, smoke=True)

    def test_load_current_pipeline_test_still_locked_without_flag(self):
        from src.pipeline.current import load_current_pipeline

        with patch("src.controller.providers.load_pipeline") as model:
            with self.assertRaisesRegex(BoundaryError, "T7"):
                load_current_pipeline(data_role="test")
            model.assert_not_called()

    def test_sealed_unlock_allows_test_role_on_boundary(self):
        locked = TeamNQBoundary(ROOT)
        self.assertNotIn("test", locked.eligible)
        opened = TeamNQBoundary(ROOT, allow_sealed_test=True)
        self.assertIn("test", opened.eligible)
        self.assertEqual(len(opened.eligible["test"]), 11114)

    def test_load_current_pipeline_sealed_unlock_passes_gate_before_gpu(self):
        """With allow_sealed_test, the T7 gate is cleared before provider load."""
        from src.pipeline.current import load_current_pipeline

        with patch("src.pipeline.current.readiness", return_value={"ready_for_gpu": True, "missing": []}):
            with patch("src.controller.providers.load_pipeline", side_effect=RuntimeError("stop_before_gpu")) as model:
                with self.assertRaisesRegex(RuntimeError, "stop_before_gpu"):
                    load_current_pipeline(
                        data_role="test",
                        allow_sealed_test=True,
                        attach_reference_rag=False,
                    )
                model.assert_called_once()

    def test_judge_preset_resolves_adaptive_product(self):
        cfg = load_judge_config()
        self.assertIn("adaptive_product", cfg["presets"])
        path = resolve_predictions_path("adaptive_product", cfg)
        self.assertTrue(str(path).replace("\\", "/").endswith(
            "baselines/outputs/adaptive_product_test/predictions.jsonl"
        ))
        # Rubric labels unchanged.
        self.assertEqual(cfg["labels"], ["CORRECT", "INCORRECT", "UNCERTAIN"])

    def test_judge_loader_reads_product_schema_fields(self):
        from baselines._lib.judge_io import build_judge_payload, is_normalized_correct

        row = build_product_row(result=_fake_result(), gold=["Paris"], prompt_style="context_fallback")
        self.assertTrue(is_normalized_correct(row) or row["normalized_match"] in (0, 1))
        payload = build_judge_payload(row)
        self.assertEqual(set(payload), {"question", "gold", "prediction"})
        self.assertEqual(payload["prediction"], "Paris")

    def test_cli_validate_split_only_reports_11114(self):
        code = product_main(
            ["--validate-split-only", "--eval-split", "test", "--allow-sealed-test"]
        )
        self.assertEqual(code, 0)

    def test_cli_refuses_test_without_unlock(self):
        with self.assertRaises(SystemExit):
            product_main(["--validate-split-only", "--eval-split", "test"])

    def test_formal_readiness_status(self):
        cfg = load_product_config()
        split_path = ROOT / cfg["split"]["default_path"]
        receipt = formal_readiness(cfg, split_path, "test")
        self.assertEqual(receipt["expected_n"], 11114)
        self.assertEqual(receipt["split"]["n"], 11114)
        # Status depends on local dataset/index presence; schema itself must be ok.
        self.assertTrue(receipt["split"]["ok"])


class ProductConfigTests(unittest.TestCase):
    """The default evaluates the current product; the historical controller_2 product needs its own config."""

    def test_default_is_the_current_product_with_its_own_output_dir(self):
        from baselines.run_product_evaluation import HISTORICAL_CONFIG
        from src.pipeline.registrations import DEFAULT_REGISTRATION, LEGACY_REGISTRATION
        current, historical = load_product_config(), load_product_config(HISTORICAL_CONFIG)
        self.assertEqual(current["_config_path"], "baselines/configs/adaptive_product_v2.json")
        self.assertEqual(current["production"]["baseline_registration"], DEFAULT_REGISTRATION)
        self.assertEqual(historical["production"]["baseline_registration"], LEGACY_REGISTRATION)
        self.assertEqual(historical["output"]["default_dir"], "baselines/outputs/adaptive_product_test")
        self.assertNotEqual(current["output"]["default_dir"], historical["output"]["default_dir"])
        self.assertNotEqual(current["output"]["smoke_dir"], historical["output"]["smoke_dir"])
        rest = lambda cfg: {k: v for k, v in cfg.items() if k not in ("description", "production", "output", "_config_path")}
        self.assertEqual(rest(current), rest(historical))
        same = lambda cfg: {k: v for k, v in cfg["production"].items() if k != "baseline_registration"}
        self.assertEqual(same(current), same(historical))

    def test_archived_product_outputs_hold_the_historical_registration(self):
        import json
        from src.pipeline.registrations import LEGACY_REGISTRATION
        run = json.loads((ROOT / "baselines/outputs/adaptive_product_test/run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(run["production"]["baseline_registration"], LEGACY_REGISTRATION)

    def test_an_output_dir_of_another_registration_is_refused_before_any_change(self):
        import json
        import tempfile
        from baselines.run_product_evaluation import HISTORICAL_CONFIG, assert_output_matches_config
        current, historical = load_product_config(), load_product_config(HISTORICAL_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            assert_output_matches_config(out, current)          # empty dir
            (out / "run_config.json").write_text(json.dumps({"production": historical["production"]}), encoding="utf-8")
            (out / "predictions.jsonl").write_text("kept\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                assert_output_matches_config(out, current)
            assert_output_matches_config(out, historical)
            self.assertEqual((out / "predictions.jsonl").read_text(encoding="utf-8"), "kept\n")

    def test_judge_preset_for_the_current_product(self):
        cfg = load_judge_config()
        path = resolve_predictions_path("adaptive_product_v2", cfg)
        self.assertTrue(str(path).replace("\\", "/").endswith("baselines/outputs/adaptive_product_v2_test/predictions.jsonl"))


if __name__ == "__main__":
    unittest.main()
