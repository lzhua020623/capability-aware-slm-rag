"""FEVER RAG matched baseline: isolation, readiness, dry-run smoke."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from baselines._lib.fever_scoring import (  # noqa: E402
    aggregate_fever_metrics,
    build_fever_output_row,
    gold_evidence_hit_at_k,
)
from baselines._lib.prompts import build_fever_baseline_rag_prompt  # noqa: E402
from baselines.run_baseline_evaluation import (  # noqa: E402
    STATUS_READY_FEVER,
    STATUS_READY_FEVER_RAG,
    fever_readiness,
    load_mode_config,
    main,
    parse_args,
)


class FeverRagConfigTests(unittest.TestCase):
    def test_config_uses_frozen_bundle_and_fever_assets(self):
        cfg = load_mode_config("fever_rag_matched")
        qwen = load_mode_config("fever_qwen_only")
        self.assertEqual(cfg["mode"], "fever_rag_matched")
        self.assertEqual(cfg["dataset"]["records_path"], qwen["dataset"]["records_path"])
        self.assertEqual(cfg["split"]["default_path"], qwen["split"]["default_path"])
        self.assertEqual(cfg["generator"]["model_name"], qwen["generator"]["model_name"])
        self.assertEqual(cfg["generator"]["quantization"], "nf4")
        self.assertFalse(cfg["generator"]["do_sample"])
        ret = cfg["retrieval"]
        self.assertEqual(ret["embedding_model"], "BAAI/bge-base-en-v1.5")
        self.assertEqual(ret["top_k"], 5)
        self.assertIn("fever_bge_ivfpq", ret["index_path"])
        self.assertIn("fever_corpus", ret["corpus_path"])
        self.assertNotIn("nq_", Path(ret["index_path"]).name)
        self.assertNotIn("nq_", Path(ret["corpus_path"]).name)
        self.assertIn(
            "fever_rag_matched_context_fallback",
            cfg["output"]["default_dir"].replace("\\", "/"),
        )

    def test_parse_mode(self):
        ns = parse_args(["--mode", "fever_rag_matched", "--smoke", "--dry-run", "--limit", "4"])
        self.assertEqual(ns.mode, "fever_rag_matched")


class FeverRagPromptAndHitTests(unittest.TestCase):
    def test_prompt_has_evidence_not_nq_question(self):
        prompt = build_fever_baseline_rag_prompt(
            "The sky is green.",
            [{"text": "The sky is blue."}],
        )
        self.assertIn("Evidence:", prompt)
        self.assertIn("Claim: The sky is green.", prompt)
        self.assertIn("SUPPORTS", prompt)
        self.assertNotIn("Question:", prompt)

    def test_hit_at_5_page_equals_passage(self):
        hits = [{"passage_id": "Foo_Bar", "rank": 1}, {"passage_id": "Other", "rank": 2}]
        self.assertTrue(gold_evidence_hit_at_k(hits, ["Foo_Bar"], k=5))
        self.assertFalse(gold_evidence_hit_at_k(hits, ["Missing"], k=5))
        self.assertIsNone(gold_evidence_hit_at_k(hits, [], k=5))

    def test_output_row_rag_fields(self):
        row = build_fever_output_row(
            sample_id="1",
            claim="c",
            prediction="SUPPORTS",
            gold="SUPPORTS",
            mode="fever_rag_matched",
            hits=[
                {"passage_id": "A", "text": "t", "score": 0.9, "rank": 1},
                {"passage_id": "B", "text": "u", "score": 0.8, "rank": 2},
            ],
            gold_page_ids=["A"],
        )
        self.assertEqual(row["retrieved_passage_ids"], ["A", "B"])
        self.assertEqual(row["retrieved_ranks"], [1, 2])
        self.assertEqual(row["retrieved_scores"], [0.9, 0.8])
        self.assertTrue(row["gold_evidence_hit_at_5"])
        m = aggregate_fever_metrics([row], mode="fever_rag_matched")
        self.assertEqual(m["evidence_hit_at_5"], 1.0)


class FeverRagReadinessTests(unittest.TestCase):
    def test_validate_readiness_when_assets_present(self):
        cfg = load_mode_config("fever_rag_matched")
        if not (ROOT / cfg["dataset"]["records_path"]).is_file():
            self.skipTest("fever bundle missing")
        ret = cfg["retrieval"]
        for key in ("index_path", "ids_path", "offsets_path", "corpus_path", "metadata_path"):
            if not (ROOT / ret[key]).is_file():
                self.skipTest(f"missing {key}")
        status, details = fever_readiness(cfg)
        self.assertEqual(status, STATUS_READY_FEVER_RAG)
        self.assertEqual(details["n"], 13332)

    def test_qwen_only_readiness_still_ok(self):
        cfg = load_mode_config("fever_qwen_only")
        if not (ROOT / cfg["dataset"]["records_path"]).is_file():
            self.skipTest("fever bundle missing")
        status, _ = fever_readiness(cfg)
        self.assertEqual(status, STATUS_READY_FEVER)

    def test_dry_run_smoke_isolated(self):
        cfg = load_mode_config("fever_rag_matched")
        if not (ROOT / cfg["dataset"]["records_path"]).is_file():
            self.skipTest("fever bundle missing")
        nq = ROOT / "baselines" / "outputs" / "qwen_only" / "predictions.jsonl"
        fever_q = ROOT / "baselines" / "outputs" / "fever_qwen_only" / "predictions.jsonl"
        nq_before = nq.read_bytes() if nq.is_file() else None
        fq_before = fever_q.read_bytes() if fever_q.is_file() else None
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fever_rag_matched_context_fallback_smoke"
            rc = main(
                [
                    "--mode",
                    "fever_rag_matched",
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
            self.assertEqual(metrics["mode"], "fever_rag_matched")
            row = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("retrieved_passage_ids", row)
            self.assertEqual(row["mode"], "fever_rag_matched")
        if nq_before is not None:
            self.assertEqual(nq.read_bytes(), nq_before)
        if fq_before is not None:
            self.assertEqual(fever_q.read_bytes(), fq_before)


class FeverRagIsolationTests(unittest.TestCase):
    def test_refuse_nq_paths_in_guard(self):
        from baselines.run_baseline_evaluation import _fever_guard_output_dir

        with self.assertRaises(SystemExit):
            _fever_guard_output_dir(
                ROOT / "baselines" / "outputs" / "qwen_only",
                mode="fever_rag_matched",
            )
        with self.assertRaises(SystemExit):
            _fever_guard_output_dir(
                ROOT / "baselines" / "outputs" / "fever_qwen_only",
                mode="fever_rag_matched",
            )


if __name__ == "__main__":
    unittest.main()
