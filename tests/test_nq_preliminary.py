"""CPU-only tests for frozen NQ C3 protocol and interrupted resumption."""

from __future__ import annotations

import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from scripts import run_nq_preliminary as runner


class NqPreliminaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = copy.deepcopy(runner.load_preliminary_config())
        manifest_name = self.config["samples"]["nq"]["manifest"]
        for name in (
            manifest_name,
            "src/generation/qwen.py",
            "src/evaluation/qa.py",
            "scripts/run_nq_preliminary.py",
        ):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runner.ROOT / name, target)
        self.manifest = json.loads((self.root / manifest_name).read_text())
        frozen_ids = list(self.manifest["ids"])
        extra = [f"extra-{number}" for number in range(3216 - len(frozen_ids))]
        self.records = [
            {
                "id": sample_id,
                "question": f"Question {sample_id}?",
                "short_answer": f"answer {sample_id}",
                "gold_long_answer_text": (
                    f"Gold paragraph containing answer {sample_id} and context."
                ),
            }
            for sample_id in frozen_ids + extra
        ]
        source = self.root / self.config["samples"]["nq"]["source"]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            "".join(json.dumps(record) + "\n" for record in self.records),
            encoding="utf-8",
        )
        self.addCleanup(patch.stopall)
        patch.object(runner, "ROOT", self.root).start()
        patch.object(runner, "load_preliminary_config", return_value=self.config).start()
        self.samples = runner.load_frozen_samples(self.config)
        self.fingerprint = runner.experiment_fingerprint(self.config, self.samples)
        self.output = self.root / self.config["result_paths"]["nq_c3_7b"]

    def invoke(self, *args):
        with (
            patch.object(sys, "argv", ["runner", *args]),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            return runner.main()

    def fake_model(self, prediction="synthetic prediction"):
        module = ModuleType("src.generation.qwen")
        module.load_qwen_nf4 = Mock(
            return_value=(object(), SimpleNamespace(hf_device_map={"model": 0, "lm_head": "cpu"}))
        )
        module.generate_greedy = Mock(return_value=(prediction, 0.1))
        transformers = ModuleType("transformers")
        transformers.set_seed = Mock()
        patch.dict(
            sys.modules,
            {"src.generation.qwen": module, "transformers": transformers},
        ).start()
        return module, transformers

    def result(self):
        sample = self.samples[0]
        prediction = sample["short_answer"]
        return {
            "sample_id": sample["id"],
            "dataset": "natural_questions",
            "condition": "C3",
            "model_name": self.config["primary_model"]["model_name"],
            "cpu_offload": False,
            "cpu_offload_modules": [],
            "question_or_claim": sample["question"],
            "gold_answer_or_label": sample["short_answer"],
            "evidence": runner.oracle_evidence(sample),
            "prediction": prediction,
            "correct": True,
            "latency": 0.1,
            "em": 1,
            "f1": 1.0,
            "experiment_fingerprint": self.fingerprint,
        }

    def test_oracle_is_annotated_controlled_long_answer_only(self):
        self.assertEqual(len(self.samples), 500)
        self.assertEqual(
            [str(sample["id"]) for sample in self.samples],
            [str(sample_id) for sample_id in self.manifest["ids"]],
        )
        sample = self.samples[0]
        evidence = runner.oracle_evidence(sample)
        self.assertEqual(evidence, [{"text": sample["gold_long_answer_text"]}])
        prompt = runner.build_prompt(
            self.config["prompts"]["nq"]["C3"], sample["question"], evidence
        )
        self.assertIn("[1] " + sample["gold_long_answer_text"], prompt)
        self.assertNotIn("passage_id", prompt)

    def test_conditions_must_be_explicit(self):
        with self.assertRaises(SystemExit) as raised:
            self.invoke()
        self.assertEqual(raised.exception.code, 2)

    def test_check_only_needs_no_torch_transformers_or_faiss(self):
        with patch.dict(
            sys.modules,
            {"torch": None, "transformers": None, "faiss": None, "sentence_transformers": None},
        ):
            self.assertEqual(self.invoke("--conditions", "C3", "--check-only"), 0)
        self.assertFalse(self.output.exists())

    def test_smoke_uses_seed_and_oracle_without_writing(self):
        model, transformers = self.fake_model()
        self.assertEqual(self.invoke("--conditions", "C3", "--smoke-test"), 0)
        transformers.set_seed.assert_called_once_with(42)
        prompt = model.generate_greedy.call_args.args[2]
        self.assertIn(self.samples[0]["gold_long_answer_text"], prompt)
        self.assertFalse(self.output.exists())

    def test_resume_rejects_duplicate_and_incompatible_c3(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text((json.dumps(self.result()) + "\n") * 2)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            runner.completed_ids(
                self.output, "C3", self.samples, self.config, self.fingerprint
            )
        for key, value in (
            ("sample_id", "not-frozen"),
            ("evidence", [{"text": "retrieved text"}]),
            ("experiment_fingerprint", "wrong"),
            ("em", 0),
        ):
            with self.subTest(key=key):
                record = self.result()
                record[key] = value
                self.output.write_text(json.dumps(record) + "\n")
                with self.assertRaises(RuntimeError):
                    runner.completed_ids(
                        self.output, "C3", self.samples, self.config, self.fingerprint
                    )

    def test_interrupted_c3_resumes_to_500_without_duplicates(self):
        model, _ = self.fake_model()
        patch.object(runner, "run_paired_analysis").start()
        model.generate_greedy.side_effect = [("wrong", 0.1), RuntimeError("interrupted")]
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.invoke("--conditions", "C3")
        self.assertEqual(len(runner.load_jsonl(self.output)), 1)

        model.generate_greedy.reset_mock(side_effect=True)
        model.generate_greedy.return_value = ("wrong", 0.1)
        self.assertEqual(self.invoke("--conditions", "C3"), 0)
        rows = runner.load_jsonl(self.output)
        self.assertEqual(len(rows), 500)
        self.assertEqual(len({str(row["sample_id"]) for row in rows}), 500)
        self.assertTrue(all(row["evidence"] for row in rows))
        self.assertTrue(
            all(len(row["evidence"]) == 1 and "score" not in row["evidence"][0] for row in rows)
        )
        model.load_qwen_nf4.reset_mock()
        self.assertEqual(self.invoke("--conditions", "C3"), 0)
        model.load_qwen_nf4.assert_not_called()


if __name__ == "__main__":
    unittest.main()
