"""Tests for the one-call plugin factory. No model, index or corpus is loaded."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.compensation import EnhancementExecutor
from src.compensation.evidence_refinement import EvidenceRefiner
from src.compensation.interface import Enhancement
from src.compensation.plugin import (
    VALIDATED_KEEP_K,
    VALIDATED_MIN_SCORE_RATIO,
    VALIDATED_TOP_N,
    build_refinement_plugin,
    build_refiner,
    register_refinement_plugin,
    validated_config,
)
from src.compensation.refinement_adapter import EvidenceRefinementAdapter
from src.compensation.refinement_schema import RefinementConfig
from src.controller.schema import ControllerAction


def generate(prompt: str) -> str:
    return "William Shakespeare"


class ValidatedConfigTest(unittest.TestCase):
    def test_it_is_the_measured_setup(self):
        config = validated_config("models/reranker/accessibility_v1")
        self.assertEqual(config.keep_k, VALIDATED_KEEP_K)
        self.assertEqual(config.reranker_model, "models/reranker/accessibility_v1")
        self.assertIsNone(config.min_score)
        self.assertEqual(config.probe_hint_weight, 0.0)
        self.assertTrue(config.require_grounded)

    def test_the_deeper_pool_is_the_default(self):
        self.assertEqual(VALIDATED_TOP_N, 50)

    def test_the_relative_floor_is_on_by_default(self):
        self.assertEqual(validated_config().min_score_ratio, VALIDATED_MIN_SCORE_RATIO)

    def test_the_floor_is_relative_not_absolute(self):
        # An absolute cut can empty a question's context; the relative one cannot,
        # so only the relative field is set.
        config = validated_config()
        self.assertIsNotNone(config.min_score_ratio)
        self.assertIsNone(config.min_score)

    def test_the_floor_takes_the_conservative_end_of_the_plateau(self):
        # 0.8 scored higher but is not distinguishable from 0.6 (p = 0.24), and
        # cuts to 1.90 passages against 2.77.
        self.assertEqual(VALIDATED_MIN_SCORE_RATIO, 0.6)

    def test_the_refiner_carries_the_floor_through(self):
        with mock.patch("src.compensation.plugin.scorer_from_name", return_value=lambda q, t: [0.0] * len(t)):
            refiner = build_refiner(generate_fn=generate, config=validated_config("shuffle:42"), top_n=None)
        self.assertEqual(refiner.config.min_score_ratio, VALIDATED_MIN_SCORE_RATIO)


class MissingArtifactTest(unittest.TestCase):
    def test_a_missing_reranker_names_the_rebuild_command(self):
        config = RefinementConfig(reranker_model="models/reranker/does-not-exist")
        with self.assertRaises(FileNotFoundError) as caught:
            build_refiner(generate_fn=generate, config=config, top_n=None)
        message = str(caught.exception)
        self.assertIn("trained reranker", message)
        self.assertIn("train_rerank_accessibility.py", message)

    def test_a_missing_pool_names_its_own_rebuild_command(self):
        with tempfile.TemporaryDirectory() as directory:
            reranker = Path(directory) / "reranker"
            reranker.mkdir()
            with self.assertRaises(FileNotFoundError) as caught:
                build_refiner(
                    generate_fn=generate,
                    config=RefinementConfig(reranker_model=str(reranker)),
                    corpus_path=Path(directory) / "nope.jsonl",
                )
            self.assertIn("build_nq15_pool_index.py", str(caught.exception))

    def test_a_control_scorer_needs_no_reranker_on_disk(self):
        refiner = build_refiner(
            generate_fn=generate, config=RefinementConfig(reranker_model="shuffle:42"), top_n=None
        )
        self.assertIsInstance(refiner, EvidenceRefiner)
        self.assertIsNone(refiner.candidate_provider)


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.reranker = Path(self.directory.name) / "reranker"
        self.reranker.mkdir()
        self.config = RefinementConfig(reranker_model=str(self.reranker))

    def test_without_a_pool_the_refiner_has_no_provider(self):
        with mock.patch("src.compensation.plugin.scorer_from_name", return_value=lambda q, t: [0.0] * len(t)):
            refiner = build_refiner(generate_fn=generate, config=self.config, top_n=None)
        self.assertIsNone(refiner.candidate_provider)
        self.assertEqual(refiner.config.keep_k, VALIDATED_KEEP_K)

    def test_with_a_pool_the_provider_is_attached(self):
        sentinel = lambda item: []  # noqa: E731 - the refiner only requires a callable
        with mock.patch("src.compensation.plugin.scorer_from_name", return_value=lambda q, t: [0.0] * len(t)), mock.patch(
            "src.compensation.plugin.IndexCandidateProvider.from_frozen_artifacts", return_value=sentinel
        ) as loader, mock.patch("src.compensation.plugin._require"):
            refiner = build_refiner(generate_fn=generate, config=self.config, top_n=25)
        self.assertIs(refiner.candidate_provider, sentinel)
        self.assertEqual(loader.call_args.kwargs["top_n"], 25)

    def test_the_plugin_satisfies_the_enhancement_protocol(self):
        with mock.patch("src.compensation.plugin.scorer_from_name", return_value=lambda q, t: [0.0] * len(t)):
            plugin = build_refinement_plugin(generate_fn=generate, config=self.config, top_n=None)
        self.assertIsInstance(plugin, EvidenceRefinementAdapter)
        self.assertIsInstance(plugin, Enhancement)

    def test_registering_puts_it_on_the_evidence_refinement_lane(self):
        executor = EnhancementExecutor()
        with mock.patch("src.compensation.plugin.scorer_from_name", return_value=lambda q, t: [0.0] * len(t)):
            plugin = register_refinement_plugin(executor, generate_fn=generate, config=self.config, top_n=None)
        self.assertEqual(executor.registered_actions(), frozenset({ControllerAction.EVIDENCE_REFINEMENT}))
        self.assertIsInstance(plugin, EvidenceRefinementAdapter)

    def test_a_custom_generator_is_used(self):
        calls = []

        def recording(prompt: str) -> str:
            calls.append(prompt)
            return "answer"

        with mock.patch("src.compensation.plugin.scorer_from_name", return_value=lambda q, t: [0.0] * len(t)):
            refiner = build_refiner(generate_fn=recording, config=self.config, top_n=None)
        self.assertEqual(refiner.generate("a prompt"), "answer")
        self.assertEqual(calls, ["a prompt"])


if __name__ == "__main__":
    unittest.main()
