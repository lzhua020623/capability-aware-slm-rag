"""Tests for runtime-safe Diagnosis features. No models are loaded."""

from __future__ import annotations

import dataclasses
import inspect
import statistics
import unittest

from src.diagnosis import features as features_module
from src.diagnosis.features import ABSENT_ANSWER_RANK, FEATURE_NAMES, extract_features
from src.diagnosis.schema import DiagnosisInput, EvidenceItem


def nq_input(answer: str = "Max Planck", query: str = "who proposed the quantum hypothesis", sample_id: str = "1") -> DiagnosisInput:
    return DiagnosisInput(
        sample_id=sample_id,
        dataset="nq",
        query=query,
        evidence=[
            EvidenceItem(rank=1, passage_id="nq_p_1", text="The quantum theory began in 1900 .", score=0.8),
            EvidenceItem(rank=2, passage_id="nq_p_2", text="Max Planck proposed the quantum hypothesis .", score=0.6),
            EvidenceItem(rank=3, passage_id="nq_p_3", text="Planck , Max Planck was German .", score=0.4),
        ],
        initial_answer=answer,
    )


def fever_input(answer: str = "REFUTES") -> DiagnosisInput:
    return DiagnosisInput(
        sample_id=106023,
        dataset="fever",
        query="Boeing 787 Dreamliner is a machine.",
        evidence=[
            EvidenceItem(rank=1, passage_id="Boeing_787_Dreamliner", text="The Boeing 787 Dreamliner is a jet airliner .", score=0.56),
            EvidenceItem(rank=2, passage_id="Boeing_707", text="The Boeing 707 is an airliner .", score=0.42),
        ],
        initial_answer=answer,
    )


class FeatureContractTests(unittest.TestCase):
    def test_feature_names_and_types(self):
        for item in (nq_input(), fever_input()):
            features = extract_features(item)
            self.assertEqual(tuple(features), FEATURE_NAMES)
            self.assertTrue(all(isinstance(value, float) for value in features.values()))

    def test_independent_of_sample_id(self):
        self.assertEqual(extract_features(nq_input(sample_id="1")), extract_features(nq_input(sample_id="999")))

    def test_only_runtime_input_is_needed(self):
        params = list(inspect.signature(extract_features).parameters)
        self.assertEqual(params, ["item"])
        self.assertEqual(
            {f.name for f in dataclasses.fields(DiagnosisInput)},
            {"sample_id", "dataset", "query", "evidence", "initial_answer"},
        )
        source = inspect.getsource(features_module)
        for forbidden in ("sample_id", "target_label", "metadata", "needs_manual_review", "gold"):
            self.assertNotIn(forbidden, source.replace("sample_id is never used", ""))

    def test_deterministic(self):
        self.assertEqual(extract_features(nq_input()), extract_features(nq_input()))
        self.assertEqual(extract_features(fever_input()), extract_features(fever_input()))


class RetrievalScoreTests(unittest.TestCase):
    def test_score_features(self):
        features = extract_features(nq_input())
        scores = [0.8, 0.6, 0.4]
        self.assertAlmostEqual(features["top1_score"], 0.8)
        self.assertAlmostEqual(features["top2_score"], 0.6)
        self.assertAlmostEqual(features["score_gap_1_2"], 0.2)
        self.assertAlmostEqual(features["score_mean_top5"], statistics.fmean(scores))
        self.assertAlmostEqual(features["score_std_top5"], statistics.pstdev(scores))
        self.assertAlmostEqual(features["score_range_top5"], 0.4)

    def test_evidence_order_follows_rank(self):
        item = nq_input()
        shuffled = dataclasses.replace(item, evidence=list(reversed(item.evidence)))
        self.assertEqual(extract_features(item), extract_features(shuffled))


class QueryCoverageTests(unittest.TestCase):
    def test_query_coverage(self):
        # Content tokens: proposed, quantum, hypothesis.
        features = extract_features(nq_input())
        self.assertAlmostEqual(features["query_coverage_top1"], 1 / 3)
        self.assertAlmostEqual(features["query_coverage_max5"], 1.0)
        self.assertAlmostEqual(features["query_coverage_mean5"], (1 / 3 + 1.0 + 0.0) / 3)


class AnswerEvidenceTests(unittest.TestCase):
    def test_answer_present_in_later_passages(self):
        features = extract_features(nq_input("Max Planck"))
        self.assertEqual(features["answer_in_top1"], 0.0)
        self.assertEqual(features["answer_in_any"], 1.0)
        self.assertEqual(features["answer_first_rank"], 2.0)
        self.assertEqual(features["answer_passage_count"], 2.0)
        self.assertEqual(features["answer_token_coverage_max"], 1.0)
        self.assertEqual(features["answer_length_tokens"], 2.0)

    def test_answer_in_top1(self):
        features = extract_features(nq_input("1900"))
        self.assertEqual(features["answer_in_top1"], 1.0)
        self.assertEqual(features["answer_first_rank"], 1.0)

    def test_absent_answer_uses_sentinel(self):
        features = extract_features(nq_input("Albert Einstein"))
        self.assertEqual(features["answer_in_any"], 0.0)
        self.assertEqual(features["answer_first_rank"], float(ABSENT_ANSWER_RANK))
        self.assertEqual(features["answer_passage_count"], 0.0)
        self.assertEqual(features["answer_token_coverage_max"], 0.0)

    def test_partial_token_match_is_not_a_span_hit(self):
        features = extract_features(nq_input("German Planck"))
        self.assertEqual(features["answer_in_any"], 0.0)
        self.assertEqual(features["answer_token_coverage_max"], 1.0)

    def test_abstention(self):
        self.assertEqual(extract_features(nq_input("It is not mentioned in the context."))["answer_is_abstention"], 1.0)
        self.assertEqual(extract_features(nq_input("Max Planck"))["answer_is_abstention"], 0.0)


class ContextAndTaskTests(unittest.TestCase):
    def test_context_lengths(self):
        features = extract_features(nq_input())
        self.assertEqual(features["query_length_tokens"], 5.0)
        self.assertEqual(features["evidence_total_tokens"], 6.0 + 6.0 + 5.0)
        self.assertAlmostEqual(features["evidence_mean_tokens"], 17 / 3)
        self.assertEqual(features["evidence_max_tokens"], 6.0)

    def test_is_fever(self):
        self.assertEqual(extract_features(fever_input())["is_fever"], 1.0)
        self.assertEqual(extract_features(nq_input())["is_fever"], 0.0)

    def test_fever_answer_features_not_applicable(self):
        features = extract_features(fever_input("REFUTES"))
        self.assertEqual(features["is_refutes_prediction"], 1.0)
        self.assertEqual(extract_features(fever_input("SUPPORTS"))["is_refutes_prediction"], 0.0)
        for name in FEATURE_NAMES:
            if name.startswith("answer_") or name in ("is_when", "is_who", "is_where", "is_what", "is_how_many"):
                self.assertEqual(features[name], 0.0, name)

    def test_nq_question_type(self):
        cases = {
            "when did the war end": "is_when",
            "who proposed the theory": "is_who",
            "where is the eiffel tower": "is_where",
            "what is the capital of france": "is_what",
            "how many players are on a team": "is_how_many",
        }
        wh_names = ("is_when", "is_who", "is_where", "is_what", "is_how_many")
        for query, expected in cases.items():
            features = extract_features(nq_input(query=query))
            self.assertEqual({name for name in wh_names if features[name] == 1.0}, {expected}, query)
        other = extract_features(nq_input(query="the taj mahal served as an example"))
        self.assertEqual(sum(other[name] for name in wh_names), 0.0)


if __name__ == "__main__":
    unittest.main()
