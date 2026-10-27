"""Tests for cross-encoder reranking of the evidence. No model is loaded."""

from __future__ import annotations

import math
import unittest

from src.compensation.rerank import (
    IdentityScorer,
    ShuffleScorer,
    scorer_from_name,
    CrossEncoderScorer,
    RelevanceScorer,
    build_candidate_pool,
    rerank_evidence,
)
from src.compensation.refinement_schema import RefinementConfig
from src.diagnosis.schema import DatasetName, DiagnosisInput, EvidenceItem
from src.probe.schema import ProbeResult

KEEP_ALL = RefinementConfig(keep_k=5)


def item(*passage_ids: str, dataset: DatasetName = DatasetName.NQ) -> DiagnosisInput:
    return DiagnosisInput(
        sample_id="s1",
        dataset=dataset,
        query="who wrote hamlet",
        evidence=[
            EvidenceItem(rank=rank, passage_id=passage_id, text=f"passage {passage_id}", score=1.0 - 0.1 * rank)
            for rank, passage_id in enumerate(passage_ids, start=1)
        ],
        initial_answer="Marlowe",
    )


class ScoreTable:
    """Scorer driven by a passage-text -> score table, recording its calls."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, query, texts):
        self.calls.append((query, tuple(texts)))
        return [self._scores[text] for text in texts]


def table_for(**by_passage: float) -> ScoreTable:
    return ScoreTable({f"passage {passage_id}": score for passage_id, score in by_passage.items()})


def probe(supporting=(), conflicting=()) -> ProbeResult:
    return ProbeResult.from_judgments("unsupported", "sufficient", "hard_to_access", supporting, conflicting)


class CandidatePoolTest(unittest.TestCase):
    def test_without_provider_the_pool_is_the_shown_evidence(self):
        request = item("p1", "p2", "p3")
        self.assertEqual(build_candidate_pool(request), tuple(request.evidence))

    def test_provider_extends_the_pool_after_the_originals(self):
        request = item("p1", "p2")
        extra = [EvidenceItem(rank=1, passage_id="p9", text="passage p9", score=0.2)]
        pool = build_candidate_pool(request, lambda _: extra)
        self.assertEqual([candidate.passage_id for candidate in pool], ["p1", "p2", "p9"])
        self.assertEqual([candidate.rank for candidate in pool], [1, 2, 3])
        self.assertEqual(pool[2].score, 0.2)

    def test_provider_duplicates_are_dropped(self):
        request = item("p1", "p2")
        extra = [
            EvidenceItem(rank=7, passage_id="p2", text="passage p2 again", score=0.9),
            EvidenceItem(rank=8, passage_id="p3", text="passage p3", score=0.1),
        ]
        pool = build_candidate_pool(request, lambda _: extra)
        self.assertEqual([candidate.passage_id for candidate in pool], ["p1", "p2", "p3"])
        self.assertEqual(pool[1].text, "passage p2")

    def test_provider_must_yield_evidence_items(self):
        request = item("p1")
        with self.assertRaises(TypeError):
            build_candidate_pool(request, lambda _: [{"passage_id": "p9"}])


class RerankEvidenceTest(unittest.TestCase):
    def test_reorders_by_score_and_renumbers_ranks(self):
        request = item("p1", "p2", "p3")
        scorer = table_for(p1=0.1, p2=0.9, p3=0.5)
        outcome = rerank_evidence(request.query, request.evidence, scorer, KEEP_ALL)
        self.assertEqual([passage.passage_id for passage in outcome.evidence], ["p2", "p3", "p1"])
        self.assertEqual([passage.rank for passage in outcome.evidence], [1, 2, 3])
        self.assertEqual(outcome.scores, (0.9, 0.5, 0.1))
        self.assertEqual(outcome.candidate_pool_size, 3)
        self.assertFalse(outcome.used_probe_hint)
        self.assertEqual(scorer.calls, [(request.query, ("passage p1", "passage p2", "passage p3"))])

    def test_original_retrieval_score_is_preserved(self):
        request = item("p1", "p2")
        outcome = rerank_evidence(request.query, request.evidence, table_for(p1=0.0, p2=1.0), KEEP_ALL)
        self.assertEqual(outcome.evidence[0].passage_id, "p2")
        self.assertAlmostEqual(outcome.evidence[0].score, request.evidence[1].score)

    def test_ties_break_on_the_original_rank(self):
        request = item("p1", "p2", "p3")
        outcome = rerank_evidence(request.query, request.evidence, table_for(p1=0.5, p2=0.5, p3=0.5), KEEP_ALL)
        self.assertEqual([passage.passage_id for passage in outcome.evidence], ["p1", "p2", "p3"])

    def test_keep_k_prunes_and_reports_dropped_in_pool_order(self):
        request = item("p1", "p2", "p3", "p4")
        outcome = rerank_evidence(
            request.query, request.evidence, table_for(p1=0.1, p2=0.9, p3=0.8, p4=0.2), RefinementConfig(keep_k=2)
        )
        self.assertEqual([passage.passage_id for passage in outcome.evidence], ["p2", "p3"])
        self.assertEqual(outcome.dropped_passage_ids, ("p1", "p4"))
        self.assertEqual(len(outcome.scores), 2)

    def test_min_score_floor_drops_weak_passages(self):
        request = item("p1", "p2", "p3")
        config = RefinementConfig(keep_k=3, min_score=0.5)
        outcome = rerank_evidence(request.query, request.evidence, table_for(p1=0.9, p2=0.4, p3=0.6), config)
        self.assertEqual([passage.passage_id for passage in outcome.evidence], ["p1", "p3"])
        self.assertEqual(outcome.dropped_passage_ids, ("p2",))

    def test_min_score_floor_always_keeps_the_best_passage(self):
        request = item("p1", "p2")
        config = RefinementConfig(keep_k=2, min_score=10.0)
        outcome = rerank_evidence(request.query, request.evidence, table_for(p1=0.1, p2=0.9), config)
        self.assertEqual([passage.passage_id for passage in outcome.evidence], ["p2"])
        self.assertEqual(outcome.dropped_passage_ids, ("p1",))

    def test_probe_hints_are_ignored_at_the_default_weight(self):
        request = item("p1", "p2")
        outcome = rerank_evidence(
            request.query, request.evidence, table_for(p1=0.6, p2=0.5), KEEP_ALL, probe(supporting=(1,), conflicting=(2,))
        )
        self.assertEqual([passage.passage_id for passage in outcome.evidence], ["p1", "p2"])
        self.assertFalse(outcome.used_probe_hint)

    def test_positive_hint_weight_promotes_the_contradicting_passage(self):
        request = item("p1", "p2")
        config = RefinementConfig(keep_k=2, probe_hint_weight=0.5)
        outcome = rerank_evidence(
            request.query, request.evidence, table_for(p1=0.6, p2=0.5), config, probe(supporting=(1,), conflicting=(2,))
        )
        self.assertEqual([passage.passage_id for passage in outcome.evidence], ["p2", "p1"])
        self.assertTrue(outcome.used_probe_hint)
        self.assertAlmostEqual(outcome.scores[0], 1.0)
        self.assertAlmostEqual(outcome.scores[1], 0.1)

    def test_negative_hint_weight_inverts_the_adjustment(self):
        request = item("p1", "p2")
        config = RefinementConfig(keep_k=2, probe_hint_weight=-0.5)
        outcome = rerank_evidence(
            request.query, request.evidence, table_for(p1=0.1, p2=0.9), config, probe(supporting=(1,), conflicting=(2,))
        )
        self.assertEqual([passage.passage_id for passage in outcome.evidence], ["p1", "p2"])

    def test_hints_that_match_nothing_are_not_reported_as_used(self):
        request = item("p1", "p2")
        config = RefinementConfig(keep_k=2, probe_hint_weight=0.5)
        outcome = rerank_evidence(request.query, request.evidence, table_for(p1=0.6, p2=0.5), config, probe())
        self.assertFalse(outcome.used_probe_hint)
        self.assertEqual(outcome.scores, (0.6, 0.5))

    def test_hints_only_match_the_ranks_the_probe_saw(self):
        request = item("p1", "p2")
        extra = [EvidenceItem(rank=1, passage_id="p9", text="passage p9", score=0.1)]
        pool = build_candidate_pool(request, lambda _: extra)
        config = RefinementConfig(keep_k=3, probe_hint_weight=1.0)
        outcome = rerank_evidence(request.query, pool, table_for(p1=0.5, p2=0.4, p9=0.3), config, probe(conflicting=(2,)))
        self.assertEqual([passage.passage_id for passage in outcome.evidence], ["p2", "p1", "p9"])

    def test_empty_pool_is_rejected(self):
        scorer = table_for()
        with self.assertRaises(ValueError):
            rerank_evidence("q", [], scorer, KEEP_ALL)

    def test_scorer_contract_is_enforced(self):
        request = item("p1", "p2")
        with self.assertRaises(ValueError):
            rerank_evidence(request.query, request.evidence, lambda query, texts: [1.0], KEEP_ALL)
        with self.assertRaises(ValueError):
            rerank_evidence(request.query, request.evidence, lambda query, texts: [1.0, math.nan], KEEP_ALL)
        with self.assertRaises(TypeError):
            rerank_evidence(request.query, request.evidence, lambda query, texts: [1.0, "high"], KEEP_ALL)

    def test_reranking_is_deterministic(self):
        request = item("p1", "p2", "p3")
        scores = {"p1": 0.3, "p2": 0.3, "p3": 0.9}
        first = rerank_evidence(request.query, request.evidence, table_for(**scores), KEEP_ALL)
        second = rerank_evidence(request.query, request.evidence, table_for(**scores), KEEP_ALL)
        self.assertEqual(first, second)


class CrossEncoderScorerTest(unittest.TestCase):
    def test_builds_query_passage_pairs_and_returns_floats(self):
        seen: list[list[list[str]]] = []

        def predict(pairs):
            seen.append(pairs)
            return [1, 2]

        scorer = CrossEncoderScorer(predict, model_name="fake")
        scores = scorer("q", ["a", "b"])
        self.assertEqual(seen, [[["q", "a"], ["q", "b"]]])
        self.assertEqual(scores, [1.0, 2.0])
        self.assertTrue(all(isinstance(score, float) for score in scores))

    def test_no_texts_means_no_model_call(self):
        def predict(pairs):
            raise AssertionError("must not be called")

        self.assertEqual(CrossEncoderScorer(predict)("q", []), [])

    def test_query_must_be_a_string(self):
        scorer = CrossEncoderScorer(lambda pairs: [])
        with self.assertRaises(TypeError):
            scorer(None, ["a"])

    def test_satisfies_the_scorer_protocol(self):
        self.assertIsInstance(CrossEncoderScorer(lambda pairs: []), RelevanceScorer)


if __name__ == "__main__":
    unittest.main()


class ShuffleScorerTest(unittest.TestCase):
    def test_same_inputs_give_the_same_scores(self):
        scorer = ShuffleScorer(42)
        self.assertEqual(scorer("who wrote hamlet", ["a", "b", "c"]), scorer("who wrote hamlet", ["a", "b", "c"]))

    def test_scores_do_not_depend_on_candidate_order(self):
        scorer = ShuffleScorer(42)
        forward = dict(zip(["a", "b", "c"], scorer("q", ["a", "b", "c"])))
        backward = dict(zip(["c", "b", "a"], scorer("q", ["c", "b", "a"])))
        self.assertEqual(forward, backward)

    def test_different_seeds_give_different_orders(self):
        texts = [f"passage {index}" for index in range(8)]
        first = sorted(range(8), key=lambda i: ShuffleScorer(1)("q", texts)[i])
        second = sorted(range(8), key=lambda i: ShuffleScorer(2)("q", texts)[i])
        self.assertNotEqual(first, second)

    def test_different_queries_give_different_orders(self):
        texts = [f"passage {index}" for index in range(8)]
        scorer = ShuffleScorer(42)
        self.assertNotEqual(scorer("query one", texts), scorer("query two", texts))

    def test_scores_are_floats_in_the_unit_interval(self):
        scores = ShuffleScorer(42)("q", [f"p{index}" for index in range(50)])
        self.assertTrue(all(isinstance(score, float) and 0.0 <= score < 1.0 for score in scores))

    def test_no_texts_means_no_scores(self):
        self.assertEqual(ShuffleScorer(42)("q", []), [])

    def test_seed_and_query_are_validated(self):
        with self.assertRaises(TypeError):
            ShuffleScorer("42")
        with self.assertRaises(TypeError):
            ShuffleScorer(True)
        scorer = ShuffleScorer(42)
        with self.assertRaises(TypeError):
            scorer(None, ["a"])

    def test_it_drives_a_real_rerank(self):
        request = item("p1", "p2", "p3")
        outcome = rerank_evidence(request.query, request.evidence, ShuffleScorer(42), KEEP_ALL)
        self.assertEqual(
            sorted(passage.passage_id for passage in outcome.evidence), ["p1", "p2", "p3"]
        )
        self.assertEqual([passage.rank for passage in outcome.evidence], [1, 2, 3])

    def test_it_satisfies_the_scorer_protocol(self):
        self.assertIsInstance(ShuffleScorer(42), RelevanceScorer)


class ScorerFromNameTest(unittest.TestCase):
    def test_a_shuffle_name_rebuilds_the_control(self):
        scorer = scorer_from_name("shuffle:7")
        self.assertIsInstance(scorer, ShuffleScorer)
        self.assertEqual(scorer.seed, 7)
        self.assertEqual(scorer("q", ["a"]), ShuffleScorer(7)("q", ["a"]))

    def test_a_negative_seed_is_accepted(self):
        self.assertEqual(scorer_from_name("shuffle:-1").seed, -1)

    def test_a_malformed_shuffle_name_is_rejected(self):
        for name in ("shuffle:", "shuffle:abc", "shuffle:1.5"):
            with self.assertRaises(ValueError):
                scorer_from_name(name)

    def test_an_empty_name_is_rejected(self):
        for name in ("", "   ", None):
            with self.assertRaises(ValueError):
                scorer_from_name(name)


class IdentityScorerTest(unittest.TestCase):
    def test_it_keeps_the_frozen_order(self):
        request = item("p1", "p2", "p3")
        outcome = rerank_evidence(request.query, request.evidence, IdentityScorer(), KEEP_ALL)
        self.assertEqual([p.passage_id for p in outcome.evidence], ["p1", "p2", "p3"])
        self.assertEqual(outcome.scores, (0.0, 0.0, 0.0))

    def test_it_resolves_by_name(self):
        self.assertIsInstance(scorer_from_name("identity"), IdentityScorer)

    def test_it_validates_the_query(self):
        with self.assertRaises(TypeError):
            IdentityScorer()(None, ["a"])

    def test_it_satisfies_the_scorer_protocol(self):
        self.assertIsInstance(IdentityScorer(), RelevanceScorer)
