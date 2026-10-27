"""Tests for sentence-level evidence pruning. No model is downloaded or loaded."""

from __future__ import annotations

import unittest

from src.compensation.pruning import ContextPruner, MatchedRandomPruner, ProvencePruner

PASSAGES = [
    "Marlowe died in 1593 . He wrote Doctor Faustus . The Globe opened later .",
    "Hamlet is a tragedy by William Shakespeare . It was written around 1600 .",
]


class FakeProvence:
    """Stands in for the real model: keeps the sentences a script says to keep."""

    def __init__(self, pruned=None, scores=None):
        self.pruned = pruned
        self.scores = scores
        self.calls = []

    def process(self, question, context, **kwargs):
        self.calls.append({"question": question, "context": context, **kwargs})
        batch = context if isinstance(context[0], list) else [context]
        pruned = self.pruned if self.pruned is not None else [[texts[0] for texts in group] for group in batch]
        scores = self.scores if self.scores is not None else [[1.0] * len(group) for group in batch]
        return {"pruned_context": pruned, "reranking_score": scores}


def provence(**kwargs) -> ProvencePruner:
    return ProvencePruner(FakeProvence(**kwargs))


class ProvencePrunerTest(unittest.TestCase):
    def test_it_satisfies_the_pruner_protocol(self):
        self.assertIsInstance(provence(), ContextPruner)

    def test_pruned_text_replaces_the_original(self):
        pruner = provence(pruned=[["Marlowe died in 1593 .", "Hamlet is a tragedy by William Shakespeare ."]])
        self.assertEqual(
            pruner.prune("who wrote hamlet", PASSAGES),
            ["Marlowe died in 1593 .", "Hamlet is a tragedy by William Shakespeare ."],
        )

    def test_an_empty_pruned_passage_falls_back_to_the_original(self):
        pruner = provence(pruned=[["", "   "]])
        self.assertEqual(pruner.prune("q", PASSAGES), PASSAGES)

    def test_the_model_is_called_without_title_forcing_and_without_reordering(self):
        model = FakeProvence()
        ProvencePruner(model, threshold=0.5, batch_size=8).prune("q", PASSAGES)
        call = model.calls[0]
        self.assertIsNone(call["title"])
        self.assertFalse(call["reorder"])
        self.assertEqual(call["threshold"], 0.5)
        self.assertEqual(call["batch_size"], 8)

    def test_a_length_mismatch_is_an_error(self):
        pruner = provence(pruned=[["only one"]])
        with self.assertRaises(RuntimeError):
            pruner.prune("q", PASSAGES)

    def test_no_texts_means_no_call(self):
        model = FakeProvence()
        self.assertEqual(ProvencePruner(model).prune("q", []), [])
        self.assertEqual(ProvencePruner(model)("q", []), [])
        self.assertEqual(model.calls, [])

    def test_it_also_serves_scores(self):
        pruner = provence(scores=[[2.5, 0.5]])
        self.assertEqual(pruner("q", PASSAGES), [2.5, 0.5])

    def test_inputs_are_validated(self):
        pruner = provence()
        with self.assertRaises(TypeError):
            pruner.prune(None, PASSAGES)
        with self.assertRaises(TypeError):
            pruner.prune("q", [1, 2])

    def test_a_model_without_process_is_rejected(self):
        with self.assertRaises(TypeError):
            ProvencePruner(object())

    def test_batched_pruning_keeps_groups_aligned(self):
        pruner = provence(pruned=[["a kept"], ["b kept"]])
        self.assertEqual(pruner.prune_many(["q1", "q2"], [["a"], ["b"]]), [["a kept"], ["b kept"]])

    def test_batched_pruning_checks_its_inputs(self):
        with self.assertRaises(ValueError):
            provence().prune_many(["q"], [["a"], ["b"]])
        self.assertEqual(provence().prune_many([], []), [])


class MatchedRandomPrunerTest(unittest.TestCase):
    def setUp(self):
        # Provence keeps 1 of 3 sentences in the first passage, 1 of 2 in the second.
        self.reference = provence(pruned=[["He wrote Doctor Faustus .", "It was written around 1600 ."]])

    def test_it_keeps_the_same_number_of_sentences(self):
        control = MatchedRandomPruner(self.reference, seed=42)
        pruned = control.prune("who wrote hamlet", PASSAGES)
        self.assertEqual(len(control.split(pruned[0])), 1)
        self.assertEqual(len(control.split(pruned[1])), 1)

    def test_it_is_deterministic(self):
        first = MatchedRandomPruner(self.reference, seed=42).prune("q", PASSAGES)
        second = MatchedRandomPruner(self.reference, seed=42).prune("q", PASSAGES)
        self.assertEqual(first, second)

    def test_different_seeds_can_pick_different_sentences(self):
        picks = {
            tuple(MatchedRandomPruner(self.reference, seed=seed).prune("q", PASSAGES)) for seed in range(12)
        }
        self.assertGreater(len(picks), 1)

    def test_kept_sentences_stay_in_their_original_order(self):
        reference = provence(pruned=[["x", "y"]])  # keeps 1 sentence per passage
        control = MatchedRandomPruner(reference, seed=3)
        text = "One . Two . Three . Four ."
        for seed in range(6):
            control = MatchedRandomPruner(reference, seed=seed)
            reference.model.pruned = [["One . Two .", "y"]]  # keep 2 of 4
            pruned = control.prune("q", [text, "a . b ."])[0]
            sentences = control.split(pruned)
            self.assertEqual(sentences, sorted(sentences, key=control.split(text).index))

    def test_keeping_everything_returns_the_original(self):
        reference = provence(pruned=[list(PASSAGES)])
        self.assertEqual(MatchedRandomPruner(reference, seed=1).prune("q", PASSAGES), PASSAGES)

    def test_it_validates_its_arguments(self):
        with self.assertRaises(TypeError):
            MatchedRandomPruner("not a pruner")
        with self.assertRaises(TypeError):
            MatchedRandomPruner(self.reference, seed="42")

    def test_no_texts_means_no_work(self):
        self.assertEqual(MatchedRandomPruner(self.reference, seed=1).prune("q", []), [])


if __name__ == "__main__":
    unittest.main()
