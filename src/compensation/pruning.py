"""Sentence-level pruning of the retrieved evidence.

Reordering passages turned out not to change answers: on this stack a trained
reranker is statistically indistinguishable from a seeded shuffle, and what
drives repair is perturbing the context at all rather than ordering it well
(see `docs/EVIDENCE_REFINEMENT_RERANK.md`). Pruning is a different mechanism
for the same diagnosis: `evidence_accessibility` says the answer is present but
"buried in long or irrelevant text", and removing that text is something
reordering cannot do.

`ProvencePruner` wraps `naver/provence-reranker-debertav3-v1` (DeBERTa-v3,
0.4B, English, CC BY-NC), which labels each token keep/drop for a given
question and returns the surviving sentences plus a reranking score from the
same forward pass.

`MatchedRandomPruner` is its control: it removes the same number of sentences
from the same passage, chosen pseudo-randomly. Pruning also perturbs the
prompt, so without this control a gain cannot be attributed to removing the
*right* sentences rather than to removing *some*.

Neither pruner ever returns an empty context: if everything would be dropped,
the original passage text is kept, so the generator is never handed nothing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

PROVENCE_MODEL = "naver/provence-reranker-debertav3-v1"
DEFAULT_THRESHOLD = 0.1


@runtime_checkable
class ContextPruner(Protocol):
    """Rewrites passage texts for one query, keeping them aligned and non-empty."""

    def prune(self, query: str, texts: Sequence[str]) -> list[str]: ...


def _check(query: str, texts: Sequence[str]) -> list[str]:
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")
    texts = list(texts)
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("texts must all be strings")
    return texts


def _fallback(pruned: str, original: str) -> str:
    """Never hand the generator an empty passage."""
    return pruned.strip() or original


class ProvencePruner:
    """Question-conditioned sentence pruning; also usable as a relevance scorer."""

    def __init__(self, model, threshold: float = DEFAULT_THRESHOLD, batch_size: int = 32):
        if not callable(getattr(model, "process", None)):
            raise TypeError("model must provide process(question, context, ...)")
        self.model = model
        self.threshold = threshold
        self.batch_size = batch_size
        self.model_name = PROVENCE_MODEL

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = PROVENCE_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | None = None,
        **kwargs,
    ):
        """`device` defaults to CUDA when available; on CPU this model is ~15x slower."""
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        return cls(model, threshold=threshold, **kwargs)

    def prune_many(self, queries: Sequence[str], contexts: Sequence[Sequence[str]]) -> list[list[str]]:
        """One batched call for many samples; the offline evaluations use this."""
        if len(queries) != len(contexts):
            raise ValueError(f"{len(queries)} queries for {len(contexts)} contexts")
        if not queries:
            return []
        output = self.model.process(
            list(queries),
            [list(texts) for texts in contexts],
            title=None,
            threshold=self.threshold,
            batch_size=self.batch_size,
            reorder=False,
            enable_warnings=False,
        )
        return [
            [_fallback(new, old) for new, old in zip(pruned, texts)]
            for pruned, texts in zip(output["pruned_context"], contexts)
        ]

    def _process(self, query: str, texts: list[str]) -> dict:
        # title=None: NQ passages have no title line, so nothing should be force-kept.
        return self.model.process(
            query,
            [texts],
            title=None,
            threshold=self.threshold,
            batch_size=self.batch_size,
            reorder=False,
            enable_warnings=False,
        )

    def prune(self, query: str, texts: Sequence[str]) -> list[str]:
        texts = _check(query, texts)
        if not texts:
            return []
        output = self._process(query, texts)
        pruned = output["pruned_context"][0]
        if len(pruned) != len(texts):
            raise RuntimeError(f"pruner returned {len(pruned)} passages for {len(texts)}")
        return [_fallback(new, old) for new, old in zip(pruned, texts)]

    def __call__(self, query: str, texts: Sequence[str]) -> list[float]:
        """Provence's own reranking score, so it can stand in for a cross-encoder."""
        texts = _check(query, texts)
        if not texts:
            return []
        return [float(score) for score in self._process(query, texts)["reranking_score"][0]]


class MatchedRandomPruner:
    """Drops as many sentences as a reference pruner did, chosen pseudo-randomly.

    The control for `ProvencePruner`: same compression, same passage, same
    sentence splitter — only *which* sentences survive differs.
    """

    def __init__(self, reference: ProvencePruner, seed: int = 42):
        if not isinstance(reference, ProvencePruner):
            raise TypeError(f"reference must be a ProvencePruner, got {type(reference).__name__}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        self.reference = reference
        self.seed = seed
        self.model_name = f"random-prune:{seed}"

    @staticmethod
    def split(text: str) -> list[str]:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)

    def _order(self, query: str, sentences: Sequence[str]) -> list[int]:
        keys = []
        for index, sentence in enumerate(sentences):
            digest = hashlib.blake2b(repr((self.seed, query, sentence)).encode("utf-8"), digest_size=8).digest()
            keys.append((int.from_bytes(digest, "big"), index))
        return [index for _, index in sorted(keys)]

    def prune(self, query: str, texts: Sequence[str]) -> list[str]:
        texts = _check(query, texts)
        if not texts:
            return []
        reference = self.reference.prune(query, texts)
        pruned = []
        for original, kept in zip(texts, reference):
            sentences = self.split(original)
            keep_n = len(self.split(kept))
            if keep_n >= len(sentences) or not sentences:
                pruned.append(original)
                continue
            chosen = sorted(self._order(query, sentences)[:keep_n])
            pruned.append(_fallback(" ".join(sentences[index] for index in chosen), original))
        return pruned
