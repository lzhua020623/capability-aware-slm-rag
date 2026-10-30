"""BM25 over the NQ Train corpus (bm25s), the candidate source of P3-BM25.

The saved index keeps the corpus line order, so a hit's position also indexes the BGE
corpus metadata. Parameters are those of configs/controller/l2_round_v1.json step_2_l2.
"""

from __future__ import annotations

import json
from pathlib import Path


def tokenize(texts):
    import bm25s
    return bm25s.tokenize(list(texts), stopwords='en', return_ids=False, show_progress=False)


class Bm25Index:
    """A saved bm25s index whose document order is the corpus line order."""

    def __init__(self, directory):
        import bm25s
        directory = Path(directory)
        self.receipt = json.loads((directory / 'receipt.json').read_text(encoding='utf-8'))
        self.passage_ids = json.loads((directory / 'passage_ids.json').read_text(encoding='utf-8'))
        self.model = bm25s.BM25.load(str(directory / 'index'))

    def search(self, query, k):
        """[(corpus position, score)] best first; zero scores (no shared term) are dropped."""
        docs, scores = self.model.retrieve(tokenize([query]), k=k, show_progress=False)
        return [(int(d), float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]
