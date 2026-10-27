"""Build cross-encoder training data from the `evidence_accessibility` annotations.

Accessibility means the needed information is inside the frozen Top-5 but
noise, conflict or ordering stopped the generator from using it. That is
exactly the supervision a reranker wants: within one sample's Top-5, the
answer-bearing passage is the positive and the distractors that outranked it
are the negatives. No new passages are retrieved, so the frozen retrieval stays
untouched.

Positives are found with the gold answer (a contiguous normalised token run in
the passage text) and fall back to the gold passage ID. Gold fields are offline
supervision: they shape the training set here and never reach runtime.

The current CLI uses the frozen team train/validation IDs and preserves the
older reserve/selection restrictions. It partitions records BEFORE building
examples, checks normalized query overlap, and writes a checksum-bound bundle
without a test file. The old grouping/splitting helpers below remain only for
historical reproduction; the current CLI never calls them.

Usage:
    python scripts/build_rerank_training_data.py
    python scripts/build_rerank_training_data.py --max-negatives 2 --label evidence_accessibility
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.qa import normalize_answer  # noqa: E402

DEFAULT_RECORDS = ROOT / "data" / "processed" / "external" / "nq_train_delivery_20261017"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "team_nq_v1" / "rerank"
JACCARD_THRESHOLD = 0.6
RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokens(text: str) -> frozenset[str]:
    return frozenset(normalize_answer(text).split())


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def contains_answer(answer: str, text: str) -> bool:
    """The answer appears as a contiguous normalised token run in the passage."""
    needle = normalize_answer(answer).split()
    if not needle:
        return False
    hay = normalize_answer(text).split()
    span = len(needle)
    return any(hay[start : start + span] == needle for start in range(len(hay) - span + 1))


def rare_token_blocks(token_sets: list[frozenset[str]], per_item: int = 2) -> dict[str, list[int]]:
    """Index each item under its rarest tokens, so only plausible pairs are compared."""
    frequency: Counter[str] = Counter()
    for item in token_sets:
        frequency.update(item)
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(token_sets):
        for token in sorted(item, key=lambda token: (frequency[token], token))[:per_item]:
            blocks[token].append(index)
    return blocks


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def group_queries(token_sets: list[frozenset[str]], threshold: float) -> list[int]:
    """Union-find group id per query: near-duplicate questions share a group."""
    groups = UnionFind(len(token_sets))
    for indices in rare_token_blocks(token_sets).values():
        if len(indices) < 2 or len(indices) > 400:  # a huge block means the token is not rare after all
            continue
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                if jaccard(token_sets[left], token_sets[right]) >= threshold:
                    groups.union(left, right)
    return [groups.find(index) for index in range(len(token_sets))]


def held_out_filter(path: Path, threshold: float):
    """Predicate: True when a query is too close to a frozen Dev/Test query."""
    rows = read_jsonl(path)
    held_out = [tokens(row.get("question") or row.get("query") or "") for row in rows]
    exact = {frozenset(item) for item in held_out}
    blocks = rare_token_blocks(held_out)

    def too_close(query_tokens: frozenset[str]) -> bool:
        if query_tokens in exact:
            return True
        frequency: Counter[str] = Counter()
        for token in query_tokens:
            frequency[token] += 1
        candidates: set[int] = set()
        for token in query_tokens:
            candidates.update(blocks.get(token, ()))
        return any(jaccard(query_tokens, held_out[index]) >= threshold for index in candidates)

    return too_close, len(rows)


def build_examples(records: list[dict], label: str, max_negatives: int) -> tuple[list[dict], Counter]:
    reasons: Counter[str] = Counter()
    examples: list[dict] = []
    for record in records:
        if record["primary_label"] != label:
            continue
        reasons["candidates"] += 1
        reference = record.get("annotation_reference") or {}
        gold_answer = str(reference.get("gold_answer") or "")
        gold_passage_id = reference.get("gold_passage_id")
        evidence = record["evidence"]

        positives = [passage for passage in evidence if gold_answer and contains_answer(gold_answer, passage["text"])]
        source = "gold_answer_text"
        if not positives and gold_passage_id:
            positives = [passage for passage in evidence if passage["passage_id"] == gold_passage_id]
            source = "gold_passage_id"
        if not positives:
            reasons["skipped_no_positive"] += 1
            continue

        positive_ids = {passage["passage_id"] for passage in positives}
        negatives = [passage for passage in evidence if passage["passage_id"] not in positive_ids]
        if gold_passage_id:
            negatives = [passage for passage in negatives if passage["passage_id"] != gold_passage_id]
        if not negatives:
            reasons["skipped_no_negative"] += 1
            continue

        best = min(positives, key=lambda passage: passage["rank"])
        # Hardest negatives first: the distractors the frozen retriever ranked above the answer.
        negatives.sort(key=lambda passage: passage["rank"])
        kept = negatives[:max_negatives] if max_negatives else negatives
        reasons[f"positive_from_{source}"] += 1
        reasons["positive_rank_%d" % best["rank"]] += 1
        examples.append(
            {
                "sample_id": record["sample_id"],
                "query": record["query"],
                "positive": {"passage_id": best["passage_id"], "rank": best["rank"], "text": best["text"]},
                "negatives": [
                    {"passage_id": passage["passage_id"], "rank": passage["rank"], "text": passage["text"]}
                    for passage in kept
                ],
                "gold_answer": gold_answer,
                "positive_source": source,
            }
        )
    return examples, reasons


def split_examples(examples: list[dict], group_ids: list[int]) -> dict[str, list[dict]]:
    by_group: dict[int, list[dict]] = defaultdict(list)
    for example, group in zip(examples, group_ids):
        by_group[group].append(example)
    groups = sorted(by_group)
    random.Random(SEED).shuffle(groups)
    train_end = int(len(groups) * RATIOS["train"])
    validation_end = train_end + int(len(groups) * RATIOS["validation"])
    assignment = {
        "train": groups[:train_end],
        "validation": groups[train_end:validation_end],
        "test": groups[validation_end:],
    }
    return {name: [example for group in ids for example in by_group[group]] for name, ids in assignment.items()}


def main(argv=None) -> None:
    from src.data.team_nq import TeamNQBoundary, prepare_training_bundle
    parser = argparse.ArgumentParser(description="Build reranker data on the frozen team train/validation roles")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default="evidence_accessibility")
    parser.add_argument("--max-negatives", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_negatives < 0:
        parser.error("--max-negatives must be nonnegative")
    boundary = TeamNQBoundary(ROOT)
    report = prepare_training_bundle(boundary, args.records, args.out_dir,
        lambda rows: build_examples(rows, args.label, args.max_negatives), component="reranker")
    print(json.dumps({"boundary": report["boundary"], "role_examples":
        {r: spec["examples"] for r, spec in report["role_files"].items()}}, indent=2))


if __name__ == "__main__":
    main()
