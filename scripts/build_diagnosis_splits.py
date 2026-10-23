"""Build the fixed Diagnosis v1 train/validation/test split.

Only trusted records (needs_manual_review == false) are split; flagged records
stay in the development dataset but are excluded from v1. Queries that are
exact duplicates or near-duplicates (content-token Jaccard >= threshold) form
one group and always land in the same split. Groups are allocated per
dataset x label stratum; when a group mixes labels it is stratified by its
rarest label. Group integrity takes priority over exact label proportions.

The output stores sample IDs only, per dataset, so it can be committed.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnosis.dev_dataset import load_jsonl
from src.diagnosis.features import content_tokens, tokenize

SEED = 42
SPLITS: tuple[str, ...] = ("train", "validation", "test")
RATIOS: dict[str, float] = {"train": 0.70, "validation": 0.15, "test": 0.15}
JACCARD_THRESHOLD = 0.6
DATASET_PATHS = {
    "nq": ROOT / "data" / "processed" / "diagnosis" / "nq_500.jsonl",
    "fever": ROOT / "data" / "processed" / "diagnosis" / "fever_500.jsonl",
}
OUTPUT_PATH = ROOT / "configs" / "splits" / "diagnosis_v1_split.json"


def trusted_records(records: Iterable[dict]) -> list[dict]:
    return [record for record in records if not record["metadata"]["needs_manual_review"]]


def query_groups(records: list[dict], threshold: float = JACCARD_THRESHOLD) -> list[list[dict]]:
    """Union records whose normalized queries are identical or have token Jaccard >= threshold."""
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    normalized = [" ".join(tokenize(record["input"]["query"])) for record in records]
    terms = [content_tokens(tokenize(record["input"]["query"])) for record in records]
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            union = terms[i] | terms[j]
            similar = normalized[i] == normalized[j] or (
                union and len(terms[i] & terms[j]) / len(union) >= threshold
            )
            if similar:
                parent[find(j)] = find(i)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[find(index)].append(record)
    return sorted(grouped.values(), key=lambda group: min(r["input"]["sample_id"] for r in group))


def split_targets(size: int) -> dict[str, int]:
    """Largest-remainder rounding of RATIOS so the targets sum to size."""
    raw = {name: size * RATIOS[name] for name in SPLITS}
    targets = {name: int(raw[name]) for name in SPLITS}
    by_remainder = sorted(SPLITS, key=lambda name: (-(raw[name] - targets[name]), SPLITS.index(name)))
    for name in by_remainder[: size - sum(targets.values())]:
        targets[name] += 1
    return targets


def assign_dataset(records: list[dict], seed: int = SEED) -> dict[str, str]:
    """Return sample_id -> split for one dataset."""
    label_counts = Counter(record["target_label"] for record in records)
    strata: dict[str, list[list[dict]]] = defaultdict(list)
    for group in query_groups(records):
        rarest = min({r["target_label"] for r in group}, key=lambda label: (label_counts[label], label))
        strata[rarest].append(group)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for label in sorted(strata):
        groups = strata[label]
        rng.shuffle(groups)
        targets = split_targets(sum(len(group) for group in groups))
        filled = dict.fromkeys(SPLITS, 0)
        for group in sorted(groups, key=len, reverse=True):
            name = max(SPLITS, key=lambda s: (targets[s] - filled[s], -SPLITS.index(s)))
            filled[name] += len(group)
            for record in group:
                assignment[record["input"]["sample_id"]] = name
    return assignment


def build_split(records_by_dataset: dict[str, list[dict]], seed: int = SEED) -> dict:
    splits = {}
    for dataset, records in records_by_dataset.items():
        assignment = assign_dataset(trusted_records(records), seed)
        splits[dataset] = {
            name: sorted(sample_id for sample_id, split in assignment.items() if split == name)
            for name in SPLITS
        }
    return {
        "version": "diagnosis_v1",
        "seed": seed,
        "ratios": RATIOS,
        "filter": "metadata.needs_manual_review == false",
        "grouping": {"method": "query_token_jaccard", "threshold": JACCARD_THRESHOLD},
        "splits": splits,
    }


def validate_split(split: dict, records_by_dataset: dict[str, list[dict]]) -> dict:
    """Raise ValueError on any integrity violation; return a summary for reporting."""
    summary: dict = {"counts": Counter(), "labels": Counter(), "groups_checked": 0}
    for dataset, records in records_by_dataset.items():
        by_id = {record["input"]["sample_id"]: record for record in records}
        trusted = {r["input"]["sample_id"] for r in trusted_records(records)}
        seen: dict[str, str] = {}
        for name in SPLITS:
            for sample_id in split["splits"][dataset][name]:
                if sample_id in seen:
                    raise ValueError(f"{dataset}:{sample_id} appears in {seen[sample_id]} and {name}")
                if sample_id not in by_id:
                    raise ValueError(f"{dataset}:{sample_id} is not in the development dataset")
                if by_id[sample_id]["metadata"]["needs_manual_review"]:
                    raise ValueError(f"{dataset}:{sample_id} needs manual review but is in {name}")
                seen[sample_id] = name
                summary["counts"][(dataset, name)] += 1
                summary["labels"][(dataset, name, by_id[sample_id]["target_label"])] += 1
        if set(seen) != trusted:
            raise ValueError(f"{dataset}: split does not cover every trusted sample exactly once")
        for group in query_groups(trusted_records(records)):
            names = {seen[r["input"]["sample_id"]] for r in group}
            if len(names) > 1:
                raise ValueError(f"{dataset}: query group crosses splits {sorted(names)}")
            summary["groups_checked"] += len(group) > 1
    return summary


def main() -> int:
    records_by_dataset = {name: load_jsonl(path) for name, path in DATASET_PATHS.items()}
    split = build_split(records_by_dataset)
    summary = validate_split(split, records_by_dataset)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    for dataset, records in records_by_dataset.items():
        excluded = len(records) - len(trusted_records(records))
        multi = [g for g in query_groups(trusted_records(records)) if len(g) > 1]
        print(f"{dataset}: excluded needs_manual_review={excluded}, multi-sample query groups={len(multi)} "
              f"covering {sum(len(g) for g in multi)} samples")
        for name in SPLITS:
            labels = {label: n for (d, s, label), n in sorted(summary["labels"].items()) if d == dataset and s == name}
            print(f"  {name}: {summary['counts'][(dataset, name)]} {labels}")
    print(f"Validation passed; {summary['groups_checked']} multi-sample groups kept within one split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
