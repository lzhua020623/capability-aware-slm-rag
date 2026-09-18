"""Build frozen Preliminary Recoverability Experiment sample manifests.

Reads existing processed NQ/FEVER files. Does not modify source data and
does not run a language model.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NQ_SRC = ROOT / "data" / "processed" / "nq_controlled" / "dev.jsonl"
FEVER_SRC = ROOT / "data" / "processed" / "retrieval" / "fever_dev.jsonl"
NQ_OUT = ROOT / "configs" / "splits" / "preliminary_nq_500.json"
FEVER_OUT = ROOT / "configs" / "splits" / "preliminary_fever_500.json"

SEED = 42
NQ_N = 500
NQ_SOURCE_N = 3216
FEVER_PER_LABEL = 250


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sample_nq(records: list[dict]) -> list[str]:
    ids = [str(record["id"]) for record in records]
    if len(ids) != NQ_SOURCE_N:
        raise RuntimeError(
            f"Expected {NQ_SOURCE_N} NQ controlled samples, found {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise RuntimeError("NQ controlled subset contains duplicate IDs")
    return random.Random(SEED).sample(ids, NQ_N)


def has_valid_gold_evidence(record: dict) -> bool:
    gold_text = record.get("gold_evidence_text")
    gold_pages = record.get("gold_page_ids")
    return bool(gold_text) and bool(gold_pages)


def sample_fever(records: list[dict]) -> list[dict]:
    supports: list[int | str] = []
    refutes: list[int | str] = []
    seen: set[str] = set()
    for record in records:
        label = record.get("label")
        if label not in {"SUPPORTS", "REFUTES"}:
            continue
        if not has_valid_gold_evidence(record):
            continue
        sample_id = record["id"]
        key = str(sample_id)
        if key in seen:
            raise RuntimeError(f"Duplicate FEVER id: {sample_id}")
        seen.add(key)
        if label == "SUPPORTS":
            supports.append(sample_id)
        else:
            refutes.append(sample_id)

    if len(supports) < FEVER_PER_LABEL or len(refutes) < FEVER_PER_LABEL:
        raise RuntimeError(
            "Not enough valid-gold FEVER samples: "
            f"SUPPORTS={len(supports)}, REFUTES={len(refutes)}"
        )

    rng = random.Random(SEED)
    sampled_supports = rng.sample(supports, FEVER_PER_LABEL)
    sampled_refutes = rng.sample(refutes, FEVER_PER_LABEL)
    return [{"id": sample_id, "label": "SUPPORTS"} for sample_id in sampled_supports] + [
        {"id": sample_id, "label": "REFUTES"} for sample_id in sampled_refutes
    ]


def verify_nq(ids: list[str]) -> None:
    if len(ids) != NQ_N or len(set(ids)) != NQ_N:
        raise RuntimeError(f"NQ manifest must contain {NQ_N} unique IDs, got {len(ids)}")


def verify_fever(items: list[dict]) -> None:
    ids = [str(item["id"]) for item in items]
    if len(ids) != 2 * FEVER_PER_LABEL or len(set(ids)) != 2 * FEVER_PER_LABEL:
        raise RuntimeError(
            f"FEVER manifest must contain {2 * FEVER_PER_LABEL} unique IDs, got {len(ids)}"
        )
    supports = sum(1 for item in items if item["label"] == "SUPPORTS")
    refutes = sum(1 for item in items if item["label"] == "REFUTES")
    if supports != FEVER_PER_LABEL or refutes != FEVER_PER_LABEL:
        raise RuntimeError(
            f"FEVER labels must be {FEVER_PER_LABEL}/{FEVER_PER_LABEL}, "
            f"got SUPPORTS={supports}, REFUTES={refutes}"
        )


def main() -> int:
    nq_records = load_jsonl(NQ_SRC)
    fever_records = load_jsonl(FEVER_SRC)
    nq_ids = sample_nq(nq_records)
    fever_items = sample_fever(fever_records)
    verify_nq(nq_ids)
    verify_fever(fever_items)

    write_json(
        NQ_OUT,
        {
            "dataset": "natural_questions",
            "seed": SEED,
            "n": NQ_N,
            "source": str(NQ_SRC.relative_to(ROOT)).replace("\\", "/"),
            "ids": nq_ids,
        },
    )
    write_json(
        FEVER_OUT,
        {
            "dataset": "fever",
            "seed": SEED,
            "n": 2 * FEVER_PER_LABEL,
            "supports": FEVER_PER_LABEL,
            "refutes": FEVER_PER_LABEL,
            "source": str(FEVER_SRC.relative_to(ROOT)).replace("\\", "/"),
            "filter": "valid gold evidence, SUPPORTS/REFUTES",
            "samples": fever_items,
        },
    )

    fever_ids = {str(item["id"]) for item in fever_items}
    supports = sum(1 for item in fever_items if item["label"] == "SUPPORTS")
    refutes = sum(1 for item in fever_items if item["label"] == "REFUTES")
    print(f"Wrote {NQ_OUT}")
    print(f"NQ unique IDs: {len(set(nq_ids))}")
    print(f"Wrote {FEVER_OUT}")
    print(f"FEVER unique IDs: {len(fever_ids)}")
    print(f"FEVER SUPPORTS: {supports}")
    print(f"FEVER REFUTES: {refutes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
