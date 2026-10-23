"""Compare production P3-20 against saved offline selections, with zero inference.

Only score journals and selection-only JSONL are inputs. No Gold, answers,
Diagnosis labels, models, FAISS indexes or evaluation outcomes are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.retrieval_enhancement.p3 import RERANKER_REVISION, select_p3_20


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_saved_scores(scores_path, selections_path, output_dir, *, expected_count=2381):
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    expected = {}
    with Path(selections_path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["policy"] != "P3" or row["depth"] != 20:
                continue
            sid = row["sample_id"]
            if not isinstance(sid, str) or sid in expected:
                raise ValueError("invalid or duplicate selection sample_id")
            expected[sid] = row["selected_ids"]
    seen = set()
    ordered_ids = []
    mismatches = []
    with Path(scores_path).open("rb") as source, (target / "selections.jsonl").open("x", encoding="utf-8") as output:
        for line in source:
            if not line.endswith(b"\n"):
                raise ValueError("score journal has an incomplete final line")
            row = json.loads(line)
            digest = row.pop("_record_sha256")
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("score journal record checksum mismatch")
            sid = row["sample_id"]
            if not isinstance(sid, str) or sid in seen:
                raise ValueError("invalid or duplicate score sample_id")
            if row["dataset"] != "nq" or row["reranker_revision"] != RERANKER_REVISION:
                raise ValueError("score journal dataset/revision mismatch")
            pool = row["candidates"][:20]
            if [e["rank"] for e in pool] != list(range(1, 21)):
                raise ValueError("saved BGE ranks must be ordered 1..20")
            # Project only runtime evidence fields; do not forward saved rows.
            candidates = [{"passage_id": e["passage_id"], "text": e["text"],
                           "score": e["retrieval_score"]} for e in pool]
            scores = [e["cross_encoder_score"] for e in pool]
            actual = [e["passage_id"] for e in select_p3_20(candidates, scores)["evidence"]]
            if sid not in expected or actual != expected[sid]:
                mismatches.append({"sample_id": sid, "actual": actual, "expected": expected.get(sid)})
            seen.add(sid)
            ordered_ids.append(sid)
            output.write(json.dumps({"sample_id": sid, "selected_ids": actual}) + "\n")
    complete = len(seen) == len(expected) == expected_count and ordered_ids == list(expected)
    report = {
        "passed": complete and not mismatches, "policy": "P3-20", "expected_count": expected_count,
        "checked": len(seen), "reference_count": len(expected), "ids_and_order_match": complete,
        "mismatch_count": len(mismatches), "mismatches": mismatches,
        "model_calls": 0, "gold_or_reference_answers_read": False,
        "scores_sha256": _sha(scores_path), "reference_sha256": _sha(selections_path),
        "selector_source_sha256": _sha(ROOT / "src/compensation/retrieval_enhancement/p3.py"),
        "selections_sha256": _sha(target / "selections.jsonl"),
    }
    with (target / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    if not report["passed"]:
        raise ValueError(f"P3-20 consistency failed; see {target / 'report.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--selections", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=2381)
    args = parser.parse_args()
    print(json.dumps(check_saved_scores(args.scores, args.selections, args.output_dir,
                                       expected_count=args.expected_count), indent=2))


if __name__ == "__main__":
    main()
