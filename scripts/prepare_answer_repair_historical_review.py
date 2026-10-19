"""Freeze a bounded sample of historical v7 changed outputs; assign no truth labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_answer_repair_experiment import encode_json, write_frozen


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(policy_path, source_paths, out_dir):
    policy_raw = policy_path.read_bytes(); policy = json.loads(policy_raw)
    sources = policy["sources"]
    if len(source_paths) != len(sources):
        raise ValueError("Provide one exact verified queue for each source in policy order")
    groups = defaultdict(list); total = 0; seen = set()
    for (identity, expected), path in zip(sources.items(), source_paths):
        raw = path.read_bytes()
        if sha(raw) != expected:
            raise ValueError("Historical source queue checksum mismatch: " + identity)
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line); total += 1
            if (row["cohort"] not in policy["eligible_cohorts"] or not row["changed"]
                    or row["strategy"] != policy["strategies"][row["dataset"]]):
                continue
            result = row["model_result"]
            if (sha(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()) != row["result_sha256"]
                    or result["changed"] is not True or row["pair_review"]["status"] != "pending"
                    or any(row["pair_review"][f] is not None for f in (
                        "initial_correct", "final_correct", "initial_supported", "final_supported"))):
                raise ValueError("Expected frozen model results with unfilled human judgments")
            key = (row["cohort"], row["dataset"], row["strategy"])
            identity_key = (*key, row["sample_id"])
            if identity_key in seen:
                raise ValueError("Duplicate historical result pair")
            seen.add(identity_key)
            rank = sha((f"E-v7-historical-changed-v1:{policy['seed']}:" +
                        ":".join(identity_key)).encode())
            groups[key].append((rank, row["sample_id"], row, identity))
    selected, counts = [], {}
    for key, rows in sorted(groups.items()):
        chosen = sorted(rows, key=lambda x: x[:2])[:policy["maximum_per_cohort_dataset_strategy"]]
        counts["/".join(key)] = {"changed_available": len(rows), "selected": len(chosen)}
        for rank, _, row, source in chosen:
            selected.append({**row, "review_number": len(selected) + 1,
                "review_set_id": policy["version"], "sampling_sha256": rank,
                "source_queue": source, "source_queue_sha256": sources[source],
                "human_verified": False, "agent_assistance_status": "pending"})
    if not selected:
        raise ValueError("No eligible historical changed outputs")
    queue_raw = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in selected)
    manifest = {"version": policy["version"], "policy_sha256": sha(policy_raw),
        "source_queues_sha256": sources, "total_source_result_records": total,
        "eligible_changed_pairs": sum(len(v) for v in groups.values()),
        "selected_pairs": len(selected), "unique_dataset_sample_ids": len({(r["dataset"], r["sample_id"]) for r in selected}),
        "by_cohort_dataset_strategy": counts, "queue_sha256": sha(queue_raw),
        "human_verified": False, "all_human_fields_pending": True,
        "agent_assistance_complete": False, "selection_uses_semantic_judgments_or_EM": False,
        "scope": policy["scope"]}
    files = {out_dir / "manifest.json": encode_json(manifest), out_dir / "pending_pairs.jsonl": queue_raw}
    inputs = {policy_path.resolve(), *[p.resolve() for p in source_paths]}
    if any(path.resolve() in inputs for path in files):
        raise ValueError("Outputs must not replace source inputs")
    for path, raw in files.items():
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Historical sample already frozen; do not silently resample")
    for path, raw in files.items():
        write_frozen(path, raw)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/answer_repair_v7_historical_review_v1.json")
    parser.add_argument("--sources", type=Path, nargs=2, help="Optional byte-identical local copies, in policy source order")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/answer_repair/v7_historical_review_v1")
    args = parser.parse_args(argv)
    policy = json.loads(args.policy.read_bytes())
    paths = args.sources or [ROOT / p for p in policy["sources"]]
    print(json.dumps(prepare(args.policy, paths, args.out_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
