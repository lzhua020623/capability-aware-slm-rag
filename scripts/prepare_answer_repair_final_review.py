"""Freeze the union of all v7/r3 changed outputs from the two final evaluations."""
from __future__ import annotations

from collections import Counter
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


def prepare(root=ROOT):
    output = root / "artifacts/answer_repair/final_review_v1"
    rows, sources = [], {}
    for name, expected in (("final_nq75", 300), ("unseen_nq125", 250)):
        directory = root / "artifacts/answer_repair" / name
        receipt_raw = (directory / "result_validation.json").read_bytes()
        receipt = json.loads(receipt_raw)
        if (receipt["research_result"] is not True or receipt["synthetic_preflight_only"] is not False
                or receipt["replayed_records"] != expected
                or receipt["selected_final_revision"] != "r3"
                or receipt["version_changed_after_evaluation"] is not False):
            raise ValueError("Both real final runs must pass frozen replay")
        for relative, expected_sha in receipt["raw_file_sha256"].items():
            if sha((directory / receipt["result_directory"] / relative).read_bytes()) != expected_sha:
                raise ValueError("Raw result changed since collection")
        raw = (directory / "review/primary_changed_pairs.jsonl").read_bytes()
        group = [json.loads(x) for x in raw.splitlines() if x.strip()]
        if len(group) != receipt["pending_primary_changed_pairs"]:
            raise ValueError("Changed-output scope differs from receipt")
        sources[name] = {"receipt_sha256": sha(receipt_raw), "queue_sha256": sha(raw)}
        for source in group:
            if (not source["changed"] or source["strategy"] not in ("short_answer_quote", "conservative_repair")
                    or source["human_verified"] is not False or source["pair_review"]["status"] != "pending"):
                raise ValueError("Only primary changed outputs with pending human labels are eligible")
            material = {k: source[k] for k in ("dataset", "sample_id", "query", "initial_answer", "final_answer", "evidence", "frozen_reference")}
            material["citations"] = source["model_result"]["citations"]
            material_sha = sha(json.dumps(material, ensure_ascii=False, sort_keys=True).encode())
            # Full output/citation material, not ID alone, defines reusable review.
            existing = next((x for x in rows if x["review_material_sha256"] == material_sha), None)
            identity = {"run": name, "strategy": source["strategy"], "result_sha256": source["result_sha256"]}
            if existing:
                existing["equivalent_result_sources"].append(identity)
                continue
            human = source.get("initial_human_review") or {}
            stratum = source.get("screening_stratum") or {}
            alias = f"N{human['review_number']}" if human else f"U{stratum['challenge_number']}"
            rows.append({**source, "review_number": len(rows) + 1,
                         "review_set_id": "answer_repair_final_changed_review_v1",
                         "source_alias": alias, "source_run": name,
                         "review_material_sha256": material_sha,
                         "source_queue_sha256": sha(raw), "equivalent_result_sources": [identity]})
    raw = b"".join((json.dumps(x, ensure_ascii=False) + "\n").encode() for x in rows)
    manifest = {"version": "answer_repair_final_changed_review_v1", "queue_sha256": sha(raw),
                "sources": sources, "selected_pairs": len(rows),
                "unique_dataset_sample_ids": len({(x['dataset'], x['sample_id']) for x in rows}),
                "primary_changed_result_pairs": sum(len(x["equivalent_result_sources"]) for x in rows),
                "by_run_strategy": dict(Counter(f"{x['source_run']}/{x['strategy']}" for x in rows)),
                "human_confirmed": 0, "all_human_fields_pending": True,
                "scope": "All changed v7/r3 outputs from final dev75 and E-only stratified challenge125; ablations excluded; no correctness-based selection"}
    write_frozen(output / "pending_pairs.jsonl", raw)
    write_frozen(output / "manifest.json", encode_json(manifest))
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
