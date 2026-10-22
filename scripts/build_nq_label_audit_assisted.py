"""Add separately validated agent notes to a derived A200 page, never human gold."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_nq_label_audit import build_page
from scripts.import_nq_train_preliminary import write_new, sha


def build(folder, assisted, out_stem="A200_assisted_review"):
    if Path(out_stem).name != out_stem:
        raise ValueError("Output stem must be a filename")
    packet = json.loads((folder / "review_packet.json").read_bytes())
    records = {r["review_number"]: r for r in packet["records"]}
    seen = set()
    for line in assisted.read_bytes().splitlines():
        if not line.strip():
            continue
        row = json.loads(line); number = row["review_number"]
        if number in seen or number not in records:
            raise ValueError("Duplicate or unknown assisted number")
        original = records[number]
        if (row["sample_id"] != original["sample_id"] or row["human_verified"] is not False
                or row["review_set_id"] != packet["review_set_id"]
                or row["source_record_sha256"] != original["source_record_sha256"]
                or row["source_review_sha256"] != sha(folder / "review_queue.jsonl")):
            raise ValueError("Assisted record does not attest exact source")
        for quote in row["agent_review"]["quoted_evidence"]:
            e = next(e for e in original["evidence"] if e["rank"] == quote["rank"])
            if (e["passage_id"] != quote["passage_id"]
                    or e["text"][quote["start"]:quote["end"]] != quote["quote"]):
                raise ValueError("Assisted quote/source interval mismatch")
        original["agent_assessment"] = copy.deepcopy(row["agent_review"])
        seen.add(number)
    if any(r["human_verified"] or r["human_review"]["status"] != "pending" for r in records.values()):
        raise ValueError("Derived assistance cannot alter human labels")
    packet.pop("packet_sha256")
    packet["packet_id"] += "_" + out_stem
    packet["sources_sha256"]["assisted"] = sha(assisted)
    packet["counts"].update(assisted=len(seen),
        disputed=sum(r["agent_assessment"]["suggested_diagnosis_label"] is None for r in records.values()),
        suggested_correct=sum(r["agent_assessment"]["suggested_diagnosis_label"] == "correct" for r in records.values()))
    packet["packet_sha256"] = hashlib.sha256(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    raw = build_page(packet)
    write_new(folder / (out_stem + ".html"), raw)
    receipt = {"assisted": len(seen), "remaining_without_semantic_assistance": len(records) - len(seen),
               "human_confirmed": 0, "assisted_numbers": sorted(seen), "packet_sha256": packet["packet_sha256"],
               "html_sha256": hashlib.sha256(raw).hexdigest(), "assisted_file_sha256": sha(assisted),
               "reference_visible_assistance_not_independent_human_gold": True}
    write_new(folder / (out_stem + ".manifest.json"), (json.dumps(receipt, indent=2) + "\n").encode())
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=ROOT / "artifacts/answer_repair/nq_label_audit_200_v1")
    p.add_argument("--assisted", type=Path, default=ROOT / "data/annotations/assisted/nq_label_audit_200_v1_batch01_20261022.jsonl")
    p.add_argument("--out-stem", default="A200_assisted_through_batch01")
    args = p.parse_args()
    print(json.dumps(build(args.folder, args.assisted, args.out_stem), indent=2))


if __name__ == "__main__":
    main()
