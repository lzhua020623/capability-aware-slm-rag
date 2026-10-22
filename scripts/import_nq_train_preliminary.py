"""Read the 15 preliminary Drive workbooks into an offline label overlay.

No spreadsheet is edited. IDs must be text; every frozen source field, shard
hash, partition and workbook summary is checked before publishing. Preliminary
labels never become human gold or model inputs. Existing human decisions remain
separate, with disagreements reported rather than overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
HEADERS = ("sample_id", "dataset", "query", "initial_answer", "evidence",
           "annotation_reference", "label", "secondary_label")
LABELS = {
    "Correct": "correct",
    "Evidence Insufficiency": "evidence_insufficiency",
    "Evidence Accessibility Failure": "evidence_accessibility",
    "Evidence Utilisation Failure": "evidence_utilisation",
}
SUMMARY_KEYS = {
    "total": ("Total records", "Records labeled (preliminary)", "Total preliminary labels", "Total provisional labeled records"),
    "semantic": ("Human-reviewed labels", "Five-passage manually adjudicated", "Fully adjudicated with Top-5",
                 "Complete Top-5 semantic adjudications", "Complete Top-5 semantic adjudications (selected examples)",
                 "Selected evidence-informed semantic reviews", "Targeted Top-5 semantic reviews", "Full Top-5 evidence semantic review"),
    "automatic": ("Automated preliminary labels", "Automatically pre-labeled, not manually adjudicated",
                  "Automatic provisional labels (NOT fully reviewed)", "Automated provisional labels (UNVERIFIED)",
                  "Automated provisional labels", "Automatic preliminary labels"),
    "review": ("Flagged for review", "Flagged review questions", "Flagged possible alternate labels",
               "Review questions / flagged uncertainties", "Doubt records"),
}


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def excel_text(text: str) -> str:
    # Decode one OOXML escape pass only (escaped literal '_xNNNN_' stays literal).
    return re.sub(r"_x([0-9A-Fa-f]{4})_", lambda m: chr(int(m[1], 16)), text)


def sheet_rows(path: Path, name: str):
    """Stream OOXML cells without coercing long sample IDs through floats."""
    with ZipFile(path) as book:
        rels = {r.attrib["Id"]: r.attrib["Target"]
                for r in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))}
        sheets = ET.fromstring(book.read("xl/workbook.xml")).find(NS + "sheets")
        sheet = next((s for s in sheets if s.attrib["name"] == name), None)
        if sheet is None:
            raise ValueError(f"{path.name}: missing sheet {name}")
        target = rels[sheet.attrib[REL + "id"]]
        target = target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            shared = [excel_text("".join(si.itertext())) for si in
                      ET.fromstring(book.read("xl/sharedStrings.xml"))]
        with book.open(target) as handle:
            for _, element in ET.iterparse(handle, events=("end",)):
                if element.tag != NS + "row":
                    continue
                cells, types = {}, {}
                for cell in element.findall(NS + "c"):
                    address = cell.attrib["r"]
                    col = re.match(r"[A-Z]+", address)[0]
                    kind = cell.attrib.get("t", "n")
                    if cell.find(NS + "f") is not None:
                        raise ValueError(f"{path.name}/{name}!{address}: formula is not source data")
                    value = cell.findtext(NS + "v")
                    if kind == "inlineStr":
                        value = excel_text("".join(t.text or "" for t in cell.findall(".//" + NS + "t")))
                    elif kind == "s":
                        value = shared[int(value)]
                    cells[col], types[col] = value, kind
                yield int(element.attrib["r"]), cells, types
                element.clear()


def annotation_rows(path: Path):
    rows = sheet_rows(path, "C1_Annotations")
    _, header, _ = next(rows)
    if tuple(header.get(chr(65 + i)) for i in range(len(HEADERS))) != HEADERS:
        raise ValueError(f"{path.name}: unexpected annotation headers")
    for number, cells, types in rows:
        if not any(v is not None for v in cells.values()):
            continue
        if types.get("A") not in {"inlineStr", "s", "str"}:
            raise ValueError(f"{path.name}!A{number}: sample_id must be stored as text")
        row = {key: cells.get(chr(65 + i)) for i, key in enumerate(HEADERS)}
        if row["label"] not in LABELS:
            raise ValueError(f"{path.name}!G{number}: unknown/empty label {row['label']!r}")
        if row["secondary_label"] not in {None, ""}:
            raise ValueError(f"{path.name}!H{number}: unexpected secondary label")
        for key in ("evidence", "annotation_reference"):
            row[key] = json.loads(row[key])
        yield number, row


def summary_values(path: Path):
    return {cells.get("A"): cells.get("B") for number, cells, _ in sheet_rows(path, "Summary")
            if number > 1 and cells.get("A") is not None}


def summary_count(summary: dict, kind: str, *, optional=False):
    found = [int(summary[key]) for key in SUMMARY_KEYS[kind] if key in summary]
    if not found and optional:
        return None
    if len(found) != 1:
        raise ValueError(f"Missing/ambiguous summary metric {kind}")
    return found[0]


def write_new(path: Path, data: bytes):
    """Allow identical replay; refuse to replace a different frozen result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing changed output: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(data)


def audit_import(*, workbook_dir: Path, source_dir: Path, delivery_manifest: Path,
                 partition_path: Path, human_path: Path, out_dir: Path, audit_path: Path):
    delivery, download = load(delivery_manifest), load(workbook_dir / "download_manifest.json")
    download_files = {row["name"]: row for row in download["files"]}
    expected_books = {f"NQ_C1_{i:03d}_preliminary_annotations.xlsx" for i in range(delivery["shard_count"])}
    if set(download_files) != expected_books or {p.name for p in workbook_dir.glob("*.xlsx")} != expected_books:
        raise ValueError("Workbook set differs from delivery")
    partitions = {}
    for role, ids in load(partition_path)["partitions"].items():
        for sample_id in ids:
            if sample_id in partitions:
                raise ValueError(f"Duplicate ID in partitions: {sample_id}")
            partitions[sample_id] = role
    human = {r["sample_id"]: r for r in jsonl(human_path)}
    seen, counts, by_role = set(), Counter(), defaultdict(Counter)
    disagreements, unresolved, matched_human, shards = [], [], [], []
    review_rows_total, summary_semantic_total, summary_auto_total = 0, 0, 0
    with tempfile.TemporaryDirectory(prefix="nq-preliminary-") as temporary:
        temp_overlay = Path(temporary) / "labels.jsonl"
        with temp_overlay.open("w", encoding="utf-8") as output:
            for i, shard in enumerate(delivery["shards"]):
                book = workbook_dir / f"NQ_C1_{i:03d}_preliminary_annotations.xlsx"
                book_hash = sha(book)
                if book_hash != download_files[book.name]["sha256"]:
                    raise ValueError(f"Workbook checksum mismatch: {book.name}")
                source = source_dir / shard["file"]
                if sha(source) != shard["sha256"]:
                    raise ValueError(f"Source checksum mismatch: {source.name}")
                originals = {r["sample_id"]: r for r in jsonl(source)}
                local_ids, local_counts = set(), Counter()
                summary = summary_values(book)
                for row_no, row in annotation_rows(book):
                    sample_id = row["sample_id"]
                    if sample_id in seen or sample_id not in originals or sample_id not in partitions:
                        raise ValueError(f"Duplicate/unknown/unpartitioned ID: {sample_id}")
                    original = originals[sample_id]
                    for key in HEADERS[:6]:
                        if json.dumps(row[key], sort_keys=True, ensure_ascii=False) != json.dumps(original[key], sort_keys=True, ensure_ascii=False):
                            raise ValueError(f"{book.name}!row {row_no}/{sample_id}: changed frozen field {key}")
                    seen.add(sample_id); local_ids.add(sample_id)
                    label = LABELS[row["label"]]
                    counts[label] += 1; local_counts[label] += 1
                    by_role[partitions[sample_id]][label] += 1
                    item = {
                        "sample_id": sample_id, "dataset": "nq", "label": label,
                        "raw_label": row["label"], "data_role": partitions[sample_id],
                        "annotation_status": "preliminary_mixed_heuristic_and_subset_semantic_review",
                        "human_verified": False, "per_record_review_provenance": "not_supplied",
                        "runtime_use": False,
                        "source": {"file": book.name, "sha256": book_hash,
                                   "sheet": "C1_Annotations", "row": row_no},
                    }
                    output.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                    if sample_id in human:
                        h = human[sample_id]
                        decision = h["human_review"]
                        comparison = {"sample_id": sample_id, "review_number": h["review_number"],
                                      "preliminary_label": label, "human_label": decision.get("primary_label"),
                                      "human_status": decision["status"], "human_decision_preserved": True}
                        if decision["status"] != "completed" or decision.get("primary_label") not in LABELS.values():
                            unresolved.append(comparison)
                        elif decision["primary_label"] != label:
                            disagreements.append(comparison)
                        else:
                            matched_human.append(comparison)
                if local_ids != set(originals) or len(local_ids) != shard["n_samples"]:
                    raise ValueError(f"Missing/extra rows in {book.name}")
                summary_total = summary_count(summary, "total", optional=True)
                if summary_total is not None and summary_total != len(local_ids):
                    raise ValueError(f"Summary total mismatch: {book.name}")
                for raw, normalized in LABELS.items():
                    if int(summary[raw]) != local_counts[normalized]:
                        raise ValueError(f"Summary label mismatch: {book.name}/{raw}")
                reviews = list(sheet_rows(book, "Review_Questions"))
                review_count = sum(1 for row_no, cells, _ in reviews if row_no > 1 and cells.get("A"))
                if review_count != summary_count(summary, "review"):
                    raise ValueError(f"Review sheet count mismatch: {book.name}")
                unknown_review_ids = {cells.get("A") for row_no, cells, _ in reviews if row_no > 1 and cells.get("A")} - local_ids
                if unknown_review_ids:
                    raise ValueError(f"Unknown review IDs: {book.name}")
                semantic, automatic = summary_count(summary, "semantic"), summary_count(summary, "automatic")
                if semantic + automatic != len(local_ids):
                    raise ValueError(f"Summary provenance count mismatch: {book.name}")
                review_rows_total += review_count; summary_semantic_total += semantic; summary_auto_total += automatic
                shards.append({"file": book.name, "sha256": book_hash, "rows": len(local_ids),
                               "label_counts": dict(sorted(local_counts.items())), "review_sheet_rows": review_count,
                               "summary": summary})
                print(f"Checked {book.name}: {len(local_ids)} rows", flush=True)
        if len(seen) != delivery["total_samples"] or seen != set(partitions):
            raise ValueError("Full ID coverage differs from delivery/partition manifest")
        audit = {
            "version": "nq_train_preliminary_import_20261022_v1", "ok": True,
            "source_folder_url": download["source_folder_url"], "received_date": download["received_date"],
            "download_zip_sha256": download["zip_sha256"], "total": len(seen), "workbook_count": len(shards),
            "duplicate_ids": 0, "missing_ids": 0, "unknown_ids": 0, "frozen_field_changes": 0,
            "all_sample_ids_stored_as_text": True, "label_counts": dict(sorted(counts.items())),
            "partition_label_counts": {k: dict(sorted(v.items())) for k, v in sorted(by_role.items())},
            "workbook_claimed_semantic_review_total": summary_semantic_total,
            "workbook_claimed_automatic_preliminary_total": summary_auto_total,
            "review_sheet_rows_total": review_rows_total,
            "review_sheet_rows_are_not_an_unresolved_count": True,
            "human_verified_labels_imported": 0,
            "per_record_semantic_review_ids_or_reviewer_provenance_supplied": False,
            "existing_human_comparison": {"matched": len(matched_human), "disagreed": len(disagreements),
                                           "unresolved": len(unresolved), "disagreements": sorted(disagreements, key=lambda r: r["review_number"])},
            "files": shards,
            "inputs_sha256": {"delivery_manifest": sha(delivery_manifest), "partition_ids": sha(partition_path),
                              "existing_human": sha(human_path)},
            "overlay": {"path": str((out_dir / "labels.jsonl").relative_to(ROOT)) if out_dir.is_relative_to(ROOT) else str(out_dir / "labels.jsonl"),
                        "sha256": sha(temp_overlay), "runtime_use": False},
            "policy": ["Preliminary labels only; not human gold or a measured accuracy.",
                       "No source/gold/evidence/workbook/partition/human decision modified.",
                       "No sample selection, prompt tuning or GPU inference performed.",
                       "Reserved split is E-only; team-wide unseen status remains unconfirmed."],
        }
        report = (json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        # Check both frozen destinations before writing either.
        destinations = [(out_dir / "labels.jsonl", temp_overlay.read_bytes()), (audit_path, report)]
        for path, data in destinations:
            if path.exists() and path.read_bytes() != data:
                raise FileExistsError(f"Refusing changed output: {path}")
        for path, data in destinations:
            write_new(path, data)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook-dir", type=Path, default=ROOT / "data/processed/external/nq_train_preliminary_20261022")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/processed/external/nq_train_delivery_20261017")
    parser.add_argument("--delivery-manifest", type=Path, default=ROOT / "data/annotations/nq_train_delivery/manifest.json")
    parser.add_argument("--partition-path", type=Path, default=ROOT / "data/processed/answer_repair/nq_train_v1/partition_ids.json")
    parser.add_argument("--human-path", type=Path, default=ROOT / "data/annotations/human/answer_repair_nq_train_dev_v1_through_batch06_20261019.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/processed/answer_repair/nq_train_preliminary_20261022")
    parser.add_argument("--audit-path", type=Path, default=ROOT / "data/annotations/nq_train_preliminary/import_audit_20261022.json")
    args = parser.parse_args()
    audit = audit_import(**vars(args))
    print(json.dumps({key: audit[key] for key in ("ok", "total", "label_counts", "existing_human_comparison")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
