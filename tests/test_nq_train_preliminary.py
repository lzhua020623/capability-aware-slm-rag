"""Guard label provenance, exact IDs and frozen inputs on spreadsheet import."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from scripts.import_nq_train_preliminary import HEADERS, NS, audit_import, sha, write_new


def worksheet(rows, numeric_id=False):
    root = ET.Element(NS + "worksheet")
    data = ET.SubElement(root, NS + "sheetData")
    for i, values in enumerate(rows, 1):
        row = ET.SubElement(data, NS + "row", r=str(i))
        for j, value in enumerate(values):
            kind = "n" if numeric_id and i == 2 and j == 0 else "inlineStr"
            cell = ET.SubElement(row, NS + "c", r=f"{chr(65+j)}{i}", t=kind)
            if value is not None:
                if kind == "n":
                    ET.SubElement(cell, NS + "v").text = str(value)
                else:
                    ET.SubElement(ET.SubElement(cell, NS + "is"), NS + "t").text = str(value)
    return ET.tostring(root, encoding="utf-8")


def fixture(root, *, mutate=None, numeric_id=False, wrong_count=False):
    raw, books = root / "raw", root / "books"
    raw.mkdir(); books.mkdir()
    sample_id = "8399475098188452058"
    original = {"sample_id": sample_id, "dataset": "nq", "query": "Who?", "initial_answer": "A",
                "evidence": [{"rank": i, "text": "A", "passage_id": str(i)} for i in range(1, 6)],
                "annotation_reference": {"gold_answer": "A"}, "label": None}
    source = raw / "nq_train_annotation_000.jsonl"
    source.write_text(json.dumps(original) + "\n")
    manifest = root / "delivery.json"
    manifest.write_text(json.dumps({"total_samples": 1, "shard_count": 1, "shards": [
        {"file": source.name, "sha256": sha(source), "n_samples": 1}]}))
    partition = root / "partition.json"
    partition.write_text(json.dumps({"partitions": {"development": [sample_id]}}))
    human = root / "human.jsonl"
    human.write_text(json.dumps({"sample_id": sample_id, "review_number": 3, "human_review": {
        "status": "completed", "primary_label": "evidence_utilisation"}}) + "\n")
    row = dict(original, label="Correct", secondary_label=None)
    if mutate:
        mutate(row)
    values = [json.dumps(row[k]) if k in {"evidence", "annotation_reference"} else row[k] for k in HEADERS]
    summary = [["Metric", "Value"], ["Total records", 1], ["Correct", 2 if wrong_count else 1],
               ["Evidence Insufficiency", 0], ["Evidence Accessibility Failure", 0],
               ["Evidence Utilisation Failure", 0], ["Doubt records", 0],
               ["Full Top-5 evidence semantic review", 0], ["Automatic preliminary labels", 1]]
    names = ["C1_Annotations", "Review_Questions", "Summary"]
    book = books / "NQ_C1_000_preliminary_annotations.xlsx"
    with ZipFile(book, "w") as z:
        sheets = ''.join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(names, 1))
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{NS[1:-1]}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{sheets}</sheets></workbook>')
        rels = ''.join(f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>' for i in range(1, 4))
        z.writestr("xl/_rels/workbook.xml.rels", f'<Relationships>{rels}</Relationships>')
        z.writestr("xl/worksheets/sheet1.xml", worksheet([HEADERS, values], numeric_id))
        z.writestr("xl/worksheets/sheet2.xml", worksheet([["sample_id", "selected_label"]]))
        z.writestr("xl/worksheets/sheet3.xml", worksheet(summary))
    (books / "download_manifest.json").write_text(json.dumps({"files": [{"name": book.name, "sha256": sha(book)}],
        "source_folder_url": "https://drive.google.com/fixture", "received_date": "2026-10-22", "zip_sha256": "fixture"}))
    return {"workbook_dir": books, "source_dir": raw, "delivery_manifest": manifest,
            "partition_path": partition, "human_path": human, "out_dir": root / "out", "audit_path": root / "audit.json"}


class PreliminaryImportTests(unittest.TestCase):
    def test_exact_text_id_offline_provenance_and_human_decision_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            human_bytes = args["human_path"].read_bytes()
            report = audit_import(**args)
            overlay = json.loads((args["out_dir"] / "labels.jsonl").read_text())
            self.assertEqual(overlay["sample_id"], "8399475098188452058")
            self.assertFalse(overlay["human_verified"])
            self.assertFalse(overlay["runtime_use"])
            self.assertNotIn("query", overlay)
            self.assertNotIn("annotation_reference", overlay)
            self.assertEqual(report["existing_human_comparison"]["disagreed"], 1)
            self.assertEqual(args["human_path"].read_bytes(), human_bytes)
            self.assertEqual(audit_import(**args), report)

    def test_rejects_precision_risk_and_mutated_frozen_fields_without_publishing(self):
        cases = [{"numeric_id": True}, {"mutate": lambda row: row.update(query="changed")},
                 {"mutate": lambda row: row["evidence"][0].update(text="changed")},
                 {"mutate": lambda row: row.update(label="unknown")}, {"wrong_count": True}]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                args = fixture(Path(directory), **case)
                with self.assertRaises(ValueError):
                    audit_import(**args)
                self.assertFalse(args["audit_path"].exists())
                self.assertFalse((args["out_dir"] / "labels.jsonl").exists())

    def test_checksum_change_and_conflicting_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            (args["source_dir"] / "nq_train_annotation_000.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                audit_import(**args)
            write_new(args["audit_path"], b"frozen")
            with self.assertRaises(FileExistsError):
                write_new(args["audit_path"], b"changed")
            self.assertEqual(args["audit_path"].read_bytes(), b"frozen")


if __name__ == "__main__":
    unittest.main()
