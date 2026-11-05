"""Notebook source cells and notebook generators stay free of Chinese prose."""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u4e00-\u9fff]")
# Real dataset path names that must stay unchanged when they appear in source.
ALLOWED = ("NQ_15\u6279", "NQ_C1_3216_\u5408\u5e76\u5168\u91cf.xlsx")
GENERATORS = (
    "scripts/prepare_answer_repair_v8.py",
    "scripts/prepare_answer_repair_v8_r2.py",
    "scripts/prepare_answer_repair_v8_r3.py",
    "scripts/prepare_answer_repair_v8_r4.py",
    "scripts/prepare_answer_repair_final_dev.py",
    "scripts/package_answer_repair_challenge.py",
    "scripts/build_reviewed_answer_repair_notebook.py",
    "scripts/build_kaggle_answer_repair_notebook.py",
    "scripts/build_controller_fever_c1_bundle.py",
    "scripts/build_answer_repair_notebook.py",
    "scripts/build_answer_repair_v8_transport.py",
    "scripts/build_answer_repair_v9_notebook.py",
    "scripts/build_answer_repair_contract_notebook.py",
)


def residual(text):
    for item in ALLOWED:
        text = text.replace(item, "")
    return sorted(set(CJK.findall(text)))


class NotebookSourceLanguageTests(unittest.TestCase):
    def test_notebook_source_cells_have_no_chinese(self):
        leftovers = []
        paths = list((ROOT / "notebooks").rglob("*.ipynb")) + list((ROOT / "artifacts").rglob("*.ipynb"))
        self.assertGreater(len(paths), 0)
        for path in paths:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                source = cell.get("source", [])
                text = "".join(source) if isinstance(source, list) else source
                hits = residual(text)
                if hits:
                    leftovers.append(f"{path.relative_to(ROOT)} cell {index}: {''.join(hits)}")
        self.assertEqual(leftovers, [])

    def test_notebook_generators_have_no_chinese(self):
        leftovers = []
        for relative in GENERATORS:
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            hits = residual(path.read_text(encoding="utf-8"))
            if hits:
                leftovers.append(f"{relative}: {''.join(hits)}")
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
