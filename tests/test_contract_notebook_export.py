"""Regression for the real Kaggle failure after successful model inference."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.build_answer_repair_contract_notebook import ROOT, fix_export_timestamps


class ExportTimestampTests(unittest.TestCase):
    def test_actual_frozen_finally_block_exports_epoch_files_without_touching_bytes(self):
        notebook = json.loads((ROOT / "notebooks/answer_repair_contract_v1.ipynb").read_bytes())
        original = "".join(notebook["cells"][1]["source"])
        fixed = fix_export_timestamps(original)
        # No inference command, input archive, prompt or model setting changes.
        self.assertEqual(fixed.replace(",strict_timestamps=False", ""), original)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"; root.mkdir()
            frozen_file = root / "epoch_file.json"
            frozen_file.write_bytes(b'{"same":"frozen bytes"}\n')
            os.utime(frozen_file, (0, 0))
            cache = root / "__pycache__"; cache.mkdir()
            (cache / "skip.pyc").write_bytes(b"not part of export")
            for code, should_fail in ((original, True), (fixed, False)):
                output = Path(temp) / ("bad.zip" if should_fail else "good.zip")
                tree = ast.parse(code)
                final = next(n.finalbody for n in tree.body if isinstance(n, ast.Try))
                module = ast.Module(body=final, type_ignores=[])

                class TemporaryOutput(ast.NodeTransformer):
                    def visit_Constant(self, node):
                        if node.value == "/kaggle/working/e_contract_v1_outputs.zip":
                            return ast.copy_location(ast.Constant(str(output)), node)
                        return node

                module = ast.fix_missing_locations(TemporaryOutput().visit(module))
                execute = lambda: exec(compile(module, "frozen_export_finally", "exec"),
                                       {"root": root, "zipfile": zipfile})
                if should_fail:
                    with self.assertRaisesRegex(ValueError, "timestamps before 1980"):
                        execute()
                else:
                    execute()
                    with zipfile.ZipFile(output) as archive:
                        self.assertEqual(archive.namelist(), ["epoch_file.json"])
                        self.assertEqual(archive.read("epoch_file.json"), frozen_file.read_bytes())
                        self.assertEqual(archive.getinfo("epoch_file.json").date_time[0], 1980)


if __name__ == "__main__":
    unittest.main()
