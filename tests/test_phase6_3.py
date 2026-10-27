"""Compatibility and explicit provenance acceptance, without model imports."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
import zipfile
from unittest.mock import Mock, patch

from scripts.evaluate_nq_p320_generation import accepted_metadata
from scripts.run_nq_p320_gpu_generation import lazy_qwen_generator
from scripts.score_phase6_3 import extract_windows_zip, summary, wilson
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/provenance/nq_p320_win2_package_manifest.json"


class WindowsCompatibilityTests(unittest.TestCase):
    def test_missing_fcntl_uses_nonblocking_windows_lock_before_generation(self):
        code = '''
import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
sys.modules['fcntl'] = None
locking = Mock()
sys.modules['msvcrt'] = SimpleNamespace(locking=locking, LK_NBLCK=99)
from src.compensation.retrieval_enhancement.generation_replay import run_generation
with tempfile.TemporaryDirectory() as t:
    p = Path(t)/'inputs.jsonl'
    row = dict(sample_id='x', question='Where?', initial_answer='wrong',
               evidence=[dict(rank=i, passage_id=str(i), text='London') for i in range(1,6)])
    p.write_text(json.dumps(row)+'\\n')
    def generate(prompt):
        assert locking.call_count == 1
        assert locking.call_args.args[1:] == (99,1)
        return 'London'
    assert run_generation(p, Path(t)/'out', generate, run_metadata={})['completed'] == 1
    locking.side_effect = OSError('lock busy')
    try:
        run_generation(p, Path(t)/'out', lambda _: (_ for _ in ()).throw(AssertionError()),
                       run_metadata={}, resume=True)
    except OSError:
        pass
    else:
        raise AssertionError('lock contention must fail')
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def fake_modules(self, tokenizer_commit=None, model_commit="a09a354"):
        qwen = ModuleType("src.generation.qwen")
        qwen.load_qwen_nf4 = Mock(return_value=(
            SimpleNamespace(init_kwargs={"_commit_hash": tokenizer_commit}),
            SimpleNamespace(config=SimpleNamespace(_commit_hash=model_commit))))
        qwen.generate_greedy = Mock(return_value=("London", 0.01))
        return {"src.generation.qwen": qwen,
                "torch": SimpleNamespace(__version__="test", cuda=SimpleNamespace(get_device_name=lambda _: "fake")),
                "transformers": SimpleNamespace(__version__="5.2.0"),
                "bitsandbytes": SimpleNamespace(__version__="test")}

    def test_missing_tokenizer_commit_records_inferred_source_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as t, patch.dict(sys.modules, self.fake_modules()):
            self.assertEqual(lazy_qwen_generator(t)("prompt"), "London")
            receipt = json.loads((Path(t)/"model_receipt.json").read_text())["identity"]
            self.assertEqual(receipt["tokenizer_commit"], "a09a354")
            self.assertEqual(receipt["tokenizer_commit_source"], "inferred_from_model_same_repo")
            self.assertEqual(lazy_qwen_generator(t)("prompt"), "London")
        with tempfile.TemporaryDirectory() as t, patch.dict(sys.modules, self.fake_modules("other")):
            with self.assertRaisesRegex(RuntimeError, "matching"):
                lazy_qwen_generator(t)("prompt")
            self.assertFalse((Path(t)/"model_receipt.json").exists())

    def test_unresolved_model_commit_is_not_accepted(self):
        hub = SimpleNamespace(scan_cache_dir=lambda: SimpleNamespace(repos=[]))
        with tempfile.TemporaryDirectory() as t, patch.dict(sys.modules, {
            **self.fake_modules(model_commit=None), "huggingface_hub": hub
        }):
            with self.assertRaisesRegex(RuntimeError, "matching"):
                lazy_qwen_generator(t)("prompt")


class ProvenanceTests(unittest.TestCase):
    def test_explicit_manifest_maps_all_source_hashes_and_rejects_modified_manifest(self):
        expected, audit = accepted_metadata(MANIFEST)
        package = json.loads(MANIFEST.read_text())
        self.assertEqual(expected["source_sha256"],
                         {p: package["files"][p] for p in expected["source_sha256"]})
        self.assertEqual(audit["mode"], "explicit_win2_manifest")
        with tempfile.TemporaryDirectory() as t:
            bad = Path(t)/"manifest.json"
            bad.write_text(MANIFEST.read_text()+" ")
            with self.assertRaisesRegex(ValueError, "unapproved"):
                accepted_metadata(bad)


class ScoringTests(unittest.TestCase):
    def test_windows_paths_and_traversal_are_handled_before_extraction(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            archive = root/"results.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("gpu_results\\results.jsonl", "ok")
            extract_windows_zip(archive, root/"out")
            self.assertEqual((root/"out/gpu_results/results.jsonl").read_text(), "ok")
            for name in ("..\\escape", "C:\\escape", "/escape"):
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr(name, "bad")
                with self.assertRaises(ValueError):
                    extract_windows_zip(archive, root/"unsafe")
                self.assertFalse((root/"unsafe").exists())

    def test_paired_scoring_counts_guard_benefit_and_preserves_unchanged_denominator(self):
        def row(base, final, raw, status):
            return dict(base_em=base, base_f1=base, final_em=final, final_f1=final,
                        raw_em=raw, raw_f1=raw, status=status)
        rows = [row(1, 1, 0, "kept_original"), row(0, 1, 1, "answer_replaced"),
                row(0, 0, 1, "kept_original"), row(1, 1, 1, "no_change")]
        with patch("scripts.score_phase6_3.REPLICATES", 100):
            group, draws = summary(rows, np.random.default_rng(0))
        self.assertEqual(group["n"], 4)
        self.assertEqual(group["guarded"]["fixed"], 1)
        self.assertEqual(group["guarded"]["broken"], 0)
        self.assertEqual(group["unguarded"]["fixed"], 2)
        self.assertEqual(group["unguarded"]["broken"], 1)
        self.assertEqual(group["metrics"]["em_delta"]["estimate"], 0.25)
        self.assertEqual(group["rejected_raw_em_correct"], 1)
        self.assertEqual(draws.shape, (100, 4))
        self.assertIsNone(wilson(0, 0))
        self.assertGreater(wilson(0, 244)[1], 0)


if __name__ == "__main__":
    unittest.main()
