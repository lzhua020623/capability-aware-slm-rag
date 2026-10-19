"""Freeze the last smoke8 audit-only revision from the verified r3 package."""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from scripts.prepare_answer_repair_v8_r2 import notebook
from scripts.run_answer_repair import prepare_records, read_jsonl
from src.compensation.conservative_repair import V8_VERSION
from src.diagnosis import DatasetName

OVERLAYS = (
    "src/compensation/conservative_repair.py", "src/compensation/line_contract.py",
    "scripts/check_answer_repair_v8_smoke.py", "scripts/audit_answer_repair_release.py",
    "scripts/prepare_answer_repair_v8_r4.py", "tests/test_answer_repair_v8.py",
    "tests/test_answer_repair_v8_r2.py", "tests/test_answer_repair_v8_r4.py",
    "tests/fixtures/answer_repair_v8r4_invariants.json",
    "configs/answer_repair_v8r4_smoke8_v1.json", "docs/ANSWER_REPAIR_V8_R4.md",
)
RUNTIME_CHANGES = {"src/compensation/conservative_repair.py", "src/compensation/line_contract.py"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def invariant_ast(raw):
    module = ast.parse(raw)
    digest = lambda n: sha(ast.dump(n, include_attributes=False).encode())
    return {
        "functions": {n.name: digest(n) for n in module.body if isinstance(n, ast.FunctionDef)
                      and n.name not in ("build_check_prompt", "parse_check")},
        "assignments": {t.id: digest(n.value) for n in module.body if isinstance(n, ast.Assign)
                        for t in n.targets if isinstance(t, ast.Name) and t.id != "V8_VERSION"},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-archive", type=Path, default=ROOT /
                        "artifacts/answer_repair/v8r3_smoke8/answer_repair_v8r3_smoke8_gpu.tar.gz")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/answer_repair/v8r4_smoke8")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/answer_repair_v8r4_smoke8.ipynb")
    args = parser.parse_args(argv)
    if V8_VERSION != "v8r4_fact_first_audit_v1":
        raise ValueError("Builder requires the frozen r4 source revision")
    protocol = json.loads((ROOT / "configs/answer_repair_v8r4_smoke8_v1.json").read_bytes())
    parent_raw = args.parent_archive.read_bytes()
    if sha(parent_raw) != protocol["parent_bundle_sha256"]:
        raise ValueError("Frozen r3 parent hash mismatch")
    prefix = "answer_repair_v8r3_smoke8_gpu/"
    with tarfile.open(fileobj=io.BytesIO(parent_raw)) as handle:
        checks = json.loads(handle.extractfile(prefix + "bundle_checksums.json").read())
        files = {p: handle.extractfile(prefix + p).read() for p in checks}
    if any(sha(raw) != checks[p] for p, raw in files.items()):
        raise ValueError("Frozen r3 payload integrity mismatch")
    parent = json.loads(files["manifest.json"])
    if parent["version"] != "answer_repair_v8r3_smoke8_v1":
        raise ValueError("Expected r3 smoke parent")
    if protocol["protocol"] != parent["protocol"] or protocol["review_numbers"] != parent["review_numbers"]:
        raise ValueError("Inputs, budgets and arms must not change")
    original_files = dict(files)
    files.update({p: (ROOT / p).read_bytes() for p in OVERLAYS})
    if invariant_ast(original_files["src/compensation/conservative_repair.py"]) != invariant_ast(
            files["src/compensation/conservative_repair.py"]):
        raise ValueError("A non-audit function or constant changed")
    if any(raw != files[p] for p, raw in original_files.items() if p.startswith("src/") and p not in RUNTIME_CHANGES):
        raise ValueError("Unexpected runtime source change")
    human = read_jsonl(files["offline/initial_reviews.jsonl"], "initial reviews")
    by_number = {r["review_number"]: r["sample_id"] for r in human}
    for case in protocol["semantic_regressions"]["cases"]:
        case["sample_id"] = by_number[case["review_number"]]
    scored = files[parent["scored_input"]]
    runtime = files[parent["runtime_input"]]
    prepared = prepare_records(read_jsonl(scored, "scored C1"), DatasetName.NQ)
    if read_jsonl(runtime, "runtime") != [json.loads(json.dumps(asdict(r))) for _, r in prepared]:
        raise ValueError("Runtime input is not the original C1 allowlist projection")
    source_hashes = {"scored_c1": sha(scored), "runtime": sha(runtime)}
    if source_hashes != parent["source_input_sha256"]:
        raise ValueError("Smoke inputs changed")
    manifest = {**parent, **protocol, "v8_prompt_version": V8_VERSION,
                "portable_tests": parent["portable_tests"] + ["tests.test_answer_repair_v8_r4"],
                "overlay_paths": list(OVERLAYS), "payload_policy": "Verified frozen r3 with audit-only overlays",
                "source_input_sha256": source_hashes, "gpu_inference_completed": False,
                "r3_non_audit_functions_and_constants_unchanged": True,
                "runtime_source_diff": {p: {"r3_sha256": sha(original_files[p]), "r4_sha256": sha(files[p])}
                                        for p in sorted(RUNTIME_CHANGES)}}
    files["manifest.json"] = encode_json(manifest)
    files["bundle_checksums.json"] = encode_json({p: sha(raw) for p, raw in sorted(files.items())})
    name = "answer_repair_v8r4_smoke8_gpu"
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as handle:
            for p, raw in sorted(files.items()):
                item = tarfile.TarInfo(name + "/" + p)
                item.size, item.mode = len(raw), 0o644
                handle.addfile(item, io.BytesIO(raw))
    archive_raw = buffer.getvalue()
    archive = args.out_dir / (name + ".tar.gz")
    book = json.loads(notebook(archive, files, archive_raw=archive_raw, revision="r4"))
    book["cells"][0]["source"] = [
        "# E v8r4：最后一轮 FACT 核验提示实验\n",
        "固定 8 条开发样本、四组 32 条输出、最多 64 次生成。只修改核验及其十行输出契约。\n",
        "保留 r3 工程门槛；只用 N31/N48/N12 事先固定的核验回归选版本，不看 EM。\n",
        "全部通过选 r4，否则按 r3 收口；不追加 r5，不自动启动 75 条或未见评测。\n",
    ]
    final = "".join(book["cells"][-1]["source"]).replace(
        "print('Failures:', report['failures'])",
        "print('Failures:', report['failures'])\nprint('Known cases:', report['known_case_regressions'])\n"
        "print('Selected final revision:', report['selected_final_revision'])")
    book["cells"][-1]["source"] = final.splitlines(keepends=True)
    for c in book["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<frozen-r4-notebook>", "exec")
    notebook_raw = encode_json(book)
    write_frozen(archive, archive_raw)
    write_frozen(args.notebook, notebook_raw)
    write_frozen(args.out_dir / "manifest.json", encode_json(manifest))
    preparation = {"version": manifest["version"], "bundle_sha256": sha(archive_raw),
                   "parent_bundle_sha256": sha(parent_raw), "notebook_sha256": sha(notebook_raw),
                   "checksums_sha256": sha(files["bundle_checksums.json"]), "files": len(files),
                   "samples_per_arm": 8, "predictions_expected": 32, "generation_call_upper_bound": 64,
                   "gpu_inference_completed": False}
    write_frozen(args.out_dir / "preparation.json", encode_json(preparation))
    print(json.dumps(preparation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
