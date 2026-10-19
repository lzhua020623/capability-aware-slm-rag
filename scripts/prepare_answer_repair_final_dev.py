"""Package final NQ75 from the selected frozen r3/r4 inference source, without tuning."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from scripts.prepare_answer_repair_v8_r2 import notebook

ORIGINAL_SHA = "906f096f78e36a4f272cdd863ce2e7383d62acb638c1e480a06c5d3f75daa7a3"
SELECTED_SHA = {
    "r3": "72ed761fd8017ebbaf439729ff318caa55c0823acf4b62429c4db8f31310b6f1",
    "r4": "d13880c5e114b5f8379f931103af249e1f10b4975de98c9a5b74a9c161900e11",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def payload(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("Input archive hash mismatch: " + str(path))
    prefix = path.name.removesuffix(".tar.gz") + "/"
    with tarfile.open(fileobj=io.BytesIO(raw)) as handle:
        checks = json.loads(handle.extractfile(prefix + "bundle_checksums.json").read())
        files = {p: handle.extractfile(prefix + p).read() for p in checks}
    if any(sha(raw) != checks[p] for p, raw in files.items()):
        raise ValueError("Input payload hash mismatch")
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, default=ROOT / "artifacts/answer_repair/v8r4_smoke8")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/answer_repair/final_nq75")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/answer_repair_final_nq75.ipynb")
    args = parser.parse_args(argv)
    validation = json.loads((args.smoke_dir / "result_validation.json").read_bytes())
    result_directory = validation.get("result_directory", "kaggle_20261019")
    if Path(result_directory).name != result_directory or not result_directory.startswith("kaggle_"):
        raise ValueError("Invalid raw result directory in validation receipt")
    report_raw = (args.smoke_dir / result_directory / "smoke_acceptance.json").read_bytes()
    report = json.loads(report_raw)
    if (validation.get("replayed_records") != 32 or
            not validation.get("frozen_engineering_gate_recalculation_matches") or
            validation["raw_file_sha256"]["smoke_acceptance.json"] != sha(report_raw)):
        raise ValueError("Final selection requires verified frozen CPU replay and the original smoke report")
    revision = report.get("selected_final_revision")
    if revision not in SELECTED_SHA:
        raise ValueError("No predeclared final revision selected")
    if (revision == "r4") != bool(report["ready_for_full_nq75_run"]):
        raise ValueError("Selection does not follow the frozen r4/fallback rule")
    if revision == "r3":
        baseline = json.loads((ROOT / "artifacts/answer_repair/v8r3_smoke8/result_validation.json").read_bytes())
        if not baseline["engineering_gate_passed"]:
            raise ValueError("Fallback r3 baseline was not verified")
    source_archive = ROOT / f"artifacts/answer_repair/v8{revision}_smoke8/answer_repair_v8{revision}_smoke8_gpu.tar.gz"
    files = payload(source_archive, SELECTED_SHA[revision])
    selected = json.loads(files["manifest.json"])
    original = payload(ROOT / "artifacts/answer_repair/v8_nq75/answer_repair_v8_nq75_gpu.tar.gz", ORIGINAL_SHA)
    initial = json.loads(original["manifest.json"])
    if selected["protocol"] != initial["protocol"]:
        raise ValueError("Final arms or budgets changed")
    original_runtime = {p: raw for p, raw in files.items() if p.startswith("src/")}
    for p in (selected["scored_input"], selected["runtime_input"]):
        files.pop(p)
    for p in ("inputs/nq75_c1.jsonl", "inputs/nq75_runtime.jsonl", "offline/initial_reviews.jsonl"):
        files[p] = original[p]
    human = [json.loads(s) for s in files["offline/initial_reviews.jsonl"].splitlines()]
    groups = {g: sum(x["group"] == g for x in human) for g in ("include_e_core", "correct_control")}
    if groups != {"include_e_core": 11, "correct_control": 64}:
        raise ValueError("Final development cohort changed")
    version = f"answer_repair_v8{revision}_nq75_final_v1"
    adapter_source = (ROOT / "scripts/run_answer_repair_portable_tests.py").read_text()
    adapter_sha = sha(adapter_source.encode())
    manifest = {k: v for k, v in selected.items() if k not in
                ("smoke_acceptance", "semantic_regressions", "failure_policy", "review_numbers", "overlay_paths")}
    manifest.update(version=version, purpose="Final development measurement; no post-run tuning or gain claim",
                    selected_final_revision=revision, final_inference_bundle_sha256=SELECTED_SHA[revision],
                    final_selection_report_sha256=sha(report_raw), final_selection=report,
                    sample_ids=initial["sample_ids"], samples=75, groups=groups,
                    review_numbers=[x["review_number"] for x in human], expected_predictions=300,
                    max_generation_calls=600, scored_input="inputs/nq75_c1.jsonl",
                    runtime_input="inputs/nq75_runtime.jsonl", source_input_sha256=initial["source_input_sha256"],
                    initial_human_review_sha256=sha(files["offline/initial_reviews.jsonl"]),
                    source_row_sha256={json.loads(line)["sample_id"]: sha(line) for line in
                                       files["inputs/nq75_c1.jsonl"].splitlines(keepends=True)},
                    frozen_final=True, tuning_allowed_after_run=False, gpu_inference_completed=False,
                    preflight_adapter_sha256=adapter_sha,
                    preflight_adapter_scope="External unittest process only; no frozen source or inference patch",
                    acceptance_policy="Report all development outputs, EM/F1 and cost; do not select another version")
    auditor = "scripts/audit_answer_repair_release.py"
    before = files[auditor]
    files[auditor] = before.replace(b"'answer_repair_v8_nq75_v1',", b"'answer_repair_v8_nq75_v1', "
        b"'answer_repair_v8r3_nq75_final_v1', 'answer_repair_v8r4_nq75_final_v1',", 1)
    if before == files[auditor]:
        raise ValueError("Cannot adapt evaluation manifest recognition")
    files["scripts/prepare_answer_repair_final_dev.py"] = Path(__file__).read_bytes()
    manifest["evaluation_only_overlay"] = {auditor: {"before": sha(before), "after": sha(files[auditor])}}
    manifest["runtime_source_sha256"] = {p: sha(raw) for p, raw in original_runtime.items()}
    if any(files[p] != raw for p, raw in original_runtime.items()):
        raise ValueError("Selected inference code changed")
    files["manifest.json"] = encode_json(manifest)
    files["bundle_checksums.json"] = encode_json({p: sha(raw) for p, raw in sorted(files.items())})
    name = f"answer_repair_final_{revision}_nq75_gpu"
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as handle:
            for p, raw in sorted(files.items()):
                info = tarfile.TarInfo(name + "/" + p)
                info.size, info.mode = len(raw), 0o644
                handle.addfile(info, io.BytesIO(raw))
    archive_raw = out.getvalue()
    archive = args.out_dir / (name + ".tar.gz")
    book = json.loads(notebook(archive, files, archive_raw=archive_raw, revision=revision))
    book["cells"][0]["source"] = ["# E 最终 NQ75 开发实验\n",
        f"最终版本 {revision} 已按预设 smoke 规则选定。C1 复用，四组各 75 条，共 300 条输出，上限 600 次生成。\n",
        "模型、证据、评分规则和推理代码保持冻结；运行完不再调参，本批不是未见收益证据。\n"]
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        test_command = "run_logged([sys.executable, '-m', 'unittest', *manifest['portable_tests']], 'tests.log')"
        if test_command in text:
            adapted = (f"adapter_source = {adapter_source!r}\n"
                       "adapter_path = OUTPUT / 'preflight_ast_compat.py'\n"
                       "adapter_path.write_text(adapter_source)\n"
                       f"assert hashlib.sha256(adapter_path.read_bytes()).hexdigest() == {adapter_sha!r}\n"
                       "(OUTPUT / 'preflight_adapter.json').write_text(json.dumps({\n"
                       f"    'adapter_sha256': {adapter_sha!r}, 'frozen_bundle_sha256': EXPECTED_SHA256,\n"
                       "    'scope': 'unittest process only; no inference changes'\n"
                       "}, indent=2) + '\\n')\n"
                       "run_logged([sys.executable, str(adapter_path), *manifest['portable_tests']], 'tests.log')")
            text = text.replace(test_command, adapted)
        text = text.replace(f"answer_repair_v8{revision}_smoke8_v1", version).replace("manifest['samples'] == 8", "manifest['samples'] == 75")
        if "scripts/check_answer_repair_v8_smoke.py" in text:
            text = text.replace("scripts/check_answer_repair_v8_smoke.py", "scripts/summarize_answer_repair_v8.py")
            text = text.replace("(OUTPUT / 'smoke_acceptance.json')", "(OUTPUT / 'comparison.json')")
            text = text.replace("print('Engineering gate passed:', report['engineering_gate_passed'])", "print('All arms complete:', report['all_arms_complete'])")
            text = text.replace("print('Failures:', report['failures'])", f"print('Final version remains {revision}; no post-run selection.')")
            text = text.replace("Semantic acceptance: pending human review. No full run launched.",
                                "Human output review pending; inference version remains frozen.")
        compile(text, "<final75-notebook>", "exec")
        cell["source"] = text.splitlines(keepends=True)
    notebook_raw = encode_json(book)
    write_frozen(archive, archive_raw)
    write_frozen(args.notebook, notebook_raw)
    write_frozen(args.out_dir / "manifest.json", encode_json(manifest))
    receipt = {"version": version, "selected_final_revision": revision, "bundle_sha256": sha(archive_raw),
               "notebook_sha256": sha(notebook_raw), "checksums_sha256": sha(files["bundle_checksums.json"]),
               "frozen_inference_source_unchanged": True, "predictions_expected": 300,
               "generation_call_upper_bound": 600, "gpu_inference_completed": False}
    write_frozen(args.out_dir / "preparation.json", encode_json(receipt))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
