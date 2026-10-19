"""Package the already frozen E-only challenge using selected final inference bytes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_answer_repair_experiment import encode_json, write_frozen
from scripts.prepare_answer_repair_final_dev import payload
from scripts.prepare_answer_repair_unseen import frozen_selection
from scripts.prepare_answer_repair_v8_r2 import notebook


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final-dir", type=Path, default=ROOT / "artifacts/answer_repair/final_nq75")
    p.add_argument("--challenge", type=Path, default=ROOT / "artifacts/answer_repair/unseen_nq125")
    p.add_argument("--notebook", type=Path, default=ROOT / "notebooks/answer_repair_final_nq125.ipynb")
    a = p.parse_args(argv)
    prep = json.loads((a.final_dir / "preparation.json").read_bytes()); revision = prep["selected_final_revision"]
    archive = a.final_dir / f"answer_repair_final_{revision}_nq75_gpu.tar.gz"
    selected = frozen_selection(a.final_dir / "preparation.json", archive)
    files = payload(archive, prep["bundle_sha256"]); parent = json.loads(files["manifest.json"])
    challenge_raw = (a.challenge / "manifest.json").read_bytes(); challenge = json.loads(challenge_raw)
    challenge_prep = json.loads((a.challenge / "preparation.json").read_bytes())
    if (sha(challenge_raw) != challenge_prep["manifest_sha256"] or challenge["samples"] != 125
            or challenge["final_selection"] != selected or challenge["groups"] != {"screen_candidate": 100, "em_match_control": 25}
            or not challenge["calibration_ids_consumed"] or challenge["post_result_tuning_allowed"]):
        raise ValueError("Challenge is not bound to this frozen final source")
    source_files = {name: (a.challenge / name).read_bytes() for name in challenge["outputs_sha256"]}
    if any(sha(source_files[name]) != digest for name, digest in challenge["outputs_sha256"].items()):
        raise ValueError("Challenge input fingerprint changed")
    for name in (parent["scored_input"], parent["runtime_input"], "offline/initial_reviews.jsonl"):
        files.pop(name)
    files.update(source_files)
    version = f"answer_repair_v8{revision}_nq125_challenge_v1"
    manifest = {k: v for k, v in parent.items() if k not in ("review_numbers", "initial_human_review_sha256", "final_selection")}
    manifest.update(version=version, purpose="One-time E-only stratified challenge; no post-result tuning",
        scope=challenge["scope"], team_registration=challenge["team_registration"],
        samples=125, sample_ids=challenge["sample_ids"], groups=challenge["groups"],
        protocol=[x for x in parent["protocol"] if x["strategy"] in ("short_answer_quote", "conservative_repair")],
        expected_predictions=250, max_generation_calls=500, scored_input="inputs/nq125_c1.jsonl",
        runtime_input="inputs/nq125_runtime.jsonl", challenge_manifest_sha256=sha(challenge_raw),
        source_input_sha256={"scored_c1": sha(source_files["inputs/nq125_c1.jsonl"]),
                             "runtime": sha(source_files["inputs/nq125_runtime.jsonl"])},
        source_row_sha256={json.loads(line)["sample_id"]: sha(line) for line in
                           source_files["inputs/nq125_c1.jsonl"].splitlines(keepends=True)},
        offline_strata_sha256=sha(source_files["offline/strata.jsonl"]),
        final_dev_bundle_sha256=prep["bundle_sha256"], gpu_inference_completed=False,
        human_initial_labels_available=False, tuning_allowed_after_run=False,
        acceptance_policy="Report frozen metrics, paired changed-output review and cost; do not select another version")
    if len(manifest["protocol"]) != 2:
        raise ValueError("Expected the original v7 and complete-v8 configurations")
    audit = "scripts/audit_answer_repair_release.py"; before = files[audit]
    files[audit] = before.replace(b"'answer_repair_v8_nq75_v1',", ("'answer_repair_v8_nq75_v1', '" + version + "',").encode(), 1)
    old_cohort = b"v8_cohort = 'nq75' if manifest.get('samples') == 75 else 'nq_smoke8'"
    if old_cohort not in files[audit] or files[audit] == before:
        raise ValueError("Cannot adapt frozen evaluation cohort recognition")
    files[audit] = files[audit].replace(old_cohort, b"v8_cohort = 'nq125' if manifest.get('samples') == 125 else ('nq75' if manifest.get('samples') == 75 else 'nq_smoke8')", 1)
    report_path = "scripts/summarize_answer_repair_challenge.py"
    files[report_path] = (ROOT / report_path).read_bytes()
    files["scripts/package_answer_repair_challenge.py"] = Path(__file__).read_bytes()
    manifest["evaluation_only_overlay"] = {audit: {"before": sha(before), "after": sha(files[audit])},
                                           report_path: {"after": sha(files[report_path])}}
    if any(sha(files[name]) != digest for name, digest in manifest["runtime_source_sha256"].items()):
        raise ValueError("Selected inference source changed")
    files["manifest.json"] = encode_json(manifest)
    files["bundle_checksums.json"] = encode_json({name: sha(raw) for name, raw in sorted(files.items())})
    name = f"answer_repair_final_{revision}_nq125_gpu"; buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as z:
            for path, raw in sorted(files.items()):
                info = tarfile.TarInfo(name + "/" + path); info.size, info.mode = len(raw), 0o644
                z.addfile(info, io.BytesIO(raw))
    archive_raw = buffer.getvalue(); output = a.challenge / (name + ".tar.gz")
    book = json.loads(notebook(output, files, archive_raw=archive_raw, revision=revision))
    book["cells"][0]["source"] = ["# E 一次性 NQ125 分层挑战\n",
        f"最终版本 {revision} 已冻结。100 筛查候选与 25 EM 对照，C1 复用，v7／完整 v8 各 125 条。\n",
        "无人工初始诊断标签；属于 E 组件分层评测，团队范围隔离待协调。结果返回后不改策略或样本。\n"]
    adapter = (ROOT / "scripts/run_answer_repair_portable_tests.py").read_text()
    if sha(adapter.encode()) != manifest["preflight_adapter_sha256"]:
        raise ValueError("Preflight adapter changed since final development freeze")
    for c in book["cells"]:
        if c["cell_type"] != "code":
            continue
        text = "".join(c["source"]).replace(f"answer_repair_v8{revision}_smoke8_v1", version)
        text = text.replace("manifest['samples'] == 8", "manifest['samples'] == 125")
        text = text.replace("manifest['review_numbers']", "manifest['samples']")
        test_cmd = "run_logged([sys.executable, '-m', 'unittest', *manifest['portable_tests']], 'tests.log')"
        if test_cmd in text:
            code = (f"adapter_source = {adapter!r}\n"
                    "adapter_path = OUTPUT / 'preflight_ast_compat.py'\nadapter_path.write_text(adapter_source)\n"
                    f"assert hashlib.sha256(adapter_path.read_bytes()).hexdigest() == {manifest['preflight_adapter_sha256']!r}\n"
                    "run_logged([sys.executable, str(adapter_path), *manifest['portable_tests']], 'tests.log')")
            text = text.replace(test_cmd, code)
        text = text.replace("scripts/check_answer_repair_v8_smoke.py", report_path)
        text = text.replace("OUTPUT / 'smoke_acceptance.json'", "OUTPUT / 'comparison.json'")
        text = text.replace("print('Engineering gate passed:', report['engineering_gate_passed'])", "print('All arms complete:', report['all_arms_complete'])")
        text = text.replace("print('Failures:', report['failures'])", "print('Human review pending; no post-run version selection.')")
        text = text.replace("Semantic acceptance: pending human review. No full run launched.", "E-only challenge complete; changed-output human review pending.")
        compile(text, "<challenge-notebook>", "exec"); c["source"] = text.splitlines(keepends=True)
    notebook_raw = encode_json(book)
    write_frozen(output, archive_raw); write_frozen(a.notebook, notebook_raw)
    write_frozen(a.challenge / "gpu_manifest.json", encode_json(manifest))
    receipt = {"version": version, "bundle_sha256": sha(archive_raw), "notebook_sha256": sha(notebook_raw),
               "checksums_sha256": sha(files["bundle_checksums.json"]), "selected_final_revision": revision,
               "runtime_source_unchanged_from_final_development": True, "expected_predictions": 250,
               "max_generation_calls": 500, "gpu_inference_completed": False}
    write_frozen(a.challenge / "gpu_preparation.json", encode_json(receipt))
    print(json.dumps(receipt, indent=2)); return 0


if __name__ == "__main__":
    main()
