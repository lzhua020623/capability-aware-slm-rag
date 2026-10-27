"""Reproducible offline scoring of the reviewed Windows P3-20 GPU archive."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evaluate_nq_p320_generation import LABELS, evaluate_generation
from src.compensation.retrieval_enhancement.generation_replay import sha256_file

REVISION = "338ff34b039ed367220b0b916b48c4f7fc715e20"
SEED, REPLICATES = 6302026, 10000


def extract_windows_zip(archive, target):
    """Normalize separators while rejecting traversal and duplicate destinations."""
    target = Path(target)
    with zipfile.ZipFile(archive) as z:
        names = set()
        members = []
        for item in z.infolist():
            path = PurePosixPath(item.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or ":" in str(path) or str(path) in names:
                raise ValueError("unsafe or duplicate ZIP member")
            names.add(str(path))
            members.append((item, path))
        target.mkdir(parents=True, exist_ok=False)
        for item, path in members:
            out = target / path
            if item.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(item))


def routing_weights(repo):
    sources = {}
    def read(path):
        raw = subprocess.check_output(["git", "show", REVISION + ":" + path], cwd=repo)
        sources[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    controller = read("results/controller/controller_v1_validation.json")
    diagnosis = read("results/diagnosis/diagnosis_v2_expanded_validation.json")
    probe = read("results/probe/probe_v2_expanded_validation.json")
    predictions = diagnosis["predictions"]
    labels = {r["sample_id"]: r["target_label"] for r in predictions}
    samples = {r["sample_id"]: r for r in probe["samples"]}
    counts = Counter()
    nq_count = eligible = 0
    for row in controller["decisions"]:
        if row["dataset"] != "nq":
            continue
        nq_count += 1
        offline = samples[row["sample_id"]]["offline_evaluation"]
        if offline["target_label"] != labels[row["sample_id"]]:
            raise ValueError("validation labels disagree")
        if offline["prompt_development_sample"]:
            continue
        eligible += 1
        if row["action"] == "retrieval_enhancement":
            counts[offline["target_label"]] += 1
    if (nq_count, eligible, [counts[l] for l in LABELS]) != (406, 405, [19, 23, 0, 7]):
        raise ValueError("Phase 5-7 routing population differs")
    return {"revision": REVISION, "source_sha256": sources, "validation_nq": nq_count,
            "after_prompt_development_exclusion": eligible, "routed": sum(counts.values()),
            "counts": {l: counts[l] for l in LABELS},
            "weights": {l: counts[l]/sum(counts.values()) for l in LABELS},
            "note": "NQ validation final Controller actions only; no test files read"}


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p, d = k/n, 1+z*z/n
    mid = (p+z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0., mid-half), min(1., mid+half)]


def summary(rows, rng):
    columns = ("base_em", "final_em", "raw_em", "base_f1", "final_f1", "raw_f1")
    values = np.array([[r[c] for c in columns] for r in rows], dtype=float)
    values = np.column_stack((values, values[:, 1]-values[:, 0], values[:, 2]-values[:, 0],
                             values[:, 4]-values[:, 3], values[:, 5]-values[:, 3]))
    names = columns + ("em_delta", "unguarded_em_delta", "f1_delta", "unguarded_f1_delta")
    draws = np.empty((REPLICATES, len(names)))
    for start in range(0, REPLICATES, 100):
        idx = rng.integers(0, len(rows), size=(min(100, REPLICATES-start), len(rows)))
        draws[start:start+len(idx)] = values[idx].mean(axis=1)
    ci = np.quantile(draws, [0.025, 0.975], axis=0)
    result = {"n": len(rows), "metrics": {name: {"estimate": float(values[:, i].mean()),
        "ci95": wilson(int(values[:, i].sum()), len(rows)) if name.endswith("_em") else ci[:, i].tolist()}
        for i, name in enumerate(names)}}
    for prefix, index in (("guarded", 1), ("unguarded", 2)):
        fixed = int(((values[:, 0] == 0) & (values[:, index] == 1)).sum())
        broken = int(((values[:, 0] == 1) & (values[:, index] == 0)).sum())
        correct = int(values[:, 0].sum())
        result[prefix] = {"fixed": fixed, "broken": broken, "net": fixed-broken,
                          "baseline_em_correct": correct,
                          "harm_rate": broken/correct if correct else None,
                          "harm_rate_ci95": wilson(broken, correct)}
    rejected = [r for r in rows if r["status"] == "kept_original"]
    result["guard_rejected"] = len(rejected)
    result["rejected_raw_em_correct"] = sum(r["raw_em"] == 1 for r in rejected)
    return result, draws[:, 6:]


def pct(value):
    return f"{100*value:.2f}"


def report_markdown(report):
    lines = ["# Phase 6-3 official GPU scoring", "",
             f"All 2,381 rows are scored; 2,313 ordered terminal results passed journal checksums and frozen guard validation. The 68 unchanged rows retain their original answers in both guarded and unguarded scoring. Generation failures: {report['generation_errors']}.", "",
             "EM means SQuAD/NQ normalized exact match against the best reference alias; F1 means token overlap F1 against the best alias. Label=correct is an AI Diagnosis stratum, not EM correctness.", "",
             "## Before / after", "",
             "Values are percentages; deltas are percentage points. Brackets are 95% CIs. EM levels use Wilson; F1 levels and paired deltas use 10,000 paired row bootstrap replicates (seed 6302026). All rows, including unchanged rows, enter the bootstrap.", "",
             "| Label | N | EM before [CI] | EM after [CI] | Delta EM [CI] | F1 before [CI] | F1 after [CI] | Delta F1 [CI] |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, group in report["groups"].items():
        cells = []
        for key in ("base_em", "final_em", "em_delta", "base_f1", "final_f1", "f1_delta"):
            m = group["metrics"][key]
            cells.append(f"{pct(m['estimate'])} [{pct(m['ci95'][0])}, {pct(m['ci95'][1])}]")
        lines.append(f"| {label} | {group['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## Guard and harm", "",
              "Fixed: baseline EM=0 to EM=1. Broken: baseline EM=1 to EM=0. Harm uses the originally EM-correct rows as denominator. Unguarded uses raw generations, with unchanged/error rows retaining baseline answers.", "",
              "| Label | Guarded fixed / broken | Unguarded fixed / broken | Guard rejected | Rejected raw EM=1 | Guarded harm [CI] |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, g in report["groups"].items():
        a, b = g["guarded"], g["unguarded"]
        harm = "N/A (0 baseline EM-correct)" if a["harm_rate"] is None else f"{a['broken']}/{a['baseline_em_correct']} = {pct(a['harm_rate'])}% [{pct(a['harm_rate_ci95'][0])}, {pct(a['harm_rate_ci95'][1])}]"
        lines.append(f"| {label} | {a['fixed']} / {a['broken']} | {b['fixed']} / {b['broken']} | {g['guard_rejected']} | {g['rejected_raw_em_correct']} | {harm} |")
    g = report["groups"]["overall"]
    lines += ["", f"Guard rejection: {g['guard_rejected']}/2313 = {pct(g['guard_rejected']/2313)}%; raw EM=1 among rejected: {g['rejected_raw_em_correct']}/{g['guard_rejected']} = {pct(g['rejected_raw_em_correct']/g['guard_rejected'])}%.",
              f"Guard prevents {g['unguarded']['broken']-g['guarded']['broken']} EM breaks and forgoes {g['unguarded']['fixed']-g['guarded']['fixed']} EM fixes; net EM benefit of guard is {g['guarded']['net']-g['unguarded']['net']} rows.", "",
              "| Policy | EM after [CI] | F1 after [CI] | Delta EM [CI] | Delta F1 [CI] | Harm [CI] |", "|---|---:|---:|---:|---:|---:|"]
    for prefix, keys in (("guarded", ("final_em", "final_f1", "em_delta", "f1_delta")), ("unguarded", ("raw_em", "raw_f1", "unguarded_em_delta", "unguarded_f1_delta"))):
        cells = [f"{pct(g['metrics'][k]['estimate'])} [{pct(g['metrics'][k]['ci95'][0])}, {pct(g['metrics'][k]['ci95'][1])}]" for k in keys]
        a = g[prefix]
        lines.append(f"| {prefix} | " + " | ".join(cells) + f" | {pct(a['harm_rate'])}% [{pct(a['harm_rate_ci95'][0])}, {pct(a['harm_rate_ci95'][1])}] |")
    lines += ["", "## Validation routing weighted estimate", "",
              f"Recomputed from commit `{REVISION}`: NQ validation 406, excluding 1 Prompt Development row gives 405; final Controller action routes 49 to retrieval_enhancement. Label counts: correct 19, insufficiency 23, accessibility 0, utilisation 7. Weights are 19/49, 23/49, 0/49, 7/49. Source paths and SHA256 are in routing_weights.json. No test records are read.", "",
              "Weighted delta = sum(label weight * label mean paired delta). CIs resample within each label and hold validation weights fixed; they do not include uncertainty in the 49-row routing composition.", "",
              "| Policy | Weighted delta EM [CI] | Weighted delta F1 [CI] |", "|---|---:|---:|"]
    for prefix, keys in (("guarded", ("em_delta", "f1_delta")), ("unguarded", ("unguarded_em_delta", "unguarded_f1_delta"))):
        cells = [f"{pct(report['weighted'][k]['estimate'])} [{pct(report['weighted'][k]['ci95'][0])}, {pct(report['weighted'][k]['ci95'][1])}]" for k in keys]
        lines.append(f"| {prefix} | " + " | ".join(cells) + " |")
    latency = report["latency_seconds"]
    lines += ["", "## Generation latency", "", f"All 2,313 calls: median {latency['median']:.6f} s, mean {latency['mean']:.6f} s; total {latency['sum']:.3f} s. First lazy call includes model load; receipt model load {report['model_receipt']['model_load_seconds']:.6f} s. Excluding the first call: median {latency['excluding_first']['median']:.6f} s, mean {latency['excluding_first']['mean']:.6f} s. These are worker generation-call wall times, not pure GPU kernel timings.", "",
              "## Provenance and source hash acceptance", "",
              "Every win2 package file was verified against its package_manifest. The evaluator retains strict hash comparison. The explicit --package-manifest option accepts only the reviewed manifest with SHA256 `3c72eb6d5ba62714711b18ae71c7a1c78943f5f18565ab6e1342005a55d79f7d`; source keys and all non-source run settings must still match. Acceptance is logged in metrics.json. The Windows lock and commit-receipt fixes change source bytes without changing prompts, generation, or the answer guard. Both fixes were copied byte-for-byte from win2. Archive and reference hashes, model identity, sessions, and source file hashes are preserved in audit.json and raw/.", "",
              "## Reference comparison", "",
              "Comparison to supplied quick computation (after rounding): " + json.dumps(report["quick_computation_matches"]), "",
              "## Limitations", "",
              "C1 reproduction has not been run yet under transformers 5.2.0 and Qwen commit a09a35458c702b33eeacc393d103063234e8bc28. Gains on failure-label rows may partly come from regeneration noise. Historical C1 was not revision pinned and initial answers are annotation-derived; this comparison does not isolate a causal retrieval effect. The tokenizer commit was inferred from the model's same-repository snapshot, not independently observed. Hardware: NVIDIA RTX 4000 SFF Ada Generation; torch 2.6.0+cu124; bitsandbytes 0.50.2.", "",
              "The validation-weighted result assumes label-conditional effects transport to actual routed queries; it is not a measured integrated Controller gain. Labels are AI initial annotations, not independently verified human labels. CIs assume exchangeable rows, do not model question clusters, historical model drift or routing-weight uncertainty, and are descriptive intervals on a fixed development cohort. This work ran offline scoring and fake-model tests only, with no new generation or C1 reproduction.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-zip", required=True, type=Path)
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--routing-repo", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    package_path = args.bundle_dir / "package_manifest.json"
    package = json.loads(package_path.read_text())
    verified = {}
    for name, digest in package["files"].items():
        path = (args.bundle_dir/name).resolve()
        if not path.is_relative_to(args.bundle_dir.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"win2 file hash mismatch: {name}")
        verified[name] = digest
    if (package["n"], package["to_generate"], package["unchanged"]) != (2381, 2313, 68):
        raise ValueError("unexpected package population")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    extract_windows_zip(args.results_zip, args.output_dir / "raw")
    gpu = args.output_dir / "raw/gpu_results"
    basic = evaluate_generation(args.reference, args.bundle_dir/"generation_inputs.jsonl", gpu,
                                args.output_dir/"scoring", package_manifest=package_path)
    rows = [json.loads(line) for line in (args.output_dir/"scoring/per_sample.jsonl").read_text().splitlines()]
    routing = routing_weights(args.routing_repo)
    rng = np.random.default_rng(SEED)
    groups, draws = {}, {}
    for label in (*LABELS, "overall"):
        group = rows if label == "overall" else [r for r in rows if r["label"] == label]
        groups[label], draws[label] = summary(group, rng)
    weighted_draws = sum(routing["weights"][l]*draws[l] for l in LABELS)
    weighted = {}
    for i, key in enumerate(("em_delta", "unguarded_em_delta", "f1_delta", "unguarded_f1_delta")):
        weighted[key] = {"estimate": sum(routing["weights"][l]*groups[l]["metrics"][key]["estimate"] for l in LABELS),
                         "ci95": np.quantile(weighted_draws[:, i], [0.025, 0.975]).tolist()}
    times = np.array([r["latency"]["generation"] for r in rows if r["latency"] is not None])
    if len(times) != 2313 or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("invalid latency records")
    receipt = json.loads((gpu/"model_receipt.json").read_text())
    report = {"groups": groups, "weighted": weighted, "generation_errors": basic["overall"]["generation_errors"],
              "latency_seconds": {"n": len(times), "median": float(np.median(times)), "mean": float(times.mean()),
                 "sum": float(times.sum()), "excluding_first": {"median": float(np.median(times[1:])), "mean": float(times[1:].mean())}},
              "model_receipt": receipt, "bootstrap": {"seed": SEED, "replicates": REPLICATES},
              "routing": routing, "source_verification": basic["source_verification"]}
    report["quick_computation_matches"] = {
        "insufficiency_em_0_to_5.74": pct(groups["evidence_insufficiency"]["metrics"]["base_em"]["estimate"]) == "0.00" and pct(groups["evidence_insufficiency"]["metrics"]["final_em"]["estimate"]) == "5.74",
        "overall_em_35.99_to_37.46": pct(groups["overall"]["metrics"]["base_em"]["estimate"]) == "35.99" and pct(groups["overall"]["metrics"]["final_em"]["estimate"]) == "37.46",
        "broken_73": groups["overall"]["guarded"]["broken"] == 73,
        "weighted_delta_em_3.90": pct(weighted["em_delta"]["estimate"]) == "3.90"}
    audit = {"zip_sha256": sha256_file(args.results_zip), "reference_sha256": sha256_file(args.reference),
             "verified_package_files": verified, "source_verification": basic["source_verification"],
             "complete_ordered_guard_verified": 2313, "unchanged": 68, "generation_errors": basic["overall"]["generation_errors"],
             "raw_file_sha256": {str(f.relative_to(gpu)): sha256_file(f) for f in gpu.iterdir() if f.is_file()}}
    for filename, value in (("report.json", report), ("audit.json", audit), ("routing_weights.json", routing)):
        (args.output_dir/filename).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    shutil.copyfile(package_path, args.output_dir/"win2_package_manifest.json")
    (args.output_dir/"report.md").write_text(report_markdown(report))
    print(json.dumps({"overall": groups["overall"], "weighted": weighted, "latency": report["latency_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
