"""Build a Gold-free GPU bundle and separate Mac-only evaluation references."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.retrieval_enhancement.generation_replay import (
    load_generation_inputs, read_results, sha256_file,
)
from src.compensation.retrieval_enhancement.p3 import RERANKER_REVISION, select_p3_20
from src.evaluation.qa import gold_answers_from_record


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).strip()


def _json(path):
    return json.loads(Path(path).read_text())


def _lines(path, rows):
    with Path(path).open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def runtime_code_paths():
    code = ["scripts/run_nq_p320_gpu_generation.py", "src/__init__.py", "src/config.py",
            "src/compensation/__init__.py", "src/evaluation/__init__.py", "src/evaluation/qa.py",
            "src/evaluation/recoverability.py", "src/generation/__init__.py",
            "src/generation/qwen.py", "src/generation/prompts.py", "configs/base.yaml"]
    return code + [str(p.relative_to(ROOT)) for p in sorted(
        (ROOT / "src/compensation/retrieval_enhancement").glob("*.py"))]


def prepare_bundle(phase5_dir, split_path, output_dir, *, diagnosis_ref="origin/reranking"):
    prior = Path(phase5_dir)
    config = _json(prior / "run_config.json")
    ids = config["sample_ids"]
    if config["n"] != 2381 or len(ids) != len(set(ids)) or len(ids) != 2381:
        raise ValueError("expected the fixed unique 2381-sample Development set")
    frozen = set(_json(ROOT / "configs/splits/preliminary_nq_500.json")["ids"])
    split = _json(split_path)
    if set(ids) & (frozen | set(split["splits"]["nq"]["test"])):
        raise ValueError("Frozen/test sample in requested Development inputs")
    if sha256_file(split_path) != config["split_sha256"]:
        raise ValueError("Diagnosis split differs from Phase5-6")
    scores_path = prior / "crossencoder_scores.jsonl"
    if sha256_file(scores_path) != _json(prior / "evaluation/completion.json")["scoring_journal_sha256"]:
        raise ValueError("saved scoring journal differs from completed Phase5-6")
    scores = read_results(scores_path)
    if [r["sample_id"] for r in scores] != ids:
        raise ValueError("scored sample IDs/order differ from fixed set")
    source_path = "data/annotations/normalized/nq_3216_diagnosis_inputs.jsonl"
    # Retain the exact trailing newline for the source receipt.
    source_blob = subprocess.check_output(["git", "show", f"{diagnosis_ref}:{source_path}"], cwd=ROOT)
    c1_rows = [json.loads(line) for line in source_blob.splitlines()]
    c1 = {r["sample_id"]: r for r in c1_rows}
    if len(c1) != len(c1_rows) or not set(ids) <= c1.keys():
        raise ValueError("missing/duplicate C1 initial answers")
    selections_path = prior / "evaluation/selection_simulations.jsonl"
    selection_rows = [json.loads(line) for line in selections_path.read_text().splitlines()]
    selected_reference = {r["sample_id"]: r["selected_ids"] for r in selection_rows
                          if r["policy"] == "P3" and r["depth"] == 20}
    if list(selected_reference) != ids:
        raise ValueError("saved P3-20 selection membership/order differs")
    runtime = []
    changed = set()
    text_representation_differences = 0
    for scored in scores:
        sid = scored["sample_id"]
        row = c1[sid]
        if row["question"] != scored["question"] or not isinstance(row["c1_prediction"], str) or not row["c1_prediction"].strip():
            raise ValueError("C1 question/initial answer mismatch")
        if scored["reranker_revision"] != RERANKER_REVISION:
            raise ValueError("saved reranker revision differs")
        pool = scored["candidates"][:20]
        hits = [{"passage_id": p["passage_id"], "text": p["text"], "score": p["retrieval_score"]} for p in pool]
        outcome = select_p3_20(hits, [p["cross_encoder_score"] for p in pool])
        selected = outcome["evidence"]
        if [p["passage_id"] for p in selected] != selected_reference[sid]:
            raise ValueError("production selection differs from Phase5-6")
        base_ids = [p["passage_id"] for p in pool[:5]]
        if base_ids != [p["passage_id"] for p in row["top5_evidence"]]:
            raise ValueError("C1 ordered Top5 IDs differ from current Base")
        text_representation_differences += any(a["text"] != b["text"] for a, b in zip(pool[:5], row["top5_evidence"]))
        if [p["passage_id"] for p in selected] != base_ids:
            changed.add(sid)
            runtime.append({"sample_id": sid, "question": scored["question"],
                            "initial_answer": row["c1_prediction"],
                            "evidence": [{"rank": rank, "passage_id": p["passage_id"], "text": p["text"]}
                                         for rank, p in enumerate(selected, 1)]})
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    bundle = target / "gpu_bundle"
    bundle.mkdir()
    input_path = bundle / "generation_inputs.jsonl"
    _lines(input_path, runtime)
    load_generation_inputs(input_path)  # strict whitelist, ranks, uniqueness

    # Only after every runtime decision has been persisted, read offline Gold/labels.
    annotations = {r["id"]: r for r in map(json.loads, (prior / "offline_annotations.jsonl").read_text().splitlines())}
    labels = {r["sample_id"]: r for r in map(json.loads, (prior / "offline_diagnosis_labels.jsonl").read_text().splitlines())}
    if set(annotations) != set(ids) or set(labels) != set(ids):
        raise ValueError("offline reference membership differs")
    offline = target / "mac_offline_only"
    offline.mkdir()
    references = [{"sample_id": sid, "initial_answer": c1[sid]["c1_prediction"],
                   "gold_answers": gold_answers_from_record(annotations[sid]),
                   "label": labels[sid]["primary_label"], "generation_required": sid in changed}
                  for sid in ids]
    if not all(r["gold_answers"] for r in references):
        raise ValueError("missing offline answer aliases")
    _lines(offline / "references.jsonl", references)

    # Explicit code/config allowlist: no data/, tests/, Gold, labels, model or index.
    for name in runtime_code_paths():
        dest = bundle / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    (bundle / "requirements-gpu.txt").write_text("torch\ntransformers\nbitsandbytes\naccelerate\npyyaml\n")
    (bundle / "README.txt").write_text(
        "NQ P3-20 GPU generation package; only changed Development Top5 inputs.\n"
        "No Gold/labels/Frozen/test inputs, models, FAISS or reranker are included.\n"
        "Use Linux + NVIDIA CUDA + bitsandbytes NF4 support. Install requirements-gpu.txt.\n"
        "From this directory, first run 20 engineering samples:\n"
        "python -B scripts/run_nq_p320_gpu_generation.py --input generation_inputs.jsonl --output-dir gpu_results --limit 20\n"
        "Then continue the SAME ordered job without rerunning completed records:\n"
        "python -B scripts/run_nq_p320_gpu_generation.py --input generation_inputs.jsonl --output-dir gpu_results --resume\n"
        "Return the entire gpu_results directory to Mac, including run_manifest.json/model_receipt.json/results.jsonl.\n"
        "The first timing includes lazy model initialization; model/API attempts are not physical batch counts.\n"
        "Generator exception: save an explicit fallback and stop; resume continues after that terminal failure.\n"
        "Original C1 did not pin Qwen weights. Resolved model/tokenizer commits and dependencies are recorded on GPU.\n"
        "Do not upload mac_offline_only/, the original annotations or any unrelated data.\n"
    )
    receipt = {
        "phase": "6-1", "code_commit": _git("rev-parse", "HEAD").decode(),
        "c1_source": {"git_commit": _git("rev-parse", diagnosis_ref).decode(), "path": source_path,
                      "sha256": hashlib.sha256(source_blob).hexdigest(), "source_records": len(c1_rows),
                      "covered": len(ids), "coverage": 1.0, "empty_initial_answers": 0,
                      "caveat": "annotation-derived C1 Diagnosis data, not canonical raw C1; complete original provenance unavailable"},
        "n": len(ids), "to_generate": len(runtime), "unchanged": len(ids) - len(runtime),
        "ordered_p3_selections_verified": len(ids), "frozen_test_overlap": 0,
        "c1_text_representation_difference_samples": text_representation_differences,
        "evidence_text_source": "Phase5-6 saved canonical Metadata texts; no Unicode decoding or source edits",
        "generator_model": "Qwen/Qwen2.5-7B-Instruct", "qwen_original_revision_pinned": False,
        "reranker_revision_used_for_saved_scores": RERANKER_REVISION,
        "files": {str(p.relative_to(bundle)): sha256_file(p) for p in sorted(bundle.rglob("*")) if p.is_file()},
    }
    with (bundle / "package_manifest.json").open("x") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    archive = target / "nq_p320_gpu_generation.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as out:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                out.write(path, str(Path("gpu_bundle") / path.relative_to(bundle)))
    summary = {**receipt, "archive_bytes": archive.stat().st_size, "archive_sha256": sha256_file(archive),
               "runtime_jsonl_bytes": input_path.stat().st_size,
               "mac_reference_sha256": sha256_file(offline / "references.jsonl"),
               "label_counts_offline_only": dict(Counter(r["label"] for r in references)),
               "generation_inference_calls": 0}
    with (target / "preparation_report.json").open("x") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(args.phase5_dir, args.split, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
