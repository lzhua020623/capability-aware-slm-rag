"""Build the full NQ official Train controlled single-passage subset.

Reuses scripts.prepare_nq.process_example / evaluate_annotation unchanged so the
filter semantics match the frozen Dev pipeline. Reads Train Parquet shards with
column projection (no document.html) and never save_to_disk() the full Train.

Outputs (never overwrites existing finals):
  data/processed/nq_controlled/train.jsonl
  data/processed/nq_controlled/train_stats.json

Modes:
  --verify-dev   re-filter local Dev into a temp file and compare to formal Dev
  (default)      process all Train shards with resume via progress JSON
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_nq import (  # noqa: E402
    STAGE_NAMES,
    evaluate_annotation,
    iter_annotations,
    process_example,
)

OUT_DIR = ROOT / "data" / "processed" / "nq_controlled"
TRAIN_JSONL = OUT_DIR / "train.jsonl"
TRAIN_STATS = OUT_DIR / "train_stats.json"
PROGRESS_PATH = OUT_DIR / "train_full_progress.json"
DEV_FORMAL = OUT_DIR / "dev.jsonl"
NQ_DEV_DIR = ROOT / "data" / "raw" / "natural_questions" / "dev"

HF_DATASET = "google-research-datasets/natural_questions"
NUM_SHARDS = 287
EXPECTED_TRAIN = 307373
COLUMNS = [
    "id",
    "question.text",
    "annotations",
    "document.title",
    "document.url",
    "document.tokens.token",
    "document.tokens.is_html",
]
TAG_RE = re.compile(r"^<([A-Za-z0-9]+)[\s>]", re.IGNORECASE)


def normalize_example(row: dict) -> dict:
    """Map projected Parquet / Arrow rows to the nested shape prepare_nq expects.

    Column projection may yield nested structs or flat leaf names
    (id, text, annotations, title, url, token, is_html).
    """
    if isinstance(row.get("question"), dict) and "text" in row["question"]:
        question_text_value = row["question"]["text"]
    elif "question.text" in row:
        question_text_value = row["question.text"]
    elif "text" in row and "token" in row:
        question_text_value = row["text"]
    else:
        question_text_value = row.get("question") or ""

    document = row.get("document")
    if isinstance(document, dict) and isinstance(document.get("tokens"), dict):
        tokens = document["tokens"]["token"]
        is_html = document["tokens"]["is_html"]
        title = document.get("title") or ""
        url = document.get("url") or ""
    elif "document.tokens.token" in row:
        tokens = row["document.tokens.token"]
        is_html = row["document.tokens.is_html"]
        title = row.get("document.title") or ""
        url = row.get("document.url") or ""
    else:
        tokens = row.get("token")
        is_html = row.get("is_html")
        title = row.get("title") or row.get("document.title") or ""
        url = row.get("url") or row.get("document.url") or ""

    if tokens is None or is_html is None:
        raise KeyError(f"Could not find document tokens in row keys={list(row.keys())}")

    return {
        "id": row["id"],
        "question": {"text": question_text_value or ""},
        "annotations": row["annotations"],
        "document": {
            "title": title,
            "url": url,
            "tokens": {"token": list(tokens), "is_html": list(is_html)},
        },
    }


def long_answer_tag(example: dict, annotation: dict) -> str:
    tokens = example["document"]["tokens"]["token"]
    is_html = example["document"]["tokens"]["is_html"]
    la = annotation["long_answer"]
    start = int(la["start_token"])
    end = int(la["end_token"])
    if start < 0 or end <= start or start >= len(tokens):
        return "none"
    if not is_html[start]:
        return "non_html"
    match = TAG_RE.match(tokens[start])
    return match.group(1).upper() if match else tokens[start].strip("<>").split()[0].upper()


def best_annotation_for_stats(example: dict, best_stage: int):
    """Pick the annotation that achieved best_stage (same preference as process_example)."""
    chosen = None
    stage_hit = -1
    for annotation in iter_annotations(example):
        stage, _ = evaluate_annotation(example, annotation)
        if stage > stage_hit:
            stage_hit = stage
            chosen = annotation
        if stage_hit == best_stage and stage == best_stage:
            chosen = annotation
            break
    return chosen


def accumulate(example: dict, stage_counts: list[int], tag_counter: Counter) -> dict | None:
    best_stage, record = process_example(example)
    stage_counts[0] += 1
    for level in range(1, best_stage + 1):
        stage_counts[level] += 1
    if best_stage >= 2:
        annotation = best_annotation_for_stats(example, best_stage)
        if annotation is not None:
            tag = long_answer_tag(example, annotation)
            if best_stage < 5:
                tag_counter[tag] += 1
            else:
                tag_counter[f"kept_{tag}"] += 1
    return record if best_stage == 5 and record is not None else None


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {
            "completed_shards": [],
            "stage_counts": [0] * len(STAGE_NAMES),
            "rejected_long_answer_tags": {},
            "kept": 0,
            "valid_question": 0,
            "duplicate_ids_skipped": 0,
            "seen_ids": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Drop full seen_ids from disk growth; keep only count + bloom via set in memory.
    payload = {k: v for k, v in progress.items() if k != "seen_ids_set"}
    if "seen_ids_set" in progress:
        payload["seen_ids_count"] = len(progress["seen_ids_set"])
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def iter_dev_examples():
    import pyarrow as pa

    for path in sorted(NQ_DEV_DIR.glob("data-*.arrow")):
        with pa.memory_map(str(path)) as source:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                for row in batch.to_pylist():
                    yield normalize_example(row)


def verify_dev(temp_dir: Path) -> int:
    if not DEV_FORMAL.exists():
        print(f"ERROR: missing formal Dev file {DEV_FORMAL}")
        return 1
    formal = [json.loads(line) for line in DEV_FORMAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    stage_counts = [0] * len(STAGE_NAMES)
    tag_counter: Counter = Counter()
    rebuilt: list[dict] = []
    for example in iter_dev_examples():
        record = accumulate(example, stage_counts, tag_counter)
        if record is not None:
            rebuilt.append(record)

    temp_dir.mkdir(parents=True, exist_ok=True)
    out = temp_dir / "dev_rebuild_from_train_pipeline.jsonl"
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rebuilt) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Rebuilt Dev: {len(rebuilt)} -> {out}")
    print("Stages:", {STAGE_NAMES[i]: stage_counts[i] for i in range(len(STAGE_NAMES))})

    ok = True
    if len(rebuilt) != 3216 or len(formal) != 3216:
        print(f"FAIL count: rebuilt={len(rebuilt)} formal={len(formal)}")
        ok = False
    formal_by_id = {str(r["id"]): r for r in formal}
    rebuilt_by_id = {str(r["id"]): r for r in rebuilt}
    if set(formal_by_id) != set(rebuilt_by_id):
        only_f = sorted(set(formal_by_id) - set(rebuilt_by_id))[:5]
        only_r = sorted(set(rebuilt_by_id) - set(formal_by_id))[:5]
        print(f"FAIL id set: only_formal={only_f} only_rebuilt={only_r}")
        ok = False
    diffs = []
    for sample_id, formal_rec in formal_by_id.items():
        rebuilt_rec = rebuilt_by_id.get(sample_id)
        if rebuilt_rec is None:
            continue
        for key in ("question", "short_answer", "gold_long_answer_text"):
            if formal_rec.get(key) != rebuilt_rec.get(key):
                diffs.append((sample_id, key))
                if len(diffs) >= 10:
                    break
        if len(diffs) >= 10:
            break
    if diffs:
        print(f"FAIL field diffs (first {len(diffs)}): {diffs}")
        ok = False
    if ok:
        print("PASS: Dev regression 7830 -> 3216 identical on id/question/short_answer/gold_long_answer_text")
        return 0
    print("STOP: Dev regression failed; do not write Train output with this pipeline.")
    return 1


def open_shard_table(shard: int, cache_dir: Path | None):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download

    filename = f"default/train-{shard:05d}-of-{NUM_SHARDS:05d}.parquet"
    # Prefer hub download into cache (resumable), then local parquet read.
    local = hf_hub_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return pq.read_table(local, columns=COLUMNS)


def process_train(cache_dir: Path | None, progress_every: int) -> int:
    if TRAIN_STATS.exists() and TRAIN_JSONL.exists() and not PROGRESS_PATH.exists():
        print(f"ERROR: final outputs already exist ({TRAIN_JSONL.name}, {TRAIN_STATS.name}); refusing to overwrite")
        return 1

    progress = load_progress(PROGRESS_PATH)
    completed = set(progress.get("completed_shards", []))
    stage_counts = list(progress.get("stage_counts", [0] * len(STAGE_NAMES)))
    while len(stage_counts) < len(STAGE_NAMES):
        stage_counts.append(0)
    tag_counter = Counter(progress.get("rejected_long_answer_tags", {}))
    # Also restore kept_* tags if stored mixed.
    for k, v in progress.get("long_answer_tags", {}).items():
        tag_counter[k] = v
    seen_ids: set[str] = set()
    # Rebuild seen ids from existing partial JSONL when resuming.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if TRAIN_JSONL.exists() and completed else "w"
    if mode == "w" and TRAIN_JSONL.exists():
        print(f"ERROR: {TRAIN_JSONL} exists without matching progress; refusing to overwrite")
        return 1
    if TRAIN_JSONL.exists():
        with TRAIN_JSONL.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    seen_ids.add(str(json.loads(line)["id"]))
        if len(seen_ids) != progress.get("kept", len(seen_ids)):
            print(f"WARNING: progress kept={progress.get('kept')} but jsonl ids={len(seen_ids)}; using jsonl count")
        progress["kept"] = len(seen_ids)

    started = time.time()
    progress["elapsed_seconds_prev"] = float(progress.get("elapsed_seconds", 0))
    valid_question = int(progress.get("valid_question", 0))
    duplicate_ids_skipped = int(progress.get("duplicate_ids_skipped", 0))

    with TRAIN_JSONL.open(mode, encoding="utf-8", newline="\n") as out:
        for shard in range(NUM_SHARDS):
            if shard in completed:
                continue
            shard_t0 = time.time()
            table = open_shard_table(shard, cache_dir)
            rows = table.to_pylist()
            kept_this = 0
            for row in rows:
                example = normalize_example(row)
                q = example["question"]["text"] if isinstance(example["question"], dict) else str(example["question"])
                if q.strip():
                    valid_question += 1
                record = accumulate(example, stage_counts, tag_counter)
                if record is None:
                    continue
                sample_id = str(record["id"])
                if sample_id in seen_ids:
                    duplicate_ids_skipped += 1
                    continue
                seen_ids.add(sample_id)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_this += 1
            out.flush()
            completed.add(shard)
            progress.update({
                "completed_shards": sorted(completed),
                "stage_counts": stage_counts,
                "kept": len(seen_ids),
                "valid_question": valid_question,
                "duplicate_ids_skipped": duplicate_ids_skipped,
                "rejected_long_answer_tags": {
                    k: v for k, v in tag_counter.items() if not k.startswith("kept_")
                },
                "long_answer_tags": dict(tag_counter),
                "last_shard": shard,
                "last_shard_rows": len(rows),
                "last_shard_kept": kept_this,
                "last_shard_seconds": round(time.time() - shard_t0, 2),
                "elapsed_seconds": round(time.time() - started + float(progress.get("elapsed_seconds_prev", 0)), 2),
            })
            save_progress(PROGRESS_PATH, progress)
            if shard % progress_every == 0 or shard + 1 == NUM_SHARDS:
                print(
                    f"shard {shard + 1}/{NUM_SHARDS}: rows={len(rows)} kept_total={len(seen_ids)} "
                    f"stage5={stage_counts[5]} elapsed={progress['elapsed_seconds']:.0f}s",
                    flush=True,
                )

    elapsed = time.time() - started + float(progress.get("elapsed_seconds_prev", 0))
    rejected = {k: v for k, v in tag_counter.items() if not k.startswith("kept_")}
    kept_tags = {k[len("kept_"):]: v for k, v in tag_counter.items() if k.startswith("kept_")}
    stats = {
        "source": f"{HF_DATASET} default/train (*.parquet, column projection)",
        "output": str(TRAIN_JSONL.relative_to(ROOT).as_posix()),
        "num_shards": NUM_SHARDS,
        "expected_train_samples": EXPECTED_TRAIN,
        "original samples": stage_counts[0],
        "valid_question": valid_question,
        "has short answer": stage_counts[1],
        "exactly one short-answer span": stage_counts[2],
        "non-yes/no": stage_counts[3],
        "short answer inside gold long answer": stage_counts[4],
        "paragraph long answer": stage_counts[5],
        "final samples": len(seen_ids),
        "pass_rate": round(len(seen_ids) / max(stage_counts[0], 1), 6),
        "duplicate_ids_skipped": duplicate_ids_skipped,
        "rejected_long_answer_tags": dict(sorted(rejected.items(), key=lambda x: -x[1])),
        "kept_long_answer_tags": dict(sorted(kept_tags.items(), key=lambda x: -x[1])),
        "elapsed_seconds": round(elapsed, 2),
        "filter_functions": "scripts.prepare_nq.process_example / evaluate_annotation",
    }
    if TRAIN_STATS.exists():
        print(f"ERROR: {TRAIN_STATS} already exists; refusing to overwrite")
        return 1
    TRAIN_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    # Clear progress only after successful stats write so resume is clean next time.
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
    print(json.dumps(stats, indent=2))
    print(f"Wrote {TRAIN_JSONL}")
    print(f"Wrote {TRAIN_STATS}")
    return 0


def overlap_report() -> None:
    if not TRAIN_JSONL.exists() or not DEV_FORMAL.exists():
        print("Overlap skipped: missing train or formal Dev")
        return
    train = [json.loads(line) for line in TRAIN_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev = [json.loads(line) for line in DEV_FORMAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_ids = {str(r["id"]) for r in train}
    dev_ids = {str(r["id"]) for r in dev}
    norm_q = lambda s: " ".join(str(s).strip().lower().split())
    train_q = {norm_q(r["question"]) for r in train}
    dev_q = {norm_q(r["question"]) for r in dev}
    train_qa = {(norm_q(r["question"]), norm_q(r["short_answer"])) for r in train}
    dev_qa = {(norm_q(r["question"]), norm_q(r["short_answer"])) for r in dev}
    report = {
        "train_candidates": len(train),
        "dev_candidates": len(dev),
        "exact_sample_id_overlap": len(train_ids & dev_ids),
        "normalized_question_overlap": len(train_q & dev_q),
        "normalized_question_plus_answer_overlap": len(train_qa & dev_qa),
    }
    path = OUT_DIR / "train_dev_overlap.json"
    if path.exists():
        print(f"Overlap file exists, printing only: {report}")
    else:
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote {path}")
    print(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-dev", action="store_true", help="Dev regression only; no Train write")
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path(os.environ.get("TEMP", "/tmp")) / "nq_train_full_verify",
        help="Temp dir for Dev rebuild JSONL",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("E:/hf_cache"),
        help="Hugging Face cache root (default E:/hf_cache to avoid filling C:)",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--overlap-only", action="store_true")
    args = parser.parse_args()

    if args.overlap_only:
        overlap_report()
        return 0
    if args.verify_dev:
        return verify_dev(args.temp_dir)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # Point hub downloads at E: before importing hub download paths again.
    os.environ.setdefault("HF_HOME", str(args.cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(args.cache_dir / "hub"))
    code = process_train(args.cache_dir / "hub", args.progress_every)
    if code == 0:
        overlap_report()
    return code


if __name__ == "__main__":
    sys.exit(main())
