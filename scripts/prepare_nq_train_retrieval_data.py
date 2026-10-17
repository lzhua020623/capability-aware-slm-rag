"""Build NQ Train retrieval queries + corpus with Dev-identical semantics.

Reuses scripts.prepare_retrieval_data.extract_paragraphs (all <P> paragraphs from
each controlled sample's Wikipedia document, global exact-text dedup, gold text
mapped to passage_id). Does not encode, index, or call any LM.

Outputs (never overwrite existing finals):
  data/processed/retrieval/nq_train.jsonl
  data/processed/retrieval/nq_train_corpus.jsonl
  data/processed/retrieval/nq_train_retrieval_stats.json

--verify-dev rebuilds Dev into a temp dir and compares to formal nq_dev/nq_corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_retrieval_data import extract_paragraphs  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "retrieval"
TRAIN_QUERY_SRC = ROOT / "data" / "processed" / "nq_controlled" / "train.jsonl"
DEV_QUERY_SRC = ROOT / "data" / "processed" / "nq_controlled" / "dev.jsonl"
DEV_QUERY_FORMAL = OUT_DIR / "nq_dev.jsonl"
DEV_CORPUS_FORMAL = OUT_DIR / "nq_corpus.jsonl"
NQ_DEV_RAW = ROOT / "data" / "raw" / "natural_questions" / "dev"

TRAIN_QUERY_OUT = OUT_DIR / "nq_train.jsonl"
TRAIN_CORPUS_OUT = OUT_DIR / "nq_train_corpus.jsonl"
TRAIN_STATS_OUT = OUT_DIR / "nq_train_retrieval_stats.json"
PROGRESS_PATH = OUT_DIR / "nq_train_retrieval_progress.json"

NUM_SHARDS = 287
HF_DATASET = "google-research-datasets/natural_questions"
DOC_COLUMNS = [
    "id",
    "document.title",
    "document.tokens.token",
    "document.tokens.is_html",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_train_shard_dir(cache_dir: Path) -> Path:
    hub = cache_dir / "hub" / f"datasets--{HF_DATASET.replace('/', '--')}"
    snaps = sorted((hub / "snapshots").glob("*")) if (hub / "snapshots").exists() else []
    if not snaps:
        raise FileNotFoundError(f"No HF snapshot under {hub}; expected Train parquet cache")
    shard_dir = snaps[-1] / "default"
    sample = shard_dir / "train-00000-of-00287.parquet"
    if not sample.exists():
        raise FileNotFoundError(f"Missing {sample}")
    return shard_dir


def normalize_doc_row(row: dict) -> tuple[str, str, list[str], list[bool]]:
    sample_id = str(row["id"])
    if isinstance(row.get("document"), dict) and "tokens" in row["document"]:
        title = row["document"].get("title") or ""
        tokens = list(row["document"]["tokens"]["token"])
        is_html = list(row["document"]["tokens"]["is_html"])
        return sample_id, title, tokens, is_html
    title = row.get("title") or row.get("document.title") or ""
    tokens = row.get("token") or row.get("document.tokens.token")
    is_html = row.get("is_html") or row.get("document.tokens.is_html")
    if tokens is None or is_html is None:
        raise KeyError(f"Missing tokens in row keys={list(row.keys())}")
    return sample_id, title, list(tokens), list(is_html)


def passage_id_for(index: int, width: int) -> str:
    return f"nq_p_{index:0{width}d}"


def build_corpus_and_queries(
    controlled_queries: list[dict],
    document_iter,
    corpus_out: Path,
    query_out: Path,
    passage_id_width: int,
    progress_cb=None,
) -> dict:
    """Dev-identical construction: extract <P> paras in document visit order, dedup by text."""
    needed = {str(record["id"]) for record in controlled_queries}
    text_to_pid: dict[str, str] = {}
    corpus_count = 0
    paras_before_dedup = 0
    docs_used = 0
    passages_per_doc: list[int] = []
    title_counts: Counter = Counter()

    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    if corpus_out.exists():
        raise FileExistsError(f"Refusing to overwrite {corpus_out}")
    if query_out.exists():
        raise FileExistsError(f"Refusing to overwrite {query_out}")

    with corpus_out.open("w", encoding="utf-8", newline="\n") as corpus_handle:
        for sample_id, title, tokens, is_html in document_iter:
            if sample_id not in needed:
                continue
            docs_used += 1
            texts = extract_paragraphs(tokens, is_html)
            paras_before_dedup += len(texts)
            new_here = 0
            for text in texts:
                if text in text_to_pid:
                    continue
                pid = passage_id_for(corpus_count, passage_id_width)
                text_to_pid[text] = pid
                corpus_handle.write(
                    json.dumps(
                        {
                            "passage_id": pid,
                            "document_id": title,
                            "document_title": title,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                corpus_count += 1
                new_here += 1
            passages_per_doc.append(len(texts))
            title_counts[title] += 1
            if progress_cb is not None and docs_used % 500 == 0:
                progress_cb(docs_used, corpus_count, paras_before_dedup)

    missing_gold = 0
    with corpus_out.open("a", encoding="utf-8", newline="\n") as corpus_handle, query_out.open(
        "w", encoding="utf-8", newline="\n"
    ) as query_handle:
        for record in controlled_queries:
            gold_text = record["gold_long_answer_text"]
            if gold_text not in text_to_pid:
                pid = passage_id_for(corpus_count, passage_id_width)
                text_to_pid[gold_text] = pid
                corpus_handle.write(
                    json.dumps(
                        {
                            "passage_id": pid,
                            "document_id": record.get("document_title"),
                            "document_title": record.get("document_title"),
                            "text": gold_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                corpus_count += 1
                missing_gold += 1
            query_handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "question": record["question"],
                        "short_answer": record.get("short_answer"),
                        "document_id": record.get("document_title"),
                        "document_title": record.get("document_title"),
                        "gold_passage_id": text_to_pid[gold_text],
                        "gold_evidence_text": gold_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    passages_per_doc_sorted = sorted(passages_per_doc) if passages_per_doc else [0]
    mid = len(passages_per_doc_sorted) // 2
    median = (
        passages_per_doc_sorted[mid]
        if len(passages_per_doc_sorted) % 2 == 1
        else (
            passages_per_doc_sorted[mid - 1] + passages_per_doc_sorted[mid]
        )
        / 2
        if passages_per_doc_sorted
        else 0
    )
    return {
        "controlled_queries": len(controlled_queries),
        "documents_used": docs_used,
        "unique_document_titles": len(title_counts),
        "paragraphs_before_dedup": paras_before_dedup,
        "corpus_passages": corpus_count,
        "missing_gold_appended": missing_gold,
        "gold_mapped": len(controlled_queries),  # every query gets a gold_passage_id
        "mean_paragraphs_per_document": round(paras_before_dedup / max(docs_used, 1), 4),
        "median_paragraphs_per_document": median,
        "min_paragraphs_per_document": min(passages_per_doc) if passages_per_doc else 0,
        "max_paragraphs_per_document": max(passages_per_doc) if passages_per_doc else 0,
        "passage_id_width": passage_id_width,
    }


def iter_dev_documents():
    from datasets import load_from_disk

    dataset = load_from_disk(str(NQ_DEV_RAW))
    for example in dataset:
        yield (
            str(example["id"]),
            example["document"]["title"],
            list(example["document"]["tokens"]["token"]),
            list(example["document"]["tokens"]["is_html"]),
        )


def iter_train_documents(shard_dir: Path, needed_ids: set[str], progress_path: Path):
    completed = set()
    if progress_path.exists():
        completed = set(json.loads(progress_path.read_text(encoding="utf-8")).get("completed_shards", []))
    for shard in range(NUM_SHARDS):
        if shard in completed:
            continue
        path = shard_dir / f"train-{shard:05d}-of-{NUM_SHARDS:05d}.parquet"
        table = pq.read_table(path, columns=DOC_COLUMNS)
        for row in table.to_pylist():
            sample_id, title, tokens, is_html = normalize_doc_row(row)
            if sample_id in needed_ids:
                yield sample_id, title, tokens, is_html
        completed.add(shard)
        progress_path.write_text(
            json.dumps({"completed_shards": sorted(completed), "last_shard": shard}) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if (shard + 1) % 5 == 0 or shard + 1 == NUM_SHARDS:
            print(f"scanned shard {shard + 1}/{NUM_SHARDS}", flush=True)


def compare_jsonl(formal: Path, rebuilt: Path, kind: str) -> list[str]:
    a = [json.loads(line) for line in formal.read_text(encoding="utf-8").splitlines() if line.strip()]
    b = [json.loads(line) for line in rebuilt.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = []
    if len(a) != len(b):
        errors.append(f"{kind} count formal={len(a)} rebuilt={len(b)}")
        return errors
    for i, (fa, fb) in enumerate(zip(a, b)):
        if fa != fb:
            diffs = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
            errors.append(f"{kind} line {i} id={fa.get('id') or fa.get('passage_id')} fields={diffs[:6]}")
            if len(errors) >= 10:
                break
    return errors


def verify_dev(temp_dir: Path) -> int:
    temp_dir.mkdir(parents=True, exist_ok=True)
    query_out = temp_dir / "nq_dev_rebuild.jsonl"
    corpus_out = temp_dir / "nq_corpus_rebuild.jsonl"
    for path in (query_out, corpus_out):
        if path.exists():
            path.unlink()
    queries = load_jsonl(DEV_QUERY_SRC)
    t0 = time.time()
    stats = build_corpus_and_queries(
        controlled_queries=queries,
        document_iter=iter_dev_documents(),
        corpus_out=corpus_out,
        query_out=query_out,
        passage_id_width=6,
    )
    print(json.dumps(stats, indent=2))
    print(f"Dev rebuild seconds={time.time() - t0:.1f}")
    errors = compare_jsonl(DEV_QUERY_FORMAL, query_out, "query") + compare_jsonl(
        DEV_CORPUS_FORMAL, corpus_out, "corpus"
    )
    if errors:
        print("FAIL Dev retrieval regression:")
        for err in errors:
            print(" ", err)
        return 1
    print("PASS: Dev retrieval regression identical (3216 queries, 99922 passages)")
    return 0


def run_train(cache_dir: Path) -> int:
    for path in (TRAIN_QUERY_OUT, TRAIN_CORPUS_OUT, TRAIN_STATS_OUT):
        if path.exists():
            print(f"ERROR: refusing to overwrite existing {path}")
            return 1
    if not TRAIN_QUERY_SRC.exists():
        print(f"ERROR: missing {TRAIN_QUERY_SRC}")
        return 1

    queries = load_jsonl(TRAIN_QUERY_SRC)
    if len(queries) != 74097:
        print(f"WARNING: expected 74097 controlled Train queries, got {len(queries)}")
    needed = {str(r["id"]) for r in queries}
    shard_dir = resolve_train_shard_dir(cache_dir)
    present = list(shard_dir.glob("train-*-of-00287.parquet"))
    if len(present) != NUM_SHARDS:
        print(f"ERROR: expected {NUM_SHARDS} train shards, found {len(present)} under {shard_dir}")
        return 1

    # Train corpus expected >> 1e6; use 7-digit ids (Dev keeps 6 for regression identity).
    passage_id_width = 7
    t0 = time.time()
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    def progress_cb(docs_used, corpus_count, paras_before):
        print(
            f"docs={docs_used}/{len(needed)} corpus={corpus_count} "
            f"paras_before_dedup={paras_before} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    stats = build_corpus_and_queries(
        controlled_queries=queries,
        document_iter=iter_train_documents(shard_dir, needed, PROGRESS_PATH),
        corpus_out=TRAIN_CORPUS_OUT,
        query_out=TRAIN_QUERY_OUT,
        passage_id_width=passage_id_width,
        progress_cb=progress_cb,
    )
    elapsed = time.time() - t0
    # Integrity: duplicate ids / texts
    pid_counts: Counter = Counter()
    text_counts: Counter = Counter()
    with TRAIN_CORPUS_OUT.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            pid_counts[rec["passage_id"]] += 1
            text_counts[rec["text"]] += 1
    gold_ok = 0
    gold_fail = 0
    pid_set = {pid for pid, n in pid_counts.items() if n == 1}
    # Remap check via query file
    with TRAIN_QUERY_OUT.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if rec["gold_passage_id"] in pid_set:
                gold_ok += 1
            else:
                gold_fail += 1

    scanned = json.loads(PROGRESS_PATH.read_text(encoding="utf-8")) if PROGRESS_PATH.exists() else {}
    report = {
        **stats,
        "elapsed_seconds": round(elapsed, 2),
        "shards_present": len(present),
        "shards_scanned": len(scanned.get("completed_shards", [])),
        "duplicate_passage_id_count": sum(1 for n in pid_counts.values() if n > 1),
        "duplicate_exact_text_count": sum(1 for n in text_counts.values() if n > 1),
        "gold_passage_mapped": gold_ok,
        "gold_passage_mapping_failures": gold_fail,
        "query_jsonl_bytes": TRAIN_QUERY_OUT.stat().st_size,
        "corpus_jsonl_bytes": TRAIN_CORPUS_OUT.stat().st_size,
        "source_controlled": str(TRAIN_QUERY_SRC.relative_to(ROOT).as_posix()),
        "source_shards": str(shard_dir),
        "called_bge_qwen_diagnosis": False,
        "read_test": False,
        "semantics": "prepare_retrieval_data.prepare_nq / extract_paragraphs + global text dedup",
    }
    TRAIN_STATS_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
    print(json.dumps(report, indent=2))
    print(f"Wrote {TRAIN_QUERY_OUT}")
    print(f"Wrote {TRAIN_CORPUS_OUT}")
    print(f"Wrote {TRAIN_STATS_OUT}")
    if gold_fail or stats["documents_used"] != len(queries):
        print(
            f"WARNING: docs_used={stats['documents_used']} queries={len(queries)} "
            f"gold_fail={gold_fail}"
        )
        return 1 if gold_fail else 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-dev", action="store_true")
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path(os.environ.get("TEMP", "/tmp")) / "nq_train_retrieval_verify",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("E:/hf_cache"))
    args = parser.parse_args()
    if args.verify_dev:
        return verify_dev(args.temp_dir)
    return run_train(args.cache_dir)


if __name__ == "__main__":
    sys.exit(main())
