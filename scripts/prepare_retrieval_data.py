"""Prepare unified retrieval inputs from NQ and FEVER.

Writes query/corpus JSONL files for a later Retriever. Does not compute
embeddings, build FAISS indexes, download data, or run a language model.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
NQ_RAW_DIR = ROOT / "data" / "raw" / "natural_questions" / "dev"
NQ_QUERY_SRC = ROOT / "data" / "processed" / "nq_controlled" / "dev.jsonl"
FEVER_TRAIN_SRC = ROOT / "data" / "raw" / "fever" / "train.jsonl"
FEVER_DEV_SRC = ROOT / "data" / "raw" / "fever" / "shared_task_dev.jsonl"
FEVER_WIKI_ROOT = ROOT / "data" / "raw" / "fever" / "wiki-pages"

OUT_DIR = ROOT / "data" / "processed" / "retrieval"
NQ_QUERY_OUT = OUT_DIR / "nq_dev.jsonl"
NQ_CORPUS_OUT = OUT_DIR / "nq_corpus.jsonl"
FEVER_DEV_OUT = OUT_DIR / "fever_dev.jsonl"
FEVER_CORPUS_OUT = OUT_DIR / "fever_corpus.jsonl"

SEED = 42
N_PREVIEW = 3
KEEP_LABELS = {"SUPPORTS", "REFUTES"}
WIKI_FILE_RE = re.compile(r"wiki-\d+\.jsonl$", re.IGNORECASE)
OPEN_P = re.compile(r"<P>", re.IGNORECASE)
CLOSE_P = re.compile(r"</P>", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_text(tokens: list[str], is_html: list[bool], start: int, end: int) -> str:
    pieces = [
        token
        for token, html in zip(tokens[start:end], is_html[start:end])
        if not html
    ]
    return " ".join(pieces).strip()


def extract_paragraphs(tokens: list[str], is_html: list[bool]) -> list[str]:
    paragraphs = []
    n = len(tokens)
    i = 0
    while i < n:
        if is_html[i] and OPEN_P.fullmatch(tokens[i]):
            j = i + 1
            while j < n and not (is_html[j] and CLOSE_P.fullmatch(tokens[j])):
                j += 1
            if j < n:
                text = extract_text(tokens, is_html, i, j + 1)
                if text:
                    paragraphs.append(text)
                i = j + 1
                continue
        i += 1
    return paragraphs


def prepare_nq() -> tuple[list[dict], list[dict]]:
    queries = load_jsonl(NQ_QUERY_SRC)
    needed_ids = {str(record["id"]) for record in queries}
    print(f"Loaded {len(queries)} NQ controlled queries from {NQ_QUERY_SRC}")

    dataset = load_from_disk(str(NQ_RAW_DIR))
    paragraphs: list[tuple[str, str]] = []
    seen_docs = 0
    for example in tqdm(dataset, desc="Extracting NQ paragraphs"):
        example_id = str(example["id"])
        if example_id not in needed_ids:
            continue
        seen_docs += 1
        title = example["document"]["title"]
        tokens = example["document"]["tokens"]["token"]
        is_html = example["document"]["tokens"]["is_html"]
        for text in extract_paragraphs(tokens, is_html):
            paragraphs.append((title, text))
    print(f"NQ documents used: {seen_docs}")
    print(f"NQ paragraphs before dedup: {len(paragraphs)}")

    corpus = []
    text_to_pid: dict[str, str] = {}
    for title, text in paragraphs:
        if text in text_to_pid:
            continue
        passage_id = f"nq_p_{len(corpus):06d}"
        text_to_pid[text] = passage_id
        corpus.append(
            {
                "passage_id": passage_id,
                "document_id": title,
                "document_title": title,
                "text": text,
            }
        )

    query_out = []
    missing_gold = 0
    for record in queries:
        gold_text = record["gold_long_answer_text"]
        if gold_text not in text_to_pid:
            passage_id = f"nq_p_{len(corpus):06d}"
            text_to_pid[gold_text] = passage_id
            corpus.append(
                {
                    "passage_id": passage_id,
                    "document_id": record.get("document_title"),
                    "document_title": record.get("document_title"),
                    "text": gold_text,
                }
            )
            missing_gold += 1
        query_out.append(
            {
                "id": record["id"],
                "question": record["question"],
                "short_answer": record.get("short_answer"),
                "document_id": record.get("document_title"),
                "document_title": record.get("document_title"),
                "gold_passage_id": text_to_pid[gold_text],
                "gold_evidence_text": gold_text,
            }
        )
    if missing_gold:
        print(f"Added {missing_gold} gold paragraphs that were missing from <P> extraction")

    write_jsonl(NQ_QUERY_OUT, query_out)
    write_jsonl(NQ_CORPUS_OUT, corpus)
    print(f"Wrote {NQ_QUERY_OUT}")
    print(f"Wrote {NQ_CORPUS_OUT}")
    return query_out, corpus


def wiki_jsonl_files() -> list[Path]:
    files = []
    for path in FEVER_WIKI_ROOT.rglob("*.jsonl"):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        if WIKI_FILE_RE.search(path.name):
            files.append(path)
    return sorted(files)


def parse_fever_lines(lines_field) -> dict[int, str]:
    if lines_field is None:
        return {}
    if isinstance(lines_field, list):
        raw = "\n".join(str(item) for item in lines_field)
    else:
        raw = str(lines_field)
    mapping: dict[int, str] = {}
    for line in raw.split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            sentence_id = int(parts[0])
        except ValueError:
            continue
        mapping[sentence_id] = parts[1].strip()
    return mapping


def iter_evidence_sentences(evidence) -> list[tuple[str, int]]:
    pairs = []
    seen = set()
    for group in evidence or []:
        for item in group:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            page_id, sentence_id = item[2], item[3]
            if page_id in (None, "") or sentence_id is None:
                continue
            pair = (str(page_id), int(sentence_id))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return pairs


def filter_fever_records(path: Path) -> list[dict]:
    kept = []
    for record in load_jsonl(path):
        if record.get("label") in KEEP_LABELS:
            kept.append(record)
    return kept


def prepare_fever() -> tuple[list[dict], int]:
    train_records = filter_fever_records(FEVER_TRAIN_SRC)
    dev_records = filter_fever_records(FEVER_DEV_SRC)
    print(f"FEVER train SUPPORTS/REFUTES: {len(train_records)}")
    print(f"FEVER dev SUPPORTS/REFUTES: {len(dev_records)}")

    needed_pages = {
        page_id
        for record in train_records + dev_records
        for page_id, _sentence_id in iter_evidence_sentences(record.get("evidence"))
    }
    print(f"FEVER gold pages to resolve: {len(needed_pages)}")

    wiki_files = wiki_jsonl_files()
    print(f"FEVER wiki jsonl files (skipping __MACOSX): {len(wiki_files)}")
    if not wiki_files:
        raise FileNotFoundError(f"No wiki-*.jsonl files under {FEVER_WIKI_ROOT}")

    page_lines: dict[str, dict[int, str]] = {}
    corpus_count = 0
    seen_pages: set[str] = set()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with FEVER_CORPUS_OUT.open("w", encoding="utf-8") as handle:
        for wiki_path in tqdm(wiki_files, desc="Building FEVER corpus"):
            with wiki_path.open("r", encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    page = json.loads(line)
                    page_id = str(page.get("id") or "").strip()
                    text = (page.get("text") or "").strip()
                    if not page_id or page_id in seen_pages:
                        continue
                    seen_pages.add(page_id)
                    if text:
                        handle.write(
                            json.dumps(
                                {
                                    "passage_id": page_id,
                                    "page_id": page_id,
                                    "title": page_id,
                                    "text": text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        corpus_count += 1
                    if page_id in needed_pages:
                        page_lines[page_id] = parse_fever_lines(page.get("lines"))

    def attach_gold(records: list[dict]) -> list[dict]:
        out = []
        for record in records:
            pairs = iter_evidence_sentences(record.get("evidence"))
            gold_page_ids = []
            gold_sentence_ids = []
            gold_evidence_text = []
            seen_pages_local = set()
            seen_text = set()
            for page_id, sentence_id in pairs:
                gold_sentence_ids.append([page_id, sentence_id])
                if page_id not in seen_pages_local:
                    gold_page_ids.append(page_id)
                    seen_pages_local.add(page_id)
                sentence = page_lines.get(page_id, {}).get(sentence_id, "")
                if sentence and sentence not in seen_text:
                    gold_evidence_text.append(sentence)
                    seen_text.add(sentence)
            out.append(
                {
                    "id": record["id"],
                    "claim": record["claim"],
                    "label": record["label"],
                    "gold_page_ids": gold_page_ids,
                    "gold_sentence_ids": gold_sentence_ids,
                    "gold_evidence_text": gold_evidence_text,
                }
            )
        return out

    fever_dev = attach_gold(dev_records)
    write_jsonl(FEVER_DEV_OUT, fever_dev)
    print(f"Wrote {FEVER_DEV_OUT}")
    print(f"Wrote {FEVER_CORPUS_OUT}")
    return fever_dev, corpus_count


def preview(records: list[dict], query_key: str, gold_key: str, title: str) -> None:
    rng = random.Random(SEED)
    sample = rng.sample(records, k=min(N_PREVIEW, len(records)))
    print(f"\nRandom {title} preview (seed={SEED}):")
    for i, record in enumerate(sample, start=1):
        print(f"[{i}] id={record['id']}")
        print(f"  {query_key}: {record[query_key]}")
        print(f"  gold evidence: {record[gold_key]}")


def main() -> int:
    print("=== Natural Questions retrieval data ===")
    nq_queries, nq_corpus = prepare_nq()
    nq_linked = sum(1 for record in nq_queries if record.get("gold_passage_id"))
    if nq_linked != len(nq_queries):
        print(f"ERROR: {len(nq_queries) - nq_linked} NQ queries missing gold_passage_id")
        return 1
    print()

    print("=== FEVER retrieval data ===")
    fever_dev, fever_corpus_n = prepare_fever()
    supports = sum(1 for record in fever_dev if record["label"] == "SUPPORTS")
    refutes = sum(1 for record in fever_dev if record["label"] == "REFUTES")
    with_gold = sum(1 for record in fever_dev if record["gold_evidence_text"])
    print()

    print("=== Retrieval data statistics ===")
    print(f"NQ query 数量: {len(nq_queries)}")
    print(f"NQ corpus passage 数量: {len(nq_corpus)}")
    print(f"FEVER dev SUPPORTS 数量: {supports}")
    print(f"FEVER dev REFUTES 数量: {refutes}")
    print(f"FEVER 有有效 gold evidence 的数量: {with_gold}")
    print(f"FEVER corpus passage 数量: {fever_corpus_n}")

    preview(nq_queries, "question", "gold_evidence_text", "NQ")
    preview(fever_dev, "claim", "gold_evidence_text", "FEVER")
    return 0


if __name__ == "__main__":
    sys.exit(main())
