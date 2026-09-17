"""Audit NQ Base RAG smoke-test outputs without rerunning the model."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.qa import exact_match, normalize_answer, token_f1

SMOKE_PATH = ROOT / "results" / "base_rag" / "nq_smoke_test.jsonl"
QUERY_PATH = ROOT / "data" / "processed" / "retrieval" / "nq_dev.jsonl"


def safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        cleaned = []
        for arg in args:
            text = str(arg)
            cleaned.append(
                text.encode(encoding, errors="replace").decode(encoding, errors="replace")
            )
        print(*cleaned, **kwargs)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def gold_rank(hits: list[dict], gold_passage_id: str | None) -> int | None:
    if not gold_passage_id:
        return None
    for i, hit in enumerate(hits, start=1):
        if hit.get("passage_id") == gold_passage_id:
            return i
    return None


def classify(gold_in_top5: bool, em: int, gold_in_answer: bool) -> str:
    if not gold_in_top5:
        return "retrieval_miss"
    if em == 1:
        return "em_success"
    if gold_in_answer:
        return "formatting_or_verbose"
    return "model_failure"


def main() -> int:
    smoke = load_jsonl(SMOKE_PATH)
    queries = {str(item["id"]): item for item in load_jsonl(QUERY_PATH)}
    print(f"Auditing {len(smoke)} smoke-test records from {SMOKE_PATH}")
    print("No model is loaded; original JSONL is not modified.")
    print()

    counts: Counter[str] = Counter()
    for i, record in enumerate(smoke, start=1):
        query_id = str(record["id"])
        query = queries.get(query_id, {})
        gold_passage_id = query.get("gold_passage_id")
        gold_evidence = query.get("gold_evidence_text", "")
        hits = record.get("retrieved_top5") or []
        rank = gold_rank(hits, gold_passage_id)
        gold_in_top5 = rank is not None

        gold_answer = record["gold_answer"]
        model_answer = record["model_answer"]
        em = exact_match(model_answer, gold_answer)
        f1 = token_f1(model_answer, gold_answer)
        norm_gold = normalize_answer(gold_answer)
        norm_pred = normalize_answer(model_answer)
        gold_in_answer = bool(norm_gold) and norm_gold in norm_pred
        label = classify(gold_in_top5, em, gold_in_answer)
        counts[label] += 1

        safe_print("=" * 80)
        safe_print(f"[{i}/{len(smoke)}] category={label}")
        safe_print(f"id: {query_id}")
        safe_print(f"question: {record['question']}")
        safe_print(f"gold_answer: {gold_answer}")
        safe_print(f"model_answer: {model_answer}")
        safe_print(f"stored exact_match: {record.get('exact_match')}")
        safe_print(f"stored F1: {record.get('token_level_f1')}")
        safe_print(f"normalized gold_answer: {norm_gold}")
        safe_print(f"normalized model_answer: {norm_pred}")
        safe_print(f"recomputed EM: {em}")
        safe_print(f"recomputed F1: {f1:.4f}")
        safe_print(f"gold_answer_in_model_answer: {str(gold_in_answer).lower()}")
        safe_print(f"gold_passage_id: {gold_passage_id}")
        safe_print(f"gold_in_top5: {str(gold_in_top5).lower()}")
        safe_print(f"gold passage 排名: {rank if rank is not None else 'none'}")
        safe_print(f"gold evidence text: {gold_evidence}")
        safe_print("Top-5 passages:")
        for rank_i, hit in enumerate(hits, start=1):
            marker = "  [GOLD]" if hit.get("passage_id") == gold_passage_id else ""
            safe_print(
                f"  {rank_i}. {hit.get('passage_id')} score={float(hit.get('score', 0.0)):.4f}{marker}"
            )
            safe_print(f"     {hit.get('text', '')}")
        print()

    print("=" * 80)
    print("Category counts:")
    print(f"  retrieval_miss: {counts['retrieval_miss']}")
    print(f"  formatting_or_verbose: {counts['formatting_or_verbose']}")
    print(f"  model_failure: {counts['model_failure']}")
    if counts["em_success"]:
        print(f"  em_success: {counts['em_success']}")
    print(f"  total: {sum(counts.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
