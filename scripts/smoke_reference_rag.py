"""Resource-level smoke: 1–2 Train/Val queries through formal Reference RAG.

Does not run Enhancement E2E or touch frozen Test IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_queries(limit: int) -> list[dict]:
    from src.data.team_nq import TeamNQBoundary

    boundary = TeamNQBoundary(ROOT)
    # Never use sealed Test IDs. Scan Train/Val membership only.
    allowed = {
        sid: role
        for role in ("train", "validation")
        for sid in boundary.splits[role]
    }
    path = ROOT / "data/processed/diagnosis/nq_train_74097_diagnosis.jsonl"
    if not path.is_file():
        raise SystemExit(f"Missing Train/Val diagnosis records: {path}")
    found: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row.get("input") or row
            sid = str(payload.get("sample_id") or "")
            role = allowed.get(sid)
            if role is None:
                continue
            found.append({"sample_id": sid, "role": role, "query": payload["query"]})
            if len(found) >= limit:
                break
    if len(found) < limit:
        raise SystemExit(
            f"Could not resolve {limit} Train/Val queries from {path}; found {len(found)}"
        )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--cpu-retriever", action="store_true", help="Force BGE on CPU")
    args = parser.parse_args(argv)

    samples = _load_queries(args.limit)
    print(json.dumps({"phase": "queries", "samples": samples}, ensure_ascii=False, indent=2))

    from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosisClassifier
    from src.generation.qwen import free_cuda_memory, load_qwen_nf4
    from src.pipeline.reference_rag import MatchedReferenceRAG

    device = "cpu" if args.cpu_retriever else None
    tokenizer, model = load_qwen_nf4()
    rag = MatchedReferenceRAG.from_config(root=ROOT, tokenizer=tokenizer, model=model, device=device)
    classifier = HierarchicalDiagnosisClassifier.load(
        ROOT / "models/diagnosis/diagnosis_v2_expanded_hierarchical.json"
    )

    results = []
    try:
        for sample in samples:
            item = rag.run(sample["query"], sample_id=sample["sample_id"], dataset="nq")
            diagnosis = classifier.diagnose(item)
            row = {
                "sample_id": item.sample_id,
                "role": sample["role"],
                "query": item.query,
                "top5_ok": len(item.evidence) == 5
                and [e.rank for e in item.evidence] == [1, 2, 3, 4, 5]
                and all(e.text.strip() for e in item.evidence)
                and all(e.score == e.score for e in item.evidence),
                "evidence_preview": [
                    {"rank": e.rank, "passage_id": e.passage_id, "score": e.score, "text": e.text[:120]}
                    for e in item.evidence
                ],
                "initial_answer": item.initial_answer,
                "initial_answer_ok": bool(item.initial_answer.strip()),
                "diagnosis_input_ok": True,
                "diagnosis_label": diagnosis.final_label.value,
                "diagnosis_confidence": diagnosis.confidence,
            }
            results.append(row)
            print(json.dumps({"phase": "sample", **row}, ensure_ascii=False, indent=2))
    finally:
        rag.close()
        free_cuda_memory()

    ok = all(r["top5_ok"] and r["initial_answer_ok"] and r["diagnosis_input_ok"] for r in results)
    print(json.dumps({"status": "SMOKE_OK" if ok else "SMOKE_FAIL", "n": len(results)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
