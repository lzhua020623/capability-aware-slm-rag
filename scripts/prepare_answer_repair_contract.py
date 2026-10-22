"""New fictional engineering cases; never augment the real benchmark with them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.import_nq_train_preliminary import write_new
from src.diagnosis import DiagnosisInput, EvidenceItem


def request_from_case(case):
    if case.get("synthetic") is not True:
        raise ValueError("This interface experiment accepts fictional synthetic cases only")
    row = case["request"]
    return DiagnosisInput(sample_id=row["sample_id"], dataset=row["dataset"], query=row["query"],
                          initial_answer=row["initial_answer"],
                          evidence=[EvidenceItem(**item) for item in row["evidence"]])


def fixtures():
    cases = [
        ("nq", "keep_alias", "Who founded Northstar Studio?", "Mira Vale",
         ["Dr. Mira Vale founded Northstar Studio in 2003."], "Mira Vale"),
        ("nq", "entity", "Who plays Captain Rowan in Glass Harbor?", "Captain Rowan",
         ["Ilan Reed plays Captain Rowan in Glass Harbor."], "Ilan Reed"),
        ("nq", "date_vs_quantity", "How many episodes are in Zephyr's second season?", "14",
         ["Zephyr's second season contains nine episodes.", "The premiere aired on May 9. A separate series had 14 episodes."], "nine"),
        ("nq", "relation_direction", "Who received the Orion grant?", "Lena Shaw",
         ["Lena Shaw awarded the Orion grant to Omar Voss."], "Omar Voss"),
        ("nq", "scope", "Who led the Lumen Council in 2011?", "Bela Moss",
         ["Arin Holt led the Lumen Council from 2010 through 2012. Bela Moss succeeded Holt in 2013."], "Arin Holt"),
        ("nq", "insufficient", "Where was the fictional engineer Nera Quill born?", "Amber Bay",
         ["Nera Quill designed the Atlas Bridge. This record gives no birthplace."], "Amber Bay"),
        ("fever", "keep_supports", "Northstar Studio was founded in 2003.", "SUPPORTS",
         ["Northstar Studio was founded in 2003 by Mira Vale."], "SUPPORTS"),
        ("fever", "repair_refutes", "Glass Harbor was directed by Ilan Reed.", "SUPPORTS",
         ["Glass Harbor was directed by Nera Quill, not Ilan Reed."], "REFUTES"),
        ("fever", "repair_supports", "Omar Voss received the Orion grant.", "REFUTES",
         ["Omar Voss received the Orion grant from Lena Shaw."], "SUPPORTS"),
        ("fever", "missing_is_not_refutes", "Nera Quill was born in Amber Bay.", "SUPPORTS",
         ["Nera Quill designed the Atlas Bridge. The archive contains no birthplace."], "SUPPORTS"),
        ("fever", "conflict", "The Atlas Bridge opened in 2007.", "REFUTES",
         ["Archive A says the Atlas Bridge opened in 2007.", "Archive B says the Atlas Bridge opened in 2008. The disagreement is unresolved."], "REFUTES"),
        ("fever", "negation", "Every vessel in the Lumen fleet used diesel engines.", "SUPPORTS",
         ["One vessel in the Lumen fleet used a steam engine instead of diesel."], "REFUTES"),
    ]
    result = []
    for i, (dataset, kind, query, initial, passages, expected) in enumerate(cases, 1):
        # Five short passages reproduce the interface shape; filler is labelled
        # fictional/irrelevant. It is not a real retrieval or augmentation result.
        passages = [*passages, *[f"Unrelated archive note {j}: the Nacre museum stores ceramic bowls."
                                 for j in range(len(passages) + 1, 6)]]
        result.append({"synthetic": True, "case_id": f"S{i:02d}", "failure_mode": kind,
            "request": {"sample_id": f"synthetic-contract-{i:02d}", "dataset": dataset,
                "query": query, "initial_answer": initial, "evidence": [
                    {"rank": rank, "passage_id": f"fictional_{i}_{rank}", "text": passage, "score": 1. / rank}
                    for rank, passage in enumerate(passages, 1)]},
            "offline_expected_final": expected,
            "scope": "synthetic engineering only; no training, model accuracy or benchmark claim"})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/answer_repair/v9_contract_v1/synthetic_cases.jsonl")
    args = p.parse_args()
    raw = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in fixtures())
    write_new(args.out, raw)
    print(json.dumps({"synthetic_cases": len(fixtures()), "real_benchmark_cases": 0, "model_training": False}))


if __name__ == "__main__":
    main()
