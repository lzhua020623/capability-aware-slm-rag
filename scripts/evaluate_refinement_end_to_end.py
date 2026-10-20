"""Does refinement actually fix the answer? EM / F1 before and after, on labelled samples.

This is the question the whole module exists for: on samples annotated
`evidence_accessibility` -- the needed information is in the frozen Top-5 but
the generator could not use it -- how many answers does reranking the evidence
and re-asking once turn from wrong to right?

Protocol
--------
* "before" is the C1 answer on the Top-5 in its original order, "after" the
  answer on the same passages reordered (and optionally pruned) by the refiner.
  Same model, same prompt text, same passages: only the ordering differs.
* With the frozen local generator (`--backend frozen`), "before" is the stored
  C1 answer, which came from exactly that setup. With any hosted backend it is
  **regenerated through the same endpoint**, because a hosted Qwen2.5-7B serves
  fp16/bf16 while the stored answers came from local 4-bit NF4: comparing one
  against the other would mix quantisation into the delta. Absolute numbers
  from a hosted run therefore will not match the frozen baseline; the
  before/after difference is what transfers.
* The Controller is bypassed and the action is set to `evidence_refinement` for
  every sample. That is deliberate: this measures what the module can do on
  the samples it is meant for, not how well Diagnosis routes to it. The
  end-to-end number through the real Controller is lower, because only a
  fraction of what it routes here is truly accessibility
  (`docs/DIAGNOSIS_PROBE_CONTROLLER_HANDOFF.md`, section F).
* Only samples from the held-out split are evaluated by default, so the
  reranker is never scored on what it trained on.
* Gold answers are read after the answers exist, for scoring only.

Run `--label correct` for the harm arm: the same pipeline on already-correct
samples, where every changed answer is a potential regression.

Usage:
    # dry run: whole pipeline, no model, no API key
    python scripts/evaluate_refinement_end_to_end.py --backend echo --limit 5

    # hosted Qwen2.5-7B (any OpenAI-compatible endpoint)
    $env:LLM_API_KEY = "..."
    python scripts/evaluate_refinement_end_to_end.py --backend openrouter \
        --generator-model qwen/qwen-2.5-7b-instruct --limit 200

    # the frozen local generator (4-bit NF4), on a machine that holds the weights
    python scripts/evaluate_refinement_end_to_end.py --backend frozen --limit 200
    python scripts/evaluate_refinement_end_to_end.py --backend frozen --label correct --limit 200
    python scripts/evaluate_refinement_end_to_end.py --backend frozen --frozen-500 \
        --records results/preliminary/nq_c1_7b.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.evidence_refinement import EvidenceRefiner, build_refinement_prompt  # noqa: E402
from src.compensation.rerank import CrossEncoderScorer  # noqa: E402
from src.compensation.schema import RefinementConfig  # noqa: E402
from src.config import load_generator_config  # noqa: E402
from src.controller.schema import ControllerAction, ControllerDecision  # noqa: E402
from src.diagnosis.labels import DiagnosisLabel  # noqa: E402
from src.diagnosis.schema import DatasetName, DiagnosisInput, EvidenceItem  # noqa: E402
from src.evaluation.qa import exact_match, token_f1  # noqa: E402
from src.generation.backend import build_backend  # noqa: E402

DEFAULT_RECORDS = ROOT / "data" / "processed" / "diagnosis" / "nq15_train_records.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "processed" / "rerank" / "test.jsonl"
DEFAULT_MODEL = "models/reranker/accessibility_v1"
DEFAULT_OUT = ROOT / "results" / "compensation" / "refinement_end_to_end.json"

ROUTED = ControllerDecision(
    action=ControllerAction.EVIDENCE_REFINEMENT,
    diagnosis_label=DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
    diagnosis_confidence=1.0,
    used_probe=False,
    probe_overrode_diagnosis=False,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_input(record: dict) -> DiagnosisInput:
    return DiagnosisInput(
        sample_id=record["sample_id"],
        dataset=DatasetName(record.get("dataset", "nq")),
        query=record["query"],
        evidence=[
            EvidenceItem(rank=passage["rank"], passage_id=passage["passage_id"], text=passage["text"], score=passage["score"])
            for passage in record["evidence"]
        ],
        initial_answer=record["initial_answer"],
    )


def build_refiner(args, config: RefinementConfig):
    """The refiner plus a flag saying whether the stored C1 answer is still comparable.

    `frozen` is the project's own generator (Qwen2.5-7B, 4-bit NF4, greedy) and
    produced the stored answers, so "before" can be read from the records. Any
    other backend is a different numeric setup, so "before" must be regenerated
    through it.
    """
    if args.backend == "frozen":
        return EvidenceRefiner.from_frozen_defaults(config=config), "frozen local Qwen2.5-7B NF4", False

    backend = build_backend(args.backend, args.generator_model, args.base_url, args.api_key)
    max_new_tokens = args.max_new_tokens

    def generate(prompt: str) -> str:
        text, _latency = backend.generate(prompt, max_new_tokens)
        return text

    scorer = CrossEncoderScorer.from_pretrained(config.reranker_model)
    return EvidenceRefiner(scorer, generate, config=config), backend.name, True


def frozen_500_records(path: Path, label_file: Path, label: str) -> list[dict]:
    """The frozen NQ C1 500 reshaped like the NQ_15 records, filtered to one annotation label."""
    labels = {str(row["sample_id"]): row["primary_label"] for row in read_jsonl(label_file)}
    records = []
    for row in read_jsonl(path):
        sample_id = str(row["sample_id"])
        if labels.get(sample_id) != label:
            continue
        records.append(
            {
                "sample_id": sample_id,
                "dataset": "nq",
                "query": row["question_or_claim"],
                "initial_answer": row["prediction"],
                "evidence": [
                    {"rank": index, "passage_id": passage["passage_id"], "text": passage["text"], "score": passage["score"]}
                    for index, passage in enumerate(row["evidence"], start=1)
                ],
                "primary_label": labels[sample_id],
                "annotation_reference": {"gold_answer": row["gold_answer_or_label"]},
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT, help="restrict to these sample_ids (held-out)")
    parser.add_argument("--frozen-500", action="store_true", help="read the frozen NQ C1 500 instead of NQ_15")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/annotations/normalized/nq_3216_labels_normalized.jsonl")
    parser.add_argument("--label", default="evidence_accessibility")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="reranker; a local path or a hub name")
    parser.add_argument("--keep-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument(
        "--backend",
        default="frozen",
        help="frozen (local Qwen2.5-7B NF4), echo (dry run), or any OpenAI-compatible provider "
        "(openrouter, together, fireworks, deepinfra, groq, local-vllm, ollama, ...)",
    )
    parser.add_argument("--generator-model", default=None, help="model id at the hosted endpoint")
    parser.add_argument("--base-url", default=None, help="override the provider base URL")
    parser.add_argument("--api-key", default=None, help="defaults to $LLM_API_KEY")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="defaults to configs/base.yaml")
    parser.add_argument(
        "--regenerate-before",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="re-ask the backend on the original order instead of reading the stored C1 answer "
        "(default: on for every backend except frozen)",
    )
    args = parser.parse_args()

    generator_config = load_generator_config()
    if args.max_new_tokens is None:
        args.max_new_tokens = int(generator_config.get("max_new_tokens", 64))
    if args.generator_model is None:
        args.generator_model = str(generator_config["model_name"])

    if args.frozen_500:
        records = frozen_500_records(args.records, args.labels, args.label)
        split_note = "frozen NQ C1 500"
    else:
        records = [row for row in read_jsonl(args.records) if row["primary_label"] == args.label]
        if args.split and args.split.exists():
            held_out = {row["sample_id"] for row in read_jsonl(args.split)}
            records = [row for row in records if row["sample_id"] in held_out]
            split_note = f"held-out split {args.split.name}"
        else:
            split_note = "all records (no split filter)"
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no records matched")
    print(f"{len(records)} {args.label} samples from {split_note}")

    config = RefinementConfig(keep_k=args.keep_k, reranker_model=args.model)
    print(f"loading reranker {config.reranker_model} and generator backend {args.backend}")
    refiner, generator_name, must_regenerate = build_refiner(args, config)
    regenerate_before = must_regenerate if args.regenerate_before is None else args.regenerate_before
    if must_regenerate and not regenerate_before:
        print("WARNING: comparing a hosted answer against the stored 4-bit NF4 answer mixes quantisation "
              "into the delta; --no-regenerate-before was asked for explicitly.")
    print(f"generator: {generator_name} | before answers: {'regenerated' if regenerate_before else 'stored C1'}")

    rows = []
    statuses: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    start = time.perf_counter()
    for index, record in enumerate(records, start=1):
        item = to_input(record)
        if regenerate_before:
            # Same backend, same frozen prompt, Top-5 in its original order.
            item = dataclasses.replace(item, initial_answer=refiner.generate(build_refinement_prompt(item, item.evidence)))
        result = refiner.run(item, ROUTED)
        gold = str((record.get("annotation_reference") or {}).get("gold_answer") or "")
        before_em, after_em = exact_match(item.initial_answer, gold), exact_match(result.final_answer, gold)
        before_f1, after_f1 = token_f1(item.initial_answer, gold), token_f1(result.final_answer, gold)
        statuses[result.status.value] += 1
        if result.reject_reason:
            rejects[result.reject_reason] += 1
        rows.append(
            {
                "sample_id": record["sample_id"],
                "status": result.status.value,
                "answer_changed": result.answer_changed,
                "evidence_changed": result.evidence_changed,
                "initial_answer": item.initial_answer,
                "final_answer": result.final_answer,
                "gold_answer": gold,
                "em_before": before_em,
                "em_after": after_em,
                "f1_before": round(before_f1, 4),
                "f1_after": round(after_f1, 4),
                "reject_reason": result.reject_reason,
            }
        )
        if index % 25 == 0 or index == len(records):
            elapsed = time.perf_counter() - start
            print(f"  {index}/{len(records)}  {elapsed / index:.2f}s/sample")

    total = len(rows)
    fixed = [row for row in rows if row["em_after"] > row["em_before"]]
    broken = [row for row in rows if row["em_after"] < row["em_before"]]
    summary = {
        "samples": total,
        "em_before": round(sum(row["em_before"] for row in rows) / total, 4),
        "em_after": round(sum(row["em_after"] for row in rows) / total, 4),
        "f1_before": round(sum(row["f1_before"] for row in rows) / total, 4),
        "f1_after": round(sum(row["f1_after"] for row in rows) / total, 4),
        "fixed": len(fixed),
        "broken": len(broken),
        "net_fixed": len(fixed) - len(broken),
        "answers_changed": sum(row["answer_changed"] for row in rows),
        "evidence_changed": sum(row["evidence_changed"] for row in rows),
        "status_distribution": dict(statuses),
        "reject_reasons": dict(rejects),
        "seconds_per_sample": round((time.perf_counter() - start) / total, 3),
    }

    payload = {
        "note": "Offline evaluation after the answers were produced. The Controller is bypassed: every sample "
        "is treated as routed to evidence_refinement, so this is the module's ceiling on its own target "
        "class, not an end-to-end pipeline number.",
        "label": args.label,
        "source": str(args.records.relative_to(ROOT)) if args.records.is_relative_to(ROOT) else str(args.records),
        "split": split_note,
        "generator": {
            "backend": args.backend,
            "name": generator_name,
            "model": args.generator_model if args.backend != "frozen" else str(generator_config["model_name"]),
            "max_new_tokens": args.max_new_tokens,
            "before_answers": "regenerated through the same backend" if regenerate_before else "stored frozen C1",
        },
        "config": config.to_dict(),
        "summary": summary,
    }
    if args.save_rows:
        payload["rows"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
