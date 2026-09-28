"""Larger-model control run: Llama-3.3-70B on the samples the 7B got wrong.

Reads the frozen-schema C0/C1/C3 result files produced by the 7B runs, selects
the failure subset, rebuilds each prompt from the `evidence` already recorded in
those files, and re-answers with a 70B model through any backend.

Because the 7B result files already carry `evidence` (Top-5 for C1, gold for C3)
this script needs no dataset, no FAISS index, and nothing under data/.

Selection bias: `--subset` conditions on the 7B being wrong, so the results
support "of the 7B's failures, X% are recovered" and NOT "the 70B is more
accurate". The missing cell is samples the 7B got right. `--control-n` draws a
random sample of those so the regression rate (7B right -> 70B wrong) can be
measured; without it, no claim about which model is better is supportable.

Examples
--------
Validate the whole pipeline with no compute and no API key:

    python scripts/run_llama70b_control.py --qwen-c1 results/preliminary/nq_c1_7b.jsonl \
        --qwen-c3 results/preliminary/nq_c3_7b.jsonl --dataset nq --backend echo

Real run against any OpenAI-compatible provider:

    $env:LLM_API_KEY = "..."
    python scripts/run_llama70b_control.py --qwen-c1 ... --qwen-c3 ... \
        --dataset nq --backend together \
        --model meta-llama/Llama-3.3-70B-Instruct-Turbo --control-n 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_preliminary_config
from src.evaluation.fever import parse_fever_prediction
from src.evaluation.qa import best_em_f1
from src.generation.backend import build_backend

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def safe_print(*args) -> None:
    """Windows consoles raise on non-cp1252 output; degrade instead of dying."""
    text = " ".join(str(arg) for arg in args)
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Missing 7B result file: {path}\n"
            "Produce the C0/C1/C3 runs first; this script only re-answers their failures."
        )
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_key(record: dict) -> str:
    for field in ("sample_id", "id"):
        if record.get(field) is not None:
            return str(record[field])
    raise RuntimeError(f"Result record has neither sample_id nor id: {record!r}")


def is_correct(record: dict) -> bool:
    """Read the frozen `correct` field, tolerating the smoke-test spellings."""
    for field in ("correct", "prediction_correct"):
        if field in record:
            return bool(record[field])
    if "em" in record:
        return bool(record["em"])
    raise RuntimeError(f"Result record has no correctness field: {sorted(record)}")


def gold_answers(record: dict) -> list[str]:
    """NQ gold may be one string or the full alias list; keep every alias."""
    gold = record.get("gold_answer_or_label", record.get("gold_answer"))
    if isinstance(gold, list):
        return [str(item) for item in gold if str(item).strip()]
    if isinstance(gold, str) and gold.strip():
        return [gold]
    return []


def evidence_texts(record: dict) -> list[str]:
    """Normalize `evidence` across the shapes the 7B runners may write."""
    evidence = record.get("evidence")
    if not evidence:
        return []
    if isinstance(evidence, str):
        return [evidence]
    texts = []
    for item in evidence:
        if isinstance(item, dict):
            text = item.get("text") or item.get("passage") or ""
        else:
            text = str(item)
        if text.strip():
            texts.append(text.strip())
    return texts


def build_prompt(
    instruction: str, question: str, passages: list[str], dataset: str
) -> str:
    """Mirror src.generation.qwen's layout so the 7B and 70B see the same shape."""
    label = "Evidence" if dataset == "fever" else "Context"
    tail = f"Claim: {question}" if dataset == "fever" else f"Question: {question}"
    if not passages:
        return f"{instruction}\n\n{tail}"
    blocks = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))
    return f"{instruction}\n\n{label}:\n{blocks}\n\n{tail}"


def score(prediction: str, record: dict, dataset: str) -> tuple[bool, dict]:
    if dataset == "fever":
        gold = str(record.get("gold_answer_or_label", record.get("gold_label", ""))).upper()
        predicted = parse_fever_prediction(prediction)
        return predicted == gold, {"parsed_prediction": predicted}
    golds = gold_answers(record)
    em, f1 = best_em_f1(prediction, golds)
    return bool(em), {"em": em, "f1": f1}


def _from_cohort(record: dict, condition: str) -> dict:
    """Present a cohort row in the shape the rest of this script expects.

    Cohort rows store the 7B's two predictions side by side and name the oracle
    evidence `oracle_evidence`; a condition result names them `prediction` and
    `evidence`. C1 evidence is not kept in a cohort row, so --condition C1 has to
    come from the C1 result file instead.
    """
    if condition == "C1":
        raise SystemExit(
            "--condition C1 needs the retrieved passages, which a cohort file does "
            "not store. Use --qwen-c1 for C1, or --condition C3 with the cohort."
        )
    return {
        **record,
        "evidence": record.get("oracle_evidence") or [],
        "prediction": record.get("c3_prediction"),
        "correct": bool(record.get("c3_correct")),
    }


def select_subset(
    c1_records: list[dict],
    c3_records: list[dict],
    rule: str,
) -> tuple[set[str], set[str]]:
    """Return (failure ids, ids the 7B answered correctly under C1)."""
    c1_wrong = {sample_key(r) for r in c1_records if not is_correct(r)}
    c1_right = {sample_key(r) for r in c1_records if is_correct(r)}
    if rule == "c1-wrong":
        return c1_wrong, c1_right
    if rule == "c1-and-c3-wrong":
        if not c3_records:
            raise SystemExit("--subset c1-and-c3-wrong needs --qwen-c3")
        c3_wrong = {sample_key(r) for r in c3_records if not is_correct(r)}
        return c1_wrong & c3_wrong, c1_right
    raise SystemExit(f"Unknown --subset {rule!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cohort",
        type=Path,
        help="A precomputed persistent-failure cohort jsonl (C1 wrong AND C3 wrong). "
        "Simplest path: it already carries the question, gold, and oracle evidence. "
        "Mutually exclusive with --qwen-c1/--qwen-c3.",
    )
    parser.add_argument("--qwen-c1", type=Path, help="7B C1 result jsonl")
    parser.add_argument("--qwen-c3", type=Path, help="7B C3 result jsonl")
    parser.add_argument("--dataset", choices=("nq", "fever"), required=True)
    parser.add_argument(
        "--condition",
        choices=("C0", "C1", "C3"),
        default="C3",
        help="Which condition the 70B runs under. C3 (gold evidence) isolates "
        "model capability from retrieval and matches the frozen 32B routing.",
    )
    parser.add_argument(
        "--subset",
        choices=("c1-wrong", "c1-and-c3-wrong"),
        default="c1-and-c3-wrong",
        help="Which failures to re-answer. Default matches the frozen 32B rule.",
    )
    parser.add_argument(
        "--control-n",
        type=int,
        default=0,
        help="Also run N randomly drawn samples the 7B got RIGHT, so the "
        "regression rate is measurable. Required for any 'the 70B is better' claim.",
    )
    parser.add_argument("--backend", default="echo", help="echo | local | together | fireworks | deepinfra | openrouter | groq | ollama | local-vllm | local-llamacpp")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", help="Override the backend's base URL")
    parser.add_argument("--api-key", help="Defaults to $LLM_API_KEY")
    parser.add_argument("--out", type=Path, help="Output jsonl (default: results/preliminary/<dataset>_<condition>_70b.jsonl)")
    parser.add_argument("--limit", type=int, help="Stop after N samples (smoke testing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_preliminary_config()
    max_new_tokens = int(config["primary_model"]["max_new_tokens"])
    instruction = config["prompts"][args.dataset][args.condition]

    correct_ids: set[str] = set()
    if args.cohort:
        if args.qwen_c1 or args.qwen_c3:
            raise SystemExit("--cohort cannot be combined with --qwen-c1/--qwen-c3")
        # A cohort file already carries question, gold, and oracle evidence, and
        # its rows are the failure set by construction.
        records = load_jsonl(args.cohort)
        by_id = {sample_key(r): _from_cohort(r, args.condition) for r in records}
        failure_ids = set(by_id)
        print(f"Cohort       : {args.cohort.name} ({len(failure_ids)} samples)")
        rules = {r.get("correctness_rule", "exact_match") for r in records}
        print(f"Correctness rule behind this cohort: {', '.join(sorted(rules))}")
    else:
        if not args.qwen_c1:
            raise SystemExit("Pass either --cohort or --qwen-c1 (with --qwen-c3)")
        c1_records = load_jsonl(args.qwen_c1)
        c3_records = load_jsonl(args.qwen_c3) if args.qwen_c3 else []
        failure_ids, correct_ids = select_subset(c1_records, c3_records, args.subset)
        # The record carrying the evidence for the condition we are running.
        source_records = c3_records if args.condition == "C3" else c1_records
        if not source_records:
            raise SystemExit(f"--condition {args.condition} needs the matching 7B result file")
        by_id = {sample_key(r): r for r in source_records}

    targets = [(sid, "failure") for sid in sorted(failure_ids) if sid in by_id]
    if args.control_n > 0:
        if not correct_ids:
            raise SystemExit("--control-n needs --qwen-c1/--qwen-c3; a cohort file has no correct samples")
        pool = sorted(sid for sid in correct_ids if sid in by_id)
        n = min(args.control_n, len(pool))
        chosen = random.Random(args.seed).sample(pool, n)
        targets += [(sid, "control") for sid in chosen]
        print(f"Control group: {n} of {len(pool)} samples the 7B answered correctly")
    if args.limit:
        targets = targets[: args.limit]

    out_path = args.out or (
        ROOT / "results" / "preliminary" / f"{args.dataset}_{args.condition.lower()}_70b.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out_path.exists():
        done = {sample_key(r) for r in load_jsonl(out_path)}
        print(f"Resuming: {len(done)} records already in {out_path}")

    backend = build_backend(args.backend, args.model, args.base_url, args.api_key)
    print(f"Backend      : {backend.name}")
    print(f"Dataset      : {args.dataset}   Condition: {args.condition}")
    if not args.cohort:
        print(f"Subset rule  : {args.subset}")
    print(f"7B failures  : {len(failure_ids)}")
    print(f"To run       : {len([t for t in targets if t[0] not in done])}")
    print(f"Output       : {out_path}\n")

    n_run = n_correct = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for sid, group in targets:
            if sid in done:
                continue
            record = by_id[sid]
            question = str(
                record.get("question_or_claim")
                or record.get("question")
                or record.get("claim")
                or ""
            )
            passages = [] if args.condition == "C0" else evidence_texts(record)
            prompt = build_prompt(instruction, question, passages, args.dataset)
            prediction, latency = backend.generate(prompt, max_new_tokens)
            correct, extra = score(prediction, record, args.dataset)

            out = {
                "sample_id": sid,
                "dataset": args.dataset,
                "condition": args.condition,
                "model_name": args.model,
                "group": group,
                "question_or_claim": question,
                "gold_answer_or_label": record.get(
                    "gold_answer_or_label", record.get("gold_label", record.get("gold_answer"))
                ),
                "evidence": passages,
                "prediction": prediction,
                "correct": correct,
                "latency": latency,
                "seven_b_correct": is_correct(record),
                **extra,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            n_run += 1
            n_correct += int(correct)
            safe_print(
                f"[{n_run}] {group:7s} {sid}  correct={correct}  "
                f"{latency:5.1f}s  pred={prediction[:60]!r}"
            )

    print()
    if n_run:
        print(f"Ran {n_run} samples, {n_correct} correct ({n_correct / n_run:.1%})")
    failures = [t for t in targets if t[1] == "failure"]
    if failures:
        print(
            f"Recovered {n_correct}/{len(failures)} of the 7B's failures "
            "(conditional on the 7B being wrong -- not an accuracy figure)"
        )
    if args.control_n == 0:
        print(
            "\nNOTE: no control group. These numbers cannot support a claim that "
            "the 70B is more accurate than the 7B -- rerun with --control-n to "
            "measure the regression rate."
        )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
