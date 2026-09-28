"""LLM-as-judge pass over NQ predictions that exact match scored wrong.

Why this exists: the frozen NQ metric is normalized Exact Match against a single
gold string, but the stored golds are often whole clauses (median 5 words, p75
11, max 41) and the NQ extractor joins tokens with spaces, so `P$C` becomes
`P $ C` and `(1983)` becomes `( 1983 )`. Among the 227 C1-wrong-and-C3-wrong
samples, 42% have the prediction contained in the gold or vice versa. Those are
scored as model failures but are arguably correct answers, and feeding them to a
larger-model control would attribute a scoring artifact to model capability.

This script asks a judge model whether each prediction actually answers the
question, so the real failure set can be separated from the metric artifacts.

Scope matters. The primary statistic is

    RFR = |C1 wrong AND C3 correct| / |C1 wrong|

and exact match decides both terms, so judging only the persistent-failure set
fixes the numerator while leaving the denominator inflated. Default scope is
every C1-wrong and every C3-wrong sample, which keeps the recomputed rate
internally consistent.

Judge validity: pass --validate-n to also judge samples exact match already
marked CORRECT. The judge should agree on nearly all of them; a low agreement
rate there means the judge is unusable and the run should be discarded.

Examples
--------
Dry run, no compute, no key:

    python scripts/judge_answers.py --c1 results/preliminary/nq_c1_7b.jsonl \
        --c3 results/preliminary/nq_c3_7b.jsonl --backend echo

Real run with Qwen2.5-32B through any OpenAI-compatible provider:

    $env:LLM_API_KEY = "..."
    python scripts/judge_answers.py --c1 ... --c3 ... \
        --backend deepinfra --model Qwen/Qwen2.5-32B-Instruct --validate-n 60
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.backend import build_backend

DEFAULT_JUDGE = "Qwen/Qwen2.5-32B-Instruct"
VERDICT_RE = re.compile(r"\b(CORRECT|INCORRECT)\b", re.IGNORECASE)

# Neutral on purpose: the judge is never told that exact match already scored
# the prediction wrong, so it cannot simply ratify the metric.
JUDGE_PROMPT = """You are grading one short-answer question against a reference answer.

Judge ONLY whether the prediction says the same thing as the reference. Do not \
use your own knowledge of the subject. A prediction that is true in the real \
world but states something the reference does not state is INCORRECT.

Mark CORRECT when the difference is only in form: wording, word order, \
capitalization, punctuation, spacing, verbosity, an abbreviation or full name \
of the same entity, a shorter span of the reference that still answers the \
question, or harmless extra words around the same content.

Mark INCORRECT when the prediction names a different entity, date, number, or \
fact than the reference, omits the information the question asks for, or says \
the answer is unavailable.

Question: {question}
Reference answer: {gold}
Predicted answer: {prediction}

Reply with exactly one word: CORRECT or INCORRECT."""


def safe_print(*args) -> None:
    text = " ".join(str(arg) for arg in args)
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing result file: {path}")
    text = path.read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_verdict(text: str) -> str:
    match = VERDICT_RE.search(text or "")
    return match.group(1).upper() if match else "UNPARSED"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--c1", type=Path, required=True, help="NQ C1 result jsonl")
    parser.add_argument("--c3", type=Path, required=True, help="NQ C3 result jsonl")
    parser.add_argument(
        "--scope",
        choices=("wrong-only", "persistent-only", "all"),
        default="wrong-only",
        help="wrong-only (default): every C1-wrong and C3-wrong sample, so RFR "
        "stays consistent. persistent-only: just C1-wrong AND C3-wrong. "
        "all: every sample in both files.",
    )
    parser.add_argument(
        "--validate-n",
        type=int,
        default=0,
        help="Also judge N samples exact match marked CORRECT, to measure judge "
        "agreement. Strongly recommended: without it the judge is unvalidated.",
    )
    parser.add_argument("--backend", default="echo")
    parser.add_argument("--model", default=DEFAULT_JUDGE)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "preliminary" / "nq_judge_verdicts.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conditions = {"C1": load_jsonl(args.c1), "C3": load_jsonl(args.c3)}
    c1_wrong = {str(r["sample_id"]) for r in conditions["C1"] if not r["correct"]}
    c3_wrong = {str(r["sample_id"]) for r in conditions["C3"] if not r["correct"]}

    targets: list[tuple[str, dict, str]] = []
    for condition, records in conditions.items():
        wrong = c1_wrong if condition == "C1" else c3_wrong
        for record in records:
            sid = str(record["sample_id"])
            if args.scope == "all":
                group = "wrong" if sid in wrong else "validation"
            elif args.scope == "persistent-only":
                if not (sid in c1_wrong and sid in c3_wrong):
                    continue
                group = "wrong"
            else:
                if sid not in wrong:
                    continue
                group = "wrong"
            targets.append((condition, record, group))

    if args.validate_n > 0 and args.scope != "all":
        rng = random.Random(args.seed)
        for condition, records in conditions.items():
            wrong = c1_wrong if condition == "C1" else c3_wrong
            pool = [r for r in records if str(r["sample_id"]) not in wrong]
            n = min(args.validate_n // 2, len(pool))
            targets += [(condition, r, "validation") for r in rng.sample(pool, n)]

    if args.limit:
        targets = targets[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if args.out.exists():
        done = {(r["condition"], str(r["sample_id"])) for r in load_jsonl(args.out)}
        print(f"Resuming: {len(done)} verdicts already in {args.out}")

    backend = build_backend(args.backend, args.model, args.base_url, args.api_key)
    n_wrong = sum(1 for _, _, g in targets if g == "wrong")
    n_val = len(targets) - n_wrong
    print(f"Judge        : {backend.name}")
    print(f"Scope        : {args.scope}")
    print(f"To judge     : {n_wrong} exact-match-wrong + {n_val} validation")
    print(f"Output       : {args.out}\n")

    counts: dict[str, int] = {}
    with args.out.open("a", encoding="utf-8") as handle:
        for condition, record, group in targets:
            sid = str(record["sample_id"])
            if (condition, sid) in done:
                continue
            prompt = JUDGE_PROMPT.format(
                question=record.get("question_or_claim", ""),
                gold=record.get("gold_answer_or_label", ""),
                prediction=record.get("prediction", ""),
            )
            raw, latency = backend.generate(prompt, 8)
            verdict = parse_verdict(raw)
            key = f"{group}/{verdict}"
            counts[key] = counts.get(key, 0) + 1
            handle.write(
                json.dumps(
                    {
                        "sample_id": sid,
                        "condition": condition,
                        "group": group,
                        "judge_model": args.model,
                        "question_or_claim": record.get("question_or_claim"),
                        "gold_answer_or_label": record.get("gold_answer_or_label"),
                        "prediction": record.get("prediction"),
                        "exact_match_correct": bool(record["correct"]),
                        "judge_verdict": verdict,
                        "judge_raw": raw,
                        "latency": latency,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            safe_print(f"  {condition} {group:10s} {sid:22s} {verdict}")

    print("\nverdict counts:")
    for key in sorted(counts):
        print(f"  {key:24s} {counts[key]}")
    val_total = sum(v for k, v in counts.items() if k.startswith("validation/"))
    val_agree = counts.get("validation/CORRECT", 0)
    if val_total:
        rate = val_agree / val_total
        print(f"\njudge agreement on exact-match-CORRECT samples: {val_agree}/{val_total} ({rate:.1%})")
        if rate < 0.9:
            print("  WARNING: the judge disagrees with exact match on samples it already")
            print("  scored correct. Inspect before trusting any verdict from this run.")
    else:
        print("\nNOTE: no validation group. Pass --validate-n to check the judge is sane.")
    print(f"\nWrote {args.out}")
    print("Recompute RFR with scripts/recompute_rfr.py once judging is complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
