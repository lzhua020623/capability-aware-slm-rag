"""Compare rerankers on the held-out accessibility split, with paired statistics.

Every model scores the same samples, so the comparison is paired: for each
sample we record the rank the model gives the answer-bearing passage inside
that sample's own frozen Top-5. A difference in rank-1 accuracy is then tested
with McNemar's exact test (the discordant pairs only) and a paired bootstrap
confidence interval, rather than being read off two independent percentages.

The first `--models` entry is the reference every other model is compared to.

Usage:
    python scripts/compare_rerank_models.py
    python scripts/compare_rerank_models.py --split validation --models BAAI/bge-reranker-base models/reranker/accessibility_v1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "team_nq_v1" / "rerank"
DEFAULT_OUT = ROOT / "results" / "compensation" / "team_nq_v1" / "rerank_model_comparison.json"
DEFAULT_MODELS = (
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-base",
    "models/reranker/accessibility_v1",
)
BOOTSTRAP_SAMPLES = 10000
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@torch.inference_mode()
def positive_ranks(model_name: str, examples: list[dict], max_length: int, device: torch.device) -> list[int]:
    """Rank of the answer-bearing passage per sample; ties count against it."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    model.eval()
    ranks: list[int] = []
    for example in examples:
        passages = [example["positive"]["text"]] + [negative["text"] for negative in example["negatives"]]
        encoded = tokenizer(
            [example["query"]] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            scores = model(**encoded).logits.squeeze(-1).float().tolist()
        if isinstance(scores, float):
            scores = [scores]
        ranks.append(1 + sum(score >= scores[0] for score in scores[1:]))
    del model
    torch.cuda.empty_cache()
    return ranks


def mcnemar_exact(reference: list[int], candidate: list[int]) -> dict:
    """Exact binomial test over the samples where exactly one model puts the positive first."""
    only_candidate = sum(c == 1 and r != 1 for r, c in zip(reference, candidate))
    only_reference = sum(r == 1 and c != 1 for r, c in zip(reference, candidate))
    discordant = only_candidate + only_reference
    if discordant == 0:
        return {"wins": 0, "losses": 0, "discordant": 0, "p_value": 1.0}
    # Two-sided exact binomial p under p=0.5.
    from math import comb

    smaller = min(only_candidate, only_reference)
    tail = sum(comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return {
        "wins": only_candidate,
        "losses": only_reference,
        "discordant": discordant,
        "p_value": round(min(1.0, 2 * tail), 6),
    }


def bootstrap_difference(reference: list[int], candidate: list[int], samples: int, seed: int) -> dict:
    """Paired bootstrap CI for the rank-1 accuracy difference (candidate - reference)."""
    rng = random.Random(seed)
    size = len(reference)
    indices = range(size)
    differences = []
    for _ in range(samples):
        picks = [rng.randrange(size) for _ in indices]
        differences.append(
            sum(candidate[i] == 1 for i in picks) / size - sum(reference[i] == 1 for i in picks) / size
        )
    differences.sort()
    return {
        "mean": round(mean(differences), 4),
        "ci95_low": round(differences[int(0.025 * samples)], 4),
        "ci95_high": round(differences[int(0.975 * samples)], 4),
    }


def summarise(ranks: list[int]) -> dict:
    total = len(ranks)
    counts: dict[int, int] = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    return {
        "rank1": round(sum(rank == 1 for rank in ranks) / total, 4),
        "mrr": round(sum(1 / rank for rank in ranks) / total, 4),
        "rank_distribution": {str(rank): counts[rank] for rank in sorted(counts)},
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from src.data.team_nq import TeamNQBoundary
    if args.out.exists():
        raise FileExistsError('Preserve existing comparison; choose a new output path')
    parts, data_receipt = TeamNQBoundary(ROOT).read_training_bundle(args.data_dir, component='reranker')
    examples = parts[args.split]
    if args.limit:
        examples = examples[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(examples)} {args.split} examples on {device}")

    ranks: dict[str, list[int]] = {}
    per_model: dict[str, dict] = {}
    for model_name in args.models:
        start = time.perf_counter()
        ranks[model_name] = positive_ranks(model_name, examples, args.max_length, device)
        per_model[model_name] = summarise(ranks[model_name])
        per_model[model_name]["seconds"] = round(time.perf_counter() - start, 1)
        print(f"  {model_name:40s} rank1={per_model[model_name]['rank1']} mrr={per_model[model_name]['mrr']}")

    reference_name = args.models[0]
    comparisons = {}
    for model_name in args.models[1:]:
        comparisons[model_name] = {
            "vs": reference_name,
            "rank1_difference": round(per_model[model_name]["rank1"] - per_model[reference_name]["rank1"], 4),
            "mcnemar": mcnemar_exact(ranks[reference_name], ranks[model_name]),
            "paired_bootstrap": bootstrap_difference(
                ranks[reference_name], ranks[model_name], args.bootstrap, SEED
            ),
        }
        print(f"  {model_name} vs {reference_name}: {json.dumps(comparisons[model_name], ensure_ascii=False)}")

    payload = {
        "note": "Paired comparison on held-out accessibility samples. rank is the position of the "
        "answer-bearing passage inside that sample's own frozen Top-5; ties count against it. "
        "This measures ordering only, not final answer accuracy.",
        "split": args.split,
        "data_boundary": data_receipt['boundary'],
        "examples": len(examples),
        "max_length": args.max_length,
        "bootstrap_samples": args.bootstrap,
        "seed": SEED,
        "models": per_model,
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
