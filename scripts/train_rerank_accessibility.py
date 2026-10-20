"""Fine-tune the evidence reranker on the `evidence_accessibility` annotations.

A binary cross-encoder: one (query, passage) pair in, one relevance logit out,
trained with BCE against the positives and negatives built by
`scripts/build_rerank_training_data.py`. The saved model is a drop-in for
`CrossEncoderScorer.from_pretrained(<path>)`, so nothing in the refinement
interface changes -- only `RefinementConfig.reranker_model`.

Only accessibility samples are trained on, as commissioned. That is a
deliberately narrow signal: the model learns to pull the answer-bearing
passage above the distractors that outranked it. Whether it also hurts
already-correct samples is not knowable from this training set and is measured
separately on the frozen Dev 500 by
`scripts/evaluate_rerank_accessibility.py`.

Metrics on held-out examples: `rank1` is how often the positive outscores every
distractor of the same sample, `mrr` the mean reciprocal rank of the positive.
Both are computed per sample over its own Top-5, which is exactly what the
refiner does at runtime.

Defaults are sized for a 6 GB GPU: `BAAI/bge-reranker-base`, max_length 320,
batch 16, fp16, 8-bit Adam when bitsandbytes is available.

Usage:
    python scripts/train_rerank_accessibility.py
    python scripts/train_rerank_accessibility.py --epochs 2 --baseline BAAI/bge-reranker-large
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "rerank"
DEFAULT_OUT_DIR = ROOT / "models" / "reranker" / "accessibility_v1"
DEFAULT_METRICS = ROOT / "results" / "compensation" / "rerank_training_metrics.json"
DEFAULT_BASE_MODEL = "BAAI/bge-reranker-base"
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    """Flattened (query, passage, label) pairs from the grouped examples."""

    def __init__(self, examples: list[dict]):
        self.pairs: list[tuple[str, str, float]] = []
        for example in examples:
            self.pairs.append((example["query"], example["positive"]["text"], 1.0))
            for negative in example["negatives"]:
                self.pairs.append((example["query"], negative["text"], 0.0))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str, float]:
        return self.pairs[index]


def collate(tokenizer, max_length: int):
    def _collate(batch):
        queries, texts, labels = zip(*batch)
        encoded = tokenizer(
            list(queries), list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.float)
        return encoded

    return _collate


@torch.inference_mode()
def score_examples(model, tokenizer, examples: list[dict], device, max_length: int, batch_size: int) -> dict:
    """Per-sample rank-1 accuracy and MRR of the positive inside its own Top-5."""
    model.eval()
    rank1 = 0
    reciprocal = 0.0
    rank_counts: dict[int, int] = {}
    for start in range(0, len(examples), batch_size):
        for example in examples[start : start + batch_size]:
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
            positive_score = scores[0]
            # Ties count against the positive: strictly better distractors plus ties.
            rank = 1 + sum(score >= positive_score for score in scores[1:])
            rank1 += rank == 1
            reciprocal += 1.0 / rank
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
    total = max(len(examples), 1)
    return {
        "examples": len(examples),
        "rank1": round(rank1 / total, 4),
        "mrr": round(reciprocal / total, 4),
        "rank_distribution": {str(rank): rank_counts[rank] for rank in sorted(rank_counts)},
    }


def build_optimizer(model, learning_rate: float) -> tuple[torch.optim.Optimizer, str]:
    try:
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate), "adamw8bit"
    except Exception:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate), "adamw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--baseline", default=None, help="also score this off-the-shelf model for comparison")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-pairs", type=int, default=None, help="cap the training pairs (smoke runs)")
    parser.add_argument("--eval-limit", type=int, default=None, help="cap the evaluated examples")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_examples = read_jsonl(args.data_dir / "train.jsonl")
    validation_examples = read_jsonl(args.data_dir / "validation.jsonl")
    test_examples = read_jsonl(args.data_dir / "test.jsonl")
    if args.eval_limit:
        validation_examples = validation_examples[: args.eval_limit]
        test_examples = test_examples[: args.eval_limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)

    print(f"device={device} model={args.model}")
    print(f"train {len(train_examples)} examples | validation {len(validation_examples)} | test {len(test_examples)}")

    before = {
        "validation": score_examples(model, tokenizer, validation_examples, device, args.max_length, args.eval_batch_size),
        "test": score_examples(model, tokenizer, test_examples, device, args.max_length, args.eval_batch_size),
    }
    print("before fine-tuning:", json.dumps({key: {k: v for k, v in value.items() if k != 'rank_distribution'} for key, value in before.items()}))

    dataset = PairDataset(train_examples)
    if args.max_train_pairs:
        dataset.pairs = dataset.pairs[: args.max_train_pairs]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate(tokenizer, args.max_length),
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer, optimizer_name = build_optimizer(model, args.learning_rate)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: step / max(warmup_steps, 1) if step < warmup_steps else max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"{len(dataset)} training pairs | {total_steps} steps | optimizer={optimizer_name}")
    start = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch).logits.squeeze(-1)
                loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 200 == 0 or step == len(loader):
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f}")
        validation = score_examples(model, tokenizer, validation_examples, device, args.max_length, args.eval_batch_size)
        history.append({"epoch": epoch, "train_loss": round(running / max(len(loader), 1), 4), "validation": validation})
        print(f"  epoch {epoch} validation rank1={validation['rank1']} mrr={validation['mrr']}")
    elapsed = time.perf_counter() - start

    after = {
        "validation": history[-1]["validation"] if history else before["validation"],
        "test": score_examples(model, tokenizer, test_examples, device, args.max_length, args.eval_batch_size),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    baseline = None
    if args.baseline:
        del model
        torch.cuda.empty_cache()
        baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline)
        baseline_model = AutoModelForSequenceClassification.from_pretrained(args.baseline, num_labels=1).to(device)
        baseline = {
            "model": args.baseline,
            "test": score_examples(baseline_model, baseline_tokenizer, test_examples, device, args.max_length, args.eval_batch_size),
        }
        print("baseline:", json.dumps({k: v for k, v in baseline["test"].items() if k != "rank_distribution"}))

    metrics = {
        "note": "Trained on evidence_accessibility annotations only. rank1/mrr are per-sample over that "
        "sample's own frozen Top-5; they measure ordering, not final answer accuracy.",
        "base_model": args.model,
        "output_dir": str(args.out_dir.relative_to(ROOT)) if args.out_dir.is_relative_to(ROOT) else str(args.out_dir),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "optimizer": optimizer_name,
            "seed": SEED,
            "train_pairs": len(dataset),
        },
        "data": {
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "test_examples": len(test_examples),
        },
        "before_fine_tuning": before,
        "after_fine_tuning": after,
        "history": history,
        "off_the_shelf_baseline": baseline,
        "train_seconds": round(elapsed, 1),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"before": before["test"]["rank1"], "after": after["test"]["rank1"], "mrr": after["test"]["mrr"]}, indent=2))
    print(f"saved model to {args.out_dir}")
    print(f"wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
