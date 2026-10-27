"""Train the answer selector: does this answer follow from this evidence?

A binary cross-encoder over `(query + answer, evidence)`. At inference the
refiner scores every candidate answer for one sample and keeps the best, which
is why the headline metric here is **pick@1**: how often the correct answer
outscores the wrong answers *of the same sample*. Plain accuracy over shuffled
examples would flatter a model that only learns which samples are easy.

The query and the answer are placed in the first segment and truncation is
restricted to the evidence, so the thing being judged always survives. The
evidence is the Top-5 concatenated and cut to the length budget, so a passage
at rank 4-5 may be lost; that is a known limit of this first version.

Defaults fit a 6 GB GPU: `BAAI/bge-reranker-base`, max_length 384, batch 16,
fp16, 8-bit Adam when bitsandbytes is present.

The current CLI requires a team-role bundle with donor provenance, validates
every row before model loading, and reports validation metrics only. No test
file is read and existing model/output directories are preserved.

Usage:
    python scripts/train_answer_selector.py
    python scripts/train_answer_selector.py --max-train-examples 2000 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "team_nq_v1" / "selector"
DEFAULT_OUT_DIR = ROOT / "models" / "selector" / "answer_selector_team_v1"
DEFAULT_METRICS = ROOT / "results" / "compensation" / "team_nq_v1" / "selector_training_metrics.json"
DEFAULT_BASE_MODEL = "BAAI/bge-reranker-base"
ANSWER_MARKER = " [ANSWER] "
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def claim_text(row: dict) -> str:
    """What is being judged: the question together with the proposed answer."""
    return f"{row['query']}{ANSWER_MARKER}{row['answer']}"


def evidence_text(row: dict) -> str:
    return " ".join(row["evidence"])


class SelectorDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def collate(tokenizer, max_length: int):
    def _collate(batch: list[dict]):
        encoded = tokenizer(
            [claim_text(row) for row in batch],
            [evidence_text(row) for row in batch],
            padding=True,
            truncation="only_second",  # never cut the question or the answer
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([float(row["label"]) for row in batch])
        return encoded

    return _collate


def summarise_scores(rows: list[dict], scores: list[float]) -> dict:
    """pick@1 within each sample, plus plain accuracy at a 0 logit threshold.

    Only samples that actually pose a choice count towards pick@1: they need a
    correct answer and at least one competitor.
    """
    if len(rows) != len(scores):
        raise ValueError(f"{len(scores)} scores for {len(rows)} rows")
    by_sample: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_sample[row["sample_id"]].append((score, row["label"]))
    decidable = [items for items in by_sample.values() if any(label for _, label in items) and len(items) > 1]
    picked = sum(max(items, key=lambda item: item[0])[1] == 1 for items in decidable)
    correct = sum((score > 0) == bool(row["label"]) for row, score in zip(rows, scores))
    return {
        "examples": len(rows),
        "samples_with_a_choice": len(decidable),
        "pick_at_1": round(picked / len(decidable), 4) if decidable else None,
        "accuracy": round(correct / len(rows), 4) if rows else None,
    }


@torch.inference_mode()
def evaluate(model, tokenizer, rows: list[dict], device, max_length: int, batch_size: int) -> dict:
    """Score every row, then summarise."""
    model.eval()
    scores: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [claim_text(row) for row in batch],
            [evidence_text(row) for row in batch],
            padding=True,
            truncation="only_second",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**encoded).logits.squeeze(-1).float()
        scores.extend(logits.reshape(-1).tolist())
    return summarise_scores(rows, scores)


def build_optimizer(model, learning_rate: float):
    try:
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate), "adamw8bit"
    except Exception:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate), "adamw"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args(argv)

    from src.data.team_nq import TeamNQBoundary
    if args.out_dir.exists() or args.metrics_out.exists():
        raise FileExistsError("Preserve existing model/metrics; choose new output paths")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    parts, data_receipt = TeamNQBoundary(ROOT).read_training_bundle(args.data_dir, component='answer_selector')
    train_rows, validation_rows = parts["train"], parts["validation"]
    if args.max_train_examples:
        train_rows = train_rows[:args.max_train_examples]
    if args.eval_limit:
        validation_rows = validation_rows[:args.eval_limit]
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    print(f"device={device} model={args.model}")
    print(f"train {len(train_rows)} examples | validation {len(validation_rows)}")

    before = evaluate(model, tokenizer, validation_rows, device, args.max_length, args.eval_batch_size)
    print("before:", json.dumps(before))

    loader = DataLoader(
        SelectorDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate(tokenizer, args.max_length),
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer, optimizer_name = build_optimizer(model, args.learning_rate)
    total_steps = len(loader) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: step / max(warmup, 1)
        if step < warmup
        else max(0.0, (total_steps - step) / max(total_steps - warmup, 1)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(f"{total_steps} steps | optimizer={optimizer_name}")

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
            if step % 500 == 0 or step == len(loader):
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f}")
        validation = evaluate(model, tokenizer, validation_rows, device, args.max_length, args.eval_batch_size)
        history.append({"epoch": epoch, "train_loss": round(running / max(len(loader), 1), 4), "validation": validation})
        print(f"  epoch {epoch} validation pick@1={validation['pick_at_1']} accuracy={validation['accuracy']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    metrics = {
        "note": "pick@1 is the inference task: the correct answer must outscore the wrong answers of the "
        "same sample. Accuracy is reported only as a sanity check.",
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
            "train_examples": len(train_rows),
        },
        "data_boundary": data_receipt["boundary"],
        "training_role_files": data_receipt["role_files"],
        "test_evaluation": "not performed; separate final protocol required",
        "before_training": before,
        "history": history,
        "train_seconds": round(time.perf_counter() - start, 1),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved model to {args.out_dir}")
    print(f"wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
