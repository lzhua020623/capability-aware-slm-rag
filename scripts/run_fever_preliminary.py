"""Run the frozen 500-sample FEVER C0/C3 preliminary experiment.

C0 gives the model no evidence. C3 gives the frozen gold evidence sentences
from fever_dev.jsonl. Every generation is appended to its result file as soon
as it finishes, so an interrupted run resumes by re-running the same command.
Model, prompts, decoding, sample IDs and output paths come from
configs/preliminary.yaml and are validated before the model is loaded.

    python scripts/run_fever_preliminary.py --check-only   # no model needed
    python scripts/run_fever_preliminary.py --smoke-test   # 1 claim, no writes
    python scripts/run_fever_preliminary.py                # C0 + C3, resumable
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_base_config, load_preliminary_config
from src.evaluation.fever import parse_fever_prediction
from src.generation.qwen import (
    FEVER_PROMPT_INSTRUCTION,
    build_fever_rag_prompt,
    generate_greedy,
    load_qwen_nf4,
)

try:  # Added by feature/nq-c0-c1; absent on main, where offload is never used.
    from src.generation.qwen import cpu_offload_enabled
except ImportError:  # pragma: no cover

    def cpu_offload_enabled() -> bool:
        return False


DATASET = "fever"
CONDITIONS = ("C0", "C3")
LABELS = ("SUPPORTS", "REFUTES")


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_frozen_settings(config: dict) -> None:
    """load_qwen_nf4 reads configs/base.yaml, so both configs must agree."""
    base_generator = load_base_config()["generator"]
    primary = config["primary_model"]
    for key in ("model_name", "quantization", "load_in_4bit", "do_sample", "max_new_tokens"):
        if base_generator.get(key) != primary.get(key):
            raise RuntimeError(
                f"Generator setting {key!r} differs between configs/base.yaml "
                "and configs/preliminary.yaml"
            )
    if primary["do_sample"] is not False:
        raise RuntimeError("The frozen experiment requires deterministic decoding")
    prompts = config["prompts"][DATASET]
    for condition in CONDITIONS:
        if not str(prompts.get(condition, "")).strip():
            raise RuntimeError(f"Missing frozen FEVER {condition} prompt in configs/preliminary.yaml")
    # C3 reuses build_fever_rag_prompt so C1 and C3 share one evidence format.
    if prompts["C3"] != FEVER_PROMPT_INSTRUCTION:
        raise RuntimeError(
            "Frozen FEVER C3 prompt in configs/preliminary.yaml does not match "
            "FEVER_PROMPT_INSTRUCTION in src/generation/qwen.py"
        )


def load_frozen_samples(config: dict) -> list[dict]:
    sample_config = config["samples"][DATASET]
    manifest_path = ROOT / sample_config["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest["samples"]
    expected_n = int(sample_config["n"])
    ids = [str(item["id"]) for item in items]
    if len(ids) != expected_n or len(set(ids)) != expected_n:
        raise RuntimeError(f"Manifest must contain {expected_n} unique FEVER IDs, found {len(ids)}")
    label_counts = Counter(item["label"] for item in items)
    expected_counts = {
        "SUPPORTS": int(sample_config["supports"]),
        "REFUTES": int(sample_config["refutes"]),
    }
    if dict(label_counts) != expected_counts:
        raise RuntimeError(
            f"Manifest label counts {dict(label_counts)} do not match frozen {expected_counts}"
        )

    source_path = ROOT / sample_config["source"]
    if not source_path.is_file():
        raise RuntimeError(
            f"Missing FEVER source file: {source_path}. Copy it from a machine that has "
            "run scripts/prepare_retrieval_data.py; C0/C3 need no other data."
        )
    by_id = {str(record["id"]): record for record in load_jsonl(source_path)}

    samples: list[dict] = []
    problems: list[str] = []
    for item in items:
        sample_id = str(item["id"])
        record = by_id.get(sample_id)
        if record is None:
            problems.append(f"{sample_id}: missing from {source_path.name}")
            continue
        if record.get("label") != item["label"]:
            problems.append(
                f"{sample_id}: manifest label {item['label']} != source label {record.get('label')}"
            )
        gold = [str(s) for s in (record.get("gold_evidence_text") or []) if str(s).strip()]
        gold_pages = [str(p) for p in (record.get("gold_page_ids") or [])]
        if not gold or not gold_pages:
            problems.append(f"{sample_id}: no valid gold evidence for C3")
        samples.append(
            {
                "id": item["id"],
                "claim": record["claim"],
                "label": item["label"],
                "gold_page_ids": gold_pages,
                "gold_evidence_text": gold,
            }
        )
    if problems:
        shown = "\n".join(problems[:10])
        raise RuntimeError(f"{len(problems)} frozen FEVER samples failed validation:\n{shown}")
    return samples


def build_prompt(config: dict, condition: str, sample: dict) -> str:
    if condition == "C0":
        instruction = config["prompts"][DATASET]["C0"]
        return f"{instruction}\n\nClaim: {sample['claim']}"
    return build_fever_rag_prompt(sample["claim"], evidence_for(condition, sample))


def evidence_for(condition: str, sample: dict) -> list[dict]:
    if condition == "C0":
        return []
    return [{"text": sentence} for sentence in sample["gold_evidence_text"]]


def completed_ids(path: Path, condition: str, frozen_ids: set[str]) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for record in load_jsonl(path):
        if record.get("dataset") != DATASET or record.get("condition") != condition:
            raise RuntimeError(f"Unexpected dataset/condition in existing result file {path}")
        sample_id = str(record["sample_id"])
        if sample_id in done:
            raise RuntimeError(f"Duplicate sample ID {sample_id} in {path}")
        if sample_id not in frozen_ids:
            raise RuntimeError(f"Non-frozen sample ID {sample_id} in {path}")
        done.add(sample_id)
    return done


def write_result(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def summarize(path: Path) -> None:
    records = load_jsonl(path) if path.exists() else []
    n = len(records)
    if not n:
        print(f"{path.name}: n=0")
        return
    accuracy = sum(1 for r in records if r["correct"]) / n
    unknown = sum(1 for r in records if r["prediction"] == "UNKNOWN")
    parts = [f"{path.name}: n={n}", f"accuracy={accuracy:.4f}", f"unknown={unknown}"]
    for label in LABELS:
        subset = [r for r in records if r["gold_answer_or_label"] == label]
        if subset:
            label_acc = sum(1 for r in subset if r["correct"]) / len(subset)
            parts.append(f"{label}_acc={label_acc:.4f} (n={len(subset)})")
    print(", ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen FEVER 500 C0/C3 preliminary experiment")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate config, manifest and source data without loading the model",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run the first frozen claim under each condition without writing formal results",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="conditions to run (default: C0 C3)",
    )
    args = parser.parse_args()

    config = load_preliminary_config()
    validate_frozen_settings(config)
    samples = load_frozen_samples(config)
    primary = config["primary_model"]
    max_new_tokens = int(primary["max_new_tokens"])
    label_counts = Counter(sample["label"] for sample in samples)
    print("Preflight passed")
    print(
        f"Frozen FEVER samples: {len(samples)} "
        f"(SUPPORTS={label_counts['SUPPORTS']}, REFUTES={label_counts['REFUTES']})"
    )
    print(f"Generator: {primary['model_name']}, 4-bit NF4, greedy, max_new_tokens={max_new_tokens}")
    print(f"Conditions: {' '.join(args.conditions)}")
    if args.check_only:
        return 0

    if args.smoke_test:
        tokenizer, model = load_qwen_nf4(primary["model_name"])
        sample = samples[0]
        for condition in args.conditions:
            prompt = build_prompt(config, condition, sample)
            raw, latency = generate_greedy(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
            prediction = parse_fever_prediction(raw)
            safe_print(
                f"SMOKE {condition}: gold={sample['label']} pred={prediction} "
                f"correct={prediction == sample['label']} latency={latency:.2f}s raw={raw!r}"
            )
        print("Smoke test passed; no formal result files were written.")
        return 0

    result_paths = config["result_paths"]
    paths = {c: ROOT / result_paths[f"fever_{c.lower()}_7b"] for c in args.conditions}
    frozen_ids = {str(sample["id"]) for sample in samples}
    done = {c: completed_ids(paths[c], c, frozen_ids) for c in args.conditions}
    total = len(samples) * len(args.conditions)
    finished = sum(len(ids) for ids in done.values())
    if finished == total:
        print("All requested conditions are already complete; nothing to run.")
        for condition in args.conditions:
            summarize(paths[condition])
        return 0
    if finished:
        print(f"Resuming: {finished}/{total} generations already saved")

    tokenizer, model = load_qwen_nf4(primary["model_name"])
    offload = cpu_offload_enabled()
    for position, sample in enumerate(samples, start=1):
        sample_id = str(sample["id"])
        for condition in args.conditions:
            if sample_id in done[condition]:
                continue
            prompt = build_prompt(config, condition, sample)
            raw, latency = generate_greedy(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
            prediction = parse_fever_prediction(raw)
            correct = prediction == sample["label"]
            record = {
                "sample_id": sample["id"],
                "dataset": DATASET,
                "condition": condition,
                "model_name": primary["model_name"],
                "cpu_offload": offload,
                "question_or_claim": sample["claim"],
                "gold_answer_or_label": sample["label"],
                "evidence": evidence_for(condition, sample),
                "gold_page_ids": sample["gold_page_ids"],
                "raw_output": raw,
                "prediction": prediction,
                "correct": correct,
                "latency": latency,
            }
            write_result(paths[condition], record)
            done[condition].add(sample_id)
            finished += 1
            print(
                f"[{finished}/{total}] sample={position}/{len(samples)} {condition} "
                f"gold={sample['label']} pred={prediction} correct={correct} latency={latency:.2f}s"
            )

    for condition in args.conditions:
        summarize(paths[condition])
    return 0


if __name__ == "__main__":
    sys.exit(main())
