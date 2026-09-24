"""Run explicitly selected frozen NQ preliminary conditions.

C3 reads the annotated paragraph extracted by ``scripts/prepare_nq.py`` directly
from the frozen controlled source. It never loads retrieval results or a FAISS
index. Results are appended and fsynced one at a time so an interrupted run can
resume without repeating valid sample IDs.

    SLM_RAG_CPU_OFFLOAD=1 python scripts/run_nq_preliminary.py --conditions C3 --check-only
    SLM_RAG_CPU_OFFLOAD=1 python scripts/run_nq_preliminary.py --conditions C3 --smoke-test
    SLM_RAG_CPU_OFFLOAD=1 python -u scripts/run_nq_preliminary.py --conditions C3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_base_config, load_preliminary_config
from src.evaluation.qa import best_em_f1, gold_answers_from_record

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DATASET = "natural_questions"
CONDITIONS = ("C0", "C1", "C3")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = []
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{number}. Back up the file and inspect "
                    "the damaged line before resuming; no results were overwritten."
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected an object at {path}:{number}")
            records.append(record)
        return records


def validate_frozen_settings(config: dict) -> None:
    base_generator = load_base_config()["generator"]
    primary = config["primary_model"]
    for key in (
        "model_name",
        "quantization",
        "load_in_4bit",
        "do_sample",
        "max_new_tokens",
    ):
        if base_generator.get(key) != primary.get(key):
            raise RuntimeError(
                f"Generator setting {key!r} differs between base and preliminary configs"
            )
    if (
        primary["model_name"] != "Qwen/Qwen2.5-7B-Instruct"
        or primary["quantization"] != "nf4"
        or primary["load_in_4bit"] is not True
        or primary["do_sample"] is not False
        or int(primary["max_new_tokens"]) != 64
        or int(config["seed"]) != 42
    ):
        raise RuntimeError("NQ preliminary settings differ from the frozen protocol")
    if int(config["retrieval"]["top_k"]) != 5:
        raise RuntimeError("The frozen experiment requires retrieval Top-5")
    samples = config["samples"]["nq"]
    if int(samples["seed"]) != 42 or int(samples["n"]) != 500:
        raise RuntimeError("The frozen NQ manifest settings require seed=42 and n=500")
    prompts = config["prompts"]["nq"]
    if prompts.get("C1") != prompts.get("C3"):
        raise RuntimeError("Frozen NQ C1 and C3 prompts must be identical")
    for condition in CONDITIONS:
        if not isinstance(prompts.get(condition), str) or not prompts[condition].strip():
            raise RuntimeError(f"Missing frozen NQ {condition} prompt")


def load_frozen_samples(config: dict) -> list[dict]:
    """Load manifest-ordered controlled NQ records and validate oracle fields."""
    sample_config = config["samples"]["nq"]
    manifest_path = ROOT / sample_config["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "dataset": DATASET,
        "seed": sample_config["seed"],
        "n": sample_config["n"],
        "source": sample_config["source"],
    }
    for key, expected in expected_metadata.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"NQ manifest {key} does not match frozen config")
    frozen_ids = [str(item) for item in manifest["ids"]]
    expected_n = int(sample_config["n"])
    if len(frozen_ids) != expected_n or len(set(frozen_ids)) != expected_n:
        raise RuntimeError(f"Expected {expected_n} unique frozen NQ IDs")

    source_path = ROOT / sample_config["source"]
    if not source_path.is_file():
        raise RuntimeError(
            f"Missing controlled NQ source: {source_path}. Run scripts/prepare_nq.py "
            "or copy the frozen processed file from the NQ C0/C1 environment."
        )
    records = load_jsonl(source_path)
    if len(records) != int(sample_config["source_size"]):
        raise RuntimeError(
            f"Expected {sample_config['source_size']} controlled NQ records, "
            f"found {len(records)}"
        )
    by_id: dict[str, dict] = {}
    for record in records:
        sample_id = str(record.get("id"))
        if sample_id in by_id:
            raise RuntimeError(f"Duplicate controlled NQ source ID {sample_id}")
        by_id[sample_id] = record

    samples: list[dict] = []
    problems: list[str] = []
    for sample_id in frozen_ids:
        record = by_id.get(sample_id)
        if record is None:
            problems.append(f"{sample_id}: missing from controlled source")
            continue
        for field in ("question", "short_answer", "gold_long_answer_text"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{sample_id}: invalid {field}")
        samples.append(record)
    if problems:
        raise RuntimeError(
            f"{len(problems)} frozen NQ samples failed validation:\n"
            + "\n".join(problems[:10])
        )
    return samples


def oracle_evidence(sample: dict) -> list[dict]:
    # prepare_nq.py extracts this exact text from the annotated paragraph-long
    # answer containing the retained short-answer span. prepare_retrieval_data.py
    # maps it unchanged to gold_evidence_text/gold_passage_id.
    return [{"text": sample["gold_long_answer_text"]}]


def load_c1_queries(samples: list[dict]) -> list[dict]:
    """Load C1-only passage IDs, verifying their gold text is the oracle text."""
    path = ROOT / "data/processed/retrieval/nq_dev.jsonl"
    if not path.is_file():
        raise RuntimeError(f"Missing NQ retrieval query file required only for C1: {path}")
    by_id: dict[str, dict] = {}
    for record in load_jsonl(path):
        sample_id = str(record["id"])
        if sample_id in by_id:
            raise RuntimeError(f"Duplicate NQ retrieval query ID {sample_id}")
        by_id[sample_id] = record
    queries = []
    for sample in samples:
        sample_id = str(sample["id"])
        query = by_id.get(sample_id)
        if query is None:
            raise RuntimeError(f"Frozen NQ ID {sample_id} missing from {path}")
        if (
            query.get("question") != sample["question"]
            or query.get("short_answer") != sample["short_answer"]
            or query.get("gold_evidence_text") != sample["gold_long_answer_text"]
            or not query.get("gold_passage_id")
        ):
            raise RuntimeError(
                f"NQ retrieval query {sample_id} does not match its controlled oracle record"
            )
        queries.append(query)
    return queries


def retrieve(config: dict, queries: list[dict]) -> list[list[dict]]:
    # Lazy imports keep C3 check-only independent of FAISS/model dependencies.
    from src.generation.qwen import free_cuda_memory
    from src.retrieval.e5_faiss import E5FaissRetriever

    retrieval = config["retrieval"]
    nq_config = retrieval["nq"]
    retriever = E5FaissRetriever(
        retrieval["embedding_model"],
        passage_prefix="",
        query_prefix=BGE_QUERY_INSTRUCTION,
    )
    retriever.load(ROOT / nq_config["index_path"], ROOT / nq_config["meta_path"])
    embeddings = retriever.encode_queries(
        [query["question"] for query in queries], batch_size=16
    )
    scores, indices = retriever.search(embeddings, top_k=int(retrieval["top_k"]))
    all_hits: list[list[dict]] = []
    for row_scores, row_indices in zip(scores, indices):
        hits = []
        for score, index in zip(row_scores, row_indices):
            faiss_id = int(index)
            if faiss_id < 0:
                continue
            item = retriever.metadata[faiss_id]
            hits.append(
                {
                    "passage_id": item["passage_id"],
                    "text": item["text"],
                    "score": float(score),
                }
            )
        all_hits.append(hits)
    retriever.unload()
    free_cuda_memory()
    return all_hits


def build_prompt(instruction: str, question: str, evidence: list[dict] | None) -> str:
    if evidence is None:
        return f"{instruction}\n\nQuestion: {question}"
    context = "\n\n".join(
        f"[{number}] {item['text']}" for number, item in enumerate(evidence, start=1)
    )
    return f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}"


def experiment_fingerprint(config: dict, samples: list[dict]) -> str:
    payload = {
        "primary_model": config["primary_model"],
        "seed": config["seed"],
        "prompt": config["prompts"]["nq"]["C3"],
        "samples": [
            {
                "id": sample["id"],
                "question": sample["question"],
                "short_answer": sample["short_answer"],
                "evidence": oracle_evidence(sample),
            }
            for sample in samples
        ],
        "code": {
            name: hashlib.sha256(
                (ROOT / name).read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
            for name in (
                "src/generation/qwen.py",
                "src/evaluation/qa.py",
                "scripts/run_nq_preliminary.py",
            )
        },
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def evidence_for(condition: str, sample: dict, c1_hits: list[dict] | None = None) -> list[dict]:
    if condition == "C0":
        return []
    if condition == "C3":
        return oracle_evidence(sample)
    if c1_hits is None:
        raise RuntimeError("C1 evidence was not retrieved")
    return c1_hits


def completed_ids(
    path: Path,
    condition: str,
    samples: list[dict],
    config: dict,
    fingerprint: str,
) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    by_id = {str(sample["id"]): sample for sample in samples}
    for record in load_jsonl(path):
        if record.get("dataset") != DATASET or record.get("condition") != condition:
            raise RuntimeError(f"Unexpected dataset/condition in existing result file {path}")
        sample_id = str(record.get("sample_id"))
        if sample_id in done:
            raise RuntimeError(f"Duplicate sample ID {sample_id} in {path}")
        sample = by_id.get(sample_id)
        if sample is None:
            raise RuntimeError(f"Non-frozen sample ID {sample_id} in {path}")
        expected = {
            "model_name": config["primary_model"]["model_name"],
            "question_or_claim": sample["question"],
            "gold_answer_or_label": sample["short_answer"],
        }
        if condition == "C3":
            expected.update(
                {
                    "experiment_fingerprint": fingerprint,
                    "evidence": oracle_evidence(sample),
                }
            )
        elif condition == "C0":
            expected["evidence"] = []
        for key, value in expected.items():
            if record.get(key) != value:
                raise RuntimeError(
                    f"Result {sample_id}: incompatible {key} in {path}; "
                    "archive incompatible results before resuming."
                )
        prediction = record.get("prediction")
        latency = record.get("latency")
        if not isinstance(prediction, str):
            raise TypeError(f"Invalid prediction for {sample_id} in {path}")
        em, f1 = best_em_f1(prediction, gold_answers_from_record(sample))
        if (
            record.get("em") != em
            or record.get("f1") != f1
            or record.get("correct") is not bool(em)
            or type(latency) not in (int, float)
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise RuntimeError(f"Invalid metrics for {sample_id} in {path}")
        done.add(sample_id)
    return done


def write_result(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            if existing.read(1) != b"\n":
                prefix = "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(prefix + json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize(path: Path) -> None:
    records = load_jsonl(path) if path.exists() else []
    n = len(records)
    em = sum(int(item["em"]) for item in records) / n if n else 0.0
    f1 = sum(float(item["f1"]) for item in records) / n if n else 0.0
    print(f"{path.name}: n={n}, Exact Match={em:.4f}, average F1={f1:.4f}")


def requested_cpu_offload() -> bool:
    value = os.environ.get("SLM_RAG_CPU_OFFLOAD", "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"", "0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        "SLM_RAG_CPU_OFFLOAD must be one of: 1/0, true/false, yes/no, on/off"
    )


def expected_nq_offload(config: dict, samples: list[dict]) -> bool | None:
    """Return the unanimous completed NQ C1 offload mode, when available."""
    path = ROOT / config["result_paths"]["nq_c1_7b"]
    if not path.is_file():
        return None
    frozen_ids = {str(sample["id"]) for sample in samples}
    records = load_jsonl(path)
    if len(records) != len(samples) or {str(r.get("sample_id")) for r in records} != frozen_ids:
        return None
    values = {r.get("cpu_offload") for r in records}
    if values <= {True, False} and len(values) == 1:
        return values.pop()
    raise RuntimeError("Completed NQ C1 results do not have one consistent cpu_offload mode")


def preflight(config: dict, samples: list[dict], conditions: list[str]) -> None:
    if len(samples) != 500:
        raise RuntimeError(f"Expected 500 frozen NQ samples, found {len(samples)}")
    if "C1" in conditions:
        nq_config = config["retrieval"]["nq"]
        required = [
            ROOT / nq_config["index_path"],
            ROOT / nq_config["meta_path"],
            ROOT / "data/processed/retrieval/nq_dev.jsonl",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("Missing C1 retrieval files:\n" + "\n".join(missing))
    if "C3" in conditions:
        expected = expected_nq_offload(config, samples)
        if expected is not None and requested_cpu_offload() is not expected:
            value = "1" if expected else "0"
            raise RuntimeError(
                "NQ C3 must match the completed NQ C1 cpu_offload mode. "
                f"Set SLM_RAG_CPU_OFFLOAD={value} and retry."
            )
    print("Preflight passed")
    print(f"Frozen NQ samples: {len(samples)} unique IDs")
    print(f"Conditions: {' '.join(conditions)}")
    if "C3" in conditions:
        print("C3 evidence: controlled gold_long_answer_text only (no retrieval)")


def run_paired_analysis(config: dict) -> None:
    from scripts.analyze_c1_c3_recoverability import analyze_dataset

    analyze_dataset("nq", config=config, root=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frozen NQ preliminary experiment (explicit conditions required)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-only", action="store_true", help="validate inputs without loading models"
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one frozen sample under requested conditions without formal writes",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        required=True,
        help="conditions to run; explicit selection is required",
    )
    args = parser.parse_args()
    args.conditions = list(dict.fromkeys(args.conditions))

    config = load_preliminary_config()
    validate_frozen_settings(config)
    samples = load_frozen_samples(config)
    fingerprint = experiment_fingerprint(config, samples)
    paths = {
        condition: ROOT
        / config["result_paths"][f"nq_{condition.lower()}_7b"]
        for condition in args.conditions
    }
    done = {} if args.smoke_test else {
        condition: completed_ids(
            paths[condition], condition, samples, config, fingerprint
        )
        for condition in args.conditions
    }
    preflight(config, samples, args.conditions)
    print(
        f"Generator: {config['primary_model']['model_name']}, 4-bit NF4, "
        f"greedy, max_new_tokens={config['primary_model']['max_new_tokens']}, "
        f"seed={config['seed']}"
    )
    print(f"Experiment fingerprint: {fingerprint}")
    if args.check_only:
        return 0

    c1_queries = load_c1_queries(samples) if "C1" in args.conditions else None
    if args.smoke_test:
        from transformers import set_seed

        from src.generation.qwen import generate_greedy, load_qwen_nf4

        c1_hits = retrieve(config, [c1_queries[0]])[0] if c1_queries else None
        set_seed(int(config["seed"]))
        primary = config["primary_model"]
        tokenizer, model = load_qwen_nf4(primary["model_name"])
        sample = samples[0]
        for condition in args.conditions:
            evidence = evidence_for(condition, sample, c1_hits)
            prompt = build_prompt(
                config["prompts"]["nq"][condition],
                sample["question"],
                None if condition == "C0" else evidence,
            )
            prediction, latency = generate_greedy(
                tokenizer,
                model,
                prompt,
                max_new_tokens=int(primary["max_new_tokens"]),
            )
            em, f1 = best_em_f1(prediction, gold_answers_from_record(sample))
            print(
                f"SMOKE {condition}: prediction={prediction!r}, EM={em}, "
                f"F1={f1:.3f}, latency={latency:.2f}s"
            )
        print("Smoke test passed; no formal result files were written.")
        return 0

    total = len(samples) * len(args.conditions)
    finished = sum(len(ids) for ids in done.values())
    if finished == total:
        print("All requested conditions are already complete; nothing to run.")
        for condition in args.conditions:
            summarize(paths[condition])
        if "C3" in args.conditions:
            run_paired_analysis(config)
        return 0
    if finished:
        print(f"Resuming: {finished}/{total} generations already saved")

    pending_c1 = "C1" in args.conditions and len(done["C1"]) < len(samples)
    all_c1_hits = retrieve(config, c1_queries) if pending_c1 else [None] * len(samples)

    from transformers import set_seed

    from src.generation.qwen import generate_greedy, load_qwen_nf4

    set_seed(int(config["seed"]))
    primary = config["primary_model"]
    tokenizer, model = load_qwen_nf4(primary["model_name"])
    offload = requested_cpu_offload()
    for position, (sample, c1_hits) in enumerate(zip(samples, all_c1_hits), start=1):
        sample_id = str(sample["id"])
        golds = gold_answers_from_record(sample)
        if not golds:
            raise RuntimeError(f"No gold answer for NQ sample {sample_id}")
        for condition in args.conditions:
            if sample_id in done[condition]:
                continue
            evidence = evidence_for(condition, sample, c1_hits)
            prompt = build_prompt(
                config["prompts"]["nq"][condition],
                sample["question"],
                None if condition == "C0" else evidence,
            )
            prediction, latency = generate_greedy(
                tokenizer, model, prompt, max_new_tokens=int(primary["max_new_tokens"])
            )
            em, f1 = best_em_f1(prediction, golds)
            record = {
                "sample_id": sample["id"],
                "dataset": DATASET,
                "condition": condition,
                "model_name": primary["model_name"],
                "cpu_offload": offload,
                "cpu_offload_modules": ["lm_head"] if offload else [],
                "question_or_claim": sample["question"],
                "gold_answer_or_label": golds[0],
                "evidence": evidence,
                "prediction": prediction,
                "correct": bool(em),
                "latency": latency,
                "em": em,
                "f1": f1,
            }
            if condition == "C3":
                record["experiment_fingerprint"] = fingerprint
            if condition == "C1":
                record["retrieved_top5_ids"] = [hit["passage_id"] for hit in evidence]
                gold_passage_id = c1_queries[position - 1]["gold_passage_id"]
                record["gold_evidence_in_top5"] = any(
                    hit["passage_id"] == gold_passage_id for hit in evidence
                )
            write_result(paths[condition], record)
            done[condition].add(sample_id)
            finished += 1
            print(
                f"[{finished}/{total}] sample={position}/{len(samples)} {condition} "
                f"EM={em} F1={f1:.3f} latency={latency:.2f}s"
            )

    for condition in args.conditions:
        summarize(paths[condition])
    if "C3" in args.conditions:
        run_paired_analysis(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
