"""Run the frozen FEVER500 C1 experiment against the full formal index.

The runner never uses the earlier 100k smoke pool.  It retrieves Top-5 from
the frozen 5,396,106-passage IndexIVFPQ, saves an atomic retrieval checkpoint,
unloads the retriever, and then runs Qwen2.5-7B-Instruct in 4-bit NF4.  Formal
results are appended and fsynced one record at a time for safe resumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_base_config, load_preliminary_config
from src.evaluation.fever import gold_evidence_in_top5, parse_fever_prediction

DATASET = "fever"
CONDITION = "C1"
LABELS = ("SUPPORTS", "REFUTES")
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RETRIEVAL_CHECKPOINT = "results/preliminary/fever_c1_top5.jsonl"


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{number}. Back up and inspect the file; "
                    "nothing was overwritten."
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected an object at {path}:{number}")
            records.append(record)
    return records


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_frozen_settings(config: dict) -> None:
    base = load_base_config()["generator"]
    primary = config["primary_model"]
    for key in ("model_name", "quantization", "load_in_4bit", "do_sample", "max_new_tokens"):
        if base.get(key) != primary.get(key):
            raise RuntimeError(f"Generator setting {key!r} differs between frozen configs")
    if (
        primary["model_name"] != "Qwen/Qwen2.5-7B-Instruct"
        or primary["quantization"] != "nf4"
        or primary["load_in_4bit"] is not True
        or primary["do_sample"] is not False
        or int(primary["max_new_tokens"]) != 64
        or int(config["seed"]) != 42
    ):
        raise RuntimeError("FEVER C1 generator settings differ from the frozen protocol")
    retrieval = config["retrieval"]
    fever = retrieval["fever"]
    expected = {
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "top_k": 5,
        "normalize_embeddings": True,
        "metric": "inner_product",
    }
    for key, value in expected.items():
        actual = retrieval.get(key)
        if actual != value:
            raise RuntimeError(f"Frozen retrieval {key} must be {value!r}, got {actual!r}")
    for key, value in {
        "index_type": "faiss_ivfpq",
        "expected_corpus_size": 5_396_106,
        "nlist": 4096,
        "m": 64,
        "nbits": 8,
        "nprobe": 16,
        "metric": "inner_product",
        "seed": 42,
    }.items():
        if fever.get(key) != value:
            raise RuntimeError(f"Frozen FEVER retrieval {key} must be {value!r}")
    prompt = config["prompts"][DATASET].get(CONDITION)
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Missing frozen FEVER C1 prompt")


def load_frozen_samples(config: dict) -> list[dict]:
    sample_cfg = config["samples"][DATASET]
    manifest = load_json(ROOT / sample_cfg["manifest"])
    for key in ("dataset", "seed", "n", "supports", "refutes", "source"):
        expected = DATASET if key == "dataset" else sample_cfg[key]
        if manifest.get(key) != expected:
            raise RuntimeError(f"FEVER manifest {key} does not match frozen config")
    items = manifest.get("samples")
    if not isinstance(items, list):
        raise TypeError("FEVER manifest samples must be a list")
    ids = [str(item["id"]) for item in items]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise RuntimeError("FEVER manifest must contain exactly 500 unique IDs")
    counts = Counter(item.get("label") for item in items)
    if counts != Counter({"SUPPORTS": 250, "REFUTES": 250}):
        raise RuntimeError(f"FEVER manifest label balance is invalid: {dict(counts)}")

    source = ROOT / sample_cfg["source"]
    if not source.is_file():
        raise RuntimeError(f"Missing frozen FEVER query source: {source}")
    by_id: dict[str, dict] = {}
    for record in load_jsonl(source):
        sample_id = str(record.get("id"))
        if sample_id in by_id:
            raise RuntimeError(f"Duplicate FEVER source ID {sample_id}")
        by_id[sample_id] = record
    samples = []
    problems = []
    for item in items:
        sample_id = str(item["id"])
        record = by_id.get(sample_id)
        if record is None:
            problems.append(f"{sample_id}: missing from source")
            continue
        if record.get("label") != item["label"]:
            problems.append(f"{sample_id}: label mismatch")
        if not isinstance(record.get("claim"), str) or not record["claim"].strip():
            problems.append(f"{sample_id}: invalid claim")
        gold_pages = record.get("gold_page_ids")
        gold_text = record.get("gold_evidence_text")
        if not isinstance(gold_pages, list) or not gold_pages:
            problems.append(f"{sample_id}: invalid gold_page_ids")
        if not isinstance(gold_text, list) or not gold_text:
            problems.append(f"{sample_id}: invalid gold_evidence_text")
        samples.append(
            {
                "id": item["id"],
                "claim": record.get("claim"),
                "label": item["label"],
                "gold_page_ids": gold_pages,
            }
        )
    if problems:
        raise RuntimeError(
            f"{len(problems)} frozen FEVER samples failed validation:\n"
            + "\n".join(problems[:10])
        )
    return samples


def artifact_paths(config: dict) -> dict[str, Path]:
    fever = config["retrieval"]["fever"]
    return {
        "corpus": ROOT / fever["corpus"],
        "index": ROOT / fever["index_path"],
        "ids": ROOT / fever["ids_path"],
        "metadata": ROOT / fever["metadata_path"],
        "checkpoint": ROOT / RETRIEVAL_CHECKPOINT,
        "results": ROOT / config["result_paths"]["fever_c1_7b"],
    }


def validate_index_metadata(config: dict, paths: dict[str, Path]) -> dict:
    missing = [str(path) for key, path in paths.items() if key not in {"checkpoint", "results"} and not path.is_file()]
    if missing:
        raise RuntimeError("Missing formal FEVER C1 artifacts:\n" + "\n".join(missing))
    meta = load_json(paths["metadata"])
    fever = config["retrieval"]["fever"]
    expected = {
        "corpus": fever["corpus"],
        "corpus_size": int(fever["expected_corpus_size"]),
        "embedding_model": config["retrieval"]["embedding_model"],
        "normalization": "l2",
        "faiss_index_type": "IndexIVFPQ",
        "metric": "inner_product",
        "nlist": int(fever["nlist"]),
        "m": int(fever["m"]),
        "nbits": int(fever["nbits"]),
        "nprobe": int(fever["nprobe"]),
        "top_k": int(config["retrieval"]["top_k"]),
        "seed": int(config["seed"]),
        "engineering_test": False,
        "index_path": fever["index_path"],
        "ids_path": fever["ids_path"],
        "ntotal": int(fever["expected_corpus_size"]),
        "valid_for_formal_c1": True,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"Formal FEVER index metadata {key}={meta.get(key)!r}; expected {value!r}"
            )
    dimension = meta.get("embedding_dimension")
    if type(dimension) is not int or dimension <= 0:
        raise RuntimeError("Formal FEVER index metadata has invalid embedding_dimension")
    return meta


def retrieval_fingerprint(config: dict, samples: list[dict], metadata: dict) -> str:
    payload = {
        "retrieval": config["retrieval"],
        "samples": samples,
        "index_metadata": metadata,
        "code": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/retrieval/e5_faiss.py",
                "src/retrieval/fever_ivfpq.py",
                "scripts/run_fever_c1_preliminary.py",
            )
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def validate_loaded_index(config: dict, index, metadata: dict) -> None:
    import faiss

    fever = config["retrieval"]["fever"]
    checks = {
        "ntotal": (int(index.ntotal), int(fever["expected_corpus_size"])),
        "dimension": (int(index.d), int(metadata["embedding_dimension"])),
        "nlist": (int(index.nlist), int(fever["nlist"])),
        "m": (int(index.pq.M), int(fever["m"])),
        "nbits": (int(index.pq.nbits), int(fever["nbits"])),
        "nprobe": (int(index.nprobe), int(fever["nprobe"])),
        "metric": (int(index.metric_type), int(faiss.METRIC_INNER_PRODUCT)),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(f"Formal FEVER index {name}={actual}; expected {expected}")
    if not index.is_trained:
        raise RuntimeError("Formal FEVER IndexIVFPQ is not trained")
    if "IVFPQ" not in type(index).__name__:
        raise RuntimeError(f"Expected IndexIVFPQ, loaded {type(index).__name__}")


def inspect_index(config: dict, path: Path, metadata: dict) -> None:
    from src.retrieval.fever_ivfpq import load_index

    fever = config["retrieval"]["fever"]
    index = load_index(path, int(fever["nprobe"]))
    validate_loaded_index(config, index, metadata)


def resolve_passage_ids(ids_path: Path, indices: list[list[int]], expected: int) -> list[list[str]]:
    needed = {index for row in indices for index in row if index >= 0}
    found: dict[int, str] = {}
    count = 0
    with ids_path.open("r", encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if not line.strip():
                raise RuntimeError(f"Blank line at {ids_path}:{position + 1}")
            if position in needed:
                record = json.loads(line)
                passage_id = str(record.get("passage_id") or "").strip()
                if not passage_id:
                    raise RuntimeError(f"Missing passage_id at {ids_path}:{position + 1}")
                found[position] = passage_id
            count += 1
    if count != expected:
        raise RuntimeError(f"FEVER ids count={count}; expected {expected}")
    missing = needed - set(found)
    if missing:
        raise RuntimeError(f"Could not resolve {len(missing)} FAISS positions")
    return [[found[index] for index in row if index >= 0] for row in indices]


def resolve_passage_texts(corpus_path: Path, passage_ids: list[list[str]], expected: int) -> dict[str, str]:
    needed = {passage_id for row in passage_ids for passage_id in row}
    found: dict[str, str] = {}
    count = 0
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            count += 1
            passage_id = str(record.get("passage_id") or "")
            if passage_id in needed:
                if passage_id in found:
                    raise RuntimeError(f"Duplicate requested passage ID {passage_id} in corpus")
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"Retrieved FEVER passage {passage_id} has no text")
                found[passage_id] = text
    if count != expected:
        raise RuntimeError(f"FEVER corpus count={count}; expected {expected}")
    missing = needed - set(found)
    if missing:
        raise RuntimeError(f"Retrieved {len(missing)} passage IDs are absent from corpus")
    return found


def retrieve_top5(
    config: dict,
    samples: list[dict],
    paths: dict[str, Path],
    fingerprint: str,
    metadata: dict,
) -> list[dict]:
    from src.generation.qwen import free_cuda_memory
    from src.retrieval.e5_faiss import E5FaissRetriever
    from src.retrieval.fever_ivfpq import load_index

    retrieval = config["retrieval"]
    fever = retrieval["fever"]
    expected = int(fever["expected_corpus_size"])
    print("Loading BGE query encoder and formal FEVER IndexIVFPQ...")
    retriever = E5FaissRetriever(
        retrieval["embedding_model"], passage_prefix="", query_prefix=BGE_QUERY_INSTRUCTION
    )
    retriever.index = load_index(paths["index"], int(fever["nprobe"]))
    validate_loaded_index(config, retriever.index, metadata)
    embeddings = retriever.encode_queries([sample["claim"] for sample in samples], batch_size=16)
    scores, raw_indices = retriever.search(embeddings, top_k=int(retrieval["top_k"]))
    indices = [[int(value) for value in row] for row in raw_indices]
    retriever.index = None
    retriever.unload()
    free_cuda_memory()
    print("Retriever unloaded; resolving Top-5 passage IDs and text...")
    passage_ids = resolve_passage_ids(paths["ids"], indices, expected)
    texts = resolve_passage_texts(paths["corpus"], passage_ids, expected)
    checkpoint = []
    for sample, ids, row_scores in zip(samples, passage_ids, scores):
        if len(ids) != int(retrieval["top_k"]):
            raise RuntimeError(f"Sample {sample['id']} returned {len(ids)} hits, expected 5")
        hits = [
            {"passage_id": passage_id, "text": texts[passage_id], "score": float(score)}
            for passage_id, score in zip(ids, row_scores)
        ]
        checkpoint.append(
            {
                "sample_id": sample["id"],
                "retrieval_fingerprint": fingerprint,
                "retrieved_top5": hits,
            }
        )
    return checkpoint


def validate_checkpoint(
    path: Path, samples: list[dict], fingerprint: str
) -> list[dict]:
    records = load_jsonl(path)
    if len(records) != len(samples):
        raise RuntimeError(f"Retrieval checkpoint has {len(records)} records; expected {len(samples)}")
    by_id: dict[str, dict] = {}
    for record in records:
        sample_id = str(record.get("sample_id"))
        if sample_id in by_id:
            raise RuntimeError(f"Duplicate retrieval checkpoint ID {sample_id}")
        hits = record.get("retrieved_top5")
        if record.get("retrieval_fingerprint") != fingerprint:
            raise RuntimeError("Retrieval checkpoint fingerprint is incompatible")
        if not isinstance(hits, list) or len(hits) != 5:
            raise RuntimeError(f"Retrieval checkpoint {sample_id} must contain Top-5")
        for hit in hits:
            if (
                not isinstance(hit.get("passage_id"), str)
                or not isinstance(hit.get("text"), str)
                or type(hit.get("score")) not in (int, float)
                or not math.isfinite(hit["score"])
            ):
                raise RuntimeError(f"Invalid retrieval hit for {sample_id}")
        by_id[sample_id] = record
    expected_ids = [str(sample["id"]) for sample in samples]
    if set(by_id) != set(expected_ids):
        raise RuntimeError("Retrieval checkpoint IDs differ from frozen FEVER IDs")
    return [by_id[sample_id] for sample_id in expected_ids]


def get_retrieval(
    config: dict,
    samples: list[dict],
    paths: dict[str, Path],
    fingerprint: str,
    metadata: dict,
) -> list[dict]:
    if paths["checkpoint"].is_file():
        print(f"Using validated retrieval checkpoint: {paths['checkpoint']}")
        return validate_checkpoint(paths["checkpoint"], samples, fingerprint)
    records = retrieve_top5(config, samples, paths, fingerprint, metadata)
    atomic_write_jsonl(paths["checkpoint"], records)
    print(f"Wrote retrieval checkpoint: {paths['checkpoint']}")
    return records


def build_prompt(instruction: str, claim: str, hits: list[dict]) -> str:
    evidence = "\n\n".join(
        f"[{number}] {hit['text']}" for number, hit in enumerate(hits, start=1)
    )
    return f"{instruction}\n\nEvidence:\n{evidence}\n\nClaim: {claim}"


def experiment_fingerprint(config: dict, retrieval_fingerprint_value: str) -> str:
    payload = {
        "primary_model": config["primary_model"],
        "seed": config["seed"],
        "prompt": config["prompts"][DATASET][CONDITION],
        "retrieval_fingerprint": retrieval_fingerprint_value,
        "evaluation": hashlib.sha256((ROOT / "src/evaluation/fever.py").read_bytes()).hexdigest(),
        "generation": hashlib.sha256((ROOT / "src/generation/qwen.py").read_bytes()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def completed_ids(
    path: Path,
    samples: list[dict],
    retrieval: list[dict],
    config: dict,
    fingerprint: str,
) -> set[str]:
    if not path.exists():
        return set()
    by_id = {str(sample["id"]): (sample, item["retrieved_top5"]) for sample, item in zip(samples, retrieval)}
    done = set()
    for record in load_jsonl(path):
        if record.get("dataset") != DATASET or record.get("condition") != CONDITION:
            raise RuntimeError(f"Unexpected dataset/condition in {path}")
        sample_id = str(record.get("sample_id"))
        if sample_id in done:
            raise RuntimeError(f"Duplicate result sample ID {sample_id}")
        if sample_id not in by_id:
            raise RuntimeError(f"Non-frozen result sample ID {sample_id}")
        sample, hits = by_id[sample_id]
        expected = {
            "model_name": config["primary_model"]["model_name"],
            "experiment_fingerprint": fingerprint,
            "question_or_claim": sample["claim"],
            "gold_answer_or_label": sample["label"],
            "evidence": hits,
            "retrieved_top5_ids": [hit["passage_id"] for hit in hits],
            "gold_evidence_in_top5": gold_evidence_in_top5(hits, sample["gold_page_ids"]),
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise RuntimeError(f"Result {sample_id} has incompatible {key}")
        raw = record.get("raw_output")
        prediction = parse_fever_prediction(raw) if isinstance(raw, str) else None
        latency = record.get("latency")
        if (
            prediction is None
            or record.get("prediction") != prediction
            or record.get("correct") is not (prediction == sample["label"])
            or type(latency) not in (int, float)
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise RuntimeError(f"Result {sample_id} has invalid prediction/metrics")
        done.add(sample_id)
    return done


def append_result(path: Path, record: dict) -> None:
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
    if not n:
        print(f"{path.name}: n=0")
        return
    accuracy = sum(record["correct"] for record in records) / n
    gold_hits = sum(record["gold_evidence_in_top5"] for record in records)
    unknown = sum(record["prediction"] == "UNKNOWN" for record in records)
    print(
        f"{path.name}: n={n}, accuracy={accuracy:.4f}, "
        f"gold_evidence_in_top5={gold_hits}/{n}, unknown={unknown}"
    )


def maybe_analyze(config: dict) -> None:
    c3_path = ROOT / config["result_paths"]["fever_c3_7b"]
    if not c3_path.is_file():
        print(
            "FEVER C3 result is not in this checkout; copy fever_c3_7b.jsonl into "
            "results/preliminary, then run: python scripts/analyze_c1_c3_recoverability.py "
            "--dataset fever"
        )
        return
    from scripts.analyze_c1_c3_recoverability import analyze_dataset

    analyze_dataset("fever", config=config, root=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen FEVER500 C1 full-index experiment")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--conditions", nargs="+", choices=[CONDITION], required=True,
        help="explicitly select C1",
    )
    args = parser.parse_args()
    config = load_preliminary_config()
    validate_frozen_settings(config)
    samples = load_frozen_samples(config)
    paths = artifact_paths(config)
    metadata = validate_index_metadata(config, paths)
    retrieval_fp = retrieval_fingerprint(config, samples, metadata)
    print("Preflight passed")
    print("Frozen FEVER samples: 500 (SUPPORTS=250, REFUTES=250)")
    print("Retriever: full 5,396,106-passage IndexIVFPQ, BGE-base, Top-5")
    print(f"Retrieval fingerprint: {retrieval_fp}")
    if args.check_only:
        inspect_index(config, paths["index"], metadata)
        if paths["checkpoint"].exists():
            retrieval = validate_checkpoint(paths["checkpoint"], samples, retrieval_fp)
            result_fp = experiment_fingerprint(config, retrieval_fp)
            completed_ids(paths["results"], samples, retrieval, config, result_fp)
        print("Formal FEVER C1 index validation passed; no model was loaded.")
        return 0

    if args.smoke_test:
        retrieval = retrieve_top5(config, samples[:1], paths, retrieval_fp, metadata)
        from transformers import set_seed

        from src.generation.qwen import generate_greedy, load_qwen_nf4

        set_seed(int(config["seed"]))
        tokenizer, model = load_qwen_nf4(config["primary_model"]["model_name"])
        sample = samples[0]
        hits = retrieval[0]["retrieved_top5"]
        raw, latency = generate_greedy(
            tokenizer, model,
            build_prompt(config["prompts"][DATASET][CONDITION], sample["claim"], hits),
            max_new_tokens=int(config["primary_model"]["max_new_tokens"]),
        )
        prediction = parse_fever_prediction(raw)
        safe_print(
            f"SMOKE C1: gold={sample['label']} pred={prediction} "
            f"correct={prediction == sample['label']} latency={latency:.2f}s raw={raw!r}"
        )
        if prediction == "UNKNOWN":
            raise RuntimeError("Smoke output contained no parseable FEVER label")
        print("Smoke test passed; no formal result/checkpoint files were written.")
        return 0

    retrieval = get_retrieval(config, samples, paths, retrieval_fp, metadata)
    result_fp = experiment_fingerprint(config, retrieval_fp)
    done = completed_ids(paths["results"], samples, retrieval, config, result_fp)
    if len(done) == len(samples):
        print("FEVER C1 is already complete; no model was loaded.")
        summarize(paths["results"])
        maybe_analyze(config)
        return 0
    if done:
        print(f"Resuming: {len(done)}/500 generations already saved")

    from transformers import set_seed

    from src.generation.qwen import generate_greedy, load_qwen_nf4

    set_seed(int(config["seed"]))
    tokenizer, model = load_qwen_nf4(config["primary_model"]["model_name"])
    offload = any(
        str(device) in {"cpu", "disk"}
        for device in getattr(model, "hf_device_map", {}).values()
    )
    for position, (sample, retrieval_record) in enumerate(zip(samples, retrieval), start=1):
        sample_id = str(sample["id"])
        if sample_id in done:
            continue
        hits = retrieval_record["retrieved_top5"]
        raw, latency = generate_greedy(
            tokenizer, model,
            build_prompt(config["prompts"][DATASET][CONDITION], sample["claim"], hits),
            max_new_tokens=int(config["primary_model"]["max_new_tokens"]),
        )
        prediction = parse_fever_prediction(raw)
        record = {
            "sample_id": sample["id"],
            "dataset": DATASET,
            "condition": CONDITION,
            "model_name": config["primary_model"]["model_name"],
            "experiment_fingerprint": result_fp,
            "cpu_offload": offload,
            "question_or_claim": sample["claim"],
            "gold_answer_or_label": sample["label"],
            "evidence": hits,
            "retrieved_top5_ids": [hit["passage_id"] for hit in hits],
            "gold_evidence_in_top5": gold_evidence_in_top5(hits, sample["gold_page_ids"]),
            "raw_output": raw,
            "prediction": prediction,
            "correct": prediction == sample["label"],
            "latency": latency,
        }
        append_result(paths["results"], record)
        done.add(sample_id)
        print(
            f"[{len(done)}/500] sample={position}/500 gold={sample['label']} "
            f"pred={prediction} correct={record['correct']} "
            f"gold_in_top5={record['gold_evidence_in_top5']} latency={latency:.2f}s"
        )
    summarize(paths["results"])
    maybe_analyze(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
