"""Phase 4A: paired NQ retrieval evaluation, no Generator/Diagnosis/Controller.

Default mode only checks local resource structure; --run explicitly retrieves.
Always uses the frozen NQ manifest order (a --limit prefix is a smoke subset).
Gold is joined only by the offline scorer after traces.jsonl has been saved.
--evaluate-only can rescore saved traces without loading BGE or FAISS.
Output directories must be new and outside the frozen results/preliminary tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.retrieval_enhancement import EnhancementBudget, NQFaissBackend, RetrievalEnhancer
from src.compensation.retrieval_enhancement.adapters import BGE_MODEL_NAME, BGE_QUERY_INSTRUCTION
from src.compensation.retrieval_enhancement.evaluation import evaluate_nq, sample_id


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path} must contain JSON objects")
    return records


def select_queries(records, manifest, limit=None) -> list[dict]:
    """Return only IDs/questions, in manifest order; never inspect Gold fields."""
    if manifest.get("dataset") != "natural_questions":
        raise ValueError("manifest must be for natural_questions")
    ids = manifest.get("ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("manifest.ids must be a nonempty list")
    ids = [sample_id(sid) for sid in ids]
    if len(set(ids)) != len(ids) or type(manifest.get("n")) is not int or manifest["n"] != len(ids):
        raise ValueError("manifest count/unique IDs disagree")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(ids)):
        raise ValueError("limit must be between 1 and manifest.n")
    by_id = {}
    for record in records:
        sid = sample_id(record["id"])
        if sid in by_id:
            raise ValueError(f"duplicate source id: {sid}")
        by_id[sid] = record
    missing = [sid for sid in ids if sid not in by_id]
    if missing:
        raise ValueError(f"source is missing {len(missing)} manifest IDs, first: {missing[0]}")
    queries = []
    for sid in ids[:limit]:
        question = by_id[sid]["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"invalid question for sample {sid}")
        queries.append({"sample_id": sid, "question": question})
    return queries


def run_query(sample_id: str, question: str, backend, budget: EnhancementBudget) -> dict:
    """One Base retrieval plus one bounded Enhancement round; no annotation input.

    Base retrieval failure aborts the run: it must not be counted as a Base miss.
    Additional retrieval failures use the Enhancer's fallback and remain in Trace.
    Latency excludes loading, serialization and offline scoring.
    """
    start = perf_counter()
    base = backend.retrieve(question, dataset="nq", top_k=5)
    base_latency = perf_counter() - start
    result = RetrievalEnhancer(backend).enhance(question, "nq", base, budget)
    return {"sample_id": sample_id, "dataset": "nq", "question": question,
            "base_evidence": deepcopy(base), "enhanced_evidence": result["evidence"],
            "candidates": result["candidates"], "trace": result["trace"],
            "base_latency": base_latency}


def default_cache_dir() -> Path:
    configured = (os.environ.get("SENTENCE_TRANSFORMERS_HOME") or os.environ.get("HF_HUB_CACHE")
                  or os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE"))
    return Path(configured) if configured else Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"


def check_model_cache(cache_dir: Path) -> dict:
    """Inspect files only, including shard presence. Does not validate weights."""
    model_dir = cache_dir / "models--BAAI--bge-base-en-v1.5"
    errors = []
    snapshot = None
    try:
        revision = (model_dir / "refs/main").read_text().strip()
        if not revision or Path(revision).name != revision:
            raise ValueError("invalid cached main revision")
        snapshot = model_dir / "snapshots" / revision
        for name in ("config.json", "modules.json", "tokenizer_config.json"):
            if not (snapshot / name).is_file():
                errors.append(f"missing BGE cache file: {snapshot / name}")
        if not any((snapshot / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
            errors.append("missing BGE tokenizer.json/vocab.txt")
        if (snapshot / "modules.json").is_file():
            modules = json.loads((snapshot / "modules.json").read_text())
            if not isinstance(modules, list) or not modules:
                raise ValueError("BGE modules.json must contain model modules")
            for module in modules:
                module_path = module["path"]
                if not isinstance(module_path, str) or Path(module_path).is_absolute() or ".." in Path(module_path).parts:
                    raise ValueError("invalid BGE module path")
                if module_path and not (snapshot / module_path / "config.json").is_file():
                    errors.append(f"missing BGE module config: {module_path}")
        if not any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            shard_index = next((snapshot / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                                if (snapshot / name).is_file()), None)
            if shard_index is None:
                errors.append("missing BGE weights/shard index")
            else:
                shards = set(json.loads(shard_index.read_text())["weight_map"].values())
                if not shards or any(not (snapshot / name).is_file() for name in shards):
                    errors.append("missing BGE weight shards")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"BGE cache structure: {exc}")
    return {"cache_dir": str(cache_dir.resolve()), "snapshot": str(snapshot.resolve()) if snapshot else None,
            "errors": errors, "weights_loaded": False}


def preflight(args) -> dict:
    """No torch/FAISS/model import, download, index build or query encoding."""
    model = check_model_cache(args.cache_dir)
    errors = list(model["errors"])
    resources = {}
    for name in ("index", "metadata", "queries", "manifest"):
        path = getattr(args, name)
        exists = path.is_file()
        resources[name] = {"path": str(path.resolve()), "exists": exists}
        if not exists:
            errors.append(f"missing {name}: {path}")
    selected_count = metadata_count = None
    try:
        if resources["queries"]["exists"] and resources["manifest"]["exists"]:
            manifest = json.loads(args.manifest.read_text())
            selected_count = len(select_queries(load_jsonl(args.queries), manifest, args.limit))
        if resources["metadata"]["exists"]:
            metadata = load_jsonl(args.metadata)
            ids = [record.get("passage_id") for record in metadata]
            if not metadata or any(not isinstance(pid, str) or not pid.strip() for pid in ids):
                raise ValueError("metadata has empty/invalid passage IDs")
            if len(set(ids)) != len(ids) or any(not isinstance(record.get("text"), str) or not record["text"].strip() for record in metadata):
                raise ValueError("metadata has duplicate IDs or invalid text")
            metadata_count = len(metadata)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"NQ input structure: {exc}")
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "sentence_transformers", "faiss", "numpy")}
    errors.extend(f"missing Python dependency: {name}" for name, present in dependencies.items() if not present)
    return {"structure_ready": not errors, "resources": resources, "model": model,
            "selected_count": selected_count, "metadata_count": metadata_count,
            "dependencies": dependencies, "errors": errors, "index_loaded": False,
            "note": "File structure only; matching artifacts and real retrieval are not yet validated."}


def load_backend(args):
    """Runner owns the existing Retriever, forces local cache and CPU by default."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME", "TRANSFORMERS_CACHE"):
        os.environ[name] = str(args.cache_dir.resolve())
    # Lazy import only after structural preflight has succeeded and --run is set.
    from src.retrieval.e5_faiss import E5FaissRetriever

    retriever = E5FaissRetriever(BGE_MODEL_NAME, device=args.device,
                               passage_prefix="", query_prefix=BGE_QUERY_INSTRUCTION)
    try:
        retriever.load(args.index, args.metadata)
        return retriever, NQFaissBackend(retriever)
    except Exception:
        retriever.unload()
        raise


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def new_output_dir(path: Path) -> Path:
    path = path.resolve()
    for protected in (ROOT / name for name in (".git", "results/preliminary", "configs", "data", "src", "scripts", "tests")):
        if path == protected or protected in path.parents:
            raise ValueError(f"output directory is inside a protected tree: {protected}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_metrics(output_dir, records, annotations):
    evaluation = evaluate_nq(records, annotations)
    write_json(output_dir / "metrics.json", evaluation["metrics"])
    with (output_dir / "offline_evaluation.jsonl").open("x", encoding="utf-8") as handle:
        for row in evaluation["per_sample"]:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return evaluation["metrics"]


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Run local BGE retrieval (default: structural preflight only)")
    mode.add_argument("--evaluate-only", type=Path, metavar="TRACES_JSONL")
    cli.add_argument("--cache-dir", type=Path, default=default_cache_dir(), help="HF Hub cache root containing models--BAAI--bge-base-en-v1.5")
    cli.add_argument("--index", type=Path, default=ROOT / "data/processed/indexes/nq_bge_base.index")
    cli.add_argument("--metadata", type=Path, default=ROOT / "data/processed/indexes/nq_bge_base_meta.jsonl")
    cli.add_argument("--queries", type=Path, default=ROOT / "data/processed/retrieval/nq_dev.jsonl")
    cli.add_argument("--manifest", type=Path, default=ROOT / "configs/splits/preliminary_nq_500.json")
    cli.add_argument("--limit", type=int, help="Prefix of manifest IDs; omit for all 500")
    cli.add_argument("--max-retrieval-calls", type=int, choices=(0, 1, 2), default=1)
    cli.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    cli.add_argument("--output-dir", type=Path, help="Required for run/scoring, must not exist")
    return cli


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if (args.run or args.evaluate_only) and args.output_dir is None:
        cli.error("--output-dir is required for --run / --evaluate-only")
    try:
        if args.evaluate_only:
            records = load_jsonl(args.evaluate_only)
            annotations = load_jsonl(args.queries)
            # Validate scoring before creating any directory.
            evaluate_nq(records, annotations)
            output = new_output_dir(args.output_dir)
            write_json(output / "evaluation_inputs.json", {"traces": fingerprint(args.evaluate_only), "annotations": fingerprint(args.queries)})
            print(json.dumps(write_metrics(output, records, annotations), indent=2))
            return 0
        report = preflight(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["structure_ready"]:
            return 2
        if not args.run:
            return 0
        manifest = json.loads(args.manifest.read_text())
        queries = select_queries(load_jsonl(args.queries), manifest, args.limit)
        output = new_output_dir(args.output_dir)
        budget = EnhancementBudget(max_retrieval_calls=args.max_retrieval_calls)
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
        write_json(output / "run_config.json", {
            "experiment": "nq_retrieval_enhancement_phase4a", "dataset": "nq",
            "scope": "full_manifest" if len(queries) == manifest["n"] else "smoke_subset",
            "sample_ids": [item["sample_id"] for item in queries], "manifest_seed": manifest.get("seed"),
            "model": BGE_MODEL_NAME, "model_cache_snapshot": report["model"]["snapshot"],
            "query_instruction": BGE_QUERY_INSTRUCTION, "device": args.device,
            "base_top_k": 5, "additional_top_k": 5, "final_top_k": 5, "rrf_k": 60,
            "max_retrieval_calls": budget.max_retrieval_calls, "rounds": 1,
            "rewriter": "RuleQueryRewriter", "llm_calls": 0,
            "git_commit": git.stdout.strip(), "git_dirty": bool(dirty.stdout.strip()),
            "inputs": {name: fingerprint(getattr(args, name)) for name in ("index", "metadata", "queries", "manifest")},
            "implementation": {str(path.relative_to(ROOT)): fingerprint(path)
                               for path in sorted((ROOT / "src/compensation/retrieval_enhancement").glob("*.py"))
                               + [Path(__file__)]},
            "note": "Every query receives the same operation budget; Base-miss strata are computed only offline. No QA generation."
        })
        owner = None
        records = []
        load_start = perf_counter()
        try:
            owner, backend = load_backend(args)
            load_latency = perf_counter() - load_start
            with (output / "traces.jsonl").open("x", encoding="utf-8") as handle:
                for query in queries:
                    record = run_query(query["sample_id"], query["question"], backend, budget)
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    print(f"[{len(records)}/{len(queries)}] {record['sample_id']} {record['trace']['status']}", flush=True)
        finally:
            if owner is not None:
                owner.unload()
        # Gold annotation access begins here, after all runtime output is saved.
        metrics = write_metrics(output, records, load_jsonl(args.queries))
        write_json(output / "completion.json", {"complete": True, "records": len(records), "resource_load_latency_seconds": load_latency})
        print(json.dumps(metrics, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(f"Phase 4A stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
