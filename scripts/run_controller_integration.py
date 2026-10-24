"""C1 -> frozen Diagnosis/Probe/router -> one enhancement -> auditable JSONL.

--smoke uses fictional questions and deterministic providers, never downloads
models. --preflight checks inputs and code fingerprints without inference.
Real runs require CUDA and explicitly selected, matching NQ index resources.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.answer_repair import Generation
from src.compensation.retrieval_enhancement.p3 import RERANKER_MODEL, RERANKER_REVISION
from src.controller.execution import ExecutionConfig, IntegratedController, MODES, validate_input
from src.diagnosis.adapter import c1_record_to_input
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosisClassifier
from src.diagnosis.schema import DiagnosisInput, EvidenceItem

DEFAULT_CLASSIFIER = ROOT / "models/diagnosis/diagnosis_v2_expanded_hierarchical.json"


def runtime_input(row):
    # C1 exports may contain offline fields: the established adapter reads only
    # its explicit whitelist. The resulting runtime object has no gold fields.
    if "condition" in row:
        item = c1_record_to_input(row)
    else:
        expected = {"sample_id", "dataset", "query", "initial_answer", "evidence"}
        if set(row) != expected:
            raise ValueError("Runtime JSONL must contain exactly the five DiagnosisInput fields")
        item = DiagnosisInput(row["sample_id"], row["dataset"], row["query"],
                              [EvidenceItem(**e) for e in row["evidence"]], row["initial_answer"])
    validate_input(item)
    return item


def read_inputs(path):
    items, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = runtime_input(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid runtime input at line {number}: {exc}") from exc
            key = (item.dataset.value, item.sample_id)
            if key in seen:
                raise ValueError(f"Duplicate dataset/sample_id: {key}")
            seen.add(key)
            items.append(item)
    if not items:
        raise ValueError("No runtime inputs")
    return items


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_manifest():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "configs/base.yaml"]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def summarize(records):
    groups = {}
    for mode in sorted({r["mode"] for r in records}):
        for dataset in sorted({r["dataset"] for r in records if r["mode"] == mode}):
            rows = [r for r in records if r["mode"] == mode and r["dataset"] == dataset]
            def tokens(field):
                values = [r["cost"][field] for r in rows]
                return sum(values) if all(v is not None for v in values) else None
            groups[f"{dataset}/{mode}"] = {
                "n": len(rows), "changed": sum(r["answer_changed"] for r in rows),
                "statuses": dict(Counter(r["status"] for r in rows)),
                "actions": dict(Counter(r["routing"]["decision"]["action"] for r in rows)),
                "llm_calls": sum(r["cost"]["llm_calls"] for r in rows),
                "retrieval_calls": sum(r["cost"]["retrieval_calls"] for r in rows),
                "scoring_calls": sum(r["cost"]["scoring_calls"] for r in rows),
                "input_tokens": tokens("input_tokens"),
                "output_tokens_estimate": tokens("output_tokens_estimate"),
                "total_seconds": sum(r["total_seconds"] for r in rows),
                "failed_calls": sum(r["cost"]["failed_calls"] for r in rows),
                "budget_blocks": sum(r["cost"]["budget_blocks"] for r in rows),
            }
    return {"groups": groups, "quality_metrics": "not_scored; evaluate separately after predictions",
            "cost_scope": "incremental_after_C1; add shared C1 cost for full-system totals"}


def smoke_pipeline(config, classifier_path=DEFAULT_CLASSIFIER):
    """Real frozen classifier and real module logic, fictional provider outputs."""
    class Generator:
        def generate(self, prompt, max_new_tokens):
            if "Output exactly three lines" in prompt:
                answer = "RANK: 1\nQUOTE: The capital of Luma is Amber.\nANSWER: Amber"
            elif "verdict" in prompt and "citations" in prompt:
                answer = json.dumps({"verdict": "repair", "answer": "SUPPORTS",
                                     "citations": [{"rank": 1, "quote": "The capital of Luma is Amber."}]})
            elif "answer_supported" in prompt or "evidence_verdict" in prompt:
                answer = "{}"  # explicit unparsable Probe: retains the frozen diagnosis
            else:
                answer = "SUPPORTS" if "Claim:" in prompt else "Amber"
            # Unknown tokens remain unknown; fictional latency is not GPU cost.
            return Generation(answer, 0.0)

    class Scorer:
        model_name, revision = RERANKER_MODEL, RERANKER_REVISION
        def __call__(self, query, texts):
            return [float(i) for i in range(len(texts))]

    class Retriever:
        supported_datasets = frozenset({'nq', 'fever'})
        def retrieve(self, query, *, dataset, top_k):
            return [{"passage_id": f"extra{i}", "text": "The capital of Luma is Amber.",
                     "score": 1.0 - i / 100} for i in range(top_k)]

    generator = Generator()
    return IntegratedController(HierarchicalDiagnosisClassifier.load(classifier_path), generator,
                                {"nq": generator, "fever": generator}, Scorer(), Retriever(), config=config)


def smoke_inputs():
    evidence = [EvidenceItem(i, f"p{i}", "The capital of Luma is Amber.", 1 - i / 10) for i in range(1, 6)]
    return [DiagnosisInput("integration-fiction-nq", "nq", "What is the capital of Luma?", evidence, "Blue"),
            DiagnosisInput("integration-fiction-fever", "fever", "The capital of Luma is Amber.", evidence, "REFUTES")]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/controller/integration_v1.json")
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--nq-index", type=Path)
    parser.add_argument("--nq-metadata", type=Path)
    parser.add_argument("--nq-resources", type=Path,
                        help="Verified NQ Train corpus/index/IDs/offsets registry")
    parser.add_argument('--fever-resources', type=Path, help='Verified FEVER IVFPQ resource registry')
    args = parser.parse_args(argv)
    if args.smoke == (args.input is not None):
        parser.error("Choose exactly one of --smoke and --input")
    config = ExecutionConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    modes = args.mode or (list(MODES) if args.smoke else ["adaptive"])
    if len(set(modes)) != len(modes):
        parser.error("Duplicate modes")
    items = smoke_inputs() if args.smoke else read_inputs(args.input)
    # New directory only: never overwrite a previous experiment or restart with
    # changed code. Interrupted predictions remain available for investigation.
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run directory")
    need_index = (not args.smoke and any(i.dataset.value == "nq" for i in items)
                  and any(m in ("adaptive", "always_retrieve") for m in modes))
    if args.nq_resources and (args.nq_index or args.nq_metadata):
        parser.error("Choose --nq-resources or the legacy --nq-index/--nq-metadata pair")
    if bool(args.nq_index) != bool(args.nq_metadata):
        parser.error("Provide both --nq-index and --nq-metadata")
    if need_index and not args.preflight and not (args.nq_resources or (args.nq_index and args.nq_metadata)):
        parser.error("NQ adaptive/retrieval runs require matching --nq-resources or --nq-index/--nq-metadata")
    need_fever = (not args.smoke and any(i.dataset.value == 'fever' for i in items)
                  and any(m in ('adaptive', 'always_retrieve') for m in modes))
    if need_fever and not args.preflight and not args.fever_resources:
        parser.error('FEVER adaptive/retrieval runs require --fever-resources')
    for path in (args.nq_index, args.nq_metadata):
        if path is not None and not path.is_file():
            parser.error(f"Missing resource: {path}")
    resource_receipt = None
    if args.nq_resources:
        from src.controller.nq_train_resources import validate_resources
        _, _, resource_receipt = validate_resources(args.nq_resources, root=ROOT)
    fever_receipt = None
    if args.fever_resources:
        from src.controller.fever_resources import validate_resources as validate_fever
        _, _, fever_receipt = validate_fever(args.fever_resources, root=ROOT)
    args.output_dir.mkdir(parents=True)
    manifest = {"schema_version": "controller_run_v1", "smoke": args.smoke,
                "preflight": args.preflight, "n_inputs": len(items), "modes": modes,
                "config": asdict(config), "source_files": code_manifest(),
                "classifier_sha256": file_sha(args.classifier),
                "input_sha256": file_sha(args.input) if args.input else None,
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "nq_index_required": need_index, "nq_train_resources": resource_receipt,
                'fever_resources_required': need_fever, 'fever_resources': fever_receipt,
                "state": "preflight" if args.preflight else "running"}
    write_json(args.output_dir / "manifest.json", manifest)
    if args.preflight:
        print(json.dumps({"preflight": "passed", "n_inputs": len(items), "nq_index_required": need_index}))
        return 0
    try:
        before = perf_counter()
        if args.smoke:
            pipeline, environment = smoke_pipeline(config, args.classifier), {"providers": "fictional CPU smoke"}
        elif modes == ["c1"]:
            pipeline, environment = IntegratedController(None, None, {}, None, config=config), {"providers": "none"}
        else:
            from src.controller.providers import load_pipeline
            pipeline, environment = load_pipeline(config, args.classifier, nq_index=args.nq_index,
                                                  nq_metadata=args.nq_metadata, nq_resources=args.nq_resources,
                                                  fever_resources=args.fever_resources)
        environment["resource_loading_seconds"] = perf_counter() - before
        # Input paths and hashes remain in the run manifest, never in Diagnosis.
        if args.nq_index:
            environment["nq_index_sha256"] = file_sha(args.nq_index)
        if args.nq_metadata:
            environment["nq_metadata_sha256"] = file_sha(args.nq_metadata)
        write_json(args.output_dir / "environment.json", environment)
        records = []
        with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as handle:
            for item in items:
                for mode in modes:
                    record = pipeline.run(item, mode=mode)
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    records.append(record)
        summary = summarize(records)
        summary["smoke"] = args.smoke
        write_json(args.output_dir / "summary.json", summary)
        failures = sum(bool(r["cost"]["failed_calls"] or r["cost"]["budget_blocks"] or r["status"].endswith("_failed") or
                            r["status"] == "budget_exceeded") for r in records)
        manifest.update(state="completed_with_errors" if failures else "completed", n_records=len(records))
        write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps({"state": manifest["state"], "records": len(records), "output_dir": str(args.output_dir)}))
        return 2 if failures else 0
    except Exception as exc:
        manifest.update(state="failed", error={"type": type(exc).__name__, "message": str(exc)})
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
