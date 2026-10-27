"""GPU-only frozen C1 generation over already selected P3-20 evidence.

No retriever, reranker, Gold or Diagnosis input. Model loads only on the first
pending generation, after strict input/resume preflight. Never run this on Mac.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compensation.retrieval_enhancement.generation_replay import run_generation, sha256_file


def verify_bundle(input_path):
    manifest_path = ROOT / "package_manifest.json"
    if not manifest_path.exists():
        return  # Running from a repository checkout rather than the transfer zip.
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"GPU bundle checksum mismatch: {name}")
    if sha256_file(input_path) != manifest["files"]["generation_inputs.jsonl"]:
        raise ValueError("input differs from the exported P3-20 bundle")


def frozen_metadata():
    from src.config import load_generator_config

    config = load_generator_config()
    required = {"model_name": "Qwen/Qwen2.5-7B-Instruct", "quantization": "nf4",
                "load_in_4bit": True, "do_sample": False, "max_new_tokens": 64}
    if any(config.get(k) != value for k, value in required.items()):
        raise ValueError("generator configuration differs from frozen C1")
    paths = ["configs/base.yaml", "src/generation/qwen.py", "src/evaluation/qa.py",
             "src/compensation/retrieval_enhancement/p3.py",
             "src/compensation/retrieval_enhancement/generation_replay.py",
             "scripts/run_nq_p320_gpu_generation.py"]
    return {"policy": "P3-20", "generator": config,
            "source_sha256": {p: sha256_file(ROOT / p) for p in paths},
            "qwen_revision": "not pinned in original C1; actual resolved commits saved in model_receipt.json"}


def lazy_qwen_generator(output_dir):
    state = {}

    def generate(prompt):
        if not state:
            # This is the only model-loading path, unreachable in unit tests.
            from src.generation.qwen import generate_greedy, load_qwen_nf4
            import torch
            import transformers
            import bitsandbytes

            load_start = perf_counter()
            tokenizer, model = load_qwen_nf4()
            model_commit = getattr(model.config, "_commit_hash", None)
            commit_source = "model.config"
            if not model_commit:
                from huggingface_hub import scan_cache_dir
                revs = [r.commit_hash for repo in scan_cache_dir().repos
                        if repo.repo_id == "Qwen/Qwen2.5-7B-Instruct" for r in repo.revisions]
                if len(revs) == 1:
                    model_commit, commit_source = revs[0], "hf_cache_single_revision"
            tokenizer_commit = tokenizer.init_kwargs.get("_commit_hash") or getattr(tokenizer, "_commit_hash", None)
            tokenizer_source = "tokenizer"
            if not tokenizer_commit:
                # Newer transformers no longer expose it; tokenizer is loaded from the same repo snapshot.
                tokenizer_commit, tokenizer_source = model_commit, "inferred_from_model_same_repo"
            if not model_commit or model_commit != tokenizer_commit:
                raise RuntimeError("cannot verify matching resolved Qwen model/tokenizer commits")
            identity = {"model_name": "Qwen/Qwen2.5-7B-Instruct", "model_commit": model_commit,
                        "tokenizer_commit": tokenizer_commit, "model_commit_source": commit_source,
                        "tokenizer_commit_source": tokenizer_source, "quantization": "nf4",
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "bitsandbytes": bitsandbytes.__version__,
                        "gpu": torch.cuda.get_device_name(0)}
            path = Path(output_dir) / "model_receipt.json"
            if path.exists():
                prior = json.loads(path.read_text())
                if prior["identity"] != identity:
                    raise RuntimeError("model/tokenizer/dependency/GPU identity differs from prior session")
            else:
                with path.open("x") as handle:
                    json.dump({"identity": identity, "model_load_seconds": perf_counter() - load_start}, handle, indent=2)
            state.update(tokenizer=tokenizer, model=model, generate=generate_greedy)
        answer, _model_seconds = state["generate"](state["tokenizer"], state["model"], prompt)
        return answer

    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="engineering smoke limit; resume continues remaining samples")
    args = parser.parse_args()
    verify_bundle(args.input)
    summary = run_generation(args.input, args.output_dir, lazy_qwen_generator(args.output_dir),
                             run_metadata=frozen_metadata(), resume=args.resume, limit=args.limit)
    print(json.dumps(summary, indent=2))
    if summary["stopped_on_error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
