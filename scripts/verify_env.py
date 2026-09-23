"""Environment verification for capability-aware-slm-rag.

Checks every dependency the RAG pipeline needs WITHOUT downloading the 15GB
Qwen2.5-7B checkpoint. Run from the repo root with the venv python.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def wrap(fn):
        try:
            detail = fn()
            results.append((name, True, detail or ""))
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn

    return wrap


@check("torch + CUDA")
def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    return (
        f"torch {torch.__version__}, cuda {torch.version.cuda}, "
        f"{props.name}, total {total / 1024**3:.2f} GiB, free {free / 1024**3:.2f} GiB"
    )


@check("numpy")
def _numpy():
    import numpy as np

    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert a.sum() == 15.0
    return f"numpy {np.__version__}"


@check("faiss IndexFlatIP")
def _faiss():
    import faiss
    import numpy as np

    rng = np.random.default_rng(42)
    vecs = rng.random((100, 32)).astype(np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(32)
    index.add(vecs)
    scores, idx = index.search(vecs[:3], 5)
    assert idx[0][0] == 0, "self should be top-1"
    return f"faiss {faiss.__version__}, {index.ntotal} vectors, top-1 self-match OK"


@check("faiss IndexIVFPQ (FEVER path)")
def _faiss_ivfpq():
    import faiss
    import numpy as np

    # mirrors configs/preliminary.yaml engineering_test: 256 passages, nlist 8
    rng = np.random.default_rng(42)
    dim, nlist, m, nbits = 64, 8, 8, 8
    vecs = rng.random((256, dim)).astype(np.float32)
    faiss.normalize_L2(vecs)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    index.nprobe = 4
    _scores, idx = index.search(vecs[:2], 5)
    assert idx.shape == (2, 5)
    return f"trained on 256 vecs, nlist={nlist}, m={m}, search OK"


@check("transformers")
def _transformers():
    import transformers

    return f"transformers {transformers.__version__}"


@check("bitsandbytes NF4 on GPU")
def _bnb():
    import bitsandbytes as bnb
    import torch

    # real 4-bit NF4 round-trip on the GPU, no model download needed
    x = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    q, state = bnb.functional.quantize_4bit(x, quant_type="nf4")
    deq = bnb.functional.dequantize_4bit(q, state, quant_type="nf4")
    err = (x - deq).abs().mean().item()
    return (
        f"bitsandbytes {bnb.__version__}, NF4 quant/dequant on CUDA OK, "
        f"mean abs err {err:.4f}"
    )


@check("sentence-transformers + BGE encoder")
def _st():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    emb = model.encode(
        ["what is the capital of France", "Paris is the capital of France"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    sim = float(emb[0] @ emb[1])
    return f"BGE-base loaded, dim={emb.shape[1]}, q/p cosine={sim:.3f}"


@check("datasets")
def _datasets():
    import datasets

    return f"datasets {datasets.__version__}"


@check("repo: src.config")
def _config():
    from src.config import load_generator_config, load_preliminary_config

    gen = load_generator_config()
    prelim = load_preliminary_config()
    return (
        f"generator={gen['model_name']}, quant={gen['quantization']}, "
        f"prelim top_k={prelim['retrieval']['top_k']}"
    )


@check("repo: src.evaluation")
def _eval():
    from src.evaluation.fever import parse_fever_prediction
    from src.evaluation.qa import best_em_f1
    from src.evaluation.recoverability import recoverable_failure_rate

    em, f1 = best_em_f1("The Paris.", ["paris", "france"])
    assert em == 1, f"expected EM=1, got {em}"
    label = parse_fever_prediction("I think this one SUPPORTS the claim")
    assert label == "SUPPORTS", label
    rfr = recoverable_failure_rate({1, 2, 3, 4}, {2, 4, 9})
    assert rfr == 0.5, rfr
    return f"EM/F1={em}/{f1:.2f}, fever parse={label}, RFR={rfr}"


@check("repo: src.retrieval / src.generation imports")
def _repo_modules():
    from src.generation.qwen import build_fever_rag_prompt, build_rag_prompt
    from src.retrieval.e5_faiss import E5FaissRetriever  # noqa: F401

    p = build_rag_prompt("who wrote Hamlet", [{"text": "Hamlet was written by Shakespeare."}])
    f = build_fever_rag_prompt("Hamlet is a play.", [{"text": "Hamlet is a tragedy."}])
    assert "Context:" in p and "Evidence:" in f
    return "prompt builders and retriever class import OK"


@check("frozen manifests")
def _manifests():
    import json

    nq = json.loads((REPO / "configs/splits/preliminary_nq_500.json").read_text("utf-8"))
    fv = json.loads((REPO / "configs/splits/preliminary_fever_500.json").read_text("utf-8"))
    n_nq = len(set(nq["ids"]))
    samples = fv["samples"]
    sup = sum(1 for s in samples if s["label"] == "SUPPORTS")
    ref = sum(1 for s in samples if s["label"] == "REFUTES")
    assert n_nq == 500, n_nq
    assert sup == 250 and ref == 250, (sup, ref)
    return f"NQ {n_nq} unique ids, FEVER {sup} SUPPORTS / {ref} REFUTES"


print("\n" + "=" * 72)
print("ENVIRONMENT VERIFICATION")
print("=" * 72)
width = max(len(name) for name, _, _ in results)
n_fail = 0
for name, ok, detail in results:
    mark = "PASS" if ok else "FAIL"
    if not ok:
        n_fail += 1
    print(f"[{mark}] {name.ljust(width)}  {detail}")
print("=" * 72)
print(f"{len(results) - n_fail}/{len(results)} passed")
sys.exit(1 if n_fail else 0)
