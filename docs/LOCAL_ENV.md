# Local environment notes

Machine-specific setup notes. The shared, authoritative instructions are in
[SETUP.md](SETUP.md); this file records what differs on a 6GB-VRAM laptop and
how to verify the install without downloading any dataset or the 15GB Qwen
checkpoint.

## Verified install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`SETUP.md` names the cu130 wheel index. cu128 was used here and works on driver
581.95; pick whichever index matches the local driver.

Resulting versions (2026-09-23):

| package | version |
|---|---|
| torch | 2.11.0+cu128 |
| transformers | 5.17.0 |
| sentence-transformers | 6.1.0 |
| faiss-cpu | 1.15.1 |
| bitsandbytes | 0.50.2 |
| datasets | 5.0.1 |
| numpy | 2.4.6 |
| accelerate | 1.15.0 |

## Verifying without downloading data

```powershell
python scripts/verify_env.py
```

Checks torch+CUDA, a real NF4 quant/dequant round-trip on the GPU, both FAISS
index types used by the project (`IndexFlatIP` for NQ, `IndexIVFPQ` for FEVER),
the BGE encoder, every `src/` module, and the two frozen manifests. It downloads
only `BAAI/bge-base-en-v1.5` (~440MB) and touches nothing under `data/` or
`results/`. Exit code 0 means all checks passed.

## 6GB VRAM: CPU offload

The frozen generator is `Qwen/Qwen2.5-7B-Instruct` at 4-bit NF4. The NF4 weights
alone are about 4.2GB. On an RTX 3060 Laptop the GPU reports 6.00 GiB total but
only ~5.00 GiB free once the Windows desktop has its share, so the model does not
fit on the GPU together with the KV cache and activations.

`load_qwen_nf4()` therefore accepts an optional CPU-offload placement, **off by
default** — with neither variable set, loading is exactly the frozen path:

```powershell
$env:SLMRAG_GPU_MAX_MEMORY = "4GiB"   # cap GPU use, spill the rest to CPU RAM
$env:SLMRAG_CPU_MAX_MEMORY = "24GiB"  # optional, defaults to 24GiB
python scripts/test_qwen_load.py
```

This changes only *where layers live*. The model, NF4 quantization, decoding
(`do_sample=false`), prompts, top-k, seed, and evaluation metrics are untouched,
so results stay comparable. Offloaded layers run on the CPU, so expect generation
to be several times slower than an all-GPU run.

Note that Hugging Face serves Qwen2.5-7B-Instruct as bf16 safetensors — the 4-bit
quantization happens at load time, so the **download is ~15GB** even though the
loaded model is ~4.2GB.

## Local state

`data/` is empty on this machine. The committed numbers in
`results/retrieval/nq_bge_metrics.json` record an index path under
`D:\Liuzheng\...`, i.e. they were produced on a teammate's machine. Rebuild
locally with the scripts in `SETUP.md`, or copy the prebuilt index across.
