# FEVER500: 7B C0 / C3 run instructions

This branch executes FEVER C0 and C3 of the frozen Preliminary Recoverability Experiment.
The research goal is to determine whether Small-Model RAG errors can be repaired with correct evidence; C1 results for the same IDs must be merged later before the Recoverable Failure Rate can be computed. C0/C3 accuracy alone cannot substitute for this statistic.

## Fixed protocol

- Config: `configs/preliminary.yaml`, with model parameters also checked against `configs/base.yaml`.
- Samples: `configs/splits/preliminary_fever_500.json`, 500 unique IDs, 250 SUPPORTS + 250 REFUTES, keeping the manifest order, no resampling.
- Model: `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4, greedy, `max_new_tokens=64`, seed 42.
- C0: instruction + claim only; the model input contains no evidence or gold label, result `evidence=[]`.
- C3: instruction + all `gold_evidence_text` of the current sample + claim, following the shared FEVER evidence format, not truncated to Top-5, no retrieval.
- Evaluation: follows the label parsing of `src/evaluation/fever.py` and computes classification accuracy; UNKNOWN counts as wrong. Smoke raises an error if it outputs an unparseable label.

The original NQ/FEVER manifests, shared configs, prompt text and decoding parameters are all unmodified. To support pre-checks without model dependencies, prompt construction moved to `src/generation/prompts.py`, and the old `src.generation.qwen` import entry point remains usable.

## Preparation before renting a GPU

First obtain the team's preprocessed `data/processed/retrieval/fever_dev.jsonl` and place it at the same path.
The file must contain the claim, label, gold_page_ids and gold_evidence_text for the frozen IDs.
It is in gitignore, so **clone will not download it**. C0/C3 do not need the FEVER corpus, FAISS index, BGE or NQ data.
To rebuild from raw data, see [SETUP.md](SETUP.md); the existing full preprocessing script also processes the NQ and FEVER corpus.

Run from the repository root; the pre-check environment needs only Python and PyYAML:

```bash
python -m pip install pyyaml
python scripts/run_fever_preliminary.py --check-only
```

The pre-check validates the model config, manifest, source data, and whether existing results allow resuming. Missing data fails explicitly; it will not automatically sample or download the model.

## Running on the GPU machine

```bash
git clone --branch feature/F500-c0c3 https://github.com/lzhua020623/capability-aware-slm-rag.git
cd capability-aware-slm-rag
```

Place the above `fever_dev.jsonl` at the specified path, create a Python environment, and install the CUDA PyTorch suitable for the machine and `requirements.txt` per [SETUP.md](SETUP.md). For an existing repository, switch to this branch and run `git pull --ff-only`.

Run in order, continuing only after the previous step passes:

```bash
python scripts/run_fever_preliminary.py --check-only
python scripts/run_fever_preliminary.py --smoke-test
python -u scripts/run_fever_preliminary.py
```

Smoke runs C0 and C3 on the first frozen sample without writing formal results; a wrong model prediction does not mean an engineering failure.
The formal run has 1000 generations in total, each written to disk; after an interruption, running the same command skips completed records.
You can also run `--conditions C0` or `--conditions C3` separately. Only one writer process runs per result file.

Outputs:

- `results/preliminary/fever_c0_7b.jsonl`: 500 items.
- `results/preliminary/fever_c3_7b.jsonl`: 500 items.

At the end it outputs the overall accuracy, the accuracy of each of the two classes and the UNKNOWN count. Each result contains the raw output, evidence, duration and experiment fingerprint, for merging with C1 by sample_id.
Re-running a completed condition only checks and summarizes the results, without loading the model.

## Resuming and result protection

The experiment fingerprint binds the frozen sample content, prompts, model config and inference/evaluation code. Old models, different data/prompts, old results without fingerprints, duplicate IDs or inconsistent metrics all cause resuming to be refused. For a new run, first move the old results to a separate archive directory; do not mix writes.

Each result is flushed + fsynced after being written; if the last JSON is complete but missing a newline, a newline is added before appending. If a power loss leaves incomplete JSON, the script reports the specific file and line number and stops, keeping the original file. Back up first, confirm that only the last line is an incomplete write, then remove that last line and resume; do not ignore corrupted lines in the middle.

Resuming uses the same software environment; the fingerprint does not lock Python package versions or the remote model revision. It is recommended to save `python -m pip freeze` and the current `git rev-parse HEAD` at run time and hand them over together with the results.

## Local validation scope

```bash
python -m unittest discover -s tests -v
```

Regression tests use the real frozen IDs with synthetic claim/evidence and a mock generator, and validate in a temporary directory the 500 × 2 interrupted resume, input isolation, data validation and protection against mixed writes; they do not generate formal experiment results.
The initial code validation was completed in a local environment without CUDA. Subsequently, the formal data pre-check and a real 7B NF4 smoke passed on a GPU machine, and the 500 generations each for C0/C3 were completed; the downloaded results were also verified item by item locally. See [experiment results and verification record](../results/preliminary/fever_c0_c3_summary.md).
