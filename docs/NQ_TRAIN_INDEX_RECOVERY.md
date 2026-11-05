# NQ Train index companion file recovery

This time the team's original 74,097-question subset and its retrieval store are kept, without expanding the NQ samples or rebuilding the full vector index. The corpus and original index come from the team's [nq_train_document](https://drive.google.com/drive/u/0/folders/1ZF9i_M-PJSzJSOW34rKPcAB1Ft-0QCQA) shared folder.

## What is recovered

| Local path (relative to the repository) | Purpose |
|---|---|
| `data/processed/retrieval/nq_train_corpus.jsonl` | Original corpus text retrieved from Drive |
| `data/processed/indexes/nq_train_bge_base.index` | Original FAISS IndexFlatIP retrieved from Drive |
| `data/processed/indexes/nq_train_bge_base_ids.jsonl` | Passage ID corresponding to the i-th vector |
| `data/processed/indexes/nq_train_bge_base_offsets.npy` | Position of the i-th original passage in the UTF-8 corpus file, int64 array |
| `data/processed/indexes/nq_train_bge_metadata.json` | Fingerprints and validation records of this recovery, clearly distinguished from the original index-build records |

The files match the `CorpusLookup.seek` reading method of the original retrieval program. The large corpus, indexes and generated sidecars are in the Git-ignored `data/processed/`; reproducible scripts, tests, hashes and validation summaries are committed.

The actual recovery on 2026-10-24 succeeded: IDs and offsets for 1,224,119 passages were all generated, and original-text location and independent BGE vector comparison at 64 positions passed. The minimum cosine similarity is 0.9999998808 and the maximum component difference is 3.39×10⁻⁷. What is counted here is the number of passages in the original retrieval store; the questions still come from the original 74,097-item subset. The actual model snapshot and full file fingerprints are in the validation directory.

## Why recovery is possible

The [original index-build code](../scripts/build_nq_train_bge_index.py) encodes the non-empty lines of the corpus file in their original order and appends vectors to the index in shard-number order. Therefore, when the original corpus and original index are truly paired, IDs and positions can be recovered by scanning the corpus, preserving the original order of every record, without re-encoding the full corpus.

The recovery tool does not automatically filter, reorder or correct records. Duplicate IDs, empty text, invalid JSON, or count or dimension mismatches all raise errors, and no partial output is published; existing companion files are not overwritten.

## Verification method and boundaries

The [recovery script](../scripts/recover_nq_train_index_sidecars.py) performs the following checks:

1. The original index is IndexFlatIP, 768-dim, with 1,224,119 vectors, all finite and L2-normalized.
2. Non-empty corpus lines match the index count, passage IDs are unique and text is non-empty; the original file SHA-256 is saved.
3. At 64 positions drawn with fixed seed 42 (including the first and last), read using the same text-file `seek` method as the original retrieval program, and check the IDs.
4. With `--verify-embeddings`, resolve and record a definite BGE revision, re-encode these 64 passages on CPU, and compare each with the vector of the corresponding row in the original index. Cosine similarity must be ≥0.9999 and the maximum component difference ≤0.001; on mismatch the companion files are not published.

Item 4 only checks numerical compatibility at the sampled positions and with the current snapshot; it cannot prove that all vectors are identical element by element, nor can this BGE revision be claimed to be the historical revision of the original C1. The original runtime environment and encoding duration were not recovered and remain unknown/null in the metadata.

## Crash check and low-memory operation

The macOS crash report of the original merge process confirms `SIGSEGV / EXC_BAD_ACCESS`, with the fault stack in `libomp.dylib` and three runtime library images of the same name in the process memory. The report did not record an OOM kill; this evidence supports runtime library compatibility as a direction of investigation, but the report alone cannot yet determine which two packages conflicted. The subsequent loss of the execution session caused by an application restart also does not equal a confirmed identical cause.

The current tool by default loads FlatIP with read-only disk mapping, relying on FAISS [`IO_FLAG_MMAP_IFC`](https://github.com/facebookresearch/faiss/blob/main/faiss/index_io.h) support; BGE sample encoding runs in a separate CPU subprocess, limiting the OpenMP, OpenBLAS, Accelerate and Torch encoding threads to one. This avoids holding the large index and the Torch encoding runtime in the same process at the same time. This method has completed the actual recovery; there is no need to switch CPU or GPU for this sidecar generation.

FAISS without Flat mmap support raises an explicit error and cannot silently fall back to large heap-memory loading. `--index-read-mode heap` is an explicit compatibility option; confirm the memory headroom yourself. The original corpus and index are not rewritten. Full-system Qwen experiments should still use the CUDA GPU platform chosen by the project.

These checks recover the usability of the retrieval resources; they do not compute QA metrics, and do not mean improved model accuracy, human review of the whole corpus, or acceptance of all Controller execution paths. FEVER resources, NQ Train Controller provider adaptation and the formal execution seal continue per the T2/T3 lists; the original evaluation held-out pool and final samples are not changed.

## Execution

First place the original corpus and index at the paths in the table, and install the dependencies required by the project, such as NumPy, FAISS CPU, Sentence Transformers and PyTorch. Recovery itself uses CPU; re-encoding the sampled passages downloads the BGE weights if they are missing from the local cache.

```bash
python scripts/recover_nq_train_index_sidecars.py --verify-embeddings --sample-count 64 --index-read-mode mmap
python -m unittest tests.test_nq_index_sidecar_recovery -v
```

The tool refuses to overwrite existing sidecars. Records without `--verify-embeddings` enabled explicitly write `performed=false` and cannot be treated as passing the vector order check.

The actual run and test results of this time are saved in the [validation directory](../artifacts/controller/nq_index_recovery_v1/README.md).

Subsequently, all 1,224,119 passages were checked item by item for ID/offset/original text, with an actual seek read-back; this and the earlier 64-passage vector comparison are two different checks. The Controller's new resource registry and offset provider are in the [resource integration notes](CONTROLLER_RESOURCE_ADAPTER_V1.md), and the original full location report is in [validation.json](../artifacts/controller/nq_mapping_full_v1/validation.json).
