"""Recover NQ Train ID/offset sidecars without re-encoding the corpus.

The tracked builder appends vectors in nonempty corpus-line order. Recovery
preserves that order, validates the existing FlatIP index and can independently
re-encode sampled passages. Historical build metadata is not fabricated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_nq_train_bge_index.py"
MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NAMES = ("nq_train_bge_base_ids.jsonl", "nq_train_bge_base_offsets.npy",
         "nq_train_bge_metadata.json")


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def sample_positions(count, limit):
    if count < 1 or limit < 1:
        raise ValueError("Positive corpus and sample counts required")
    if limit >= count:
        return list(range(count))
    chosen = {0, count - 1} if limit >= 2 else {0}
    rng = random.Random(42)
    while len(chosen) < limit:
        chosen.add(rng.randrange(count))
    return sorted(chosen)


def encode_bge_in_child(texts, provenance):
    """Keep Torch/OpenMP and the large FAISS allocation in separate processes."""
    with tempfile.TemporaryDirectory(prefix="nq_bge_verify_") as folder:
        folder = Path(folder)
        source, result, info = folder / "texts.json", folder / "vectors.npy", folder / "model.json"
        source.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        subprocess.run([sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()),
                        "--encode-samples-input", str(source), "--encode-samples-output", str(result),
                        "--encode-samples-info", str(info)], check=True, env=env)
        provenance.update(json.loads(info.read_text(encoding="utf-8")))
        return np.load(result, allow_pickle=False)


def encode_sample_files(source, output, info):
    # This branch never imports FAISS. It reads only sampled corpus passages,
    # never gold answers or final evaluation questions.
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(1)
    revision = HfApi().model_info(MODEL).sha
    model = SentenceTransformer(MODEL, revision=revision, device="cpu", trust_remote_code=False)
    texts = json.loads(source.read_text(encoding="utf-8"))
    vectors = model.encode(texts, batch_size=8, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    with output.open("xb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    with info.open("x", encoding="utf-8") as handle:
        json.dump({"model_name": MODEL, "revision": revision, "device": "cpu",
                   "passage_prefix": "", "max_seq_length": model.max_seq_length,
                   "historical_revision_claim": False,
                   "execution": "isolated CPU process; no large FAISS index loaded"}, handle, indent=2)


def recover(corpus, index_path, output_dir, *, expected_count=1_224_119,
            dimension=768, sample_count=64, encoder=None, encoder_provenance=None,
            index_read_mode="mmap", progress=None):
    import faiss

    corpus, index_path, output_dir = map(Path, (corpus, index_path, output_dir))
    if expected_count < 1 or dimension < 1:
        raise ValueError("Positive expected count/dimension required")
    targets = [output_dir / name for name in NAMES]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError("Existing sidecars will not be overwritten")
    emit = progress or (lambda stage, **details: None)
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in (corpus, index_path)}
    if index_read_mode not in ("mmap", "heap"):
        raise ValueError("Index read mode must be mmap or heap")
    flags = 0
    if index_read_mode == "mmap":
        if not hasattr(faiss, "IO_FLAG_MMAP_IFC"):
            raise RuntimeError("Read-only Flat mmap requires a supporting FAISS version; use --index-read-mode heap explicitly if memory permits")
        flags = faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY
    emit("loading_index", mode=index_read_mode)
    index = faiss.read_index(str(index_path), flags)
    if (type(index).__name__ != "IndexFlatIP" or index.metric_type != faiss.METRIC_INNER_PRODUCT
            or index.d != dimension or index.ntotal != expected_count):
        raise ValueError("Index type, metric, dimension or count differs from expected FlatIP")
    output_dir.mkdir(parents=True, exist_ok=True)
    emit("scanning_corpus", expected_count=expected_count)
    published = []
    with tempfile.TemporaryDirectory(prefix=".recover_nq_", dir=output_dir) as temp:
        staging = Path(temp)
        offsets = np.empty(expected_count, dtype=np.int64)
        seen, count, digest = set(), 0, hashlib.sha256()
        with corpus.open("rb") as source, (staging / NAMES[0]).open("w", encoding="utf-8", newline="\n") as ids:
            line_number = 0
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                line_number += 1
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"Invalid JSON/UTF-8 at corpus line {line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object at corpus line {line_number}")
                pid, text = record.get("passage_id"), record.get("text")
                if not isinstance(pid, str) or not pid.strip() or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing passage_id/text at corpus line {line_number}")
                if pid in seen:
                    raise ValueError(f"Duplicate passage_id at corpus line {line_number}")
                if count >= expected_count:
                    raise ValueError("Corpus contains more passages than index")
                seen.add(pid)
                offsets[count] = offset
                ids.write(json.dumps({"passage_id": pid}, ensure_ascii=False) + "\n")
                count += 1
                if count % 200_000 == 0:
                    emit("scanning_corpus", passages=count)
        if count != expected_count:
            raise ValueError(f"Corpus contains {count} passages; expected {expected_count}")
        with (staging / NAMES[1]).open("wb") as handle:
            np.save(handle, offsets, allow_pickle=False)

        # Verify every stored vector is finite and normalized, without loading
        # an additional full copy of the 3.76 GB vector matrix.
        emit("checking_vector_norms", count=count)
        norm_min, norm_max = float("inf"), float("-inf")
        for start in range(0, count, 8192):
            vectors = index.reconstruct_n(start, min(8192, count - start))
            if not np.isfinite(vectors).all():
                raise ValueError("Index contains nonfinite vectors")
            norms = np.linalg.norm(vectors, axis=1)
            norm_min, norm_max = min(norm_min, float(norms.min())), max(norm_max, float(norms.max()))
            if not np.allclose(norms, 1, atol=1e-4, rtol=0):
                raise ValueError("Index vectors are not L2 normalized")

        positions = sample_positions(count, sample_count)
        texts, sampled_ids = [], []
        # Text-mode seeking deliberately matches the original CorpusLookup.
        with corpus.open("r", encoding="utf-8") as handle:
            for position in positions:
                handle.seek(int(offsets[position]))
                record = json.loads(handle.readline())
                if record["passage_id"] not in seen:
                    raise ValueError("Recovered offset cannot locate a known passage")
                sampled_ids.append(record["passage_id"])
                texts.append(record["text"])
        # Verify selected ID rows independently of the corpus seek.
        selected_set, selected_ids = set(positions), []
        with (staging / NAMES[0]).open(encoding="utf-8") as handle:
            for position, line in enumerate(handle):
                if position in selected_set:
                    selected_ids.append(json.loads(line)["passage_id"])
        if selected_ids != sampled_ids:
            raise ValueError("Recovered ID and offset rows disagree")

        embedding_check = {"performed": encoder is not None, "sample_count": len(positions),
                           "positions": positions, "scope": "Sampled positions only; not every passage embedding"}
        if encoder is not None:
            emit("checking_sample_embeddings", samples=len(positions), execution="isolated CPU child")
            actual = np.asarray(encoder(texts), dtype=np.float32)
            expected = np.stack([index.reconstruct(p) for p in positions])
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError("Verification encoder returned invalid embeddings")
            lengths = np.linalg.norm(actual, axis=1)
            if not np.allclose(lengths, 1, atol=1e-4, rtol=0):
                raise ValueError("Verification encoder must return normalized vectors")
            cosine = np.einsum("ij,ij->i", actual, expected) / (lengths * np.linalg.norm(expected, axis=1))
            differences = np.max(np.abs(actual - expected), axis=1)
            embedding_check.update({"model": encoder_provenance,
                "minimum_cosine_similarity": float(cosine.min()),
                "maximum_absolute_component_difference": float(differences.max()),
                "thresholds": {"minimum_cosine": 0.9999, "maximum_component_difference": 0.001}})
            if np.any(cosine < 0.9999) or np.any(differences > 0.001):
                raise ValueError("Sampled embeddings disagree with index row order or encoder configuration")
            embedding_check["passed"] = True

        corpus_info = {"path": str(corpus.resolve()), "bytes": before[corpus][0], "sha256": digest.hexdigest()}
        emit("hashing_original_index")
        index_info = fingerprint(index_path)
        for path, stamp in before.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != stamp:
                raise ValueError("Input changed during recovery")
        metadata = {
            "schema_version": "nq_train_index_recovery_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "recovered_sidecars": True,
            "original_build_metadata_recovered": False,
            "model_name": MODEL,
            "model_name_provenance": "Declared by tracked builder; sampled compatibility check recorded separately",
            "original_model_revision": None,
            "original_build_environment": None,
            "embedding_dimension": index.d, "index_type": type(index).__name__,
            "index_access": {"mode": index_read_mode, "faiss_io_flags": flags,
                             "faiss_version": faiss.__version__},
            "similarity_metric": "inner_product", "normalize_embeddings": True,
            "query_instruction": QUERY_PREFIX, "passage_prefix": "",
            "corpus_size": count, "corpus_fingerprint": corpus_info,
            "index_fingerprint": index_info,
            "ids_fingerprint": {**fingerprint(staging / NAMES[0]), "path": str(targets[0].resolve())},
            "offsets_fingerprint": {**fingerprint(staging / NAMES[1]), "path": str(targets[1].resolve())},
            "order_rule": "Nonempty corpus lines in original file order; no filtering, sorting or re-encoding",
            "offset_format": "int64 byte positions at UTF-8 line boundaries; compatible with original text-mode CorpusLookup.seek",
            "builder_source": fingerprint(BUILDER),
            "vector_norm_validation": {"count": count, "minimum": norm_min, "maximum": norm_max},
            "offset_id_readback": {"checked": len(positions), "passed": True},
            "embedding_alignment": embedding_check,
            "original_encode_elapsed_seconds": None,
            "full_T2_ready": False,
        }
        (staging / NAMES[2]).write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        try:
            # Same-directory hard links publish complete files exclusively.
            # If another process creates a destination, preserve its file.
            for name, destination in zip(NAMES, targets):
                os.link(staging / name, destination)
                published.append(destination)
        except BaseException:
            for destination in published:
                destination.unlink()
            raise
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/retrieval/nq_train_corpus.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / "data/processed/indexes/nq_train_bge_base.index")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/indexes")
    parser.add_argument("--expected-count", type=int, default=1_224_119)
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--index-read-mode", choices=("mmap", "heap"), default="mmap",
                        help="Default read-only mmap keeps the large vector matrix out of the Python heap")
    parser.add_argument("--verify-embeddings", action="store_true", help="CPU re-encode sampled passages using a resolved BGE snapshot")
    parser.add_argument("--encode-samples-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--encode-samples-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--encode-samples-info", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.encode_samples_input is not None:
        if args.encode_samples_output is None or args.encode_samples_info is None:
            parser.error("Sample encoding requires input, output and model-info paths")
        encode_sample_files(args.encode_samples_input, args.encode_samples_output, args.encode_samples_info)
        return 0
    encoder, provenance = None, None
    if args.verify_embeddings:
        provenance = {}
        def encoder(texts):
            return encode_bge_in_child(texts, provenance)
    report = recover(args.corpus, args.index, args.output_dir,
        expected_count=args.expected_count, dimension=args.dimension,
        sample_count=args.sample_count, encoder=encoder, encoder_provenance=provenance,
        index_read_mode=args.index_read_mode,
        progress=lambda stage, **details: print(json.dumps({"stage": stage, **details}), file=sys.stderr, flush=True))
    print(json.dumps({"created": [str(args.output_dir / name) for name in NAMES],
        "passages": report["corpus_size"], "embedding_alignment": report["embedding_alignment"],
        "full_T2_ready": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
