"""Read-only corpus lookups and complete ID/offset validation; no model imports."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def passage(record, *, location):
    if not isinstance(record, dict):
        raise ValueError(f"Expected a corpus object at {location}")
    pid, text = record.get("passage_id"), record.get("text")
    if not isinstance(pid, str) or not pid.strip() or not isinstance(text, str) or not text.strip():
        raise ValueError(f"Invalid passage_id/text at {location}")
    # Annotation/source/answer fields in a corpus row never leave this module.
    return {"passage_id": pid, "text": text}


def read_id(line, *, location):
    row = json.loads(line)
    if (not isinstance(row, dict) or set(row) != {"passage_id"}
            or not isinstance(row["passage_id"], str) or not row["passage_id"].strip()):
        raise ValueError(f"Invalid ID record at {location}")
    return row["passage_id"]


def load_offsets(path):
    offsets = np.load(path, mmap_mode="r", allow_pickle=False)
    if offsets.ndim != 1 or offsets.dtype != np.dtype("int64") or not len(offsets):
        raise ValueError("Offsets must be a nonempty one-dimensional int64 array")
    if np.any(offsets < 0) or np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("Offsets must be nonnegative and strictly increasing")
    return offsets


def validate_complete_mapping(corpus, ids_path, offsets_path, metadata_path, *,
                              index_path=None, progress=None):
    """Check every ID and actual text-mode seek, independently of 64-vector QA.

    A second binary cursor establishes the true nonempty line boundaries, so
    matching IDs pointing to the wrong/duplicate line cannot pass. No embeddings
    are calculated and no question/gold files are read.
    """
    corpus, ids_path, offsets_path, metadata_path = map(Path, (corpus, ids_path, offsets_path, metadata_path))
    paths = [corpus, ids_path, offsets_path, metadata_path]
    if index_path is not None:
        index_path = Path(index_path)
        paths.append(index_path)
    before = {p: stamp(p) for p in paths}
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != "nq_train_index_recovery_v1":
        raise ValueError("Expected the registered NQ recovery metadata")
    offsets = load_offsets(offsets_path)
    if len(offsets) != meta["corpus_size"]:
        raise ValueError("Offsets count differs from recovery metadata")
    checks = {"offsets": (offsets_path, meta["offsets_fingerprint"]),
              "ids": (ids_path, meta["ids_fingerprint"])}
    if index_path is not None:
        checks["index"] = (index_path, meta["index_fingerprint"])
    fingerprints = {}
    for name, (path, expected) in checks.items():
        actual = file_sha(path)
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"{name} fingerprint differs from recovery metadata")
        fingerprints[name] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}

    count, seen, digest = 0, set(), hashlib.sha256()
    with corpus.open("rb") as sequential, corpus.open("r", encoding="utf-8") as lookup, ids_path.open(encoding="utf-8") as ids:
        while True:
            offset = sequential.tell()
            raw = sequential.readline()
            if not raw:
                break
            digest.update(raw)
            if not raw.strip():
                continue
            if count >= len(offsets):
                raise ValueError("Corpus has more nonempty rows than offsets")
            if int(offsets[count]) != offset:
                raise ValueError(f"Offset disagrees with original corpus line boundary at row {count}")
            expected = passage(json.loads(raw), location=f"corpus row {count}")
            line = ids.readline()
            if not line or read_id(line, location=f"ID row {count}") != expected["passage_id"]:
                raise ValueError(f"ID mapping disagrees at row {count}")
            if expected["passage_id"] in seen:
                raise ValueError(f"Duplicate corpus ID at row {count}")
            seen.add(expected["passage_id"])
            lookup.seek(int(offsets[count]))
            actual = passage(json.loads(lookup.readline()), location=f"seek row {count}")
            if actual != expected:
                raise ValueError(f"Seek readback disagrees at row {count}")
            count += 1
            if progress and count % 200_000 == 0:
                progress(count)
        if ids.read():
            raise ValueError("IDs file has extra rows")
    if count != len(offsets):
        raise ValueError("Corpus has fewer nonempty rows than offsets")
    expected = meta["corpus_fingerprint"]
    if digest.hexdigest() != expected["sha256"] or corpus.stat().st_size != expected["bytes"]:
        raise ValueError("Corpus fingerprint differs from recovery metadata")
    fingerprints["corpus"] = {"path": str(corpus.resolve()), "bytes": corpus.stat().st_size, "sha256": digest.hexdigest()}
    fingerprints["recovery_metadata"] = {"path": str(metadata_path.resolve()), "bytes": metadata_path.stat().st_size,
                                          "sha256": file_sha(metadata_path)}
    if any(stamp(path) != old for path, old in before.items()):
        raise ValueError("A resource changed during validation")
    return {"schema_version": "nq_complete_mapping_validation_v1", "passed": True,
            "records_checked": count, "actual_seek_reads": count,
            "index_sha256_verified": index_path is not None,
            "embedding_reencoding_performed": False,
            "scope": "All ID/offset/text mappings, not all vector-to-text embeddings",
            "prior_sampled_embedding_check": meta["embedding_alignment"], "fingerprints": fingerprints}


class OffsetCorpus(Sequence):
    """Bounded text cache; offsets stay memory-mapped, IDs are loaded once.

    Implements the existing backend's metadata Sequence API. All returned rows
    have exactly passage_id/text, and corpus/sidecar mutations fail explicitly.
    """
    def __init__(self, corpus, ids_path, offsets_path, *, cache_size=128):
        if type(cache_size) is not int or cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.paths = tuple(map(Path, (corpus, ids_path, offsets_path)))
        self.stamps = {p: stamp(p) for p in self.paths}
        self.offsets = load_offsets(self.paths[2])
        self.passage_ids = []
        seen = set()
        with self.paths[1].open(encoding="utf-8") as handle:
            for number, line in enumerate(handle):
                pid = read_id(line, location=f"ID row {number}")
                if pid in seen:
                    raise ValueError("Duplicate passage ID")
                self.passage_ids.append(pid)
                seen.add(pid)
        if len(self.passage_ids) != len(self.offsets):
            raise ValueError("ID and offset counts disagree")
        if int(self.offsets[-1]) >= self.paths[0].stat().st_size:
            raise ValueError("Offset outside corpus bounds")
        self.cache_size, self.cache = cache_size, OrderedDict()
        self.handle = self.paths[0].open("rb")

    def __len__(self):
        return len(self.passage_ids)

    def __getitem__(self, position):
        if isinstance(position, slice):
            raise TypeError("Read individual corpus rows; bulk text materialization is disabled")
        if isinstance(position, bool) or not isinstance(position, Integral):
            raise TypeError("Corpus position must be an integer")
        if self.handle.closed:
            raise RuntimeError("Corpus has been closed")
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(position)
        if any(stamp(path) != before for path, before in self.stamps.items()):
            raise ValueError("Loaded corpus or sidecars have changed")
        if position in self.cache:
            self.cache.move_to_end(position)
            return dict(self.cache[position])
        self.handle.seek(int(self.offsets[position]))
        row = passage(json.loads(self.handle.readline()), location=f"corpus row {position}")
        if row["passage_id"] != self.passage_ids[position]:
            raise ValueError(f"ID and corpus disagree at position {position}")
        self.cache[position] = row
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return dict(row)

    def close(self):
        self.cache.clear()
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
