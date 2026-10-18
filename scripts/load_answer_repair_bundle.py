"""Stdlib bootstrap for intact archives and Kaggle's automatically extracted inputs.

This source is embedded in the transport notebook before the frozen code exists
on disk. The checksums document itself is pinned outside the uploaded directory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_files(root: Path, expected_checksums_sha256: str) -> dict[str, str]:
    checksum_path = root / "bundle_checksums.json"
    if _digest(checksum_path) != expected_checksums_sha256:
        raise ValueError("Frozen checksums document hash mismatch")
    files = json.loads(checksum_path.read_bytes())
    if not isinstance(files, dict) or not files:
        raise ValueError("Invalid checksums document")
    for name, digest in files.items():
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
                or not isinstance(digest, str) or len(digest) != 64):
            raise ValueError("Invalid frozen file entry")
        target = root / name
        if not target.resolve().is_relative_to(root.resolve()) or not target.is_file() or target.is_symlink():
            raise ValueError(f"Frozen file missing or unsafe: {name}")
        if _digest(target) != digest:
            raise ValueError(f"Frozen file hash mismatch: {name}")
    return files


def load_frozen_bundle(*, work: Path, input_root: Path, bundle_name: str,
                       archive_sha256: str, checksums_sha256: str) -> tuple[Path, Path | None, dict]:
    """Copy only verified payloads to a writable directory, without changing them."""
    if not bundle_name.replace("_", "").isalnum():
        raise ValueError("Unsafe bundle name")
    archive_name = bundle_name + ".tar.gz"
    local_archive = work / archive_name
    archives = ([local_archive] if local_archive.is_file() else
                sorted(input_root.rglob(archive_name)) if input_root.exists() else [])
    if len(archives) > 1:
        raise ValueError("Multiple matching archives; attach one frozen bundle")
    staging = work / ("e_v8_" + archive_sha256[:12])
    destination = staging / bundle_name
    archive = archives[0] if archives else None
    if archive is not None:
        if _digest(archive) != archive_sha256:
            raise ValueError("Archive hash mismatch")
        if not destination.exists():
            staging.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as handle:
                handle.extractall(staging, filter="data")
        files = _verified_files(destination, checksums_sha256)
        mode = "intact_archive"
        source = str(archive)
    else:
        roots = ([p.parent for p in input_root.rglob("bundle_checksums.json")
                  if p.parent.name == bundle_name] if input_root.exists() else [])
        if len(roots) != 1:
            raise ValueError(f"Expected one archive or auto-extracted bundle, found {len(roots)} extracted roots")
        source_root = roots[0]
        files = _verified_files(source_root, checksums_sha256)
        destination.mkdir(parents=True, exist_ok=True)
        # Ignore unverified extra files rather than adding them to Python's import path.
        for name in [*files, "bundle_checksums.json"]:
            source_file, target = source_root / name, destination / name
            if target.exists():
                if _digest(target) != _digest(source_file):
                    raise ValueError(f"Existing writable bundle differs: {name}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, target)
        _verified_files(destination, checksums_sha256)
        mode, source = "kaggle_auto_extracted", str(source_root)
    report = {"transport": mode, "source": source, "bundle": str(destination),
              "expected_archive_sha256": archive_sha256,
              "archive_hash_verified": archive is not None,
              "checksums_sha256": checksums_sha256, "payload_hash_checked_files": len(files),
              "algorithm_or_input_modified": False}
    return destination, archive, report
