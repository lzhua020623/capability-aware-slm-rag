"""Import selected archive members and map ordered IDs to UTF-8 corpus lines."""
from array import array
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from src.retrieval.offset_corpus import file_sha, passage, read_id, stamp


def extract_member(archive, member, destination):
    """No archive-controlled output paths; never replace an existing file.

    Reading the selected member to EOF also verifies its ZIP CRC. A failed
    extraction deletes only the partial file created by this call.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        matches = [i for i in source.infolist() if i.filename == member]
        if len(matches) != 1 or matches[0].is_dir():
            raise ValueError("Expected exactly one regular archive member")
        if (matches[0].external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("Archive member is a symbolic link")
        with destination.open("xb") as output:
            try:
                with source.open(matches[0]) as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        output.write(chunk)
            except BaseException:
                destination.unlink()
                raise
    return {"member": member, "bytes": destination.stat().st_size, "sha256": file_sha(destination)}


def build_ordered_offsets(corpus, ids, output, *, expected_count, progress=None):
    """Prove full corpus/ID order before publishing offsets; never encode vectors."""
    corpus, ids, output = map(Path, (corpus, ids, output))
    if output.exists():
        raise FileExistsError(output)
    before = {p: stamp(p) for p in (corpus, ids)}
    offsets, seen = array('q'), set()
    corpus_hash, ids_hash = hashlib.sha256(), hashlib.sha256()
    with corpus.open('rb') as body, ids.open('rb') as mapping:
        while True:
            position = body.tell()
            raw = body.readline()
            if not raw:
                break
            corpus_hash.update(raw)
            if not raw.strip():
                continue
            row = passage(json.loads(raw), location=f"corpus row {len(offsets)}")
            id_line = mapping.readline()
            ids_hash.update(id_line)
            if not id_line or read_id(id_line, location=f"ID row {len(offsets)}") != row['passage_id']:
                raise ValueError(f"Corpus/ID order mismatch at position {len(offsets)}")
            if row['passage_id'] in seen:
                raise ValueError("Duplicate corpus ID")
            seen.add(row['passage_id'])
            offsets.append(position)
            if progress and len(offsets) % 500_000 == 0:
                progress(len(offsets))
        if mapping.read():
            raise ValueError("IDs file has extra rows")
    if len(offsets) != expected_count:
        raise ValueError("Corpus/ID count differs from expected_count")
    if any(stamp(p) != old for p, old in before.items()):
        raise ValueError("Source changed during offset generation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        np.save(handle, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    return {"records_checked": len(offsets), "full_order_verified": True,
            "embedding_reencoding_performed": False,
            "fingerprints": {"corpus": {"bytes": corpus.stat().st_size, "sha256": corpus_hash.hexdigest()},
                             "ids": {"bytes": ids.stat().st_size, "sha256": ids_hash.hexdigest()},
                             "offsets": {"bytes": output.stat().st_size, "sha256": file_sha(output)}}}
