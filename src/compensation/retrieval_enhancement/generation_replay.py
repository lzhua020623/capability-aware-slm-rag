"""Model-free GPU replay worker: strict inputs, answer guard, durable resume.

Gold/labels belong exclusively to the separate Mac evaluator. The generation
provider is injected; importing or testing this module never loads a model.
"""

from __future__ import annotations

try:
    import fcntl  # POSIX
except ImportError:  # Windows
    fcntl = None
    import msvcrt
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import uuid

from .p3 import answer_rejection, build_c1_prompt


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def payload_sha(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_generation_inputs(path):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if set(row) != {"sample_id", "question", "initial_answer", "evidence"}:
                raise ValueError("generation input must contain only the four runtime fields")
            for key in ("sample_id", "question", "initial_answer"):
                if not isinstance(row[key], str) or (key != "initial_answer" and not row[key].strip()):
                    raise ValueError(f"{key} must be a string (nonempty except initial_answer)")
            if row["sample_id"] in seen:
                raise ValueError("duplicate sample_id")
            seen.add(row["sample_id"])
            evidence = row["evidence"]
            if not isinstance(evidence, list) or len(evidence) != 5:
                raise ValueError("exactly five evidence records are required")
            ids = set()
            for rank, passage in enumerate(evidence, 1):
                if set(passage) != {"rank", "passage_id", "text"}:
                    raise ValueError("evidence must contain only rank/passage_id/text")
                if type(passage["rank"]) is not int or passage["rank"] != rank:
                    raise ValueError("evidence ranks must be ordered 1..5")
                for key in ("passage_id", "text"):
                    if not isinstance(passage[key], str) or not passage[key].strip():
                        raise ValueError(f"evidence {key} must be nonempty")
                if passage["passage_id"] in ids:
                    raise ValueError("duplicate evidence ID")
                ids.add(passage["passage_id"])
            rows.append(row)
    if not rows:
        raise ValueError("empty generation input")
    return rows


def _append(path, row):
    data = (json.dumps({**row, "_record_sha256": payload_sha(row)}, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            if n <= 0:
                raise OSError("journal append made no progress")
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)


def read_results(path, *, repair_tail=False):
    """Interior/checksummed complete corruption fails; only unfinished tail repairs.

    Repair is allowed only by the output writer under its exclusive lock. The
    Mac evaluator always uses repair_tail=False and never alters GPU results.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            try:
                row = json.loads(raw)
                digest = row.pop("_record_sha256")
                if payload_sha(row) != digest:
                    raise ValueError("checksum mismatch")
            except Exception as exc:
                if raw.endswith(b"\n") or not repair_tail:
                    raise ValueError(f"corrupt result journal at byte {offset}") from exc
                backup = path.with_name(path.name + f".partial.{uuid.uuid4().hex}.bin")
                with backup.open("xb") as out:
                    out.write(raw); out.flush(); os.fsync(out.fileno())
                with path.open("r+b") as out:
                    out.truncate(offset); out.flush(); os.fsync(out.fileno())
                break
            if not raw.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError("unterminated final result line")
                with path.open("ab") as out:
                    out.write(b"\n"); out.flush(); os.fsync(out.fileno())
            rows.append(row)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate result sample_id")
    return rows


def validate_result(row, source):
    """Recheck terminal answer decisions; used both at resume and offline scoring."""
    if row["sample_id"] != source["sample_id"] or row["generation_calls"] != 1:
        raise ValueError("result ID/call count mismatch")
    if row["status"] == "generation_failed":
        if row["raw_answer"] is not None or row["final_answer"] != source["initial_answer"] or not row["error"]:
            raise ValueError("invalid failed-generation fallback")
        return
    if not isinstance(row["raw_answer"], str) or row["error"] is not None:
        raise ValueError("invalid successful generation record")
    reason = answer_rejection(row["raw_answer"], source["evidence"])
    status = "kept_original" if reason else "answer_replaced"
    answer = source["initial_answer"] if reason else row["raw_answer"].strip()
    if (row["status"], row["final_answer"], row["reject_reason"]) != (status, answer, reason):
        raise ValueError("saved answer decision does not match frozen answer guard")


def run_generation(input_path, output_dir, generate_fn, *, run_metadata, resume=False, limit=None):
    """One append+fsync per sample; exceptions are terminal explicit fallbacks.

    KeyboardInterrupt/process termination leaves completed lines intact. A failed
    provider call is saved then this session stops; explicit resume moves on,
    never silently reruns/overwrites that failure. limit is engineering-only.
    Provider may load lazily on its first call; all preflight checks run first.
    """
    rows = load_generation_inputs(input_path)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    target = Path(output_dir)
    existing = target.exists()
    if existing and not resume:
        raise FileExistsError("output directory exists; use explicit --resume")
    if not existing:
        target.mkdir(parents=True, exist_ok=False)
    manifest = {"input_sha256": sha256_file(input_path), "n": len(rows), "run_metadata": run_metadata}
    with (target / "writer.lock").open("a+") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        config_path = target / "run_manifest.json"
        if existing:
            if json.loads(config_path.read_text()) != manifest:
                raise ValueError("resume input/config/source identity mismatch")
        else:
            with config_path.open("x") as out:
                json.dump(manifest, out, ensure_ascii=False, indent=2, allow_nan=False)
                out.flush(); os.fsync(out.fileno())
        results_path = target / "results.jsonl"
        prior = read_results(results_path, repair_tail=True)
        if len(prior) > len(rows) or [r["sample_id"] for r in prior] != [r["sample_id"] for r in rows[:len(prior)]]:
            raise ValueError("resume results must be a unique ordered input prefix")
        for result, source in zip(prior, rows):
            validate_result(result, source)
        started = perf_counter()
        done = 0
        failed = False
        pending = rows[len(prior):]
        for source in pending[:limit] if limit is not None else pending:
            t = perf_counter()
            prompt = build_c1_prompt(source["question"], source["evidence"])
            result = {"sample_id": source["sample_id"], "raw_answer": None,
                      "final_answer": source["initial_answer"], "status": "generation_failed",
                      "reject_reason": None, "error": None, "generation_calls": 1,
                      "latency": {"generation": 0.0, "validation": 0.0, "total": 0.0}}
            g = perf_counter()
            try:
                raw = generate_fn(prompt)
                result["latency"]["generation"] = perf_counter() - g
                if not isinstance(raw, str):
                    raise TypeError("generator must return a string")
                v = perf_counter()
                reason = answer_rejection(raw, source["evidence"])
                result["latency"]["validation"] = perf_counter() - v
                result.update(raw_answer=raw, reject_reason=reason,
                              final_answer=source["initial_answer"] if reason else raw.strip(),
                              status="kept_original" if reason else "answer_replaced")
            except Exception as exc:
                result["latency"]["generation"] = perf_counter() - g
                result["error"] = {"type": type(exc).__name__, "message": str(exc)}
                failed = True
            result["latency"]["total"] = perf_counter() - t
            _append(results_path, result)
            done += 1
            if (len(prior) + done) % 100 == 0:
                print(f"completed={len(prior)+done}/{len(rows)} session_seconds={perf_counter()-started:.1f}", flush=True)
            if failed:
                break
        summary = {"total": len(rows), "resumed": len(prior), "new_completed": done,
                   "completed": len(prior) + done, "remaining": len(rows) - len(prior) - done,
                   "stopped_on_error": failed, "session_seconds": perf_counter() - started}
        with (target / f"session_{uuid.uuid4().hex}.json").open("x") as out:
            json.dump(summary, out, indent=2)
        return summary
