"""Independently reconstruct Phase 5-7 weights from validation-only Git blobs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REVISION = "338ff34b039ed367220b0b916b48c4f7fc715e20"
LABELS = ("correct", "evidence_insufficiency", "evidence_accessibility", "evidence_utilisation")


def unique(rows):
    indexed = {row["sample_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate validation sample_id")
    return indexed


def recompute(repo):
    hashes = {}
    def read(path):
        raw = subprocess.check_output(["git", "show", REVISION + ":" + path], cwd=repo)
        hashes[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    decisions = unique(read("results/controller/controller_v1_validation.json")["decisions"])
    predictions = unique(read("results/diagnosis/diagnosis_v2_expanded_validation.json")["predictions"])
    samples = unique(read("results/probe/probe_v2_expanded_validation.json")["samples"])
    if decisions.keys() != predictions.keys() or decisions.keys() != samples.keys():
        raise ValueError("validation membership differs across sources")
    counts = Counter()
    nq, eligible, excluded, selected = 0, 0, [], []
    for sid, decision in decisions.items():
        diagnosis, probe = predictions[sid], samples[sid]
        if decision["dataset"] != diagnosis["dataset"] or decision["dataset"] != probe["dataset"]:
            raise ValueError("validation dataset mismatch")
        if decision["dataset"] != "nq":
            continue
        nq += 1
        offline = probe["offline_evaluation"]
        label = diagnosis["target_label"]
        if label not in LABELS or label != offline["target_label"]:
            raise ValueError("validation target_label mismatch")
        if type(offline["prompt_development_sample"]) is not bool:
            raise ValueError("invalid Prompt Development flag")
        if offline["prompt_development_sample"]:
            excluded.append(sid)
            continue
        eligible += 1
        if decision["action"] == "retrieval_enhancement":
            counts[label] += 1
            selected.append({"sample_id": sid, "target_label": label, "action": decision["action"]})
    observed = {"validation_nq": nq, "eligible": eligible, "excluded": len(excluded),
                "routed": len(selected), "counts": {label: counts[label] for label in LABELS}}
    # These numbers are an acceptance gate, never the source of the weights.
    if (nq, eligible, len(excluded), len(selected), [counts[l] for l in LABELS]) != (406, 405, 1, 49, [19, 23, 0, 7]):
        raise ValueError("STOP: routing recount differs; no weights computed: " + json.dumps(observed))
    weights = {label: counts[label] / len(selected) for label in LABELS}
    return {"revision": REVISION, "source_sha256": hashes,
            "validation_nq": nq, "after_prompt_development_exclusion": eligible,
            "prompt_development_excluded_ids": excluded, "routed": len(selected),
            "counts": dict(observed["counts"]), "weights": weights,
            "note": "Counts recomputed from final Controller actions joined to Diagnosis targets; Probe exclusion flags applied. Validation only; no test files read."}, selected


def export(repo, output_dir):
    weights, selected = recompute(repo)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "routing_weights.json").write_text(json.dumps(weights, indent=2) + "\n")
    (output_dir / "routed_samples.json").write_text(json.dumps(selected, indent=2) + "\n")
    script = output_dir / Path(__file__).name
    shutil.copyfile(__file__, script)
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output_dir.iterdir()}
    (output_dir / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path,
                        help="Local repository containing the pinned origin/reranking commit")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New directory for weights, selected IDs, this script, and checksums")
    args = parser.parse_args()
    print(json.dumps(export(args.repo, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
