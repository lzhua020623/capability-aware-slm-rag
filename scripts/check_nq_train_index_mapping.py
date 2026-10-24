"""Read-only full ID/offset/corpus audit; does not load FAISS, Torch or gold."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.offset_corpus import validate_complete_mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/retrieval/nq_train_corpus.jsonl")
    parser.add_argument("--ids", type=Path, default=ROOT / "data/processed/indexes/nq_train_bge_base_ids.jsonl")
    parser.add_argument("--offsets", type=Path, default=ROOT / "data/processed/indexes/nq_train_bge_base_offsets.npy")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/processed/indexes/nq_train_bge_metadata.json")
    parser.add_argument("--index", type=Path, default=ROOT / "data/processed/indexes/nq_train_bge_base.index")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output file; existing validation records are preserved")
    report = validate_complete_mapping(args.corpus, args.ids, args.offsets, args.metadata, index_path=args.index,
        progress=lambda count: print(json.dumps({"checked": count}), file=sys.stderr, flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": True, "records_checked": report["records_checked"],
                      "actual_seek_reads": report["actual_seek_reads"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
