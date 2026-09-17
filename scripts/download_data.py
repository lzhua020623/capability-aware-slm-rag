"""Download official Natural Questions and FEVER raw files."""

from __future__ import annotations

import json
import random
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from datasets import load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
NQ_DIR = ROOT / "data" / "raw" / "natural_questions"
NQ_DEV_DIR = NQ_DIR / "dev"
FEVER_DIR = ROOT / "data" / "raw" / "fever"
WIKI_ZIP = FEVER_DIR / "wiki-pages.zip"
WIKI_DIR = FEVER_DIR / "wiki-pages"
WIKI_URLS = [
    "https://fever.ai/download/fever/wiki-pages.zip",
    "https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip",
]
MIN_WIKI_JSONL = 50
WIKI_REQUIRED_FIELDS = ("id", "text", "lines")

NQ_DATASET = "google-research-datasets/natural_questions"
NQ_EXPECTED = 7830
NQ_REQUIRED_FIELDS = ("question", "document", "annotations", "long_answer_candidates")

FEVER_FILES = [
    ("https://fever.ai/download/fever/train.jsonl", "train.jsonl"),
    (
        "https://fever.ai/download/fever/shared_task_dev.jsonl",
        "shared_task_dev.jsonl",
    ),
]


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def make_reporthook(filename: str):
    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            downloaded = min(downloaded, total_size)
            percent = downloaded * 100.0 / total_size
            status = (
                f"{filename}: {percent:6.2f}% "
                f"({format_bytes(downloaded)} / {format_bytes(total_size)})"
            )
        else:
            status = f"{filename}: {format_bytes(downloaded)} downloaded"
        print(f"\r  {status}", end="", flush=True)

    return reporthook


def download_fever_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"SKIP (already exists): {dest.name}")
        print(f"  Saved to: {dest}")
        return

    tmp = dest.with_name(dest.name + ".part")
    print(f"Downloading {dest.name}")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=make_reporthook(dest.name))
        print()
        tmp.replace(dest)
        print(f"  Saved to: {dest}")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print()
        print(f"ERROR: failed to download {dest.name}")
        print(f"  URL: {url}")
        print(f"  {type(exc).__name__}: {exc}")
        if tmp.exists():
            tmp.unlink()


def download_with_fallback(urls: list[str], dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP (already exists): {dest.name} ({format_bytes(dest.stat().st_size)})")
        print(f"  Saved to: {dest}")
        return True

    tmp = dest.with_name(dest.name + ".part")
    for url in urls:
        print(f"Downloading {dest.name}")
        print(f"  URL: {url}")
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=make_reporthook(dest.name))
            print()
            if not tmp.exists() or tmp.stat().st_size <= 0:
                print(f"ERROR: downloaded file is empty from {url}")
                if tmp.exists():
                    tmp.unlink()
                continue
            tmp.replace(dest)
            print(f"  Saved to: {dest}")
            print(f"  Size: {format_bytes(dest.stat().st_size)}")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print()
            print(f"ERROR: failed to download {dest.name} from {url}")
            print(f"  {type(exc).__name__}: {exc}")
            if tmp.exists():
                tmp.unlink()
    print(f"ERROR: all download URLs failed for {dest.name}")
    return False


def wiki_jsonl_files(wiki_dir: Path) -> list[Path]:
    if not wiki_dir.exists():
        return []
    return sorted(path for path in wiki_dir.rglob("*.jsonl") if path.is_file())


def extract_wiki_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if names and all(
            Path(name).parts and Path(name).parts[0] == dest_dir.name for name in names
        ):
            archive.extractall(dest_dir.parent)
        else:
            archive.extractall(dest_dir)


def read_first_jsonl_record(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                if isinstance(record, dict):
                    return record
    raise ValueError(f"no JSON object found in {path}")


def download_fever_wikipedia() -> bool:
    print("=== FEVER Wikipedia corpus ===")
    if not download_with_fallback(WIKI_URLS, WIKI_ZIP):
        return False
    print()

    existing = wiki_jsonl_files(WIKI_DIR)
    if len(existing) >= MIN_WIKI_JSONL:
        print(
            f"SKIP extract (found {len(existing)} .jsonl files in {WIKI_DIR})"
        )
    else:
        if not WIKI_ZIP.exists() or WIKI_ZIP.stat().st_size <= 0:
            print(f"ERROR: missing zip for extract: {WIKI_ZIP}")
            return False
        print(f"Extracting {WIKI_ZIP} -> {WIKI_DIR}")
        try:
            extract_wiki_zip(WIKI_ZIP, WIKI_DIR)
        except (OSError, zipfile.BadZipFile) as exc:
            print(f"ERROR: failed to extract {WIKI_ZIP}: {type(exc).__name__}: {exc}")
            return False

    files = wiki_jsonl_files(WIKI_DIR)
    print(f"FEVER Wikipedia corpus files: {len(files)}")
    if len(files) < MIN_WIKI_JSONL:
        print(
            f"ERROR: expected many wiki .jsonl files "
            f"(at least {MIN_WIKI_JSONL}), found {len(files)}"
        )
        return False

    sample_path = random.choice(files)
    try:
        record = read_first_jsonl_record(sample_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: failed to read wiki page from {sample_path}: {exc}")
        return False

    missing = [field for field in WIKI_REQUIRED_FIELDS if field not in record]
    if missing:
        print(f"ERROR: wiki page missing fields: {', '.join(missing)}")
        print(f"  file: {sample_path}")
        return False

    print(f"Sample wiki file: {sample_path.name}")
    print(f"first page id: {record.get('id')}")
    print("Wiki page fields OK: id, text, lines")
    return True


def count_jsonl(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify_nq_dataset(dataset) -> tuple[bool, int]:
    count = len(dataset)
    if count != NQ_EXPECTED:
        print(f"ERROR: Natural Questions expected {NQ_EXPECTED} samples, got {count}")
        return False, count
    first = dataset[0]
    missing = [field for field in NQ_REQUIRED_FIELDS if field not in first]
    if missing:
        print(f"ERROR: first sample missing fields: {', '.join(missing)}")
        return False, count
    print("Natural Questions first-sample fields OK:")
    for field in NQ_REQUIRED_FIELDS:
        print(f"  {field}")
    return True, count


def load_existing_nq():
    if not NQ_DEV_DIR.exists():
        return None
    try:
        dataset = load_from_disk(str(NQ_DEV_DIR))
    except Exception as exc:
        print(f"Existing NQ data could not be loaded ({exc}); will re-download.")
        return None
    if len(dataset) != NQ_EXPECTED:
        print(
            f"Existing NQ data has {len(dataset)} samples "
            f"(expected {NQ_EXPECTED}); will re-download."
        )
        return None
    print(f"SKIP Natural Questions (found {NQ_EXPECTED} samples at {NQ_DEV_DIR})")
    return dataset


def download_natural_questions():
    existing = load_existing_nq()
    if existing is not None:
        return existing

    print(f"Downloading Natural Questions from Hugging Face: {NQ_DATASET}")
    try:
        dataset = load_dataset(NQ_DATASET, "dev", split="validation")
    except Exception as exc:
        print(f"Config 'dev' unavailable ({type(exc).__name__}: {exc})")
        print("Falling back to default config, split='validation'.")
        dataset = load_dataset(NQ_DATASET, split="validation")

    NQ_DEV_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving Natural Questions to {NQ_DEV_DIR}")
    dataset.save_to_disk(str(NQ_DEV_DIR))
    print("Reloading Natural Questions with load_from_disk() for verification.")
    return load_from_disk(str(NQ_DEV_DIR))


def main() -> int:
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "capability-aware-slm-rag/1.0")]
    urllib.request.install_opener(opener)

    print("=== Natural Questions ===")
    nq_ok = False
    nq_count: int | None = None
    try:
        nq_dataset = download_natural_questions()
        nq_ok, nq_count = verify_nq_dataset(nq_dataset)
    except Exception as exc:
        print(f"ERROR: Natural Questions download/verification failed: {type(exc).__name__}: {exc}")
    print()

    print("=== FEVER ===")
    for url, name in FEVER_FILES:
        download_fever_file(url, FEVER_DIR / name)
        print()

    wiki_ok = False
    wiki_count: int | None = None
    try:
        wiki_ok = download_fever_wikipedia()
        if wiki_ok:
            wiki_count = len(wiki_jsonl_files(WIKI_DIR))
    except Exception as exc:
        print(
            "ERROR: FEVER Wikipedia corpus download/verification failed: "
            f"{type(exc).__name__}: {exc}"
        )
    print()

    fever_train = count_jsonl(FEVER_DIR / "train.jsonl")
    fever_dev = count_jsonl(FEVER_DIR / "shared_task_dev.jsonl")
    fever_ok = fever_train is not None and fever_dev is not None

    nq_display = nq_count if nq_count is not None else "FAILED"
    fever_train_display = fever_train if fever_train is not None else "FAILED"
    fever_dev_display = fever_dev if fever_dev is not None else "FAILED"
    wiki_display = wiki_count if wiki_count is not None else "FAILED"

    print(f"Natural Questions validation: {nq_display} samples")
    print(f"FEVER train: {fever_train_display}")
    print(f"FEVER dev: {fever_dev_display}")
    print(f"FEVER Wikipedia corpus files: {wiki_display}")

    if nq_ok and fever_ok and wiki_ok:
        print("DOWNLOAD COMPLETE")
        return 0
    print("DOWNLOAD PARTIALLY COMPLETE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
