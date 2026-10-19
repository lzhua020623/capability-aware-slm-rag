"""Prepare the separately frozen r3 candidate-contract smoke; never overwrite r2."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_answer_repair_v8_r2 import main as prepare_revision


def main(argv=None):
    return prepare_revision(argv, revision="r3")

if __name__ == "__main__":
    raise SystemExit(main())
