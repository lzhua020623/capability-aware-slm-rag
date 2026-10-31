"""Check the selected production registration without loading models or running inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.pipeline.current import readiness
from src.pipeline.registrations import DEFAULT_REGISTRATION, REGISTRATIONS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--registration', choices=REGISTRATIONS, default=DEFAULT_REGISTRATION)
    args = parser.parse_args(argv)
    receipt = readiness(ROOT, args.registration)
    if args.output:
        if args.output.exists():
            raise FileExistsError('Preserve the previous readiness receipt')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt['ready_for_gpu'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
