"""Offline completed-pilot audit and the registered 100-item semantic review."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.distillation.teacher_audit import audit_local


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    summary,gate=audit_local(args.output,ROOT)
    print(json.dumps({'summary':summary,'gate':gate,'output':str(args.output)},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
