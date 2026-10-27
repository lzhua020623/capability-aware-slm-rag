"""Validate E teacher pilot; network requires --execute and a local credential."""
import argparse
import getpass
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distillation.teacher_pilot import preflight, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--total-requests', type=int, default=3,
                        help='Cumulative cap in the one registered journal, not additional calls')
    parser.add_argument('--prompt-key', action='store_true', help='Read an existing API key without echo or storage')
    args = parser.parse_args(argv)
    if args.prompt_key and not args.execute:
        parser.error('--prompt-key requires --execute')
    if not args.execute:
        spec, tasks = preflight(ROOT)
        result = {'tasks_verified': len(tasks), 'model': spec['response_model'],
                  'API_calls': 0, 'training_runs': 0}
    else:
        environment = dict(os.environ)
        if args.prompt_key:
            environment['AZURE_OPENAI_API_KEY'] = getpass.getpass('Existing Azure API key (hidden): ')
        result = run(ROOT, total_requests=args.total_requests, environment=environment)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
