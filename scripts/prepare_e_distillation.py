"""Prepare frozen E pilot IDs and offline tasks, without API or model calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.team_nq import TeamNQBoundary, read_json
from src.distillation.e_prep import (build_sample_manifest, export_pilot,
                                    load_protocol, validate_sample_manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--export-pilot', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args(argv)
    if args.source and not args.export_pilot:
        parser.error('--source requires --export-pilot')
    boundary, protocol = TeamNQBoundary(ROOT), load_protocol(ROOT)
    if args.manifest.exists():
        manifest = validate_sample_manifest(boundary, protocol, read_json(args.manifest))
    else:
        manifest = build_sample_manifest(boundary, protocol)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    result = {'manifest': str(args.manifest), 'ID_counts':
              {role: len(ids) for role, ids in manifest['sample_ids'].items()},
              'teacher_API_calls': 0, 'training_runs': 0}
    if args.export_pilot:
        source = args.source or ROOT / boundary.policy['source_delivery']['default_local_directory']
        result['pilot'] = export_pilot(boundary, protocol, manifest, source, args.export_pilot)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
