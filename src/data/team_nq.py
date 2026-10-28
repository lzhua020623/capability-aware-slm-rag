"""Strict NQ team roles, checked before decoding supervision or loading models.

Only IDs and registered receipts define membership. No runtime gold or score
decides whether a row belongs to train, validation or the sealed final test.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = 'configs/controller/current_baseline.json'
POLICY = 'configs/splits/team_nq_data_v1.json'
EXPECTED_COUNTS = {'train': 51866, 'validation': 11117, 'test': 11114}
# Byte hash of baselines/splits/diagnosis_nq_train_74097_split.json on the
# local formal checkout (CRLF-normalized content is identical to the LF
# historical fingerprint 7da01c94cd7b5f92076edebc80911ac891d3861a01cf6b40a1e5e6efb6482dc4).
TEAM_SPLIT_SHA256 = 'e6572c9b1c8f92dccceef2a85200a891ab33c01fe4775e4541c55e86f64de9db'
TEAM_SOURCE_COMMIT = '387ddafc82bc939ebd27b3137d580db6ecb2e04a'
ID_FIELD = re.compile(rb'(?<!\\)"sample_id"\s*:\s*"(-?\d+)"')
ID_ARRAY = re.compile(rb'(?<!\\)"(?:sample_ids|ids)"\s*:\s*\[([^\[\]]*)\]')


class BoundaryError(ValueError):
    """Unregistered data or a row outside its declared role."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BoundaryError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_object)


def numeric_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'-?\d+', value):
        raise BoundaryError('sample_id must be an exact decimal string, never a float/int')
    return value


def ids_only(path):
    """Read ID fields from saved reports without decoding questions/answers."""
    raw = Path(path).read_bytes()
    found = {v.decode('ascii') for v in ID_FIELD.findall(raw)}
    for body in ID_ARRAY.findall(raw):
        try:
            values = json.loads(b'[' + body + b']')
            if all(isinstance(v, str) and re.fullmatch(r'-?\d+', v) for v in values):
                found.update(values)
        except (ValueError, TypeError):
            pass
    return found


def query_key(value):
    if not isinstance(value, str) or not value.strip():
        raise BoundaryError('Missing query in training data')
    normalized = unicodedata.normalize('NFKC', value).casefold()
    key = ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c
                           for c in normalized).split())
    if not key:
        raise BoundaryError('Query normalizes to empty')
    return key


class TeamNQBoundary:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        try:
            registration = read_json(self.root / REGISTRATION)['data_boundary']
            if registration['path'] != POLICY:
                raise BoundaryError('Unregistered team boundary path')
            self.policy = self._checked(registration)
            if self.policy['schema_version'] != 'team_nq_boundary_v1':
                raise BoundaryError('Unknown team boundary version')
            if (self.policy['team_split']['sha256'] != TEAM_SPLIT_SHA256
                    or self.policy['source_commit'] != TEAM_SOURCE_COMMIT):
                raise BoundaryError('Frozen team split fingerprint changed; never re-split')
            split = self._checked(self.policy['team_split'])
            if split['version'] != self.policy['team_split']['version']:
                raise BoundaryError('Team split version changed')
            self.splits = split['splits']['nq']
            if set(self.splits) != set(EXPECTED_COUNTS):
                raise BoundaryError('Team split roles changed')
            for role, count in EXPECTED_COUNTS.items():
                ids = self.splits[role]
                if not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count:
                    raise BoundaryError('Team split count/uniqueness mismatch: ' + role)
                for sid in ids:
                    numeric_id(sid)
            self.membership = {}
            for role, ids in self.splits.items():
                for sid in ids:
                    if sid in self.membership:
                        raise BoundaryError('Team split ID overlap')
                    self.membership[sid] = role
            protected = self._checked(self.policy['legacy_reserve'])
            selected = self._checked(self.policy['legacy_selection'])
            self.legacy = set(protected['nq']) | set(selected['nq']['selection'])
            exposure = self._checked(self.policy['known_exposure'])
            self.exposed = set(exposure['observed_canonical_ids'])
            if not self.exposed <= self.membership.keys():
                raise BoundaryError('Exposure IDs outside registered population')
            for source in exposure['sources']:
                self._checked(source, decode=False)
            self._checked(self.policy['label_index'], decode=False)
            self.delivery = self._checked(self.policy['source_delivery']['manifest'])
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            raise BoundaryError('Team boundary missing or malformed: ' + str(exc)) from exc
        self.eligible = {
            'train': set(self.splits['train']) - self.legacy,
            'validation': set(self.splits['validation']) - self.legacy - self.exposed,
        }
        self._labels = None

    def source_paths(self, path):
        path = Path(path)
        if path.is_file():
            return [path]
        if not path.is_dir():
            raise BoundaryError('Source data path does not exist: ' + str(path))
        result = []
        for item in self.delivery['shards']:
            name = item['file']
            if Path(name).name != name:
                raise BoundaryError('Invalid delivery filename')
            source = path / name
            if not source.is_file() or sha(source) != item['sha256']:
                raise BoundaryError('Original C1 shard missing or changed: ' + name)
            result.append(source)
        return result

    def source_lines(self, path):
        # Complete checksum preflight occurs before any supervision is decoded.
        paths = self.source_paths(path)
        for source in paths:
            with source.open('rb') as handle:
                yield from handle

    @property
    def labels(self):
        if self._labels is None:
            self._labels = {}
            with (self.root / self.policy['label_index']['path']).open(encoding='utf-8') as handle:
                for line in handle:
                    row = json.loads(line)
                    self._labels[row['sample_id']] = row['primary_label']
        return self._labels

    def _checked(self, spec, *, decode=True):
        path = (self.root / spec['path']).resolve()
        if not path.is_relative_to(self.root.resolve()) or sha(path) != spec['sha256']:
            raise BoundaryError('Registered data missing or changed: ' + spec['path'])
        return read_json(path) if decode else None

    def allowed_ids(self, role):
        if role not in self.eligible:
            raise BoundaryError('Final test execution needs a separately sealed T7 plan')
        return [sid for sid in self.splits[role] if sid in self.eligible[role]]

    def require(self, ids, role):
        allowed = self.eligible.get(role)
        if allowed is None:
            raise BoundaryError('Only train/validation are available before final freeze')
        ids = list(ids)
        for sid in ids:
            numeric_id(sid)
            if sid not in self.membership:
                raise BoundaryError('Unknown source ID; external data needs separate registration')
            if sid not in allowed:
                raise BoundaryError('ID is outside allowed ' + role + ' boundary')
        return ids

    def require_record(self, row, role):
        if row.get('dataset', 'nq') != 'nq':
            raise BoundaryError('This registration covers NQ only; FEVER is not aligned yet')
        ids = [row.get('sample_id')]
        if (row.get('synthetic') is True or row.get('source_type') in ('synthetic', 'augmentation', 'counterfactual')):
            if not any(row.get(k) for k in ('source_sample_ids', 'parent_sample_ids')):
                raise BoundaryError('Synthetic data requires registered parent IDs')
        for key in ('source_sample_ids', 'parent_sample_ids'):
            if key in row:
                if not isinstance(row[key], list) or not row[key]:
                    raise BoundaryError('Invalid parent provenance: ' + key)
                ids.extend(row[key])
        if row.get('source') == 'borrowed_gold':
            if 'donor_sample_id' not in row:
                raise BoundaryError('Borrowed answer has no donor provenance; rebuild data')
        if 'donor_sample_id' in row:
            ids.append(row['donor_sample_id'])
        self.require(ids, role)

    def iter_role_records(self, path, roles=('train', 'validation'), *, counts=None):
        """Stream admitted rows; skipped roles never have supervision decoded."""
        if not roles or any(r not in self.eligible for r in roles):
            raise BoundaryError('Only train/validation source reads are permitted')
        seen = set()
        counts = Counter() if counts is None else counts
        for line in self.source_lines(path):
            if not line.strip():
                continue
            matches = ID_FIELD.findall(line)
            if len(matches) != 1:
                raise BoundaryError('Source row requires exactly one string sample_id')
            sid = matches[0].decode('ascii')
            if sid in seen or sid not in self.membership:
                raise BoundaryError('Duplicate or unknown source ID')
            seen.add(sid)
            role = self.membership[sid]
            counts['source_' + role] += 1
            if role not in roles or sid not in self.eligible[role]:
                counts['skipped_' + role] += 1
                continue
            row = json.loads(line, object_pairs_hook=_object)
            self.require_record(row, role)
            if 'primary_label' not in row:
                row['primary_label'] = self.labels[sid]
            yield role, row

    def read_role_records(self, path, roles=('train', 'validation')):
        """Skip test/protected rows by ID before decoding their supervision."""
        result, counts = {r: [] for r in roles}, Counter()
        for role, row in self.iter_role_records(path, roles, counts=counts):
            result[role].append(row)
        return result, dict(counts)

    def source_sha256(self, path):
        paths = self.source_paths(path)
        return {p.name: sha(p) for p in paths}

    def teacher_input(self, row):
        """Future teacher requests may only contain the five runtime fields."""
        self.require_record(row, 'train')
        if not isinstance(row.get('initial_answer'), str) or not row['initial_answer'].strip():
            raise BoundaryError('Teacher input has no original answer')
        query_key(row.get('query'))
        evidence = row.get('evidence')
        if not isinstance(evidence, list) or len(evidence) != 5:
            raise BoundaryError('Teacher input needs the exact Top-5 evidence')
        fields = ('rank', 'passage_id', 'text', 'score')
        ids = set()
        for rank, passage in enumerate(evidence, 1):
            if (not isinstance(passage, dict) or not set(fields) <= passage.keys()
                    or type(passage['rank']) is not int or passage['rank'] != rank
                    or not isinstance(passage['passage_id'], str) or not passage['passage_id']
                    or passage['passage_id'] in ids
                    or not isinstance(passage['text'], str) or not passage['text'].strip()
                    or type(passage['score']) not in (int, float) or not math.isfinite(passage['score'])):
                raise BoundaryError('Teacher evidence structure/ranks/score invalid')
            ids.add(passage['passage_id'])
        return {'sample_id': row['sample_id'], 'dataset': 'nq', 'query': row['query'],
                'initial_answer': row['initial_answer'],
                'evidence': [{k: passage[k] for k in fields} for passage in evidence]}

    def read_training_pair(self, directory):
        """Validate every row, including donors/parents, before model loading."""
        result = {}
        for role in ('train', 'validation'):
            rows, primary = [], set()
            with (Path(directory) / (role + '.jsonl')).open('rb') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    matches = ID_FIELD.findall(line)
                    if len(matches) != 1:
                        raise BoundaryError('Training row requires one exact sample_id')
                    self.require([matches[0].decode('ascii')], role)
                    row = json.loads(line, object_pairs_hook=_object)
                    self.require_record(row, role)
                    query_key(row.get('query'))
                    rows.append(row)
                    primary.add(row['sample_id'])
            if not rows:
                raise BoundaryError('Empty ' + role + ' data')
            for row in rows:
                if 'donor_sample_id' in row and row['donor_sample_id'] not in primary:
                    raise BoundaryError('Donor is absent from its own role file')
            result[role] = rows
        train_queries = {query_key(r['query']) for r in result['train']}
        if train_queries & {query_key(r['query']) for r in result['validation']}:
            raise BoundaryError('Normalized query crosses train/validation')
        return result

    def receipt(self):
        return {'policy_path': POLICY, 'policy_sha256': sha(self.root / POLICY),
                'split_sha256': self.policy['team_split']['sha256'],
                'official_counts': dict(EXPECTED_COUNTS),
                'eligible_counts': {r: len(v) for r, v in self.eligible.items()},
                'test_sealed': True, 'test_known_exposed': len(set(self.splits['test']) & self.exposed),
                'labels_semantically_verified': False, 'external_training_IDs_verified': False}

    def read_training_bundle(self, directory, *, component=None):
        directory = Path(directory)
        if not (directory / 'build_manifest.json').is_file():
            raise BoundaryError('Unregistered local training split; rebuild on the team roles')
        manifest = read_json(directory / 'build_manifest.json')
        if (manifest.get('schema_version') != 'team_nq_training_bundle_v1'
                or manifest.get('boundary') != self.receipt()
                or set(manifest.get('role_files', {})) != {'train', 'validation'}):
            raise BoundaryError('Training bundle has no matching team boundary receipt')
        if component is not None and manifest.get('component') != component:
            raise BoundaryError('Training bundle component does not match the trainer')
        if (directory / 'test.jsonl').exists():
            raise BoundaryError('Training bundle must not contain a test file')
        for role, spec in manifest['role_files'].items():
            if sha(directory / (role + '.jsonl')) != spec['sha256']:
                raise BoundaryError('Training file changed after role validation')
        rows = self.read_training_pair(directory)
        for role, values in rows.items():
            if sorted({r['sample_id'] for r in values}) != manifest['role_files'][role]['sample_ids']:
                raise BoundaryError('Training manifest ID membership changed')
        return rows, manifest


def prepare_training_bundle(boundary, records_path, output, builder, *, component):
    """Partition FIRST, then build positives/negatives within each team role."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve existing training data; choose a new output directory')
    source, counts = boundary.read_role_records(records_path)
    parts, reasons = {}, {}
    for role in ('train', 'validation'):
        parts[role], why = builder(source[role])
        reasons[role] = dict(why)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='team-nq-check-', dir=output.parent) as directory:
        staging = Path(directory)
        for role, rows in parts.items():
            (staging / (role + '.jsonl')).write_text(''.join(
                json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
        boundary.read_training_pair(staging)
        manifest = {'schema_version': 'team_nq_training_bundle_v1', 'component': component,
                    'boundary': boundary.receipt(), 'source_path': str(records_path),
                    'source_sha256': boundary.source_sha256(records_path), 'source_role_counts': counts,
                    'builder_reasons': reasons, 'offline_supervision_only': True,
                    'role_files': {role: {'sha256': sha(staging / (role + '.jsonl')),
                        'sample_ids': sorted({r['sample_id'] for r in rows}), 'examples': len(rows)}
                        for role, rows in parts.items()}}
        # Validate everything before claiming the fresh destination. No old
        # local split or experiment output is ever overwritten.
        output.mkdir(exist_ok=False)
        for role in parts:
            with (output / (role + '.jsonl')).open('xb') as handle:
                handle.write((staging / (role + '.jsonl')).read_bytes())
        with (output / 'build_manifest.json').open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
    return manifest
