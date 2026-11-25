"""Shared paths and CSV helpers for sampled UDFS candidate analysis."""

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
METHOD = 'UDFS_final'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def update_summary(root, name, rows, keys=('instruments', 'seed')):
    path = Path(root) / name
    old = []
    if path.exists():
        with path.open(encoding='utf-8', newline='') as handle:
            old = list(csv.DictReader(handle))
    merged = {tuple(str(row[k]) for k in keys): row for row in old + rows}
    fields = list(dict.fromkeys(k for row in rows + old for k in row))
    write_csv(path, (merged[k] for k in sorted(merged)), fields)


def add_arguments(parser):
    parser.add_argument('--instruments', nargs='+', default=['csi300'])
    parser.add_argument('--seed', '--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--result-root', type=Path, default=PROJECT / 'result')
    parser.add_argument('--analysis-root', type=Path, default=None,
                        help='Defaults to result-root; output by instruments/seed')
    parser.add_argument('--input', type=Path, help='Candidate CSV for a single group')
    parser.add_argument('--dry-run', action='store_true')


def jobs(args, parser, filename='result_final.csv'):
    if args.analysis_root is None:
        args.analysis_root = args.result_root
    if args.input and len(args.instruments) * len(args.seed) != 1:
        parser.error('--input requires a single market/seed')
    tasks = []
    for market in dict.fromkeys(args.instruments):
        if market not in ('csi300', 'csi500'):
            parser.error('Expected csi300 or csi500')
        for seed in dict.fromkeys(args.seed):
            if seed < 0:
                parser.error('seed must be nonnegative')
            root = args.result_root.expanduser().resolve() / market / str(seed)
            path = args.input.expanduser().resolve() if args.input else root / filename
            if not path.is_file():
                parser.error('Missing candidate result: ' + str(path))
            tasks.append(dict(instruments=market, seed=seed, root=root, input=path,
                              output=args.analysis_root.expanduser().resolve() / market / str(seed)))
    return tasks


def load_candidates(path, instruments, seed):
    """Read sampled expressions, retaining every row and stable candidate IDs."""
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if 'expr' not in (reader.fieldnames or ()):
            raise ValueError(f'Missing expr column in {path}')
        rows = []
        for line, original in enumerate(reader, 2):
            for key, expected in (('instruments', instruments), ('seed', seed),
                                  ('method', METHOD)):
                if original.get(key) not in (None, '') and str(original[key]) != str(expected):
                    raise ValueError(f'Input group/method mismatch: {key} at {path}:{line}')
            expression = (original['expr'] or '').strip()
            if not expression:
                raise ValueError(f'Missing expression at {path}:{line}')
            eid = hashlib.sha256((METHOD + '\0' + expression).encode()).hexdigest()
            rows.append(dict(method=METHOD, instruments=instruments, seed=seed,
                             expr=expression, expr_id=eid, source_row=line,
                             candidate_id=f'{eid}:{line}'))
    if not rows:
        raise ValueError(f'Empty candidate result: {path}')
    return rows, {'source': str(path.resolve()), 'source_sha256': digest(path),
                  'expression_column': 'expr',
                  'ic_policy': 'sampled expressions only; IC requires calculate_ic.py'}
