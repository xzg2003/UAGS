"""Analyze each framework's own combination runs by signed test IC/RIC."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


METHODS = {'alphaforge', 'alphagen', 'gp', 'UDFS_final'}
DETAIL_FIELDS = ['instruments', 'pool_size', 'seed', 'test_ic', 'test_ric']
TOP_DETAIL_FIELDS = ['rank', *DETAIL_FIELDS]
SUMMARY_FIELDS = ['instruments', 'ic_mean', 'ic_std', 'ric_mean', 'ric_std']
BY_POOL_FIELDS = ['instruments', 'pool_size', 'ic_mean', 'ic_std', 'ric_mean', 'ric_std']


def read_csv(path: Path, hashes: dict[str, str]) -> list[dict[str, str]]:
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_integer(value: str, name: str, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be an integer: {value!r}') from error
    if parsed < minimum:
        raise ValueError(f'{name} must be >= {minimum}: {value!r}')
    return parsed


def parse_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric: {value!r}') from error
    if not math.isfinite(parsed):
        raise ValueError(f'{name} must be finite: {value!r}')
    return parsed


def numeric_dirs(path: Path) -> list[Path]:
    return sorted((p for p in path.iterdir() if p.is_dir() and p.name.isdigit()),
                  key=lambda p: int(p.name)) if path.is_dir() else []


def issue(issues: list[dict], path: Path, reason: str) -> None:
    issues.append({'source': str(path), 'reason': reason})


def load_forge(project: Path, hashes: dict, issues: list[dict]) -> list[dict]:
    root = project / 'rolling_backtest_result'
    candidates = []
    for market in sorted(root.iterdir()) if root.is_dir() else []:
        if not market.is_dir():
            continue
        path = market / 'ic_summary_all.csv'
        if not path.is_file():
            continue
        seen = set()
        try:
            rows = read_csv(path, hashes)
        except (OSError, UnicodeError, csv.Error) as error:
            issue(issues, path, str(error))
            continue
        for line, row in enumerate(rows, 2):
            try:
                seed = parse_integer(row['seed'], 'seed')
                size = parse_integer(row['n_factors'], 'n_factors', 1)
                ic = parse_float(row['mean_ic'], 'mean_ic')
                ric = parse_float(row['mean_rank_ic'], 'mean_rank_ic')
                days = parse_integer(row['days'], 'days', 1)
                key = seed, size
                if key in seen:
                    raise ValueError(f'duplicate seed/pool_size: {key}')
                per_seed = market / f'seed_{seed}' / f'pool_{size}'
            except (KeyError, ValueError) as error:
                issue(issues, path, f'line {line}: {error}')
                continue
            if not (per_seed / 'daily_ic_all_years.csv').is_file():
                issue(issues, path, f'line {line}: missing native daily combination metrics for seed={seed}, pool={size}')
                continue
            seen.add(key)
            candidates.append(dict(method='alphaforge', instruments=market.name,
                                   pool_size=size, seed=seed, test_ic=ic, test_ric=ric,
                                   source_kind='native_rolling_combination',
                                   run_id=f'seed_{seed}/pool_{size}', checkpoint_step='',
                                   years=row.get('years', ''), days=days,
                                   actual_k=row.get('actual_n_factors_by_year', ''),
                                   source=str(path),
                                   selected_artifact=str(per_seed / 'daily_ic_all_years.csv')))
    return candidates


def load_alphagen_run(path: Path, market: str, size: int, seed: int,
                      hashes: dict, issues: list[dict], source_kind: str) -> dict | None:
    try:
        rows = read_csv(path, hashes)
    except (OSError, UnicodeError, csv.Error) as error:
        issue(issues, path, str(error))
        return None
    valid = []
    for line, row in enumerate(rows, 2):
        try:
            step = parse_integer(row['step'], 'step', 1)
            ic = parse_float(row['ic_test'], 'ic_test')
            ric = parse_float(row['rank_ic_test'], 'rank_ic_test')
            valid.append((ic, -step, ric, step))
        except (KeyError, ValueError) as error:
            issue(issues, path, f'line {line}: {error}')
    if not valid:
        issue(issues, path, 'no valid checkpoint metrics')
        return None
    ic, _, ric, step = max(valid)
    pool = path.parent / f'{step}_steps_pool.json'
    if not pool.is_file():
        issue(issues, path, f'missing pool checkpoint for selected step {step}: {pool}')
        return None
    return dict(method='alphagen', instruments=market, pool_size=size, seed=seed,
                test_ic=ic, test_ric=ric, source_kind=source_kind,
                run_id=path.parent.name, checkpoint_step=step, years='', days='',
                actual_k='', source=str(path), selected_artifact=str(pool))


def load_alphagen(project: Path, hashes: dict, issues: list[dict]) -> list[dict]:
    candidates = []
    for base_name, kind in [('result', 'native_pool_current'), ('out', 'native_pool_historical')]:
        root = project / base_name
        if not root.is_dir():
            continue
        for market in sorted(p for p in root.iterdir() if p.is_dir()):
            for capacity in numeric_dirs(market):
                size = int(capacity.name)
                for run in sorted(p for p in capacity.iterdir() if p.is_dir()):
                    if base_name == 'result':
                        if not run.name.isdigit():
                            continue
                        seed = int(run.name)
                    else:
                        seed_part = run.name.split('_', 1)[0]
                        if not seed_part.isdigit():
                            continue
                        seed = int(seed_part)
                    path = run / 'test_results.csv'
                    if not path.is_file():
                        continue
                    candidate = load_alphagen_run(path, market.name, size, seed, hashes, issues, kind)
                    if candidate:
                        candidates.append(candidate)
    # One value per market/capacity/seed. Prefer the current result/ run;
    # otherwise use the latest historical out/ run by its timestamped name.
    chosen = {}
    for row in candidates:
        key = row['instruments'], row['pool_size'], row['seed']
        priority = (row['source_kind'] == 'native_pool_current', row['run_id'])
        if key not in chosen or priority > chosen[key][0]:
            chosen[key] = priority, row
    return [value[1] for _, value in sorted(chosen.items())]


def load_gp_udfs(project: Path, method: str, hashes: dict, issues: list[dict]) -> list[dict]:
    root = project / 'result'
    candidates = []
    for market in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        for seed_dir in numeric_dirs(market):
            seed = int(seed_dir.name)
            combination_dir = seed_dir / 'combination' if method == 'UDFS_final' else seed_dir
            for path in sorted(combination_dir.glob('ic_*.csv')):
                tail = path.stem[3:]
                if not tail.isdigit() or int(tail) < 1:
                    continue
                size = int(tail)
                artifact = combination_dir / f'selected_factors_{size}.csv'
                if not artifact.is_file():
                    issue(issues, path, f'missing native selected factors: {artifact}')
                    continue
                try:
                    rows = read_csv(path, hashes)
                    if len(rows) != 1:
                        raise ValueError(f'expected one combination row, found {len(rows)}')
                    ic = parse_float(rows[0]['ic'], 'ic')
                    ric = parse_float(rows[0]['ric'], 'ric')
                except (OSError, UnicodeError, csv.Error, KeyError, ValueError) as error:
                    issue(issues, path, str(error))
                    continue
                candidates.append(dict(method=method, instruments=market.name,
                                       pool_size=size, seed=seed, test_ic=ic, test_ric=ric,
                                       source_kind=('native_combination' if method == 'UDFS_final'
                                                    else 'native_legacy_combination'),
                                       run_id=f'seed_{seed}/' + ('combination/' if method == 'UDFS_final' else '')
                                              + f'pool_{size}', checkpoint_step='',
                                       years='', days='', actual_k='', source=str(path),
                                       selected_artifact=str(artifact)))
    return candidates


def stats(rows: list[dict]) -> dict:
    ic = [row['test_ic'] for row in rows]
    ric = [row['test_ric'] for row in rows]
    return dict(ic_mean=statistics.mean(ic) if ic else None,
                ic_std=statistics.pstdev(ic) if ic else None,
                ric_mean=statistics.mean(ric) if ric else None,
                ric_std=statistics.pstdev(ric) if ric else None)


def rank_runs(candidates: list[dict], top_n: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    all_rows, top, summary, by_pool = [], [], [], []
    for market in sorted({row['instruments'] for row in candidates}):
        group = sorted((row for row in candidates if row['instruments'] == market),
                       key=lambda row: (-row['test_ic'], row['pool_size'], row['seed'],
                                        row['source_kind'], row['run_id'], row['source']))
        ranked = [dict(rank=i, **row) for i, row in enumerate(group, 1)]
        selected = ranked[:top_n]
        all_rows.extend(ranked)
        top.extend(selected)
        report = dict(instruments=market)
        report.update(stats(selected))
        summary.append(report)
        for size in sorted({r['pool_size'] for r in group}):
            subset = [r for r in group if r['pool_size'] == size]
            pool = dict(instruments=market, pool_size=size)
            pool.update(stats(subset))
            by_pool.append(pool)
    return all_rows, top, summary, by_pool


def run(project: Path, top_n: int, markets: list[str] | None = None,
        output_dir: Path | None = None, dry_run: bool = False) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    project = project.expanduser().resolve()
    method = project.name
    if method not in METHODS:
        raise ValueError(f'Unknown framework directory: {project}')
    if top_n != 10:
        raise ValueError('This report uses exactly ten top runs')
    hashes: dict[str, str] = {}
    issues: list[dict] = []
    if method == 'alphaforge':
        candidates = load_forge(project, hashes, issues)
    elif method == 'alphagen':
        candidates = load_alphagen(project, hashes, issues)
    else:
        candidates = load_gp_udfs(project, method, hashes, issues)
    if markets is not None:
        candidates = [r for r in candidates if r['instruments'] in markets]
        issues = [r for r in issues if any(f'/{market}/' in r['source'] for market in markets)]
        unavailable = sorted(set(markets) - {r['instruments'] for r in candidates})
        if unavailable:
            raise ValueError('No native combination results for: ' + ', '.join(unavailable))
    if not candidates:
        raise ValueError(f'No native combination results found for {method} in {project}')
    all_rows, top, summary, by_pool = rank_runs(candidates, top_n)
    print(json.dumps({'method': method, 'top_rows': len(top),
                      'pool_groups': len(by_pool), 'issues': len(issues)},
                     ensure_ascii=False, allow_nan=False))
    if not dry_run:
        output = (output_dir or (project / 'result' / 'native_combination_analysis')).expanduser().resolve()
        if output == project / 'result' or output == project:
            raise ValueError('Choose a dedicated output directory')
        write_csv(output / 'details_by_all.csv', all_rows, DETAIL_FIELDS)
        write_csv(output / 'details_by_top10.csv', top, TOP_DETAIL_FIELDS)
        write_csv(output / 'summary_by_top10.csv', summary, SUMMARY_FIELDS)
        write_csv(output / 'summary_by_pool_size.csv', by_pool, BY_POOL_FIELDS)
    return all_rows, top, summary, by_pool


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, help='Default: PROJECT/result/native_combination_analysis')
    parser.add_argument('--instruments', nargs='+', help='Default: discover all markets')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        run(args.project_root, 10, args.instruments, args.output_dir, args.dry_run)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
