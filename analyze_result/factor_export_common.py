"""Output contract shared by the four independent factor exporters."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

SOURCE_POOL_SIZE = 100


def add_common_arguments(parser: argparse.ArgumentParser, project: Path) -> None:
    parser.add_argument("--instruments", choices=("csi300", "csi500"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--rank", type=int, help="Rank in result/native_combination_analysis/details_by_top10.csv")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--output-root", type=Path,
                        default=project.parent / "backtest_result")
    parser.add_argument("--layout", choices=("legacy", "grouped"), default="legacy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def validate_arguments(args: argparse.Namespace) -> None:
    from datetime import date

    if args.seed is not None and args.seed < 0:
        raise ValueError("seed must be >= 0")
    if args.pool_size is not None and args.pool_size < 1:
        raise ValueError("pool-size must be >= 1")
    if args.rank is not None and not 1 <= args.rank <= 10:
        raise ValueError("rank must be between 1 and 10")
    if date.fromisoformat(args.start_date) > date.fromisoformat(args.end_date):
        raise ValueError("start-date must be <= end-date")


def output_paths(args: argparse.Namespace, kind: str) -> tuple[Path, Path]:
    root = Path(args.output_root).expanduser().resolve()
    if args.rank is not None:
        base = root / kind / args.instruments / str(args.rank)
        return base, base / "factor"
    if args.layout == "grouped":
        base = (root / "unified" / args.instruments /
                f"seed_{args.seed}" / f"pool_{args.pool_size}")
    else:
        base = root
    return base, base / kind / "factor"


def ranked_jobs(project: Path, kind: str, args: argparse.Namespace) -> list[dict]:
    path = project / "result/native_combination_analysis/details_by_top10.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing top-ten ranking: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    jobs = []
    for row in rows:
        market, rank = row["instruments"], int(row["rank"])
        seed, size = int(row["seed"]), int(row["pool_size"])
        if market not in ("csi300", "csi500") or not 1 <= rank <= 10:
            continue
        if args.instruments is not None and market != args.instruments:
            continue
        if args.rank is not None and rank != args.rank:
            continue
        if args.seed is not None and seed != args.seed:
            continue
        if args.pool_size is not None and size != args.pool_size:
            continue
        jobs.append(dict(instruments=market, rank=rank, seed=seed,
                         pool_size=size, test_ic=float(row["test_ic"]),
                         test_ric=float(row["test_ric"])))
    if not jobs:
        raise ValueError(f"No ranking rows match filters in {path}")
    keys = [(job["instruments"], job["rank"]) for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate market/rank in {path}")
    return sorted(jobs, key=lambda job: (job["instruments"], job["rank"]))


def expected_ranked_jobs(project: Path, kind: str) -> list[dict]:
    args = argparse.Namespace(instruments=None, rank=None, seed=None, pool_size=None)
    jobs = ranked_jobs(project, kind, args)
    for market in ("csi300", "csi500"):
        ranks = {job["rank"] for job in jobs if job["instruments"] == market}
        if ranks != set(range(1, 11)):
            raise ValueError(f"{project}: {market} ranking must contain ranks 1..10")
    return jobs


def job_arguments(args: argparse.Namespace, job: dict) -> argparse.Namespace:
    result = argparse.Namespace(**vars(args))
    for key in ("instruments", "rank", "seed", "pool_size"):
        setattr(result, key, job[key])
    return result


def preview(args: argparse.Namespace, kind: str, source: Path,
            extra: dict | None = None) -> None:
    base, factor_dir = output_paths(args, kind)
    print(json.dumps({
        "framework": kind,
        "source": str(source),
        "layout": args.layout, "source_pool_size": SOURCE_POOL_SIZE,
        "backtest_base_dir": str(base),
        "factor_dir": str(factor_dir),
        **(extra or {}),
    }, ensure_ascii=False, indent=2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_output_available(args: argparse.Namespace, kind: str) -> None:
    _, factor_dir = output_paths(args, kind)
    if factor_dir.is_dir() and any(factor_dir.glob("????-??-??.csv")) and not args.overwrite:
        raise FileExistsError(
            f"Daily factor files already exist in {factor_dir}; use --overwrite to replace this group")


def write_daily_values(frame, args: argparse.Namespace, kind: str,
                       source: Path, extra: dict | None = None) -> Path:
    import numpy as np
    import pandas as pd

    required = {"date", "stock_code", "factor_value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Factor values are missing {sorted(required - set(frame.columns))}")
    frame = frame[["date", "stock_code", "factor_value"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame = frame[(frame["date"] >= args.start_date) &
                  (frame["date"] <= args.end_date)].copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["factor_value"] = pd.to_numeric(frame["factor_value"], errors="raise")
    frame = frame[np.isfinite(frame["factor_value"])].copy()
    if frame.empty:
        raise ValueError("No finite factor values in the requested date range")
    if frame.duplicated(["date", "stock_code"]).any():
        raise ValueError("Duplicate date/stock_code factor values")
    frame = frame.sort_values(["date", "factor_value", "stock_code"],
                              ascending=[True, False, True]).reset_index(drop=True)
    base, factor_dir = output_paths(args, kind)
    existing = sorted(factor_dir.glob("????-??-??.csv")) if factor_dir.is_dir() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Daily factor files already exist in {factor_dir}; use --overwrite to replace this group")
    if args.overwrite:
        for path in existing:
            path.unlink()
    factor_dir.mkdir(parents=True, exist_ok=True)
    for date, daily in frame.groupby("date", sort=True):
        daily[["stock_code", "factor_value"]].to_csv(
            factor_dir / f"{date}.csv", index=False)
    dates = frame["date"].unique().tolist()
    manifest = {
        "framework": kind, "instruments": args.instruments, "seed": args.seed,
        "pool_size": args.pool_size, "source_pool_size": SOURCE_POOL_SIZE,
        "start_date": args.start_date,
        "end_date": args.end_date, "layout": args.layout,
        "backtest_base_dir": str(base), "factor_dir": str(factor_dir),
        "source": str(source), "source_sha256": sha256(source),
        "n_dates": len(dates), "first_date": dates[0], "last_date": dates[-1],
        "n_rows": len(frame), **(extra or {}),
    }
    if args.rank is not None:
        manifest["rank"] = args.rank
    (factor_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {kind}: {len(frame)} rows across {len(dates)} dates to {factor_dir}")
    return factor_dir
