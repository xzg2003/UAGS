"""Export one UDFS native selected-factor combination as UAGS backtest signals."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from factor_export_common import (
    add_common_arguments, ensure_output_available, preview,
    validate_arguments, write_daily_values, ranked_jobs, job_arguments,
    SOURCE_POOL_SIZE,
)
from analyze_result.combine_factors import parse_expression

KIND = "UAGS"


def source_path(args: argparse.Namespace) -> Path:
    path = (PROJECT / "result" / args.instruments / str(args.seed) /
            "combination" / f"selected_factors_{SOURCE_POOL_SIZE}.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Missing UDFS combination selected factors: {path}")
    return path


def compute_combined_values(source: Path, args: argparse.Namespace):
    import numpy as np
    import pandas as pd
    from expr_cal import expression as expressions
    from expr_cal.stock_data import Stock_Data
    from torch_utils.pytorch_utils import normalize_by_day

    factors = pd.read_csv(source)
    if "expr" not in factors or factors.empty or factors["expr"].isna().any():
        raise ValueError(f"{source} must contain nonempty expr values")
    if "weight" not in factors:
        factors["weight"] = 1.0  # Historical K=1 files contain only expr.
    factors["weight"] = pd.to_numeric(factors["weight"], errors="raise")
    if not np.isfinite(factors["weight"]).all():
        raise ValueError(f"Non-finite weights in {source}")

    warmup_start = (pd.Timestamp(args.start_date) -
                    pd.Timedelta(days=args.warmup_days)).strftime("%Y-%m-%d")
    data = Stock_Data(stock_list=args.instruments, k_line="1d",
                      start_date=warmup_start, end_date=args.end_date)
    if data.data.empty:
        raise ValueError("No price rows in the requested date range")
    columns = []
    for expr in factors["expr"].astype(str):
        # Combination files contain sampled expressions with explicit rolling
        # windows, for example ``Std(x_3, 2)``.  The old transform() helper
        # wraps every numeric literal as a Constant, which turns the window
        # into an expression object and makes pandas reject it.  Use the same
        # native AST parser as calculate_ic/combine_factors so window literals
        # remain integer delta times while ordinary constants are preserved.
        value = parse_expression(expr, expressions).evaluate(data)
        array = np.asarray(value.values if hasattr(value, "values") else value,
                           dtype=float).reshape(-1).copy()
        if len(array) != len(data.data):
            raise ValueError(f"Expression length mismatch: {expr}")
        array[~np.isfinite(array)] = 0.0
        columns.append(np.asarray(normalize_by_day(array, data), dtype=float).reshape(-1))
    matrix = np.column_stack(columns)
    matrix[~np.isfinite(matrix)] = 0.0
    combined = matrix @ factors["weight"].to_numpy(dtype=float)
    combined[~np.isfinite(combined)] = 0.0
    combined = np.asarray(normalize_by_day(combined, data), dtype=float).reshape(-1)
    frame = data.data[["date", "stock_code"]].copy()
    frame["factor_value"] = combined
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, PROJECT)
    parser.add_argument("--warmup-days", type=int, default=365)
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
        if args.warmup_days < 0:
            raise ValueError("warmup-days must be >= 0")
        for job in ranked_jobs(PROJECT, KIND, args):
            current = job_arguments(args, job)
            source = source_path(current)
            extra = {"combination": "native_selected_factors", "warmup_days": args.warmup_days,
                     "ranking_test_ic": job["test_ic"], "ranking_test_ric": job["test_ric"],
                     "ranking_pool_size": job["pool_size"]}
            if args.dry_run:
                preview(current, KIND, source, extra)
                continue
            ensure_output_available(current, KIND)
            write_daily_values(compute_combined_values(source, current), current, KIND, source, extra)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
