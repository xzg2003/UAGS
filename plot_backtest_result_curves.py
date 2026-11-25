"""Plot native combination test IC/RIC against factor-pool size.

Each framework stores its current seed-aggregated results at
``result/native_combination_analysis/summary_by_pool_size.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHOD_DIRS = {
    "UAGS": "UDFS_final",
    "Alphagen": "alphagen",
    "Alphaforge": "alphaforge",
    "GP": "gp",
}
REQUIRED_COLUMNS = {
    "instruments", "pool_size", "ic_mean", "ic_std", "ric_mean", "ric_std"
}
METRICS = ("ic", "ric")


def load_results(
    kind: Sequence[str] | str = tuple(METHOD_DIRS),
    base_dir: str | Path = "/public/home/wzx2003",
) -> pd.DataFrame:
    """Read each framework's native combination summary, not backtest returns."""
    root_dir = Path(base_dir).expanduser()
    methods = [kind] if isinstance(kind, str) else list(kind)
    if not methods:
        raise ValueError("At least one method is required")

    frames: list[pd.DataFrame] = []
    for method in methods:
        if method not in METHOD_DIRS:
            raise ValueError(f"Unknown method {method!r}; choose from {list(METHOD_DIRS)}")
        csv_path = (
            root_dir / METHOD_DIRS[method] / "result"
            / "native_combination_analysis" / "summary_by_pool_size.csv"
        )
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing native combination summary: {csv_path}")

        frame = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
        frame = frame[list(sorted(REQUIRED_COLUMNS))].copy()
        frame["instruments"] = frame["instruments"].astype("string").str.strip()
        if frame.empty or frame["instruments"].isna().any() or frame["instruments"].eq("").any():
            raise ValueError(f"{csv_path}: empty results or instrument name")
        for column in ("pool_size", "ic_mean", "ic_std", "ric_mean", "ric_std"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[list(REQUIRED_COLUMNS - {"instruments"})].isna().any().any():
            raise ValueError(f"{csv_path}: missing pool size or IC/RIC statistic")
        if (frame["pool_size"] < 1).any() or (frame["pool_size"] % 1 != 0).any():
            raise ValueError(f"{csv_path}: pool sizes must be positive integers")
        if (frame[["ic_std", "ric_std"]] < 0).any().any():
            raise ValueError(f"{csv_path}: standard deviations must be nonnegative")
        if frame.duplicated(["instruments", "pool_size"]).any():
            raise ValueError(f"{csv_path}: duplicate instrument/pool size")
        frame["pool_size"] = frame["pool_size"].astype(int)
        frame["method"] = method
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["instruments", "pool_size", "method"]
    ).reset_index(drop=True)


def plot_metric(ax: plt.Axes, instrument_df: pd.DataFrame, metric: str) -> None:
    """Draw one mean curve per method for a market."""
    for method, method_df in instrument_df.groupby("method", sort=False):
        values = method_df.sort_values("pool_size")
        x = values["pool_size"].to_numpy(dtype=float)
        mean = values[f"{metric}_mean"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", linewidth=2, markersize=5, label=method)

    market = str(instrument_df["instruments"].iloc[0]).upper()
    ax.set_title(f"{market} test {metric.upper()} vs pool size")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(sorted(instrument_df["pool_size"].unique()))
    ax.grid(alpha=0.25)


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_instrument_curves(instrument_df: pd.DataFrame, output_dir: Path) -> Path:
    market = str(instrument_df["instruments"].iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, metric in zip(axes, METRICS):
        plot_metric(ax, instrument_df, metric)
    axes[-1].set_xlabel("Factor pool size")
    axes[0].legend(ncol=min(instrument_df["method"].nunique(), 4), loc="best")
    return _save_figure(fig, output_dir / f"{market}_ic_ric_vs_pool_size.png")


def plot_metric_curves(df: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    markets = list(df["instruments"].unique())
    fig, axes = plt.subplots(len(markets), 1, figsize=(10, 4 * len(markets)), squeeze=False)
    for ax, market in zip(axes[:, 0], markets):
        plot_metric(ax, df.loc[df["instruments"] == market], metric)
        ax.set_xlabel("Factor pool size")
    axes[0, 0].legend(ncol=min(df["method"].nunique(), 4), loc="best")
    return _save_figure(fig, output_dir / f"{metric}_vs_pool_size.png")


def plot_backtest_result_curves(
    kind: Sequence[str] | str = tuple(METHOD_DIRS),
    base_dir: str | Path = "/public/home/wzx2003",
    output_dir: str | Path | None = None,
) -> list[Path]:
    results = load_results(kind=kind, base_dir=base_dir)
    destination = (
        Path(output_dir).expanduser() if output_dir is not None else
        Path(base_dir).expanduser() / "backtest_result"
    )
    destination.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_instrument_curves(group, destination)
        for _, group in results.groupby("instruments", sort=True)
    ]
    paths.extend(plot_metric_curves(results, metric, destination) for metric in METRICS)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("/public/home/wzx2003"),
                        help="Parent directory of the four framework projects")
    parser.add_argument("--output-dir", type=Path,
                        help="Plot directory (default: <base-dir>/backtest_result)")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHOD_DIRS),
                        default=list(METHOD_DIRS), help="Methods to compare")
    args = parser.parse_args()
    for path in plot_backtest_result_curves(args.methods, args.base_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
