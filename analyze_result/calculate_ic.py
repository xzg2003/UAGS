"""Calculate native signed daily cross-sectional IC for sampled UDFS candidates.

The input is ``result/{market}/{seed}/result_final.csv`` produced by
the UDFS search. This stage computes train/valid/test Pearson IC with the
project's native expression evaluator; it does not calculate RankIC.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_result.result_io import (  # noqa: E402
    PROJECT, add_arguments, jobs, load_candidates, update_summary, write_csv, write_json,
)
from analyze_result.combine_factors import parse_expression  # noqa: E402
from expr_cal import expression as native  # noqa: E402
from expr_cal import stock_data as stock  # noqa: E402


SPLITS = ("train", "valid", "test")
METRIC_FIELDS = (
    "method", "instruments", "seed", "expr", "expr_id", "source_row", "candidate_id",
    *[field for split in SPLITS for field in
      (f"ic_{split}", f"status_{split}", f"error_{split}")],
)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_signature(root, market):
    root = Path(root).expanduser().resolve()
    constituent = root / f"{market}_constituents.csv"
    files = sorted((root / "1d").glob("*/*.csv"))
    if not files:
        raise ValueError("No daily stock CSV files found in " + str(root))
    state = hashlib.sha256()
    for path in files:
        stat = path.stat()
        state.update(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return {
        "root": str(root),
        "stock_file_count": len(files),
        "stock_stats_sha256": state.hexdigest(),
        "instruments_sha256": file_digest(constituent),
    }


class NativeData:
    """Load one split and calculate native factor values and forward labels."""

    def __init__(self, args, market, start, end):
        import numpy as np

        old_dir = stock.DATA_DIR
        try:
            stock.DATA_DIR = str(args.data_dir)
            self.data = stock.Stock_Data(stock_list=market, k_line="1d",
                                         start_date=start, end_date=end)
        finally:
            stock.DATA_DIR = old_dir
        close = native.Feature("close")
        label = ((native.Ref(close, -2) / native.Ref(close, -1) - 1) * 1000).evaluate(self.data)
        self.target = np.asarray(label, dtype=float).reshape(-1)
        self.indices = list(self.data.test_slice[:-2])
        if not self.indices:
            raise ValueError(f"No evaluation dates after removing final two: {start} to {end}")
        self.split_info = {
            "days": len(self.indices),
            "first_date": str(self.data.data.loc[self.indices[0], "date"].iloc[0].date()),
            "last_date": str(self.data.data.loc[self.indices[-1], "date"].iloc[0].date()),
            "stock_count_max": max(len(index) for index in self.indices),
        }

    def factor(self, source):
        import numpy as np

        value = parse_expression(source, native).evaluate(self.data)
        value = np.asarray(value.values if hasattr(value, "values") else value,
                           dtype=float).reshape(-1)
        if len(value) != len(self.target):
            raise ValueError("Factor length differs from target")
        return value

    def ic(self, value):
        import numpy as np

        daily = []
        for index in self.indices:
            x, y = value[index], self.target[index]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 2:
                continue
            x, y = x[valid], y[valid]
            x = x - x.mean()
            y = y - y.mean()
            denominator = np.linalg.norm(x) * np.linalg.norm(y)
            if denominator <= 1e-12:
                continue
            daily.append(float(np.dot(x, y) / denominator))
        if not daily:
            raise ValueError("Constant factor or no valid daily cross sections")
        return float(np.mean(daily))


def completed_evaluations(indices, executor, evaluate, workers, heartbeat):
    iterator = iter(indices)
    running = {}

    def submit():
        try:
            index = next(iterator)
        except StopIteration:
            return False
        running[executor.submit(evaluate, index)] = index
        return True

    for _ in range(max(1, workers) * 2):
        if not submit():
            break
    while running:
        done, _ = wait(running, timeout=30, return_when=FIRST_COMPLETED)
        if not done:
            heartbeat()
            continue
        for future in done:
            index = running.pop(future)
            yield index, future.result()
            submit()


def evaluate_one(row, data, split):
    try:
        value = data.factor(row["expr"])
        ic = data.ic(value)
        if not math.isfinite(ic):
            raise ValueError("Non-finite IC")
        return {f"ic_{split}": ic, f"status_{split}": "ok", f"error_{split}": ""}
    except Exception as error:
        return {f"ic_{split}": "", f"status_{split}": "evaluation_error",
                f"error_{split}": f"{type(error).__name__}: {error}"}


def run(args, task):
    rows, source = load_candidates(task["input"], task["instruments"], task["seed"])
    stock_root = args.data_dir.expanduser().resolve() if args.data_dir else Path(stock.DATA_DIR).resolve()
    effective_args = argparse.Namespace(**vars(args))
    effective_args.data_dir = stock_root
    dates = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test": (args.test_start, args.test_end),
    }
    signature = data_signature(stock_root, task["instruments"])
    source_names = (
        "analyze_result/calculate_ic.py", "analyze_result/combine_factors.py",
        "analyze_result/result_io.py", "expr_cal/expression.py",
        "expr_cal/stock_data.py", "torch_utils/pytorch_utils.py",
    )
    contract = {
        "schema_version": 1,
        "input": str(task["input"]),
        "input_sha256": file_digest(task["input"]),
        "arguments": {"instruments": task["instruments"], "seed": task["seed"],
                      "data_dir": str(stock_root), **{f"{s}_{side}": value
                      for s, (start, end) in dates.items() for side, value in
                      (("start", start), ("end", end))}},
        "sources_sha256": {name: file_digest(PROJECT / name) for name in source_names},
        "data_signature": signature,
    }
    output = Path(task["output"])
    path, metadata_path = output / "factor_ic.csv", output / "ic_metadata.json"
    output_rows = [{key: row[key] for key in METRIC_FIELDS[:7]} for row in rows]
    old = None
    if not args.recompute and (path.exists() or metadata_path.exists()):
        if not path.exists() or not metadata_path.exists():
            raise ValueError(f"Incomplete IC files: {output}; use --recompute")
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("contract") != contract:
            raise ValueError(f"IC input/data/settings/code changed: {path}; use --recompute")
        if old.get("complete") and old.get("output_sha256") != file_digest(path):
            raise ValueError(f"Completed IC file was modified: {path}; use --recompute")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            cached = list(csv.DictReader(handle))
        if [row["candidate_id"] for row in output_rows] != [row.get("candidate_id") for row in cached]:
            raise ValueError(f"Cached IC candidates differ from input: {path}; use --recompute")
        for row, previous in zip(output_rows, cached):
            for field in METRIC_FIELDS[7:]:
                row[field] = previous.get(field, "")
        if old.get("complete"):
            print(json.dumps({"output": str(path), "candidates": len(output_rows), "cached": True}), flush=True)
            return
    for row in output_rows:
        for field in METRIC_FIELDS[7:]:
            row.setdefault(field, "")
    metadata = {
        "contract": contract, "complete": False, "instruments": task["instruments"],
        "seed": task["seed"], "candidates": len(output_rows), "source": source,
        "target": "(Ref(close,-2)/Ref(close,-1)-1)*1000",
        "metric_policy": "signed mean of daily cross-sectional Pearson IC; omit invalid pairs and final two dates",
        "split_policy": "exclude final two trading dates and invalid factor/label pairs; no zero filling",
        "splits": (old or {}).get("splits", {}), "workers": args.ic_workers,
        "checkpoint_every": args.checkpoint_every,
    }
    write_csv(path, output_rows, METRIC_FIELDS)
    write_json(metadata_path, metadata)

    workers = max(1, args.ic_workers)
    label = f"{task['instruments']}/{task['seed']}"
    print(f"{label}: {len(output_rows)} candidates, IC only, {workers} worker(s)", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for split, (start, end) in dates.items():
            pending = [i for i, row in enumerate(output_rows)
                       if row[f"status_{split}"] not in ("ok", "evaluation_error")]
            resumed = len(output_rows) - len(pending)
            print(f"{label} {split}: {resumed}/{len(output_rows)} already saved; {len(pending)} remaining", flush=True)
            if not pending:
                continue
            print(f"{label} {split}: loading market data and preparing labels", flush=True)
            data = NativeData(effective_args, task["instruments"], start, end)
            metadata["splits"][split] = data.split_info
            print(f"{label} {split}: market data ready; evaluating factors", flush=True)
            completed = 0
            last_notice = time.monotonic()

            def progress():
                nonlocal last_notice
                print(f"{label} {split}: {resumed + completed}/{len(output_rows)} completed "
                      f"({completed}/{len(pending)} this run)", flush=True)
                last_notice = time.monotonic()

            for index, values in completed_evaluations(
                    pending, executor, lambda i: evaluate_one(output_rows[i], data, split),
                    workers, progress):
                output_rows[index].update(values)
                completed += 1
                if completed % args.checkpoint_every == 0:
                    write_csv(path, output_rows, METRIC_FIELDS)
                    write_json(metadata_path, metadata)
                    progress()
                elif time.monotonic() - last_notice >= 30:
                    progress()
            write_csv(path, output_rows, METRIC_FIELDS)
            write_json(metadata_path, metadata)
            if completed % args.checkpoint_every:
                progress()
    metadata["split_counts"] = {
        split: {"ok": sum(row[f"status_{split}"] == "ok" for row in output_rows),
                "errors": sum(row[f"status_{split}"] != "ok" for row in output_rows)}
        for split in SPLITS
    }
    metadata.update(complete=True, output_sha256=file_digest(path))
    write_json(metadata_path, metadata)
    report = {"instruments": task["instruments"], "seed": task["seed"],
              "candidates": len(output_rows), "output": str(path)}
    for split in SPLITS:
        report[f"{split}_ok"] = metadata["split_counts"][split]["ok"]
        report[f"{split}_errors"] = metadata["split_counts"][split]["errors"]
    update_summary(args.analysis_root, "ic_summary.csv", [report])
    print(json.dumps(report), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--data-dir", type=Path, default=None)
    for key, value in zip(
            ("train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"),
            ("2010-01-01", "2018-12-31", "2019-01-01", "2019-12-31", "2020-01-01", "2021-12-31")):
        parser.add_argument("--" + key.replace("_", "-"), default=value)
    parser.add_argument("--ic-workers", "--workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)
    if args.ic_workers < 1 or args.checkpoint_every < 1:
        parser.error("--ic-workers/--workers and --checkpoint-every must be positive")
    try:
        spans = [datetime.fromisoformat(getattr(args, key)) for key in (
            "train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end")]
    except ValueError:
        parser.error("Dates must use YYYY-MM-DD")
    if not spans[0] <= spans[1] < spans[2] <= spans[3] < spans[4] <= spans[5]:
        parser.error("Train/validation/test dates must be ordered and disjoint")
    tasks = jobs(args, parser, "result_final.csv")
    if args.dry_run:
        print(json.dumps(tasks, default=str, indent=2))
        return
    for task in tasks:
        run(args, task)


if __name__ == "__main__":
    main()
