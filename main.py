"""Search UDFS structures and save one sampled-window expression per graph.

The search still scores implicit rolling window 5. All saved candidates are
rescored by analyze_result/calculate_ic.py after window sampling.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from DAG_search import dag_search
from DAG_search.window_sampling import sample_windows
from expr_cal.expression import Feature, Ref
from expr_cal.stock_data import Stock_Data
from torch_utils.pytorch_utils import normalize_by_day


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", choices=("csi300", "csi500"), default="csi300")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 5, 10, 30],
                        help="Positive trading-day windows, sampled uniformly per rolling call")
    parser.add_argument("--result-root", type=Path, default=Path(__file__).resolve().parent / "result")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing result.csv files")
    args = parser.parse_args(argv)
    if not args.windows or any(window < 1 for window in args.windows) or len(set(args.windows)) != len(args.windows):
        parser.error("--windows must contain distinct positive integers")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be nonnegative")
    seeds = tuple(dict.fromkeys(args.seeds))
    root = args.result_root.expanduser().resolve() / args.instruments

    def paths(seed):
        directory = root / str(seed)
        return directory, directory / "result.csv", directory / "result_final.csv"

    # Avoid loading market data when every requested seed already has raw search output.
    needs_search = args.overwrite or any(
        not paths(seed)[1].exists() for seed in seeds)
    if needs_search:
        features = ["open", "close", "high", "low", "volume"]
        close = Feature("close")
        target = (Ref(close, -2) / Ref(close, -1) - 1) * 1000
        train_data = Stock_Data(stock_list=args.instruments, k_line="1d",
                                start_date="2010-01-01", end_date="2018-12-31")
        y_train = target.evaluate(train_data).fillna(0)
        y_train = normalize_by_day(y_train.values, train_data).reshape(1, -1)

    for seed in seeds:
        directory, raw_output, final_output = paths(seed)
        if raw_output.exists() and final_output.exists() and not args.overwrite:
            print(f"{args.instruments}/{seed}: result.csv and result_final.csv exist; skipped", flush=True)
            continue

        if raw_output.exists() and not args.overwrite:
            raw = pd.read_csv(raw_output)
            if "expr" not in raw.columns:
                raise ValueError(f"Missing expr column in {raw_output}")
            expressions = raw["expr"].fillna("").astype(str).tolist()
            print(f"{args.instruments}/{seed}: result.csv exists; skipped search and sampled result_final.csv", flush=True)
        else:
            est = dag_search.DAGRegressor(n_calc_nodes=5, processes=50,
                                          random_state=seed, max_orders=int(1e4),
                                          data=train_data, topk=100000)
            est.fit(len(features), y_train)
            expressions = [graph.evaluate_symbolic()[0] for graph in est.results["graphs"]]
            directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"expr": expressions}).to_csv(raw_output, index=False)
            print(f"{args.instruments}/{seed}: search saved result.csv", flush=True)

        rng = np.random.default_rng(seed)
        sampled = [sample_windows(expression, args.windows, rng) for expression in expressions]
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"expr": sampled}).to_csv(final_output, index=False)
        print(f"{args.instruments}/{seed}: saved {len(sampled)} sampled expressions to {final_output}",
              flush=True)


if __name__ == "__main__":
    main()
