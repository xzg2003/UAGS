# UDFS：股票因子表达式搜索与评估

本项目使用 DAG 搜索股票因子表达式，并用原生表达式计算器评估因子。搜索先以默认窗口 5 评估图，再为结果中的滑动算子采样窗口长度；采样后的因子需要重新计算 IC，之后才能用于组合拟合。

## 环境安装

需要 Python 3.9 和本地股票日线数据。核心依赖列在 [`environment.yml`](environment.yml)。GPU 组合拟合需要另行安装与本机 CUDA 匹配的 CuPy；没有 GPU 时使用 `--device cpu`。Qlib 回测和数据采集脚本使用额外依赖，不包含在核心环境中。

```bash
conda env create -f environment.yml
conda activate UAGS
python -m unittest discover -s analyze_result/tests -v
```

## 行情数据

设置 `UDFS_DATA_DIR` 指向数据根目录。未设置时沿用原实验路径 `/public/home/wzx2003/stock_dataset`。`calculate_ic.py` 和 `combine_factors.py` 也接受 `--data-dir`，用于覆盖各自的行情目录；搜索使用环境变量。

```text
$UDFS_DATA_DIR/
├── csi300_constituents.csv
├── csi500_constituents.csv
└── 1d/
    └── 000001.XSHE/
        └── 000001.XSHE.csv
```

成分股 CSV 需要 `stock_code` 列。每只股票的日线 CSV 需要 `date`（或 `datetime`）、`open`、`close`、`high`、`low`、`volume`；股票代码可由文件中的 `stock_code` / `order_book_id` 或目录名提供。默认实验日期为训练集 2010–2018、验证集 2019、测试集 2020–2021。运行前请确认数据覆盖这些日期。

```bash
export UDFS_DATA_DIR=/path/to/stock_dataset
```

## 搜索、复算 IC、拟合组合

下面的命令为 CSI 300 的一个随机种子创建独立的结果目录。搜索使用 5 个计算节点、最多 10000 个图构建顺序和 50 个进程，可能耗时较长；详细参数可用 `python main.py --help` 查看。

```bash
python main.py --instruments csi300 --seeds 0 \
  --windows 1 2 5 10 30 --result-root result_window_sampled

python analyze_result/calculate_ic.py --instruments csi300 --seed 0 \
  --result-root result_window_sampled --ic-workers 4

python analyze_result/combine_factors.py --instruments csi300 --seed 0 \
  --result-root result_window_sampled --n-factors 1 10 20 50 100 --device cpu
```

结果保存在 `result_window_sampled/{instruments}/{seed}/`：

| 文件或目录 | 含义 |
| --- | --- |
| `result.csv` | 搜索生成的原始表达式，只有 `expr` 列 |
| `result_final.csv` | 对原始表达式中的每个滑动算子采样窗口后得到的表达式，只有 `expr` 列 |
| `factor_ic.csv`、`ic_metadata.json` | 训练、验证、测试集的复算 Pearson IC、计算状态和输入校验 |
| `combination/` | 已选因子、组合权重、逐日指标、组合 IC 和检查点 |

搜索使用隐含窗口 5 评估原始图；`result_final.csv` 的窗口可能不同，所以应以 `calculate_ic.py` 复算后的 `factor_ic.csv` 作为组合输入。IC 脚本计算每日截面 Pearson 相关系数的平均值，目前不计算单因子 Rank IC。组合脚本按验证集 `abs(IC)` 筛选，在训练集过滤高相关因子并拟合权重，最后评估验证集和测试集的组合指标。

若 `result.csv` 已存在而 `result_final.csv` 缺失，重新运行 `main.py` 会从原始结果补做窗口采样。两者都存在时默认跳过；`--overwrite` 会重新搜索并覆盖。IC 和组合脚本支持检查点续算，具体规则见 [`analyze_result/WINDOW_SAMPLING.md`](analyze_result/WINDOW_SAMPLING.md)。

## 代码结构

- `main.py`：搜索与窗口采样入口。
- `DAG_search/`：计算图和 DAG 搜索。
- `expr_cal/`：行情载入和原生表达式求值。
- `analyze_result/`：IC、组合、组合统计和因子值导出。
- `backtest_pipeline/`：可选 Qlib 回测。
- `scripts/`：可选数据采集、转换和检查工具。

仓库未附带开源许可证；公开发布前应确定授权条款并添加 `LICENSE`。
