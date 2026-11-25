import torch
import pandas as pd
import numpy as np
from torch import Tensor
from scipy.stats import pearsonr,spearmanr, entropy
from torch_utils.pytorch_utils import normalize_by_day, masked_mean_std_tensor


# ========== Tensor 版本的相关系数计算函数 ==========

def _mask_either_nan(x: Tensor, y: Tensor, fill_with: float = torch.nan):
    """处理 NaN 值，创建掩码"""
    x = x.clone()                       # [days, stocks]
    y = y.clone()                       # [days, stocks]
    nan_mask = x.isnan() | y.isnan()
    x[nan_mask] = fill_with
    y[nan_mask] = fill_with
    n = (~nan_mask).sum(dim=1)
    return x, y, n, nan_mask


def _rank_data_1d(x: Tensor) -> Tensor:
    """对一维数据进行排名"""
    _, inv, counts = x.unique(return_inverse=True, return_counts=True)
    cs = counts.cumsum(dim=0)
    cs = torch.cat((torch.zeros(1, dtype=x.dtype, device=x.device), cs))
    rmin = cs[:-1]
    rmax = cs[1:] - 1
    ranks = (rmin + rmax) / 2
    return ranks[inv]


def _rank_data(x: Tensor, nan_mask: Tensor) -> Tensor:
    """对数据进行排名（批量处理）"""
    rank = torch.stack([_rank_data_1d(row) for row in x])
    rank[nan_mask] = 0
    return rank  # [d, s]


def _batch_pearsonr_given_mask(
    x: Tensor, y: Tensor,
    n: Tensor, mask: Tensor
) -> Tensor:
    """给定掩码，批量计算 Pearson 相关系数"""
    x_mean, x_std = masked_mean_std_tensor(x, n, mask)
    y_mean, y_std = masked_mean_std_tensor(y, n, mask)
    cov = (x * y).sum(dim=1) / n - x_mean * y_mean
    stdmul = x_std * y_std
    stdmul[(x_std < 1e-3) | (y_std < 1e-3)] = 1
    corrs = cov / stdmul
    if corrs.isnan().any():
        corrs[corrs.isnan()] = 0
    return corrs


def batch_spearmanr_tensor(x: Tensor, y: Tensor) -> Tensor:
    """批量计算 Spearman 相关系数（Tensor 版本）

    Args:
        x: Tensor of shape (days, stocks)
        y: Tensor of shape (days, stocks)

    Returns:
        Tensor of shape (days,) containing correlation for each day
    """
    x, y, n, nan_mask = _mask_either_nan(x, y)
    rx = _rank_data(x, nan_mask)
    ry = _rank_data(y, nan_mask)
    return _batch_pearsonr_given_mask(rx, ry, n, nan_mask)


def batch_pearsonr_tensor(x: Tensor, y: Tensor) -> Tensor:
    """批量计算 Pearson 相关系数（Tensor 版本）

    Args:
        x: Tensor of shape (days, stocks)
        y: Tensor of shape (days, stocks)

    Returns:
        Tensor of shape (days,) containing correlation for each day
    """
    return _batch_pearsonr_given_mask(*_mask_either_nan(x, y, fill_with=0.))


