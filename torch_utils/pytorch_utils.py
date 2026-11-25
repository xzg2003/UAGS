from typing import Tuple, Optional
import torch
import pandas as pd
from torch import Tensor
import numpy as np
import sys
sys.path.append('.')
from expr_cal.stock_data import Stock_Data

# ========== Tensor 版本的函数 ==========

def masked_mean_std_tensor(
    x: Tensor,
    n: Optional[Tensor] = None,
    mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    计算带掩码的均值和标准差（Tensor 版本）

    `x`: [days, stocks], input data
    `n`: [days], should be `(~mask).sum(dim=1)`, provide this to avoid unnecessary computations
    `mask`: [days, stocks], data masked as `True` will not participate in the computation, \
    defaults to `torch.isnan(x)`
    """
    if mask is None:
        mask = torch.isnan(x)
    if n is None:
        n = (~mask).sum(dim=1)
    x = x.clone()
    x[mask] = 0.
    mean = x.sum(dim=1) / n
    std = ((((x - mean[:, None]) * ~mask) ** 2).sum(dim=1) / n).sqrt()
    return mean, std


def normalize_by_day_tensor(value: Tensor) -> Tensor:
    """按天归一化（Tensor 版本）

    The shape of the input and the output is (days, stocks)
    """
    mean, std = masked_mean_std_tensor(value)
    value = (value - mean[:, None]) / std[:, None]
    nan_mask = torch.isnan(value)
    value[nan_mask] = 0.
    return value


# ========== DataFrame 版本的函数（保持向后兼容）==========

def masked_mean_std(
    x: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    `x`: [days, stocks], input data
    `n`: [days], should be `(~mask).sum(dim=1)`, provide this to avoid unnecessary computations
    `mask`: [days, stocks], data masked as `True` will not participate in the computation, \
    defaults to `torch.isnan(x)`
    """
    x = x.copy().dropna()
    n = len(x)
    mean = x.sum() / n
    variance = (((x.sub(mean)) ** 2) ).sum() / n  # 按天计算方差
    std = variance.pow(0.5)  # 标准差为方差的平方根
    return mean, std


def normalize_by_day(factor: np.ndarray, data: Stock_Data) -> np.ndarray:
    """
    按天对因子做截面归一化（z-score）。

    Args:
        factor: (n_samples,) 或 (n_samples, n_factors) 的因子数组
    Returns:
        与输入形状相同的归一化数组
    """
    result = factor.copy().astype(float)
    is_1d = result.ndim == 1
    if is_1d:
        result = result.reshape(-1, 1)

    for idx_slice in data.test_slice:
        section = result[idx_slice]
        scale = np.nanmax(np.abs(section), axis=0)
        scale[scale < 1e-30] = 1.0  # 全零截面，避免除零
        section_scaled = section / scale
        mean = np.nanmean(section_scaled, axis=0)
        std = np.nanstd(section_scaled, axis=0)
        std[std < 1e-8] = 1.0  # 相对阈值，截面近似常数时跳过
        result[idx_slice] = (section_scaled - mean) / (std * np.sqrt(len(idx_slice)))

    return result.ravel() if is_1d else result
