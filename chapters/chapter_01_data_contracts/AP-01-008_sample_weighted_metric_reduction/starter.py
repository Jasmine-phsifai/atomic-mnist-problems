"""Starter for AP-01-008: Epoch Means Are Sample-Weighted.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def sample_weighted_mean(batch_means: np.ndarray, batch_sizes: np.ndarray) -> float:
    """Aggregate a sample mean from batch summaries. / 从批次摘要聚合样本均值。

    English API notes:
    - ``np.asarray(..., dtype=np.float64)`` makes accumulation precision explicit.
    - ``np.dot(weights, values)`` expresses the weighted numerator.
    - Validate integer sizes before converting them to floating weights.

    中文 API 提示：
    - ``np.asarray(..., dtype=np.float64)`` 可显式指定累积精度。
    - ``np.dot`` 可表达加权分子。
    - 转换为浮点权重之前，必须先校验批大小为整数。
    """
    if not isinstance(batch_means,np.ndarray) or not isinstance(batch_sizes,np.ndarray):
        raise TypeError("batch_means and batch_sizes must be numpy arrays, got batch_means type: {}, batch_sizes type: {}".format(batch_means.__class__.__name__, batch_sizes.__class__.__name__))
    if batch_means.size == 0 or batch_sizes.size == 0:
        raise ValueError(f"batch_means and batch_sizes must not be empty, got batch_means size: {batch_means.size}, batch_sizes size: {batch_sizes.size}")
    if not np.issubdtype(batch_sizes.dtype, np.integer):
        raise ValueError(f"batch_sizes must be integers, got batch_sizes: {batch_sizes}")
    if np.any(batch_sizes <= 0):
        raise ValueError(f"batch_sizes must be positive, got batch_sizes: {batch_sizes}")
    if batch_means.ndim != 1 or batch_sizes.ndim != 1:
        raise ValueError(f"batch_means and batch_sizes must be 1-dimensional, got batch_means.ndim: {batch_means.ndim}, batch_sizes.ndim: {batch_sizes.ndim}")
    if batch_means.shape[0] != batch_sizes.shape[0]:
        raise ValueError(f"batch_means and batch_sizes must have the same length, got batch_means length: {batch_means.shape[0]}, batch_sizes length: {batch_sizes.shape[0]}")
    if np.any(np.isnan(batch_means)) or np.any(np.isnan(batch_sizes)):
        raise ValueError(f"batch_means and batch_sizes must not contain NaN, got batch_means: {batch_means}, batch_sizes: {batch_sizes}")
    if np.any(np.isinf(batch_means)) or np.any(np.isinf(batch_sizes)):
        raise ValueError(f"batch_means and batch_sizes must not contain Inf, got batch_means: {batch_means}, batch_sizes: {batch_sizes}")
    SUM_IN_FLOAT: np.float64 = 0.0
    TOTAL_WEIGHT: np.float64 = 0.0
    # TODO: validate the summaries and weight by sample count. / 校验摘要并按样本数加权。
    batch_means = np.asarray(batch_means, dtype=np.float64)
    batch_sizes = np.asarray(batch_sizes, dtype=np.float64)
    SUM_IN_FLOAT = np.dot(batch_sizes, batch_means)
    TOTAL_WEIGHT = np.sum(batch_sizes, dtype=np.float64)
    return float(SUM_IN_FLOAT / TOTAL_WEIGHT)
