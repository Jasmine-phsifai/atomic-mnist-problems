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
    # TODO: validate the summaries and weight by sample count. / 校验摘要并按样本数加权。
    raise NotImplementedError("AP-01-008")
