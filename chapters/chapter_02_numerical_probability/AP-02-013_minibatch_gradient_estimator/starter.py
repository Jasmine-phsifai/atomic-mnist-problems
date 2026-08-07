"""Starter for AP-02-013: Measure Minibatch Gradient Unbiasedness and Variance.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def sample_batch_means(
    per_sample_gradients: np.ndarray,
    *,
    batch_size: int,
    repeats: int,
    seed: int,
) -> np.ndarray:
    """Sample repeated without-replacement gradient means. / 重复采样无放回梯度均值。

    English API notes:
    - ``rng.choice(n, size=b, replace=False)`` draws one uniform subset.
    - Preallocate ``(repeats, d)`` float64 output to make the contract visible.
    - ``batch.mean(axis=0, dtype=np.float64)`` computes each estimator.

    中文 API 提示：
    - ``rng.choice(..., replace=False)`` 每次抽取一个均匀子集。
    - 预分配 ``(repeats, d)`` 的 float64 输出可明确契约。
    - ``mean(axis=0, dtype=np.float64)`` 计算每次估计量。
    """
    # TODO: validate, own the RNG, and fill every repeated estimate. / 校验、独立管理随机源并填充全部重复估计。
    raise NotImplementedError("AP-02-013")
