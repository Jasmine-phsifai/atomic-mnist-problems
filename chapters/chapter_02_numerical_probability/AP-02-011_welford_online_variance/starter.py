"""Starter for AP-02-011: One-Pass Variance Without Catastrophic Cancellation.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def online_mean_variance(values: np.ndarray, *, ddof: int = 0) -> tuple[float, float]:
    """Compute Welford mean and variance. / 使用 Welford 法计算均值与方差。

    English API notes:
    - Enumerate observations from one so the recurrence denominator is explicit.
    - Keep ``mean`` and ``m2`` in float64 scalar state.
    - Divide only once after the pass using ``n - ddof``.

    中文 API 提示：
    - 从一开始枚举样本，使递推分母清晰可见。
    - ``mean`` 与 ``m2`` 应保持 float64 标量状态。
    - 单次遍历结束后，再统一除以 ``n - ddof``。
    """
    # TODO: validate and implement the recurrence exactly. / 校验并严格实现递推式。
    raise NotImplementedError("AP-02-011")
