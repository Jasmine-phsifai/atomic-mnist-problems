"""Starter for AP-02-003: Softmax With Shift Invariance and a Named Axis.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def stable_softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Compute stable normalized exponentials. / 计算稳定的归一化指数。

    English API notes:
    - ``np.max(logits, axis=axis, keepdims=True)`` preserves broadcast shape.
    - ``np.sum(..., keepdims=True)`` keeps the same named normalization axis.
    - Normalize the axis with ``np.core.numeric.normalize_axis_index`` or validate it yourself.

    中文 API 提示：
    - ``np.max(..., keepdims=True)`` 保留可广播的降维形状。
    - ``np.sum(..., keepdims=True)`` 应沿同一明确归一化轴执行。
    - 可规范化轴索引，也可自行进行清晰的轴范围校验。
    """
    # TODO: validate, subtract the maximum, exponentiate, and normalize. / 校验后减去最大值、指数化并归一化。
    raise NotImplementedError("AP-02-003")
