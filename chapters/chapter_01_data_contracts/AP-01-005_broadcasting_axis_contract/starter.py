"""Starter for AP-01-005: Broadcasting With a Named Feature Axis.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def standardize_features(
    x: np.ndarray,
    mean: np.ndarray | float,
    scale: np.ndarray | float,
) -> np.ndarray:
    """Standardize along the named feature axis. / 沿明确指定的特征轴标准化。

    English API notes:
    - ``np.asarray`` lets you inspect scalar versus one-dimensional inputs.
    - ``reshape(1, d)`` records feature-axis intent before arithmetic.
    - ``np.isfinite`` and elementwise comparisons validate scale values.

    中文 API 提示：
    - ``np.asarray`` 便于区分标量与一维输入。
    - 运算前用 ``reshape(1, d)`` 明确记录特征轴语义。
    - 使用 ``np.isfinite`` 与逐元素比较校验缩放量。
    """
    # TODO: validate semantic shapes before relying on broadcasting. / 在广播前校验语义形状。
    raise NotImplementedError("AP-01-005")
