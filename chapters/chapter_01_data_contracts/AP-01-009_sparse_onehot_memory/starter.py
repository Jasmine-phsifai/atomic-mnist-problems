"""Starter for AP-01-009: Sparse Labels and One-Hot Storage.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def one_hot(
    labels: np.ndarray,
    *,
    num_classes: int,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Create a checked dense indicator matrix. / 创建经过校验的稠密指示矩阵。

    English API notes:
    - ``np.eye(num_classes, dtype=dtype)[labels]`` is one vectorized option.
    - Advanced indexing returns a copy; still verify shape, dtype, and layout.
    - Validate labels before indexing so negative values cannot wrap from the end.

    中文 API 提示：
    - ``np.eye(...)[labels]`` 是一种向量化方案。
    - 高级索引返回副本，但仍需检查形状、类型与布局。
    - 索引前必须校验标签，避免负索引从末尾回绕。
    """
    # TODO: validate domain and construct the indicator matrix. / 校验定义域并构造指示矩阵。
    raise NotImplementedError("AP-01-009")
