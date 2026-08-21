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
    
    if not isinstance(labels, np.ndarray):
        raise ValueError(f'labels vector is supposed to be numpy array, got {labels.__class__.__name__} unexpected instead.')
    if not isinstance(num_classes, int):
        raise ValueError(f'num_classes is supposed to be int, got {num_classes.__class__.__name__} unexpected instead.')
    if num_classes <= 0:
        raise ValueError(f'num_classes is supposed to be positive, got {num_classes} unexpected instead.')
    if labels.ndim != 1:
        raise ValueError(f'labels vector is supposed to be 1-dimensional, got {labels.ndim} unexpected instead.')
    if labels.size == 0:
        raise ValueError(f'labels vector is supposed to be non-empty, got vectors created empty instead.')
    num_sample_count = labels.shape[0]
    Result_matrix = np.zeros((num_sample_count, num_classes), dtype=np.float64)
    Result_matrix[np.arange(num_sample_count), labels] = 1.0
    return Result_matrix.astype(dtype)
    