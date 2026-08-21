"""Starter for AP-02-001: Map the Floating-Point Lattice.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def local_spacing(values: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    """Measure forward representable spacing. / 测量向前可表示间距。

    English API notes:
    - ``np.dtype(dtype)`` normalizes a dtype argument for validation.
    - ``np.nextafter(x, +inf)`` returns the adjacent representable value toward infinity.
    - Both operands should have the requested dtype to avoid unintended promotion.

    中文 API 提示：
    - ``np.dtype(dtype)`` 可规范化并校验类型参数。
    - ``np.nextafter(x, +inf)`` 返回朝正无穷方向的相邻可表示值。
    - 两个操作数都应使用目标类型，避免意外提升精度。
    """
    # TODO: validate the finite nonnegative domain and preserve dtype. / 校验有限非负定义域并保持类型。
    if not isinstance(values, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(values)}")
    if np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"Expected floating-point array, got integer array with dtype {values.dtype}")
    if np.isinf(values).any() or np.isnan(values).any():
        raise ValueError("Input array contains NaN or infinite values")
    if values.size == 0 or values.ndim > 1:
        raise ValueError("Input array must be a non-empty 1D array")
    if values.shape[0] != 1:
        raise ValueError("Input array must be a 1D array")
    if np.any(values < 0):
        raise ValueError("Input array must contain non-negative values only")
    NextNearestNumber: np.ndarray = np.nextafter(values, np.inf, dtype=dtype)
    return (NextNearestNumber - values).astype(dtype)
