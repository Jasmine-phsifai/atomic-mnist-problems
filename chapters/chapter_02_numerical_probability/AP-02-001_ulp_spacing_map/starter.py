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
    raise NotImplementedError("AP-02-001")
