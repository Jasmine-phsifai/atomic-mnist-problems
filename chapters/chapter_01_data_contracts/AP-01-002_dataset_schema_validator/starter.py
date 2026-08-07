"""Starter for AP-01-002: Executable Dataset Schema.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def validate_split(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    expected_size: int | None = None,
) -> dict[str, object]:
    """Validate and summarize one split. / 校验并汇总一个数据划分。

    English API notes:
    - ``array.ndim``, ``array.shape``, and ``array.dtype`` are independent checks.
    - ``np.issubdtype(labels.dtype, np.integer)`` recognizes integer families.
    - ``np.bincount(labels, minlength=10)`` is useful only after range validation.

    中文 API 提示：
    - ``ndim``、``shape`` 与 ``dtype`` 是彼此独立的检查。
    - ``np.issubdtype(..., np.integer)`` 可识别整数类型族。
    - 只有先校验标签范围，才能安全使用 ``np.bincount``。
    """
    # TODO: reject instead of silently reshaping or casting. / 应拒绝错误输入，而非静默整形或转换。
    raise NotImplementedError("AP-01-002")
