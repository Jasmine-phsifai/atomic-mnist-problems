"""Starter for AP-02-008: The Finite-Difference Step Has a U-Curve.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def central_difference(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    *,
    index: int,
    h: float,
) -> float:
    """Approximate one coordinate derivative. / 近似一个坐标方向的导数。

    English API notes:
    - ``x.copy()`` creates independent plus/minus evaluation points.
    - ``np.ndim(result) == 0`` checks that the callable returned a scalar-like value.
    - Convert the final quotient to ``float`` only after evaluating both sides.

    中文 API 提示：
    - ``x.copy()`` 可建立彼此独立的正负扰动点。
    - ``np.ndim(result) == 0`` 可检查函数返回标量值。
    - 两侧都求值后，再把最终商转换为 Python ``float``。
    """
    # TODO: validate, copy twice, perturb, and divide. / 校验后复制两次、扰动并计算差商。
    raise NotImplementedError("AP-02-008")
