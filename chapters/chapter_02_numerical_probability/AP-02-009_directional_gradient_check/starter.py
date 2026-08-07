"""Starter for AP-02-009: A Random Direction Checks the Whole Gradient.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def directional_gradient_check(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    direction: np.ndarray,
    *,
    h: float,
) -> dict[str, float]:
    """Compare analytic and numeric directional derivatives. / 比较解析与数值方向导数。

    English API notes:
    - ``np.linalg.norm(direction)`` supplies the normalization scalar.
    - ``np.vdot(g, d)`` computes a flattened inner product for real arrays.
    - Use independent expressions ``x + h*d`` and ``x - h*d`` to avoid mutation.

    中文 API 提示：
    - ``np.linalg.norm`` 给出方向归一化系数。
    - 对实数组，``np.vdot`` 可计算展平后的内积。
    - 使用独立的正负表达式，避免修改调用方数组。
    """
    # TODO: validate, normalize, project, difference, and report. / 校验、归一化、投影、差分并报告。
    raise NotImplementedError("AP-02-009")
