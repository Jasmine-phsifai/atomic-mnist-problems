"""Starter for AP-02-005: Entropy and KL at the Boundary of the Simplex.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def entropy(p: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Compute Shannon entropy with zero conventions. / 按零值约定计算香农熵。

    English API notes:
    - ``scipy.special.xlogy(x, y)`` defines the product as zero when ``x == 0``.
    - Validate simplex membership before reducing.
    - ``np.sum(..., axis=axis)`` should preserve every other dimension.

    中文 API 提示：
    - ``scipy.special.xlogy`` 在 ``x == 0`` 时将乘积定义为零。
    - 降维前必须先校验单纯形成员条件。
    - ``np.sum(..., axis=axis)`` 应保留其余所有维度。
    """
    # TODO: validate and reduce -xlogy(p, p). / 校验后对 -xlogy(p, p) 降维。
    raise NotImplementedError("AP-02-005 entropy")


def kl_divergence(p: np.ndarray, q: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Compute extended-real KL divergence. / 计算扩展实数意义下的 KL 散度。

    English: ``xlogy(p, p) - xlogy(p, q)`` encodes support mismatch as infinity.
    中文：``xlogy(p, p) - xlogy(p, q)`` 可把支持集不匹配编码为正无穷。
    """
    # TODO: validate matching shapes/simplexes and preserve infinity. / 校验同形单纯形并保留正无穷。
    raise NotImplementedError("AP-02-005 kl_divergence")
