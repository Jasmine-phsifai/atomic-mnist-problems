"""Starter for AP-02-002: Stable Softplus Across Extreme Inputs.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def stable_softplus(x: np.ndarray) -> np.ndarray:
    """Evaluate log(1 + exp(x)) stably. / 稳定计算 log(1 + exp(x))。

    English API notes:
    - ``np.maximum(x, 0)`` and ``np.abs(x)`` vectorize the stable identity.
    - ``np.log1p(u)`` is accurate when ``u`` is close to zero.
    - Wrap no warning suppression around a correct formula; unexpected warnings matter.

    中文 API 提示：
    - ``np.maximum`` 与 ``np.abs`` 可向量化稳定恒等式。
    - 当 ``u`` 接近零时，``np.log1p(u)`` 比直接取对数更准确。
    - 正确公式不应依赖屏蔽警告；意外警告本身就是证据。
    """
    # TODO: validate dtype/finiteness and apply the stable identity. / 校验类型与有限性后应用稳定恒等式。
    raise NotImplementedError("AP-02-002")
