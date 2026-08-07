"""Starter for AP-02-006: Label Smoothing Is a Convention, Not a Slogan.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def smooth_targets(
    labels: np.ndarray,
    *,
    num_classes: int,
    epsilon: float,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Build all-class-uniform smoothed targets. / 构造“全类别均匀混合”平滑目标。

    English API notes:
    - ``np.full((n, k), epsilon / k, dtype=dtype)`` creates the uniform component.
    - Advanced indexing can add ``1 - epsilon`` at each true class.
    - Validate before indexing so negative labels do not wrap.

    中文 API 提示：
    - ``np.full`` 可建立均匀混合部分。
    - 可用高级索引在真实类别处加上 ``1 - epsilon``。
    - 索引前必须校验，避免负标签回绕。
    """
    # TODO: implement exactly the stated convention. / 严格实现题面指定的约定。
    raise NotImplementedError("AP-02-006")
