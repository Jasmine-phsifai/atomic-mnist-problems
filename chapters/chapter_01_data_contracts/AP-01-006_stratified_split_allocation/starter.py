"""Starter for AP-01-006: Deterministic Stratification by Integer Apportionment.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def stratified_split(
    labels: np.ndarray,
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an exact reproducible stratified partition. / 创建精确且可复现的分层划分。

    English API notes:
    - ``np.unique(labels, return_counts=True)`` gives sorted classes and counts.
    - ``np.lexsort`` can encode a deterministic remainder tie-break.
    - ``rng.permutation(indices)`` returns a shuffled copy without global state.

    中文 API 提示：
    - ``np.unique(..., return_counts=True)`` 返回排序后的类别及计数。
    - ``np.lexsort`` 可表达确定性的余数并列规则。
    - ``rng.permutation`` 返回乱序副本，不污染全局随机状态。
    """
    # TODO: implement apportionment, sampling, and partition checks. / 实现整数分配、采样与划分检查。
    raise NotImplementedError("AP-01-006")
