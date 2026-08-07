"""Starter for AP-02-004: Sparse Cross-Entropy Directly From Logits.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def sparse_cross_entropy(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    reduction: str = "mean",
) -> np.ndarray | float:
    """Compute stable sparse NLL from logits. / 从 logits 稳定计算稀疏负对数似然。

    English API notes:
    - Keep row maxima with ``keepdims=True`` before exponentiation.
    - ``logits[np.arange(n), labels]`` gathers true-class scores without one-hot storage.
    - Accumulate the mean in float64 even if per-example losses preserve input dtype.

    中文 API 提示：
    - 指数化前用 ``keepdims=True`` 保留每行最大值。
    - 使用高级索引直接取得真实类别分数，无需 one-hot。
    - 即使逐样本损失保持输入类型，均值也应以 float64 累积。
    """
    # TODO: validate and implement the fused logit-domain expression. / 校验并实现融合的 logit 域表达式。
    raise NotImplementedError("AP-02-004")
