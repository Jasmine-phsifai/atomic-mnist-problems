"""Starter for AP-02-015: PyTorch Cross-Entropy: Logits, Labels, Reduction, Gradient.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def pytorch_cross_entropy_probe(logits: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Evaluate fused PyTorch CE and its mean gradient. / 计算 PyTorch 融合交叉熵及均值梯度。

    English API notes:
    - ``torch.tensor(logits, dtype=torch.float64, requires_grad=True)`` creates a leaf copy.
    - ``torch.nn.functional.cross_entropy(scores, targets, reduction="none")`` expects raw logits.
    - ``mean_loss.backward()`` fills ``scores.grad``; use ``detach().cpu().numpy().copy()`` for evidence.

    中文 API 提示：
    - ``torch.tensor(..., requires_grad=True)`` 可建立带梯度的叶子副本。
    - ``F.cross_entropy(..., reduction="none")`` 期望原始 logits，而不是概率。
    - ``backward`` 后从 ``scores.grad`` 取梯度，并通过 detach/cpu/numpy/copy 返回独立证据。
    """
    # TODO: validate NumPy contracts, import torch locally, call fused loss, and snapshot outputs. / 校验 NumPy 契约、局部导入 torch、调用融合损失并快照输出。
    raise NotImplementedError("AP-02-015")
