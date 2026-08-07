"""Starter for AP-02-016: TensorFlow Cross-Entropy: GradientTape and Logit Semantics.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def tensorflow_cross_entropy_probe(logits: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Evaluate fused TensorFlow CE and its mean gradient. / 计算 TensorFlow 融合交叉熵及均值梯度。

    English API notes:
    - ``tf.Variable(logits, dtype=tf.float64)`` is watched automatically by ``GradientTape``.
    - ``tf.nn.sparse_softmax_cross_entropy_with_logits(labels=..., logits=...)`` returns one loss per row.
    - Compute ``tf.reduce_mean`` inside the tape; then call ``tape.gradient(mean_loss, scores)``.

    中文 API 提示：
    - ``tf.Variable`` 会被 ``GradientTape`` 自动监视。
    - ``tf.nn.sparse_softmax_cross_entropy_with_logits`` 返回逐样本损失。
    - 必须在 tape 范围内计算 ``tf.reduce_mean``，再求均值损失对 scores 的梯度。
    """
    # TODO: validate NumPy inputs, import TensorFlow locally, record the tape, and snapshot evidence. / 校验 NumPy 输入、局部导入 TensorFlow、记录 tape 并快照证据。
    raise NotImplementedError("AP-02-016")
