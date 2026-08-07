"""Starter for AP-01-012: NumPy to TensorFlow: A Snapshot Boundary.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import tensorflow as tf


def numpy_to_tensor_snapshot(array: np.ndarray, *, dtype: Any) -> "tf.Tensor":
    """Create an isolated eager tensor snapshot. / 创建隔离的 eager 张量快照。

    English API notes:
    - ``np.array(array, copy=True)`` makes the source snapshot explicit.
    - ``tf.convert_to_tensor(snapshot, dtype=dtype)`` performs TensorFlow conversion.
    - A ``tf.Tensor`` is not a ``tf.Variable``; keep this boundary immutable.

    中文 API 提示：
    - ``np.array(..., copy=True)`` 可显式建立源快照。
    - ``tf.convert_to_tensor(..., dtype=...)`` 完成 TensorFlow 转换。
    - ``tf.Tensor`` 不等于 ``tf.Variable``，本题边界应保持不可变。
    """
    # TODO: import TensorFlow locally, copy first, and validate the result. / 局部导入 TensorFlow，先复制再校验结果。
    raise NotImplementedError("AP-01-012 numpy_to_tensor_snapshot")


def tensor_to_numpy_snapshot(tensor: "tf.Tensor") -> np.ndarray:
    """Return an independent NumPy value snapshot. / 返回独立的 NumPy 数值快照。

    English: eager tensors expose ``tensor.numpy()``; wrap it in an explicit copy.
    中文：eager 张量提供 ``tensor.numpy()``，还需再显式复制以固定隔离语义。
    """
    # TODO: reject variables/non-eager values and return owned NumPy storage. / 拒绝变量或非 eager 值，并返回独立存储。
    raise NotImplementedError("AP-01-012 tensor_to_numpy_snapshot")
