"""Starter for AP-01-011: NumPy to PyTorch: Shared View Versus Owned Copy.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def numpy_to_torch_pair(array: np.ndarray) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return shared and owned CPU tensors. / 返回共享与独立所有权的 CPU 张量。

    English API notes:
    - ``torch.from_numpy(array)`` shares memory with a supported CPU ndarray.
    - ``torch.tensor(array)`` copies data; ``clone()`` also creates owned storage.
    - ``tensor.device``, ``tensor.dtype``, and ``tensor.data_ptr()`` inspect boundaries.

    中文 API 提示：
    - ``torch.from_numpy`` 与受支持的 CPU ndarray 共享内存。
    - ``torch.tensor`` 会复制数据，``clone`` 也可创建独立存储。
    - ``device``、``dtype`` 与 ``data_ptr`` 可检查边界属性。
    """
    # Import locally so NumPy-only chapters do not require PyTorch. / 在函数内导入，避免纯 NumPy 题强制依赖 PyTorch。
    # TODO: validate layout/writeability, then build one shared and one owned tensor. / 校验布局与可写性，再构造共享和独立张量。
    raise NotImplementedError("AP-01-011")
