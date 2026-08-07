"""Starter for AP-02-014: Initialization by Propagated Energy.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def initialize_weights(
    fan_in: int,
    fan_out: int,
    *,
    scheme: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one Xavier or He normal matrix. / 采样一个 Xavier 或 He 正态权重矩阵。

    English API notes:
    - ``rng.normal(loc=0.0, scale=std, size=(fan_in, fan_out))`` owns randomness explicitly.
    - Xavier normal uses ``sqrt(2 / (fan_in + fan_out))``.
    - He normal uses ``sqrt(2 / fan_in)`` for a following ReLU.

    中文 API 提示：
    - ``rng.normal`` 通过调用方传入的生成器显式管理随机性。
    - Xavier 正态标准差为 ``sqrt(2 / (fan_in + fan_out))``。
    - ReLU 前的 He 正态标准差为 ``sqrt(2 / fan_in)``。
    """
    # TODO: validate dimensions/scheme/generator and sample float64 weights. / 校验维度、方案与生成器后采样 float64 权重。
    raise NotImplementedError("AP-02-014")
