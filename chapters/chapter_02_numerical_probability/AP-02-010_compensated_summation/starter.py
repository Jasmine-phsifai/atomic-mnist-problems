"""Starter for AP-02-010: Compensated Summation Recovers Lost Increments.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def kahan_sum(values: np.ndarray) -> float:
    """Accumulate with one compensation variable. / 使用一个补偿变量进行累积。

    English API notes:
    - Iterate over a one-dimensional float64 ndarray; scalar state is the lesson here.
    - Keep ``total`` and ``compensation`` as Python/NumPy float64 values.
    - Do not sort the input: order sensitivity is part of the diagnostic.

    中文 API 提示：
    - 对一维 float64 数组逐项迭代；本题要学习的正是标量状态。
    - ``total`` 与 ``compensation`` 都应保持 float64 精度。
    - 不要排序输入；顺序敏感性本身就是诊断对象。
    """
    # TODO: validate and implement the four-line recurrence. / 校验并实现四行补偿递推。
    raise NotImplementedError("AP-02-010")
