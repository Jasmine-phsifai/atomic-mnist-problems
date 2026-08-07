"""Starter for AP-02-012: Find the Exponential Overflow Frontier.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def exp_range_audit(dtype: np.dtype | type) -> dict[str, float]:
    """Bracket exp's finite/overflow transition. / 括定 exp 从有限到溢出的边界。

    English API notes:
    - ``np.finfo(dtype).max`` and ``np.log`` provide the initial estimate.
    - ``np.nextafter(x, +/-inf)`` walks one representable input at a time.
    - ``np.errstate(over="ignore")`` should wrap only the expected probe.

    中文 API 提示：
    - ``np.finfo(dtype).max`` 与 ``np.log`` 给出初始估计。
    - ``np.nextafter`` 每次只移动一个可表示输入。
    - ``np.errstate(over="ignore")`` 只应包围预期的溢出探测。
    """
    # TODO: validate dtype, locate adjacent sides, and return the audit. / 校验类型、定位相邻两侧并返回审计结果。
    raise NotImplementedError("AP-02-012")
