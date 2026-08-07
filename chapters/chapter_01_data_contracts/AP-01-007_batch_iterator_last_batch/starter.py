"""Starter for AP-01-007: A Minibatch Iterator With an Explicit Tail Policy.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def iter_minibatches(
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool,
) -> Iterator[np.ndarray]:
    """Yield exactly one epoch. / 精确地产生一个训练轮次。

    English API notes:
    - A function containing ``yield`` returns an iterator and resumes after each yield.
    - ``rng.permutation(indices)`` protects the caller's ordering.
    - ``range(0, stop, batch_size)`` helps express slice boundaries.

    中文 API 提示：
    - 含 ``yield`` 的函数返回迭代器，并在每次产出后恢复执行。
    - ``rng.permutation(indices)`` 可保护调用方原始顺序。
    - ``range(0, stop, batch_size)`` 便于表达批次切片边界。
    """
    # TODO: validate, permute once, and implement the tail policy. / 校验后只置换一次，并实现尾批策略。
    raise NotImplementedError("AP-01-007")
    yield np.empty(0, dtype=np.int64)  # Keeps the starter typed as a generator. / 保持生成器类型。
