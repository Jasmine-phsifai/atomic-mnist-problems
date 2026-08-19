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
    #verification
    if not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {type(seed)}")
    if isinstance(seed, bool):
        raise ValueError("seed must be an int, got bool")
    if not isinstance(shuffle, bool):
        raise ValueError(f"shuffle must be a bool, got {type(shuffle)}")
    if not isinstance(drop_last, bool):
        raise ValueError(f"drop_last must be a bool, got {type(drop_last)}")
    if not isinstance(indices, np.ndarray):
        raise ValueError(f"indices must be a np.ndarray, got {type(indices)}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got {indices.ndim}D")
    if indices.size ==0:
        raise ValueError("indices must not be empty")
    if np.isinf(indices).any():
        raise ValueError("indices must not contain infinite values")
    if not isinstance(batch_size, int):
        raise ValueError(f"batch_size must be an int, got {type(batch_size)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    # 
    Randomizer_007 = np.random.default_rng(seed)
    if shuffle:
        indices_processed = Randomizer_007.permutation(indices.copy())
    else:
        indices_processed = indices.copy()
    indices_Length = len(indices_processed)
    cursor = 0
    while cursor + batch_size <= indices_Length:
        current_batch = indices_processed[cursor : cursor + batch_size]
        cursor += batch_size
        yield current_batch
    if indices_Length - cursor > 0 and not drop_last:
        current_batch = indices_processed[cursor : ]
        yield current_batch
    # TODO: validate, permute once, and implement the tail policy. / 校验后只置换一次，并实现尾批策略。