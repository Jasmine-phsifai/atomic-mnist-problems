"""Starter for AP-01-006: Deterministic Stratification by Integer Apportionment.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def stratified_split(
    labels: np.ndarray,
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an exact reproducible stratified partition. / 创建精确且可复现的分层划分。

    English API notes:
    - ``np.unique(labels, return_counts=True)`` gives sorted classes and counts.
    - ``np.lexsort`` can encode a deterministic remainder tie-break.
    - ``rng.permutation(indices)`` returns a shuffled copy without global state.

    中文 API 提示：
    - ``np.unique(..., return_counts=True)`` 返回排序后的类别及计数。
    - ``np.lexsort`` 可表达确定性的余数并列规则。
    - ``rng.permutation`` 返回乱序副本，不污染全局随机状态。
    """
    # TODO: implement apportionment, sampling, and partition checks. / 实现整数分配、采样与划分检查。
    # My assumptions: Labels[i] stored the number-class of the i-th index sample.
    if labels.size == 0 or labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'unexpected labels datatype')
    if valid_fraction <= 0 or valid_fraction >= 1:
        raise ValueError(f'valid fraction is invalid out of range (0,1), got unexpected {valid_fraction}')
    if np.any(labels < 0):
        raise ValueError('labels contain negative numbers')
    classes, counts = np.unique(labels, return_counts=True)# classes is in set(range(10)) supposedly,but not necessarily of 10-length;  while counts is of same length positive but may contain big positive integers. 
    CategoryCount = len(classes)
    RawWeights = valid_fraction * counts
    BasicQuotas = np.floor(np.round(RawWeights,9)).astype(np.uint64)
    LeftOveredWeights = RawWeights - BasicQuotas # may contain negative numbers very close to zero due to numerical precision bias, cautious
    WholeQuotas = np.round(valid_fraction * len(labels)).astype(int) # Rounded whole quotas for validation dataset, divided into a  <10-lengthed array afterwards. 
    LeftOveredQuotas_tobedistributed = WholeQuotas - np.sum(BasicQuotas).astype(np.int32)#Sum for those that cannot be allocated
    if not 0 <= LeftOveredQuotas_tobedistributed <= CategoryCount :
        raise ValueError('Unexpected Precision Bias')
    Distributed_numbers = np.lexsort((np.arange(CategoryCount), -LeftOveredWeights))[:LeftOveredQuotas_tobedistributed] # deterministic tie-breaker for the remaining quotas
    # allocate the remaining quotas to the classes
    BasicQuotas[Distributed_numbers] += 1
    #initialize the train and validation indices
    train_indices = np.array([], dtype=np.int64)
    valid_indices = np.array([], dtype=np.int64)
    rng_local_006 = np.random.default_rng(seed)
    for c, quota in zip(classes, WholeQuotas):
        c_indexed_original = np.where(labels == c)[0]
        shuffled_permutation = rng_local_006.permutation(c_indexed_original)
        valid_indices = np.concatenate((valid_indices, shuffled_permutation[:quota]))
        train_indices = np.concatenate((train_indices, shuffled_permutation[quota:]))
    rng_local_006.shuffle(train_indices)
    rng_local_006.shuffle(valid_indices)
    return train_indices, valid_indices