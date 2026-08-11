"""Starter for AP-01-005: Broadcasting With a Named Feature Axis.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def standardize_features(
    x: np.ndarray,
    mean: np.ndarray | float,
    scale: np.ndarray | float,
) -> np.ndarray:
    """Standardize along the named feature axis. / 沿明确指定的特征轴标准化。

    English API notes:
    - ``np.asarray`` lets you inspect scalar versus one-dimensional inputs.
    - ``reshape(1, d)`` records feature-axis intent before arithmetic.
    - ``np.isfinite`` and elementwise comparisons validate scale values.

    中文 API 提示：
    - ``np.asarray`` 便于区分标量与一维输入。
    - 运算前用 ``reshape(1, d)`` 明确记录特征轴语义。
    - 使用 ``np.isfinite`` 与逐元素比较校验缩放量。
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"expected x to be a np.ndarray, got {x.__class__.__name__}")
    if x.ndim != 2:
        raise ValueError(f"x is an numpy array, but expected x of shape (batchsize,features), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NAN or inf values.")
    N, D = x.shape
    if not isinstance(mean, (np.ndarray,float)):
        raise TypeError(f"expected mean to be a np.ndarray or float, got {mean.__class__.__name__}")
    if not isinstance(scale, (np.ndarray,float)):
        raise TypeError(f"expected scale to be a np.ndarray or float, got {scale.__class__.__name__}")

    if isinstance(mean, np.ndarray) and mean.ndim != 1:
        raise ValueError(f"mean is a numpy array, but expected mean of shape (features,), got {mean.shape}")
    if isinstance(scale, np.ndarray) and scale.ndim != 1:
        raise ValueError(f"scale is a numpy array, but expected scale of shape (features,), got {scale.shape}")

    if isinstance(mean, np.ndarray):
        if mean.shape[0] != D:
            raise ValueError(f"mean is a numpy array, but expected mean of shape ({D},), got {mean.shape}")
        if not np.all(np.isfinite(mean)):
            raise ValueError("mean is a legal-sized numpy array, but contains NAN or inf values.")
    else:
        if not np.isfinite(mean):
            raise ValueError(f"mean is a float, got illegal {mean}")

    ## MeanMatrix = (np.ones((N,1)) @ mean.reshape((1,D))) if isinstance(mean,np.ndarray) else np.ones((N,D)) * (np.float64(mean))

    if isinstance(scale, np.ndarray):
        if scale.shape[0] != D:
            raise ValueError(f"scale is a numpy array, but expected scale of shape ({D},), got {scale.shape}")
        if not np.all(np.isfinite(scale)):
            raise ValueError("scale is a legal-sized numpy array, but contains NAN or inf values.")
        if np.any(scale <= 0):
            raise ValueError(f"scale is a legal-sized numpy array, but contains non-positive values {scale}")
    else:
        if not np.isfinite(scale):
            raise ValueError(f"scale is a float, got illegal {scale}")
        elif scale < 0:
            raise ValueError(f"scale cannot be negative, got {scale}")
        elif scale == 0: # float equality . Why t is this even suggested? 
            raise ValueError(f"scale is zero, got {scale}")

    ## ScaleMatrix = (np.diag([(np.float64(1.0)/np.float64(i)) for i in scale])) if isinstance(scale,np.ndarray) else np.eye(D) * (np.float64(1.0)/np.float64(scale))

    ## Formerreturn = ((x - MeanMatrix) @ ScaleMatrix)

    #Focus on BroadCasting instead of matrical tricks.

    Vectorized = lambda V: np.ones((1,D)) * (np.float64(V)) if isinstance(V,float) else V.reshape((1,D))

    return (x-Vectorized(mean)) / Vectorized(scale)
    


