"""Contract tests for AP-02-007: Derive and Implement the Softmax-Loss Gradient.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import logsumexp

from starter import softmax_cross_entropy_gradient


def scalar_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Independent stable scalar objective. / 独立的稳定标量目标。"""
    return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(labels.size), labels]))


class AnalyticGradientTests(unittest.TestCase):
    """Invariant and coordinate-difference tests. / 不变量与坐标差分测试。"""

    def test_all_coordinates_against_central_difference(self) -> None:
        logits = np.array([[0.2, -0.7, 1.1], [2.0, -1.0, 0.3]], dtype=np.float64)
        labels = np.array([2, 0], dtype=np.int64)
        analytic = softmax_cross_entropy_gradient(logits, labels)
        numeric = np.empty_like(logits)
        h = 1e-6
        for index in np.ndindex(logits.shape):
            plus = logits.copy()
            minus = logits.copy()
            plus[index] += h
            minus[index] -= h
            numeric[index] = (scalar_loss(plus, labels) - scalar_loss(minus, labels)) / (2 * h)
        self.assertEqual(analytic.shape, logits.shape)
        self.assertEqual(analytic.dtype, logits.dtype)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-9)
        np.testing.assert_allclose(analytic.sum(axis=1), 0.0, rtol=0.0, atol=2e-16)

    def test_extreme_logits_stay_finite(self) -> None:
        logits = np.array([[700.0, 0.0, -700.0], [-700.0, 0.0, 700.0]], dtype=np.float64)
        labels = np.array([0, 2], dtype=np.int64)
        analytic = softmax_cross_entropy_gradient(logits, labels)
        self.assertTrue(np.all(np.isfinite(analytic)))
        stable = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        expected = (stable - np.eye(3)[labels]) / 2.0
        np.testing.assert_allclose(analytic, expected, rtol=1e-12, atol=0.0)

    def test_rejects_invalid_labels(self) -> None:
        with self.assertRaises(ValueError):
            softmax_cross_entropy_gradient(np.ones((2, 3)), np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            softmax_cross_entropy_gradient(np.ones((2, 3)), np.array([0, 3]))
        with self.assertRaises(ValueError):
            softmax_cross_entropy_gradient(np.ones((2, 3)), np.array([-1, 0]))

    def test_rejects_out_of_domain_inputs(self) -> None:
        with self.assertRaises(ValueError):
            softmax_cross_entropy_gradient(np.array([[np.nan, 0.0], [0.0, 1.0]]), np.array([0, 1]))
        with self.assertRaises(ValueError):
            softmax_cross_entropy_gradient(np.array([[np.inf, 0.0], [0.0, 1.0]]), np.array([0, 1]))
        with self.assertRaises(ValueError):
            softmax_cross_entropy_gradient(np.ones(3), np.array([0]))
        with self.assertRaises(ValueError):
            softmax_cross_entropy_gradient(np.ones((2, 3)), np.array([[0, 1]]))
        with self.assertRaises(ValueError):
            softmax_cross_entropy_gradient(np.ones((2, 3)), np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
