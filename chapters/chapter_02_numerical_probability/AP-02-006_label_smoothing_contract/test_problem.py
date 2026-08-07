"""Contract tests for AP-02-006: Label Smoothing Is a Convention, Not a Slogan.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import smooth_targets


class LabelSmoothingTests(unittest.TestCase):
    """Convention tests. / 约定一致性测试。"""

    def test_exact_all_class_convention(self) -> None:
        labels = np.array([0, 2], dtype=np.int64)
        actual = smooth_targets(labels, num_classes=4, epsilon=0.2, dtype=np.float64)
        expected = np.array([[0.85, 0.05, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05]])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-16)
        np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=0.0, atol=2e-16)

    def test_epsilon_endpoints(self) -> None:
        labels = np.array([1], dtype=np.int64)
        np.testing.assert_array_equal(
            smooth_targets(labels, num_classes=3, epsilon=0.0),
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            smooth_targets(labels, num_classes=3, epsilon=1.0, dtype=np.float64),
            np.full((1, 3), 1 / 3),
        )

    def test_rejects_invalid_contract(self) -> None:
        cases = [
            (np.array([0.0]), 3, 0.1, np.float32),
            (np.array([-1]), 3, 0.1, np.float32),
            (np.array([3]), 3, 0.1, np.float32),
            (np.array([0]), 0, 0.1, np.float32),
            (np.array([0]), 3, -0.1, np.float32),
            (np.array([0]), 3, 1.1, np.float32),
            (np.array([0]), 3, 0.1, np.int32),
        ]
        for labels, classes, epsilon, dtype in cases:
            with self.assertRaises(ValueError):
                smooth_targets(labels, num_classes=classes, epsilon=epsilon, dtype=dtype)


if __name__ == "__main__":
    unittest.main(verbosity=2)
