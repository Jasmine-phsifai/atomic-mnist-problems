"""Contract tests for AP-01-005: Broadcasting With a Named Feature Axis.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import standardize_features


class BroadcastingContractTests(unittest.TestCase):
    """Axis-semantics tests. / 轴语义测试。"""

    def test_square_axis_ambiguity_uses_features(self) -> None:
        x = np.arange(16, dtype=np.float64).reshape(4, 4)
        mean = np.array([0.0, 10.0, 20.0, 30.0])
        scale = np.array([1.0, 2.0, 4.0, 8.0])
        actual = standardize_features(x, mean, scale)
        expected = (x - mean.reshape(1, 4)) / scale.reshape(1, 4)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_scalar_statistics(self) -> None:
        x = np.array([[1.0, 3.0], [5.0, 7.0]])
        np.testing.assert_allclose(standardize_features(x, 1.0, 2.0), (x - 1.0) / 2.0)

    def test_rejects_accidental_shapes_and_bad_scale(self) -> None:
        x = np.zeros((3, 4))
        for mean, scale in [
            (np.zeros((3, 1)), np.ones(4)),
            (np.zeros(4), np.ones((1, 4))),
            (np.zeros(4), np.array([1.0, 0.0, 1.0, 1.0])),
            (np.zeros(4), np.array([1.0, np.inf, 1.0, 1.0])),
        ]:
            with self.subTest(mean_shape=np.shape(mean), scale_shape=np.shape(scale)):
                with self.assertRaises(ValueError):
                    standardize_features(x, mean, scale)


if __name__ == "__main__":
    unittest.main(verbosity=2)
