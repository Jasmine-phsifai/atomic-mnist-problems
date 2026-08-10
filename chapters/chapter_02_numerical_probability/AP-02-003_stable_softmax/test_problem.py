"""Contract tests for AP-02-003: Softmax With Shift Invariance and a Named Axis.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import softmax as scipy_softmax

from starter import stable_softmax


class StableSoftmaxTests(unittest.TestCase):
    """Simplex, shift, and axis tests. / 单纯形、平移与轴测试。"""

    def test_extreme_rows_and_shift_invariance(self) -> None:
        logits = np.array([[10000.0, 10001.0, 9999.0], [-10000.0, -9999.0, -10002.0]], dtype=np.float64)
        actual = stable_softmax(logits, axis=1)
        shifted = stable_softmax(logits + np.array([[12345.0], [-5432.0]]), axis=1)
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertTrue(np.all(actual >= 0))
        np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=0.0, atol=2e-16)
        np.testing.assert_allclose(actual, shifted, rtol=0.0, atol=2e-13)
        np.testing.assert_allclose(actual, scipy_softmax(logits, axis=1), rtol=2e-15, atol=0.0)

    def test_default_axis_and_keyword_only(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        defaulted = stable_softmax(logits)
        np.testing.assert_array_equal(defaulted, stable_softmax(logits, axis=-1))
        with self.assertRaises(TypeError):
            stable_softmax(logits, 1)  # axis is keyword-only. / axis 仅接受关键字传参。

    def test_nonlast_axis_and_dtype(self) -> None:
        logits = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual = stable_softmax(logits, axis=1)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual.sum(axis=1), np.ones((2, 4)), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(actual, scipy_softmax(logits, axis=1), rtol=2e-6, atol=2e-7)

    def test_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            stable_softmax(np.array(1.0))
        with self.assertRaises(ValueError):
            stable_softmax(np.array([1, 2]))
        with self.assertRaises((ValueError, np.AxisError)):
            stable_softmax(np.ones((2, 3)), axis=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
