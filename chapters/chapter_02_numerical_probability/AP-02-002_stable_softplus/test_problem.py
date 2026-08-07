"""Contract tests for AP-02-002: Stable Softplus Across Extreme Inputs.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import stable_softplus


class SoftplusTests(unittest.TestCase):
    """Extreme-value stability tests. / 极值稳定性测试。"""

    def test_extremes_match_logaddexp(self) -> None:
        for dtype, atol in ((np.float32, 1e-6), (np.float64, 1e-14)):
            x = np.array([-1000.0, -50.0, -1e-8, 0.0, 1e-8, 50.0, 1000.0], dtype=dtype)
            actual = stable_softplus(x)
            expected = np.logaddexp(np.array(0, dtype=dtype), x)
            self.assertEqual(actual.dtype, np.dtype(dtype))
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertTrue(np.all(actual >= 0))
            self.assertTrue(np.all(np.diff(actual) >= 0))
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)

    def test_rejects_bad_domain(self) -> None:
        with self.assertRaises(ValueError):
            stable_softplus(np.array([1, 2], dtype=np.int64))
        with self.assertRaises(ValueError):
            stable_softplus(np.array([np.inf], dtype=np.float64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
