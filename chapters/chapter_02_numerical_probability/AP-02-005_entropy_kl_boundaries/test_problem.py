"""Contract tests for AP-02-005: Entropy and KL at the Boundary of the Simplex.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import entropy, kl_divergence


class InformationBoundaryTests(unittest.TestCase):
    """Boundary-convention tests. / 边界约定测试。"""

    def test_entropy_point_mass_and_uniform(self) -> None:
        p = np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]], dtype=np.float64)
        actual = entropy(p, axis=1)
        np.testing.assert_allclose(actual, [0.0, np.log(3.0)], rtol=0.0, atol=2e-16)

    def test_kl_matching_zeros_and_support_mismatch(self) -> None:
        p = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
        q = np.array([[0.25, 0.75, 0.0], [0.0, 0.5, 0.5]])
        actual = kl_divergence(p, q, axis=1)
        expected_first = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
        self.assertAlmostEqual(float(actual[0]), float(expected_first), places=15)
        self.assertTrue(np.isinf(actual[1]))
        self.assertFalse(np.any(np.isnan(actual)))
        self.assertAlmostEqual(float(kl_divergence(p[:1], p[:1])[0]), 0.0, places=15)

    def test_rejects_invalid_probability_arrays(self) -> None:
        invalid = [np.array([0.2, 0.2]), np.array([1.1, -0.1]), np.array([np.nan, 1.0])]
        for p in invalid:
            with self.assertRaises(ValueError):
                entropy(p)
        with self.assertRaises(ValueError):
            kl_divergence(np.array([0.5, 0.5]), np.array([1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
