"""Contract tests for AP-02-009: A Random Direction Checks the Whole Gradient.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import directional_gradient_check


class DirectionalCheckTests(unittest.TestCase):
    """Projection-check tests. / 投影检查测试。"""

    def test_quadratic_and_input_ownership(self) -> None:
        matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
        x = np.array([0.4, -1.2])
        direction = np.array([2.0, -3.0])
        x_snapshot, d_snapshot = x.copy(), direction.copy()
        grad_calls = []

        def counted_grad(v: np.ndarray) -> np.ndarray:
            grad_calls.append(1)
            return matrix @ v

        report = directional_gradient_check(
            lambda v: float(0.5 * v @ matrix @ v),
            counted_grad,
            x,
            direction,
            h=1e-6,
        )
        self.assertEqual(len(grad_calls), 1)
        d_hat = direction / np.linalg.norm(direction)
        expected_analytic = float((matrix @ x) @ d_hat)
        self.assertAlmostEqual(report["analytic"], expected_analytic, places=12)
        self.assertLess(report["relative_error"], 1e-9)
        self.assertAlmostEqual(report["analytic"], report["numeric"], places=9)
        np.testing.assert_array_equal(x, x_snapshot)
        np.testing.assert_array_equal(direction, d_snapshot)

    def test_rejects_invalid_contract(self) -> None:
        x = np.ones(3)
        good_f = lambda v: float(v.sum())  # noqa: E731
        good_grad = lambda v: v  # noqa: E731
        with self.assertRaises(ValueError):
            directional_gradient_check(good_f, good_grad, x, np.zeros(3), h=1e-5)
        with self.assertRaises(ValueError):
            directional_gradient_check(good_f, good_grad, x, np.ones(2), h=1e-5)
        with self.assertRaises(ValueError):
            directional_gradient_check(good_f, good_grad, x, np.ones(3), h=0.0)
        with self.assertRaises(ValueError):
            directional_gradient_check(good_f, good_grad, x, np.ones(3), h=-1e-5)
        with self.assertRaises(ValueError):
            directional_gradient_check(good_f, good_grad, np.array([1.0, np.nan, 1.0]), np.ones(3), h=1e-5)
        with self.assertRaises(ValueError):
            directional_gradient_check(good_f, good_grad, x, np.array([1.0, np.inf, 1.0]), h=1e-5)
        with self.assertRaises(ValueError):
            directional_gradient_check(good_f, lambda v: np.array([np.nan, 0.0, 0.0]), x, np.ones(3), h=1e-5)
        with self.assertRaises(ValueError):
            directional_gradient_check(lambda v: float("nan"), good_grad, x, np.ones(3), h=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
