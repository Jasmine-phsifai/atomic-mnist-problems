"""Contract tests for AP-02-013: Measure Minibatch Gradient Unbiasedness and Variance.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import sample_batch_means


class MinibatchEstimatorTests(unittest.TestCase):
    """Statistical-contract tests. / 统计契约测试。"""

    def test_reproducibility_and_standardized_mean_error(self) -> None:
        rng = np.random.default_rng(13)
        gradients = rng.normal(size=(40, 5)) + np.linspace(-1.0, 1.0, 5)
        repeats, batch_size = 20_000, 7
        samples = sample_batch_means(gradients, batch_size=batch_size, repeats=repeats, seed=99)
        again = sample_batch_means(gradients, batch_size=batch_size, repeats=repeats, seed=99)
        different = sample_batch_means(gradients, batch_size=batch_size, repeats=repeats, seed=100)
        self.assertEqual(samples.shape, (repeats, 5))
        self.assertEqual(samples.dtype, np.float64)
        np.testing.assert_array_equal(samples, again)
        self.assertFalse(np.array_equal(samples, different))

        true_mean = gradients.mean(axis=0)
        population_s2 = gradients.var(axis=0, ddof=1)
        estimator_var = (1 - batch_size / gradients.shape[0]) * population_s2 / batch_size
        standard_error = np.sqrt(estimator_var / repeats)
        z = np.abs(samples.mean(axis=0) - true_mean) / standard_error
        self.assertLess(float(np.max(z)), 5.0)

    def test_full_batch_has_zero_variance(self) -> None:
        gradients = np.arange(24, dtype=np.float64).reshape(8, 3)
        samples = sample_batch_means(gradients, batch_size=8, repeats=10, seed=1)
        np.testing.assert_allclose(samples, np.broadcast_to(gradients.mean(axis=0), samples.shape), rtol=0.0, atol=0.0)

    def test_rejects_invalid_contract(self) -> None:
        with self.assertRaises(ValueError):
            sample_batch_means(np.ones(5), batch_size=2, repeats=3, seed=0)
        with self.assertRaises(ValueError):
            sample_batch_means(np.ones((5, 2)), batch_size=6, repeats=3, seed=0)
        with self.assertRaises(ValueError):
            sample_batch_means(np.ones((5, 2)), batch_size=2, repeats=0, seed=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
