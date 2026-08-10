"""Contract tests for AP-02-004: Sparse Cross-Entropy Directly From Logits.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import logsumexp

from starter import sparse_cross_entropy


class SparseCrossEntropyTests(unittest.TestCase):
    """Logit-domain loss tests. / logit 域损失测试。"""

    def test_extreme_logits_and_reductions(self) -> None:
        logits = np.array([[1000.0, 0.0, -1000.0], [-1000.0, 0.0, 1000.0], [4.0, 4.0, 4.0]])
        labels = np.array([0, 0, 2], dtype=np.int64)
        expected = logsumexp(logits, axis=1) - logits[np.arange(3), labels]
        actual = sparse_cross_entropy(logits, labels, reduction="none")
        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)
        self.assertAlmostEqual(sparse_cross_entropy(logits, labels, reduction="mean"), float(expected.mean()), places=13)

    def test_float32_logits_accumulate_mean_in_float64(self) -> None:
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((257, 5), dtype=np.float32)
        labels = rng.integers(0, 5, size=257).astype(np.int64)
        losses = sparse_cross_entropy(logits, labels, reduction="none")
        mean = sparse_cross_entropy(logits, labels, reduction="mean")
        oracle = float(losses.astype(np.float64).mean())
        self.assertAlmostEqual(float(mean), oracle, places=9)

    def test_rejects_invalid_contract(self) -> None:
        logits = np.ones((2, 3), dtype=np.float64)
        cases = [
            (logits.reshape(1, 2, 3), np.array([0, 1]), "mean"),
            (logits, np.array([0.0, 1.0]), "mean"),
            (logits, np.array([0]), "mean"),
            (logits, np.array([0, 3]), "mean"),
            (logits, np.array([-1, 1]), "mean"),
            (logits, np.array([0, 1]), "sum"),
        ]
        for scores, labels, reduction in cases:
            with self.assertRaises(ValueError):
                sparse_cross_entropy(scores, labels, reduction=reduction)


if __name__ == "__main__":
    unittest.main(verbosity=2)
