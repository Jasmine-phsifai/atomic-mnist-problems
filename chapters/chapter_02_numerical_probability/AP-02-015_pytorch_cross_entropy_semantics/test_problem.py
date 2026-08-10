"""Contract tests for AP-02-015: PyTorch Cross-Entropy: Logits, Labels, Reduction, Gradient.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import logsumexp, softmax

try:
    import torch
except ImportError as exc:  # Dependency message is deliberate. / 此依赖提示是有意设计的。
    raise SystemExit("Install requirements/pytorch-cpu.txt before running AP-02-015") from exc

from starter import pytorch_cross_entropy_probe


class PyTorchCrossEntropyTests(unittest.TestCase):
    """Framework-semantics tests. / 框架语义测试。"""

    def test_loss_reduction_and_gradient_oracle(self) -> None:
        logits = np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0], [-4.0, -4.0, -4.0]])
        labels = np.array([0, 1, 2], dtype=np.int64)
        report = pytorch_cross_entropy_probe(logits, labels)
        self.assertIsInstance(report["losses"], np.ndarray)
        self.assertIsInstance(report["gradient"], np.ndarray)
        losses = report["losses"]
        gradient = report["gradient"]
        self.assertEqual(losses.dtype, np.float64)
        self.assertEqual(gradient.dtype, np.float64)
        self.assertEqual(losses.shape, (3,))
        self.assertEqual(gradient.shape, (3, 3))
        expected_losses = logsumexp(logits, axis=1) - logits[np.arange(3), labels]
        expected_gradient = softmax(logits, axis=1)
        expected_gradient[np.arange(3), labels] -= 1.0
        expected_gradient /= 3
        np.testing.assert_allclose(losses, expected_losses, rtol=2e-15, atol=0.0)
        self.assertAlmostEqual(float(report["mean_loss"]), float(expected_losses.mean()), places=14)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-15, atol=2e-16)
        self.assertFalse(np.shares_memory(losses, logits))
        self.assertFalse(np.shares_memory(gradient, logits))

    def test_rejects_bad_numpy_contract(self) -> None:
        with self.assertRaises(ValueError):
            pytorch_cross_entropy_probe(np.ones((2, 3), dtype=np.float32), np.array([0, 1]))
        with self.assertRaises(ValueError):
            pytorch_cross_entropy_probe(np.ones((2, 3)), np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            pytorch_cross_entropy_probe(np.ones((2, 3)), np.array([0, 1], dtype=np.int32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
