"""Contract tests for AP-02-014: Initialization by Propagated Energy.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import initialize_weights


class InitializationTests(unittest.TestCase):
    """Scale and propagated-energy tests. / 尺度与传播能量测试。"""

    def test_reproducibility_shape_and_scale(self) -> None:
        for scheme, expected_std in [
            ("xavier_normal", np.sqrt(2 / (1024 + 512))),
            ("he_normal", np.sqrt(2 / 1024)),
        ]:
            a = initialize_weights(1024, 512, scheme=scheme, rng=np.random.default_rng(14))
            b = initialize_weights(1024, 512, scheme=scheme, rng=np.random.default_rng(14))
            self.assertEqual(a.shape, (1024, 512))
            self.assertEqual(a.dtype, np.float64)
            np.testing.assert_array_equal(a, b)
            self.assertLess(abs(float(a.std()) / expected_std - 1.0), 0.02)

    def test_propagated_energy_envelope(self) -> None:
        rng_x = np.random.default_rng(140)
        rng_h = np.random.default_rng(141)
        inputs = np.random.default_rng(142).normal(size=(4000, 256))
        xavier = initialize_weights(256, 256, scheme="xavier_normal", rng=rng_x)
        he = initialize_weights(256, 256, scheme="he_normal", rng=rng_h)
        input_energy = float(np.mean(inputs**2))
        xavier_ratio = float(np.mean((inputs @ xavier) ** 2) / input_energy)
        he_ratio = float(np.mean(np.maximum(inputs @ he, 0.0) ** 2) / input_energy)
        self.assertTrue(0.90 <= xavier_ratio <= 1.10, xavier_ratio)
        self.assertTrue(0.90 <= he_ratio <= 1.10, he_ratio)

    def test_rejects_invalid_contract(self) -> None:
        rng = np.random.default_rng(1)
        for fan_in, fan_out, scheme in [(0, 2, "he_normal"), (2, -1, "he_normal"), (2, 2, "unknown")]:
            with self.assertRaises(ValueError):
                initialize_weights(fan_in, fan_out, scheme=scheme, rng=rng)


if __name__ == "__main__":
    unittest.main(verbosity=2)
