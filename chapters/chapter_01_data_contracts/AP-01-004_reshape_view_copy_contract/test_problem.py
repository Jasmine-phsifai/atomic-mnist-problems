"""Contract tests for AP-01-004: Flattening: Shape Is Not Ownership.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import flatten_images


class FlattenContractTests(unittest.TestCase):
    """Value/layout tests. / 数值与布局测试。"""

    def test_contiguous_alias_policy(self) -> None:
        images = np.arange(2 * 28 * 28, dtype=np.float32).reshape(2, 28, 28)
        shared = flatten_images(images, require_independent=False)
        owned = flatten_images(images, require_independent=True)
        expected = images.reshape(2, 784)
        np.testing.assert_array_equal(shared, expected)
        np.testing.assert_array_equal(owned, expected)
        self.assertTrue(np.shares_memory(images, shared))
        self.assertFalse(np.shares_memory(images, owned))

    def test_noncontiguous_values(self) -> None:
        base = np.arange(3 * 28 * 28, dtype=np.int32).reshape(3, 28, 28)
        images = base[:, :, ::-1]
        self.assertFalse(images.flags.c_contiguous)
        actual = flatten_images(images, require_independent=False)
        np.testing.assert_array_equal(actual, images.reshape(3, 784))
        self.assertEqual(actual.shape, (3, 784))

    def test_rejects_wrong_image_extent(self) -> None:
        with self.assertRaises(ValueError):
            flatten_images(np.zeros((2, 14, 56)), require_independent=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
