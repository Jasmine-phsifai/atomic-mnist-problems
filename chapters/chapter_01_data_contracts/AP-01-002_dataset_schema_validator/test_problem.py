"""Contract tests for AP-01-002: Executable Dataset Schema.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import validate_split


class SchemaContractTests(unittest.TestCase):
    """Schema tests with one defect at a time. / 每次只注入一种缺陷的模式测试。"""

    def setUp(self) -> None:
        self.images = np.arange(20 * 28 * 28, dtype=np.uint32).reshape(20, 28, 28).astype(np.uint8)
        self.labels = np.repeat(np.arange(10, dtype=np.int64), 2)

    def test_valid_summary(self) -> None:
        report = validate_split(self.images, self.labels, expected_size=20)
        self.assertEqual(report["n"], 20)
        self.assertEqual(tuple(report["image_shape"]), (20, 28, 28))
        np.testing.assert_array_equal(report["class_counts"], np.full(10, 2))
        self.assertEqual(report["pixel_min"], 0)
        self.assertEqual(report["pixel_max"], 255)

    def test_rejects_shape_cardinality_and_size(self) -> None:
        for bad_images, bad_labels, size in [
            (self.images.reshape(20, -1), self.labels, None),
            (self.images, self.labels[:-1], None),
            (self.images, self.labels, 21),
        ]:
            with self.subTest(shape=bad_images.shape, labels=bad_labels.shape, size=size):
                with self.assertRaises(ValueError):
                    validate_split(bad_images, bad_labels, expected_size=size)

    def test_rejects_dtype_and_label_domain(self) -> None:
        with self.assertRaises(ValueError):
            validate_split(self.images.astype(np.float32), self.labels)
        with self.assertRaises(ValueError):
            validate_split(self.images, self.labels.astype(np.float32))
        bad = self.labels.copy()
        bad[-1] = 10
        with self.assertRaises(ValueError):
            validate_split(self.images, bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
