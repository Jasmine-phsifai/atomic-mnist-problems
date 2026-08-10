"""Contract tests for AP-01-007: A Minibatch Iterator With an Explicit Tail Policy.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from starter import iter_minibatches


class BatchIteratorTests(unittest.TestCase):
    """Epoch coverage and tail-policy tests. / 轮次覆盖与尾批策略测试。"""

    def test_complete_epoch_with_partial_tail(self) -> None:
        source = np.arange(23, dtype=np.int64)
        snapshot = source.copy()
        batches = list(iter_minibatches(source, batch_size=5, shuffle=False, seed=0, drop_last=False))
        self.assertEqual([b.size for b in batches], [5, 5, 5, 5, 3])
        joined = np.concatenate(batches)
        np.testing.assert_array_equal(joined, source)
        self.assertEqual(np.unique(joined).size, 23)
        np.testing.assert_array_equal(source, snapshot)

    def test_yields_a_generator(self) -> None:
        epoch = iter_minibatches(np.arange(4), batch_size=2, shuffle=False, seed=0, drop_last=False)
        self.assertIsInstance(epoch, types.GeneratorType)
        self.assertEqual([b.size for b in epoch], [2, 2])

    def test_drop_last_and_seed_behavior(self) -> None:
        source = np.arange(23, dtype=np.int64)
        dropped = list(iter_minibatches(source, batch_size=5, shuffle=False, seed=0, drop_last=True))
        self.assertEqual([b.size for b in dropped], [5, 5, 5, 5])
        self.assertEqual(sum(b.size for b in dropped), 20)
        a = np.concatenate(list(iter_minibatches(source, batch_size=5, shuffle=True, seed=9, drop_last=False)))
        b = np.concatenate(list(iter_minibatches(source, batch_size=5, shuffle=True, seed=9, drop_last=False)))
        c = np.concatenate(list(iter_minibatches(source, batch_size=5, shuffle=True, seed=10, drop_last=False)))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        np.testing.assert_array_equal(np.sort(a), source)

    def test_rejects_invalid_inputs(self) -> None:
        for indices, batch_size in [
            (np.arange(4).reshape(2, 2), 2),
            (np.array([], dtype=np.int64), 2),
            (np.arange(4), 0),
        ]:
            with self.assertRaises(ValueError):
                list(iter_minibatches(indices, batch_size=batch_size, shuffle=False, seed=0, drop_last=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
