"""Diagnostic benchmark for AP-02-016: TensorFlow Cross-Entropy: GradientTape and Logit Semantics.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax

try:
    import tensorflow as tf
except ImportError as exc:  # Keep failure actionable. / 让依赖失败可操作。
    raise SystemExit("Install requirements/tensorflow-cpu.txt before running AP-02-016") from exc

from starter import tensorflow_cross_entropy_probe


def main() -> None:
    """Expose finite-but-wrong probability input. / 暴露“有限但错误”的概率输入。
    """
    logits = np.array([[9.0, 0.0, -2.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    labels = np.array([0, 2], dtype=np.int64)
    try:
        report = tensorflow_cross_entropy_probe(logits, labels)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    oracle = logsumexp(logits, axis=1) - logits[np.arange(2), labels]
    probabilities = softmax(logits, axis=1)
    misuse = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.convert_to_tensor(labels, dtype=tf.int64),
        logits=tf.convert_to_tensor(probabilities, dtype=tf.float64),
    ).numpy()
    losses = np.asarray(report["losses"])
    print(f"oracle_max_error={float(np.max(np.abs(losses - oracle))):.12e}")
    print(f"mean_matches_unreduced={abs(float(report['mean_loss']) - float(losses.mean())) <= 1e-15}")
    print(f"probability_as_logits_is_finite={bool(np.all(np.isfinite(misuse)))}")
    print(f"probability_as_logits_max_discrepancy={float(np.max(np.abs(misuse - oracle))):.12e}")


if __name__ == "__main__":
    main()
