"""The same two-layer MLP used by the NumPy and PyTorch implementations."""

from __future__ import annotations

import tensorflow as tf


class SmallMLP(tf.keras.Model):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(128, activation="relu")
        self.output_layer = tf.keras.layers.Dense(classes)

    def call(self, images: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.flatten(images)
        x = self.hidden(x)
        return self.output_layer(x)
