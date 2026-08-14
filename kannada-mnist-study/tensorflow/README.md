# TensorFlow 版本

这一版本使用 `tf.data`、Keras layers 和 `tf.GradientTape`，但不把学习目标交给
`model.fit()`。训练步、验证步和 epoch 聚合由学习者手写。

## CPU 环境

```bash
python3.12 -m venv .venv-tensorflow-cpu
source .venv-tensorflow-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tensorflow/requirements-cpu.txt
```

固定直接依赖：`tensorflow==2.21.0`。TensorFlow 会安装与自身元数据兼容的 NumPy
版本，因此这里不重复强行覆盖它。

## NVIDIA GPU 环境（Linux / WSL2）

```bash
python3.12 -m venv .venv-tensorflow-gpu
source .venv-tensorflow-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tensorflow/requirements-gpu.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

固定直接依赖：`tensorflow[and-cuda]==2.21.0`。需要兼容的 NVIDIA 驱动。现代
TensorFlow 不支持 macOS GPU；Windows 原生 GPU 支持止于 TensorFlow 2.10，Windows
上的此版本应使用 WSL2。

## 文件与练习

- `data.py`：从外置 `.npz` 创建确定性的 `tf.data.Dataset`；
- `model.py`：与 NumPy/PyTorch 版本同构的 `784 -> 128 -> 10` MLP；
- `train.py`：`GradientTape`、梯度应用、指标与 epoch 循环的练习入口。

先以 eager mode 写对一个 batch，再考虑 `@tf.function`。过早编译会让形状和梯度
错误更难观察。

先用同构 MLP 对齐三套实现；CNN 是后续扩展。否则模型结构不同，速度和精度就不能
直接归因于框架或手写算法。
