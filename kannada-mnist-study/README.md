# Kannada-MNIST Study Lab

一个刻意保持小而透明的深度学习学习仓库。目标是用同一份 Kannada-MNIST
数据，依次完成三套可比较的实现：

1. NumPy + SciPy：手写前向传播、损失、反向传播、SGD 和训练循环；
2. PyTorch：使用张量与 autograd，但手写训练循环；
3. TensorFlow：使用 `tf.GradientTape`，但手写训练循环。

当前版本是 **v0.1 课程骨架**，不是三个已经完成的答案。数据读取和模型外壳已经
准备好；关键学习点会明确抛出 `NotImplementedError`，等待后续逐个实现和验证。

## 目录

| 路径 | 内容 |
| --- | --- |
| `numpy_scipy/` | 纯数组实现，SciPy 用于独立梯度检查 |
| `pytorch/` | 小型 CNN 与自定义 PyTorch 训练循环练习 |
| `tensorflow/` | 小型 CNN 与 `GradientTape` 训练循环练习 |
| `data/` | 随仓库提供的官方 Kannada-MNIST NPZ 数据 |
| `references/` | 官方 PyTorch/TensorFlow 教程快照与 D2L 离线副本 |
| `DATASET.md` | Kannada-MNIST 数据契约、布局与来源 |
| `check_dataset.py` | 在开始训练前验证形状、标签和像素范围 |

仓库没有 `src/`、包构建系统、配置框架或隐藏的训练器。每个实现目录只有几份直接
可读的 Python 文件。

## 固定环境

- Python: `3.12.13`（见 `.python-version`）
- NumPy/SciPy: `2.5.1` / `1.18.0`
- PyTorch: `2.13.0`
- TensorFlow: `2.21.0`

三套实现应使用三个独立虚拟环境，避免框架对 NumPy 和 GPU 运行时的依赖互相污染。

```bash
python3.12 -m venv .venv-numpy
source .venv-numpy/bin/activate
python -m pip install --upgrade pip
python -m pip install -r numpy_scipy/requirements.txt
```

PyTorch 和 TensorFlow 的 CPU/GPU 安装命令分别写在各自目录的 README 中。第一次
学习建议从 CPU 环境开始；模型很小，GPU 并不是理解训练循环的前置条件。

## 数据随仓库提供

仓库已包含 Zenodo 发布的官方 `.npz` 数据。克隆后可以直接验证：

```bash
python check_dataset.py --data-dir data/Kannada_MNIST_npz/Kannada_MNIST
```

三套训练程序使用同一个 `--data-dir` 路径。完整文件名、来源、许可证和校验规则见
[`DATASET.md`](DATASET.md)。Dig-MNIST 作为分布外测试集保存在相邻目录。

## 建议的实现顺序

先完成 `numpy_scipy/model.py` 中的数学原语，再用 SciPy 做梯度检查；随后完成
PyTorch 和 TensorFlow 的自定义训练循环。三套实现应共享相同的数据划分、随机种子
和指标定义，最终才比较速度与精度。

## 参考资料

`references/` 中保存的是与本项目直接相关的官方教程源码快照，而不是把两个框架的
整站文档塞进仓库。D2L 以官方 PDF 完整保存。来源、上游提交和各自许可证见
[`references/README.md`](references/README.md)。

## 版本控制约定

- 官方 Kannada-MNIST/Dig-MNIST 数据进入 Git；虚拟环境、checkpoint 和输出结果不进入；
- 小型官方教程快照进入 Git并保留上游许可证；
- D2L PDF 在本仓库中没有重复副本，因此连同校验和与许可证一起进入 Git；
- 每个学习里程碑单独提交，例如 `implement numpy softmax` 或
  `add pytorch train_one_epoch`。
