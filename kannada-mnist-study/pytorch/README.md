# PyTorch 版本

这一版本允许 PyTorch 负责张量和自动微分，但不使用 Lightning、Trainer 或
封装好的训练器。`train_one_epoch`、`evaluate`、device 迁移和指标累计由学习者完成。

## CPU 环境

```bash
python3.12 -m venv .venv-pytorch-cpu
source .venv-pytorch-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r pytorch/requirements-cpu.txt
```

直接依赖：`torch==2.13.0+cpu`、`numpy==2.5.1`。

## NVIDIA GPU 环境

固定 GPU 基线是 PyTorch 2.13.0 的 CUDA 13.0 wheel：

```bash
python3.12 -m venv .venv-pytorch-gpu
source .venv-pytorch-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r pytorch/requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

直接依赖：`torch==2.13.0+cu130`、`numpy==2.5.1`。需要兼容 CUDA 13.0 的 NVIDIA
驱动。AMD ROCm、Intel XPU 或较旧 NVIDIA 驱动不要硬套这个文件，应在 PyTorch 官方
安装选择器中选择与机器一致的构建，再把解析后的版本另存为新的 requirements 文件。

## 文件与练习

- `data.py`：直接从外置 `.npz` 构造 `Dataset` 和 `DataLoader`，不依赖 torchvision；
- `model.py`：与 NumPy 版本同构的 `784 -> 128 -> 10` MLP；
- `train.py`：设备选择、损失、优化器和训练循环的练习入口。

建议先在 CPU 上用 1 个 epoch 和少量 batch 验证损失下降，再切 GPU。不要把
`model.train()`、`zero_grad()`、`backward()`、`step()` 隐藏进辅助框架。

先用同构 MLP 对齐三套实现；CNN 是后续扩展。否则模型结构不同，速度和精度就不能
直接归因于框架或手写算法。
