# NumPy + SciPy 版本

这一目录承担“真正手写”的部分。NumPy 负责数组计算；SciPy 只用于独立数值检查，
不替你实现神经网络或优化器。

## 环境

```bash
python3.12 -m venv .venv-numpy
source .venv-numpy/bin/activate
python -m pip install --upgrade pip
python -m pip install -r numpy_scipy/requirements.txt
```

固定直接依赖：

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| NumPy | 2.5.1 | 参数、前向传播、梯度和小批次 |
| SciPy | 1.18.0 | 用 `scipy.optimize.check_grad` 检查解析梯度 |

这个版本不使用 GPU；它是数学基线，不是速度基线。

## 已提供

- `data.py`：读取官方 `.npz`，展平并归一化；
- `model.py`：参数结构与初始化；
- `gradient_check.py`：SciPy 梯度检查入口；
- `train.py`：命令行与训练流程外壳。

## 留给学习者的实现顺序

1. `relu` 与 `relu_backward`；
2. 数值稳定的 `log_softmax` 与交叉熵；
3. 两层 MLP 的 `forward`；
4. 手工链式法则 `backward`；
5. 用 SciPy 对一小批样本做梯度检查；
6. 打乱、小批次、SGD 更新、epoch 指标；
7. 最后再加入动量或 L2，避免一次混入太多变量。

先让梯度检查通过，再开始完整训练。不要用 `scipy.special.softmax` 或自动微分绕过
核心练习。
