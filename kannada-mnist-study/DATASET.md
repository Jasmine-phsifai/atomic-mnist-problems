# Kannada-MNIST 数据契约

## 官方来源

- 永久记录与下载：<https://doi.org/10.5281/zenodo.3359691>
- 上游仓库：<https://github.com/vinayprabhu/Kannada_MNIST>
- 许可证：Creative Commons Attribution 4.0 International（CC BY 4.0）
- 引用：Vinay Uday Prabhu, *Kannada-MNIST: A new handwritten digits dataset for
  the Kannada language*, 2019.

本项目使用官方 `Kannada_MNIST_npz.zip`，而不是 Kaggle 重新打包版本。主数据集包含
60,000 个训练样本和 10,000 个测试样本，每张图像为 `28 x 28` 灰度图，标签为
`0..9`。官方包还包含可作为分布外测试集的 Dig-MNIST。

## 仓库内目录布局

官方压缩包已按原始层次解压到本仓库：

```text
data/Kannada_MNIST_npz/
├── Kannada_MNIST/
│   ├── X_kannada_MNIST_train.npz
│   ├── y_kannada_MNIST_train.npz
│   ├── X_kannada_MNIST_test.npz
│   └── y_kannada_MNIST_test.npz
└── Dig_MNIST/
    ├── X_dig_MNIST.npz
    └── y_dig_MNIST.npz
```

训练程序的 `--data-dir` 应指向 `data/Kannada_MNIST_npz/Kannada_MNIST`。
`Dig_MNIST/` 是独立的分布外测试集，不属于标准 train/test 划分。

## 下载示例

以下命令用于从官方来源重建仓库内的数据目录：

```bash
curl -L 'https://zenodo.org/records/3359691/files/Kannada_MNIST_npz.zip?download=1' \
  -o Kannada_MNIST_npz.zip
unzip Kannada_MNIST_npz.zip -d data
```

Zenodo 对原始压缩包公布的 MD5 为：

```text
28423b1ce60e01c00fd63b3c1a05d10e  Kannada_MNIST_npz.zip
```

## 训练前检查

在任意一个已安装 NumPy 的虚拟环境中运行：

```bash
python check_dataset.py --data-dir data/Kannada_MNIST_npz/Kannada_MNIST
```

检查器会验证：文件存在、每个 `.npz` 只有一个数组、图像尺寸是 `28 x 28`、样本数
匹配、标签覆盖 `0..9`、像素范围处于 `0..255`，以及官方训练/测试样本数。

训练代码只读取这个目录。所有输出应写到被忽略的 `outputs/`，不要写回数据目录。
