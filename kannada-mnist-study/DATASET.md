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

## 外置目录布局

把压缩包解压到仓库之外，并让 `KANNADA_MNIST_DATA_DIR` 指向含有这些文件的目录：

```text
Kannada_MNIST_npz/
├── X_kannada_MNIST_train.npz
├── y_kannada_MNIST_train.npz
├── X_kannada_MNIST_test.npz
├── y_kannada_MNIST_test.npz
├── X_dig_MNIST.npz          # 可选
└── y_dig_MNIST.npz          # 可选
```

如果解压后多了一层目录，只需把环境变量指向真正包含 `.npz` 文件的那一层。

## 下载示例

以下命令只是示例；请把目标位置换成仓库外的目录：

```bash
mkdir -p /absolute/path/to/kannada-mnist-data
cd /absolute/path/to/kannada-mnist-data
curl -L 'https://zenodo.org/records/3359691/files/Kannada_MNIST_npz.zip?download=1' \
  -o Kannada_MNIST_npz.zip
unzip Kannada_MNIST_npz.zip
```

Zenodo 对原始压缩包公布的 MD5 为：

```text
28423b1ce60e01c00fd63b3c1a05d10  Kannada_MNIST_npz.zip
```

## 训练前检查

在任意一个已安装 NumPy 的虚拟环境中运行：

```bash
export KANNADA_MNIST_DATA_DIR=/absolute/path/to/Kannada_MNIST_npz
python check_dataset.py
```

检查器会验证：文件存在、每个 `.npz` 只有一个数组、图像尺寸是 `28 x 28`、样本数
匹配、标签覆盖 `0..9`、像素范围处于 `0..255`，以及官方训练/测试样本数。

训练代码只读取这个目录。所有输出应写到仓库根目录下被忽略的 `outputs/`，不要写回
数据目录。
