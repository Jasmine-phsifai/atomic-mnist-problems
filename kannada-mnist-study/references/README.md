# Offline references

这个目录保存学习本仓库时会实际用到的上游资料，并保留各自许可证。

## 内容范围

| 路径 | 内容 | 形式 |
| --- | --- | --- |
| `pytorch-official/` | PyTorch 官方 Basics 教程源码 | 上游提交的离线快照 |
| `tensorflow-official/` | TensorFlow 张量、自动微分、数据与自定义循环教程 | 上游提交的离线快照 |
| `d2l/` | *Dive into Deep Learning* 1.0.3 | 官方完整 PDF |

PyTorch 和 TensorFlow 的“官方文档副本”是精选的官方源文件，不是整站镜像。这样能
离线学习张量、数据、autograd、模型与自定义训练循环，同时不会让一个小型练习仓库
膨胀成数 GB。在线 API 细节仍应以对应版本的官方网站为准。

## 许可与归属

- PyTorch 教程保留上游 BSD-style license；
- TensorFlow 文档保留上游 Apache 2.0 license，站点文字/媒体另遵循其注明的许可；
- D2L 正文为 CC BY-SA 4.0，示例代码为上游注明的 modified MIT license；
- 本仓库根目录的 MIT license 不覆盖这些第三方文件。

精确 URL、快照提交、获取日期与文件清单见 [`SOURCES.md`](SOURCES.md)。
