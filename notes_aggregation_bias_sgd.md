# Aggregation Bias 与随机梯度下降：以 AP-01-008 / AP-02-001 为锚点

这份笔记要解决的不是“SGD 公式怎么背”，而是一个更具体的问题：当我们把很多样本级量聚合成一个训练信号或一个日志数字时，代码中的聚合规则是否仍然对应数学目标？全文沿着一条固定路线阅读：

1. 第 1 节先用样本均值、批均值和浮点累加定义 aggregation bias。
2. 第 2 节从总体风险和经验风险开始，逐步推到 softmax 回归的 minibatch 梯度。
3. 第 3 节把抽样、epoch、随机状态和归约公式翻译成 NumPy 操作。
4. 第 4--5 节区分偏差、方差、正则化、标准化和 BatchNorm 的作用。
5. 第 6 节把这些概念映射回仓库中的原子问题。

阅读公式时，先问“这个平均对谁均匀、在什么随机性上取期望、用什么精度实现”，再问公式如何变形。这个顺序能避免把样本测度、批测度和浮点近似混成一个“平均”。

两份锚点材料：

- **AP-01-008**（sample_weighted_metric_reduction）：批 $j$ 有 $n_j$ 个样本、批均值 $m_j$，则 epoch 均值必须是
  $$\bar m=\frac{\sum_{j} n_j\, m_j}{\sum_j n_j},\qquad
  \Delta_{\mathrm{batch}}=\frac1B\sum_j m_j-\bar m .$$
  $\Delta_{\mathrm{batch}}$ 就是 statement 里命名的 **aggregation bias**：对"批"均匀加权 vs 对"样本"均匀加权，是两种不同的测度，二者的差是系统性误差。
- **AP-02-001**（ulp_spacing_map）：浮点格式 $\tau$ 下的前向间距
  $$\operatorname{ulp}_\tau^{+}(x)=\operatorname{nextafter}_\tau(x,+\infty)-x ,$$
  在 binade $[2^e,2^{e+1})$ 内为常数 $2^{e-p+1}$（$p$ 为精度位数）。它刻画了任何"聚合/累加"实际运行的离散格点——加数小于当前 ulp 的一半时会被完全吸收（absorption），且方向确定，这又是另一种 aggregation bias 的来源。

相关后续问题：AP-02-010（Kahan 补偿求和）、AP-02-011（Welford 在线方差）、AP-02-013（minibatch 梯度估计量的无偏性与方差）、AP-01-006（分层分配）、AP-01-007（尾批策略）。

## 名词表

| English | 中文 | 本文语境中的具体含义 |
| --- | --- | --- |
| sample / example | 样本 / 数据实例 | 一次输入与其标签的组合 $(x_i,y_i)$ |
| feature | 特征 | 模型接收的输入坐标；MNIST 中可以是像素值 |
| label / target | 标签 / 目标值 | 希望模型预测的答案；分类时可以写成整数或 one-hot 向量 |
| model parameter | 模型参数 | 由训练算法调整的未知量，例如 softmax 回归中的 $W$ |
| loss | 损失 | 衡量一个样本上的预测与标签差异的数值函数 |
| population distribution | 总体分布 | 新样本 $(X,Y)$ 的联合概率分布，记为 $\mathcal P$ |
| empirical risk | 经验风险 | 在已经收集的 $N$ 个样本上计算的平均损失 $\hat R_N$ |
| batch / minibatch | 批 / 小批量 | 一次参数更新实际使用的样本子集 |
| conditional expectation | 条件期望 | 已知当前参数 $w_t$ 后，只对本步抽样随机性取平均 |
| covariance | 协方差 | 描述随机向量各坐标共同波动方向和大小的矩阵 |
| aggregation / reduction | 聚合 / 归约 | 把 $N$ 个样本级量压成一个标量或向量（sum、mean、加权 mean） |
| measure | 测度 | "哪些对象等权"：对样本均匀 vs 对批均匀，是两个不同测度 |
| unbiased estimator | 无偏估计量 | $\mathbb E[\hat\theta]=\theta$；对单次实现不作任何承诺 |
| finite-population correction (FPC) | 有限总体修正 | 无放回抽样的方差因子 $(1-b/N)$ |
| ulp | 末位单位 | 相邻可表示浮点数的间距，随量级增长 |
| binade | 二进量级区间 | $[2^e,2^{e+1})$，ulp 在其中为常数，跨界加倍 |
| absorption | 吸收 | 加数 $< \tfrac12\operatorname{ulp}(s)$ 时 $s+x$ 舍入回 $s$ |
| compensated summation | 补偿求和 | Kahan/Neumaier：用补偿项回收低位损失 |
| control variate | 控制变量 | 用已知期望的辅助量抵消估计量方差（SVRG 的核心） |
| Horvitz–Thompson weighting | HT 加权 | 非均匀抽样按 $1/(Nq_i)$ 加权恢复无偏 |
| Robbins–Monro conditions | RM 条件 | $\sum_t\eta_t=\infty,\ \sum_t\eta_t^2<\infty$ 的步长条件 |
| noise ball | 噪声球 | 恒定步长 SGD 收敛到的稳态误差邻域 |
| loss scaling | 损失缩放 | 混合精度中把损失乘 $S$ 再反缩放，防止 fp16 吸收 |
| EMA | 指数滑动平均 | $m_t=\beta m_{t-1}+(1-\beta)x_t$，一种有"启动偏差"的聚合 |

---

## 1. Aggregation bias 是什么，有什么用

### 1.1 先区分三种“平均”

设有 $N$ 个样本级量 $a_1,\ldots,a_N$。最基本的样本平均是

$$
\bar a_{\mathrm{sample}}=\frac1N\sum_{i=1}^{N}a_i.
$$

如果样本被分成 $B$ 个批次，第 $j$ 批有 $n_j$ 个样本，其批均值为

$$
m_j=\frac1{n_j}\sum_{i\in B_j}a_i,
$$

那么对所有样本等权的整体平均必须是

$$
\bar a_{\mathrm{sample}}
=\frac{\sum_{j=1}^{B}n_jm_j}{\sum_{j=1}^{B}n_j}.
$$

而把每个批次当成一个对象再取平均，得到的是

$$
\bar a_{\mathrm{batch}}=\frac1B\sum_{j=1}^{B}m_j.
$$

这两个量只有在批次大小相等，或批次均值恰好满足特殊抵消时才相等。它们的差定义为

$$
\Delta_{\mathrm{batch}}
:=\bar a_{\mathrm{batch}}-\bar a_{\mathrm{sample}}.
$$

这里“测度”可以先理解为“谁获得一个等权名额”：第一种平均让每个样本有一个名额，第二种平均让每个批次有一个名额。批次大小不等时，这就是两个不同的加权规则。

一个小例子：令批次大小为 $(64,64,2)$，批均值为 $(0.2,0.4,1.0)$。批均值的平均是

$$
\bar a_{\mathrm{batch}}=\frac{0.2+0.4+1.0}{3}=0.5333\ldots,
$$

但样本加权平均是

$$
\bar a_{\mathrm{sample}}
=\frac{64\times0.2+64\times0.4+2\times1.0}{130}
=0.3107\ldots.
$$

尾批只有两个样本，却在批均值中得到和满批一样的权重，所以差异非常明显。这个例子不是特殊的训练技巧，而是 AP-01-008 所说的“epoch 均值必须按样本加权”的直接数值版本。

### 1.2 同一个问题在训练流水线的三层出现

**统计测度层。** MNIST 的 $60000$ 个样本用批大小 $128$ 切分时，$60000=468\times128+96$。如果把 469 个批均值直接平均，最后 96 个样本会获得一个完整批次的票数；如果要估计全体样本的平均损失，就必须按批次样本数加权。对于 F1、AUC 这类非线性指标，还要额外注意“先对每批算指标再平均”通常不等于“先汇总样本再算指标”。

**估计器层。** minibatch 梯度

$$
g_B=\frac1b\sum_{i\in B}\nabla\ell_i(w)
$$

也是一次聚合。如果抽样概率和目标样本权重不一致，$\mathbb E[g_B]$ 就不再是全梯度；第 2.4 和第 4 节会把这个偏差写成公式。

**数值层。** 在有限精度中，理想实数加法被相邻可表示浮点数之间的格点和舍入规则替代。若当前累加器为 $s$，而增量小于当前间距的一半，可能出现

$$
\operatorname{fl}(s+x)=s.
$$

这叫 absorption（吸收）。它不是从样本分布产生的零均值抽样噪声，而是由精度、累加顺序和当前量级共同决定的数值误差；AP-02-001、AP-02-010 和第 3 节讨论这一层。

### 1.3 这个概念为什么值得单独拿出来

聚合本身不可避免：minibatch 能把样本级计算变成矩阵乘法，降低单步内存和计算开销。aggregation bias 的作用不是禁止聚合，而是提醒我们在每次 `mean`、`sum`、批次拼接、日志归约、混合精度累加和分布式 all-reduce 时追问：

1. 目标量对哪些对象均匀？样本、批次、类别还是设备？
2. 抽样是否改变了每个对象被看到的概率？若改变，是否有权重校正？
3. 理想实数聚合在有限精度和具体顺序下是否仍然足够接近目标？

如果只报告一个数字而不说明这三个问题，数字可能很稳定，却不代表它测量的是想测量的量。

---

## 2. 数学上：为什么随机梯度下降

### 2.1 两种目标函数

这一节先回答一个训练问题：我们手里有一批带标签的图像，怎样选择参数，使模型在**新样本**上的分类损失尽量小？要回答这个问题，必须先把“样本”“模型”“损失”和“平均”分别说明。否则 $R$、$P$、$w$ 和 $\ell$ 会在同一行里出现，却不知道它们各自代表什么。

#### 2.1.1 先固定训练问题

在监督学习（supervised learning）中，一个数据样本写成 $(x,y)$：

- $x$ 是输入特征。例如 MNIST 图像可以展开成一个长度为 $784$ 的像素向量。
- $y$ 是与 $x$ 对应的标签。例如数字分类时，$y\in\{0,1,\ldots,9\}$。
- 模型由参数 $w$ 控制。给定输入 $x$，模型产生预测 $f_w(x)$。
- 损失函数（loss function）$\ell(w;x,y)$ 把一次预测的好坏变成一个数。它越小，表示这次预测按照选定标准越好。

这里的 $w$ 表示“所有可学习参数的集合”。为了先把优化问题写成统一形式，可以把矩阵参数展平成一个向量；后面讲 softmax 回归时会把它重新写成矩阵 $W$。不要把这里的 $w$ 和“样本权重”混为一谈。

设已经收集到 $N$ 个训练样本：

$$
\mathcal D_N=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}.
$$

对固定的参数 $w$，第 $i$ 个样本的损失记为

$$
\ell_i(w):=\ell(w;x_i,y_i).
$$

所以，接下来的问题不是“每个样本都让损失为零”，而是：在一个参数 $w$ 下，如何把这些样本的损失按照明确的规则汇总起来，再最小化这个汇总量？

#### 2.1.2 总体风险：想要的量

训练数据只是可能遇到的样本中的有限一部分。把一个未来样本写成随机变量 $(X,Y)$，用 $\mathcal P$ 表示它们的联合分布：

$$
(X,Y)\sim\mathcal P.
$$

这里用大写 $(X,Y)$ 表示随机变量，用小写 $(x,y)$ 表示它的一次具体取值。分布 $\mathcal P$ 描述了“未来样本以什么概率出现”；它不是前面那个训练数据集合 $\mathcal D_N$。用 $\mathcal P$ 而不是 $P$，是为了给后面 softmax 输出的概率矩阵 $P$ 留出不冲突的记号。

总体风险（population risk，也常被称为 generalization error 的期望形式）定义为

$$
R(w)=\mathbb E_{(X,Y)\sim\mathcal P}\big[\ell(w;X,Y)\big].
$$

逐项读这个式子：

1. 先从真实分布 $\mathcal P$ 中抽取一个未来样本 $(X,Y)$。
2. 用参数 $w$ 在这个样本上计算损失 $\ell(w;X,Y)$。
3. 对所有可能的样本，按它们在 $\mathcal P$ 下的概率做平均。

因此，$R(w)$ 衡量的是模型在“继续遇到同一类数据”时的平均表现，而不是只在已经见过的训练集上表现如何。这个定义还需要一个基本的可积性假设：对正在讨论的 $w$，损失是可测且 $\mathbb E|\ell(w;X,Y)|<\infty$，这样期望才是有限数。

这里的“不可直接计算”有一个具体含义：我们通常不知道完整的 $\mathcal P$，因此无法把所有可能的 $(X,Y)$ 及其概率逐一列出。它不是说 $R(w)$ 在数学上不存在，而是说我们没有访问它所需的分布信息。

> **教材补充（依据 [D2L §3.6.1](kannada-mnist-study/references/d2l/dive-into-deep-learning.pdf)，以下为改写）。** 课本把训练数据上计算的量称为 training error，把对底层数据分布取期望的量称为 generalization error。训练集上的低损失只说明模型拟合了这批数据；我们关心总体风险，是因为部署时输入通常来自尚未见过的数据。

#### 2.1.3 经验风险：手里能算的量

对已经给定的训练集 $\mathcal D_N$，可以把每个训练样本的损失直接相加并取平均：

$$
\hat R_N(w)=\frac1N\sum_{i=1}^{N}\ell_i(w)
          =\frac1N\sum_{i=1}^{N}\ell(w;x_i,y_i).
$$

这个量叫经验风险（empirical risk），也可以理解为训练集上的平均损失。符号上的帽子 $\hat{\ }$ 提醒我们：它是用有限样本构造出来的量，不是未知总体风险本身。

如果训练样本满足 IID（independent and identically distributed，独立同分布）假设，即每个 $(x_i,y_i)$ 都独立地来自同一个 $\mathcal P$，那么在数据集尚未观察到之前，对固定的 $w$ 有

$$
\mathbb E_{\mathcal D_N}\big[\hat R_N(w)\big]
 =\frac1N\sum_{i=1}^{N}\mathbb E\big[\ell(w;X_i,Y_i)\big]
 =R(w).
$$

第二个等号只用了两件事：期望的线性性，以及每个 $(X_i,Y_i)$ 都与 $(X,Y)$ 具有同一个分布。它说明 $\hat R_N(w)$ 在固定 $w$ 时是 $R(w)$ 的一个无偏估计量；它不说明一次具体训练集上的 $\hat R_N(w)$ 必然等于 $R(w)$，也不说明最小化二者会得到同一个参数。

可以用一个小例子把“均值”和“抽样均值”分开。假设某个固定 $w$ 在四个训练样本上的损失为

$$
(\ell_1,\ell_2,\ell_3,\ell_4)=(2,0,1,3).
$$

那么经验风险是

$$
\hat R_4(w)=\frac{2+0+1+3}{4}=1.5.
$$

若随机抽取批次 $B=\{1,4\}$，这个批次的平均损失是 $(2+3)/2=2.5$。这个一次抽样结果不是 $1.5$，但如果所有大小为 $2$ 的批次都按相同概率抽取，它们的平均值会回到 $1.5$。SGD 使用的正是这个“单次可能不同、重复平均后目标对齐”的性质。

#### 2.1.4 GD 和 SGD 到底分别改变了什么

先假设每个 $\ell_i$ 都对 $w$ 可微。经验风险的梯度是

$$
\nabla\hat R_N(w)=\frac1N\sum_{i=1}^{N}\nabla\ell_i(w).
$$

梯度 $\nabla\hat R_N(w)$ 是一个向量；它的每个分量都是经验风险对某个参数分量的偏导数。步长 $\eta>0$ 决定沿负梯度方向走多远。

**梯度下降（gradient descent, GD）** 每一步都计算全部 $N$ 个样本的梯度：

$$
w_{t+1}=w_t-\eta\,\nabla\hat R_N(w_t).
$$

给定初始值 $w_0$、步长和固定数据集后，这个更新没有“抽哪个批次”的随机选择，因此是确定性的（暂时忽略浮点运算的实现差异）。

**随机梯度下降（stochastic gradient descent, SGD）** 不把目标函数从 $\hat R_N$ 换成另一个永久目标。它只是不在每一步都计算上式中的全部 $N$ 项，而是随机选出一个批次来估计这次梯度。

设 $B_t\subseteq\{1,\ldots,N\}$ 是第 $t$ 步抽到的批次，大小为 $|B_t|=b$。先定义这个批次自己的平均损失：

$$
\hat R_{B_t}(w)=\frac1b\sum_{i\in B_t}\ell_i(w).
$$

它的梯度是

$$
g_t=\nabla\hat R_{B_t}(w_t)
    =\frac1b\sum_{i\in B_t}\nabla\ell_i(w_t),
$$

然后用它更新参数：

$$
w_{t+1}=w_t-\eta g_t.
$$

这里必须区分三个对象：

| 对象 | 公式 | 它是什么 |
| --- | --- | --- |
| 目标梯度 | $\nabla\hat R_N(w_t)$ | 全部训练样本梯度的平均值 |
| 一次随机估计 | $g_t$ | 一个批次梯度的平均值，通常不等于目标梯度 |
| SGD 更新 | $w_t-\eta g_t$ | 使用这次估计量得到的参数下一步 |

“随机”发生在 $B_t$ 的选择上，而不是说损失函数的定义突然改变了。

#### 2.1.5 为什么批次梯度在平均意义上对齐

为了把上面的数值直觉写成等式，先固定一个参数 $w$，并假设 $B$ 在所有大小为 $b$ 的无放回子集（without replacement）中均匀抽取。对批次风险取期望：

$$
\mathbb E_B[\hat R_B(w)]
=\mathbb E_B\left[\frac1b\sum_{i\in B}\ell_i(w)\right].
$$

对有限和使用期望的线性性：

$$
\mathbb E_B[\hat R_B(w)]
=\frac1b\sum_{i=1}^{N}\Pr(i\in B)\,\ell_i(w).
$$

为什么 $\Pr(i\in B)=b/N$？总共有 $\binom Nb$ 个大小为 $b$ 的子集；包含指定索引 $i$ 的子集有 $\binom{N-1}{b-1}$ 个。因此

$$
\Pr(i\in B)=\frac{\binom{N-1}{b-1}}{\binom Nb}=\frac bN.
$$

代回即可得到

$$
\mathbb E_B[\hat R_B(w)]
=\frac1b\sum_{i=1}^{N}\frac bN\ell_i(w)
=\frac1N\sum_{i=1}^{N}\ell_i(w)
=\hat R_N(w).
$$

因为这里是有限个可微函数的有限和，可以逐项求导，所以同样有

$$
\mathbb E_B\left[\nabla\hat R_B(w)\right]
=\nabla\hat R_N(w).
$$

在训练过程中，$w_t$ 本身由前面抽到的批次决定，因而它通常是随机的。严谨的说法是在给定当前参数之后取条件期望：

$$
\mathbb E\left[g_t\mid w_t\right]=\nabla\hat R_N(w_t).
$$

这就是 minibatch 梯度在均匀抽样下的**条件无偏性**。它不表示每个 $g_t$ 都等于全梯度，也不表示 SGD 和 GD 会沿着同一条参数轨迹；它只表示在同一个当前点上，抽样机制没有系统性地把梯度方向推向某一侧。

因此，“可以随机”的数学依据不是一句泛泛的“均值可以抽样”，而是下面这条可检查的链：

$$
\mathrm{全梯度}
=\text{有限个单样本梯度的平均}
=\text{均匀索引抽样下单样本梯度的期望}.
$$

如果改成非均匀抽样而不加权，这条链就会断开；这正是后面讨论 aggregation bias 的入口。

#### 2.1.6 把抽象符号落到 MNIST 的 softmax 回归

现在把前面的 $w$、$\ell_i$ 和梯度具体化。MNIST 的每张图像是 $28\times28=784$ 个像素。把像素展平，并在末尾增加一个恒为 $1$ 的坐标来吸收偏置后，每个输入有 $d=785$ 个坐标；类别数为 $K=10$。

为避免行向量和列向量混在一起，下面约定：

- $x_i\in\mathbb R^{785}$ 是第 $i$ 个样本的**列向量**；数据矩阵 $X\in\mathbb R^{N\times785}$ 的第 $i$ 行是 $x_i^{\top}$。
- $y_i\in\mathbb R^{10}$ 是 one-hot 标签列向量：真实类别的位置为 $1$，其他位置为 $0$，所以 $\sum_{k=1}^{10}y_{ik}=1$。
- $Y\in\{0,1\}^{N\times10}$ 的第 $i$ 行是 $y_i^{\top}$。
- $W\in\mathbb R^{785\times10}$ 是参数矩阵。它有 $785\times10$ 个可学习参数。

对样本 $i$，先计算 10 个未归一化分数（logits）：

$$
z_i=W^{\top}x_i\in\mathbb R^{10}.
$$

对每个类别 $k$ 使用逐行 softmax：

$$
p_{ik}=\frac{\exp(z_{ik})}{\sum_{r=1}^{10}\exp(z_{ir})},
\qquad k=1,\ldots,10.
$$

于是 $p_i\in\mathbb R^{10}$ 是模型给出的类别概率：每个分量非负，且 $\sum_kp_{ik}=1$。把所有样本的 logits 和概率写成矩阵就是

$$
Z=XW\in\mathbb R^{N\times10},
\qquad
P=\operatorname{softmax}(Z)\in\mathbb R^{N\times10},
$$

其中 softmax 对 $Z$ 的每一行单独计算。

> **教材补充（依据 [D2L §4.1.1--§4.1.2](kannada-mnist-study/references/d2l/dive-into-deep-learning.pdf)，以下为改写）。** 线性分类器先把输入映射为各类别的 logits，再用 softmax 把 logits 变成概率；one-hot 标签只在真实类别坐标上取值 $1$。交叉熵用模型给真实类别分配的概率来衡量损失：真实类别概率越小，损失越大。

对 one-hot 标签，softmax 回归的单样本交叉熵（cross-entropy）为

$$
\ell_i(W)=-\sum_{k=1}^{10}y_{ik}\log p_{ik}.
$$

由于只有真实类别 $c_i$ 的 $y_{ic_i}=1$，它也可以写成

$$
\ell_i(W)=-\log p_{i,c_i}.
$$

为了得到 $p_i-y_i$，先对 logits 求导。由 softmax 定义和 $\sum_k y_{ik}=1$，有

$$
\begin{aligned}
\ell_i
&=-\sum_{k=1}^{10}y_{ik}
  \log\left(\frac{\exp(z_{ik})}{\sum_{r=1}^{10}\exp(z_{ir})}\right)\\
&=\log\left(\sum_{r=1}^{10}\exp(z_{ir})\right)
  -\sum_{k=1}^{10}y_{ik}z_{ik}.
\end{aligned}
$$

对任意类别 $k$，第一项的偏导数是

$$
\frac{\partial}{\partial z_{ik}}
\log\left(\sum_{r=1}^{10}\exp(z_{ir})\right)
=\frac{\exp(z_{ik})}{\sum_{r=1}^{10}\exp(z_{ir})}
=p_{ik},
$$

第二项的偏导数是 $y_{ik}$，因此

$$
\frac{\partial\ell_i}{\partial z_{ik}}=p_{ik}-y_{ik}.
$$

这句话的含义是：每个类别的 logit 梯度等于“模型给出的概率”减去“one-hot 形式的实际结果”。模型给某个错误类别的概率越大，该类别的梯度越倾向于把它压低；模型给真实类别的概率越小，真实类别坐标的梯度越偏向于增加它。

接下来把对 logits 的导数传回 $W$。因为

$$
z_{ik}=\sum_{r=1}^{785}W_{rk}x_{ir},
$$

所以

$$
\frac{\partial z_{ik}}{\partial W_{rk}}=x_{ir}.
$$

链式法则给出矩阵中的一个坐标：

$$
\frac{\partial\ell_i}{\partial W_{rk}}
=\frac{\partial\ell_i}{\partial z_{ik}}
  \frac{\partial z_{ik}}{\partial W_{rk}}
=(p_{ik}-y_{ik})x_{ir}.
$$

把所有 $r,k$ 坐标放回矩阵，单样本梯度是一个外积：

$$
\nabla_W\ell_i(W)=x_i(p_i-y_i)^{\top}
\in\mathbb R^{785\times10}.
$$

最后对 $N$ 个样本做经验风险的平均：

$$
\begin{aligned}
\nabla_W\hat R_N(W)
&=\frac1N\sum_{i=1}^{N}x_i(p_i-y_i)^{\top}\\
&=\frac1N X^{\top}(P-Y).
\end{aligned}
$$

第二个等号可以做维度检查：

$$
X^{\top}\in\mathbb R^{785\times N},
\qquad
P-Y\in\mathbb R^{N\times10},
\qquad
X^{\top}(P-Y)\in\mathbb R^{785\times10},
$$

正好与 $W$ 的形状相同。

现在令索引随机变量 $I$ 在 $\{1,\ldots,N\}$ 上均匀分布。有限样本空间上的期望按定义就是

$$
\mathbb E_I\big[x_I(p_I-y_I)^{\top}\big]
=\frac1N\sum_{i=1}^{N}x_i(p_i-y_i)^{\top}
=\nabla_W\hat R_N(W).
$$

这就是“全梯度可以用随机批次估计”的精确理由：全梯度本身已经是均匀索引下单样本梯度的期望。对大小为 $b$ 的批次 $B$，对应的估计量为

$$
g_B
=\frac1b\sum_{i\in B}x_i(p_i-y_i)^{\top}
=\frac1bX_B^{\top}(P_B-Y_B),
$$

其中 $X_B$、$P_B$、$Y_B$ 只保留批次中的 $b$ 行，所以

$$
X_B^{\top}\in\mathbb R^{785\times b},
\qquad
P_B-Y_B\in\mathbb R^{b\times10},
$$

乘积仍然是 $785\times10$，可以直接与 $W$ 相减。均匀抽样时，它满足

$$
\mathbb E_B[g_B\mid W]=\nabla_W\hat R_N(W),
$$

但一次具体的 $g_B$ 通常不等于全梯度。

**本节结论。** 总体风险 $R$ 是关于未知数据分布的目标；经验风险 $\hat R_N$ 是训练集上的可计算替代目标；GD 计算经验风险的完整梯度；SGD 仍然针对经验风险更新，只是用均匀批次梯度作为完整梯度的条件无偏估计。softmax 回归中的矩阵式

$$
\frac1N X^{\top}(P-Y)
$$

不是凭空出现的：它依次来自交叉熵对 logits 的导数 $p_i-y_i$、线性映射对参数的链式法则，以及对样本梯度取平均。

### 2.2 为什么不直接 GD

上一节已经说明，GD 和 SGD 的目标都是经验风险 $\hat R_N$。现在的问题变成：既然 GD 使用的是精确梯度，为什么还要接受一个随机估计？答案不是“随机一定更好”，而是每次更新的计算成本、数据访问方式和所需统计精度之间存在取舍。

#### 2.2.1 单步计算量

先看本文件的 softmax 回归。若输入维度为 $d$、类别数为 $K$，一次完整梯度主要包含两个矩阵乘法：

$$
XW\quad\text{和}\quad X^{\top}(P-Y).
$$

它们的样本规模分别是 $N\times d$ 和 $N\times K$，在稠密实现中，计算量的主阶可以写成 $O(NdK)$。如果只使用一个大小为 $b$ 的批次，对应主阶是 $O(bdK)$。因此单步计算量大约缩小为 $b/N$，但每个随机步通常只利用了数据的一小部分。

这里的 $O(\cdot)$ 是渐近量级记号：它只保留规模增长的主阶，不表示具体运行时间，也没有计入内存访问、并行效率和 softmax 的稳定实现等常数因素。

| 方法 | 一步看多少样本 | softmax 回归的主要计算量 | 这一步得到的梯度 |
| --- | --- | --- | --- |
| GD | 全部 $N$ 个 | $O(NdK)$ | 全梯度，确定 |
| SGD | 一个批次 $b$ 个 | $O(bdK)$ | 批次梯度，随机估计 |

#### 2.2.2 经验风险本身也只是有限样本信息

从总体风险的角度看，$\hat R_N$ 不是 $R$ 的完整替身，而是用 $N$ 个样本得到的统计量。在 IID、有限方差等条件下，样本平均的典型误差尺度随 $N$ 增大而按 $1/\sqrt N$ 的量级下降；这是抽样误差的常见尺度，不是没有假设时成立的精确上界。

于是训练误差可以粗略拆成三部分：

$$
\mathrm{总体风险与模型族的差异}
\;+
\mathrm{用有限数据估计总体的误差}
\;+
\mathrm{优化尚未充分进行的误差}.
$$

GD 主要减少第三项，但继续把训练集上的经验风险压得很低，并不会自动消除第二项。这个事实解释了为什么在大数据问题中，人们常常比较“每单位计算量带来的统计收益”，而不是只比较某一步的梯度是否精确。

#### 2.2.3 内存、数据访问和在线学习

“数据太大所以 GD 无法定义”是不准确的。只要能逐批读取数据并把每个批次的梯度累加起来，完整梯度在数学上仍然可以计算；只是一次 GD 更新必须完成整个数据集的扫描，并且需要保存或重新读取足够的信息。

SGD 的优势更具体：它可以在只访问一小部分数据后立即更新参数，也可以在数据以数据流（data stream）形式到达时工作。代价是每一步的梯度含有抽样噪声，并且同样数量的样本可能被多次访问。

#### 2.2.4 随机性带来的附带效果

随机梯度的噪声有时能帮助算法离开某些鞍点或狭窄区域，并可能与损失曲率共同产生所谓的隐式正则化。但这是训练动力学的附带效果，不是“SGD 目标函数不同”的理由，也不能替代对无偏性、方差和步长的分析。小数据集或需要可重复精确迭代时，GD 完全可能是更合适的选择。

### 2.3 为什么“能”随机：一个带前提的不等式

这一节只说明一个局部事实：在适当条件下，随机更新的**期望损失**可以下降。它不是完整的 SGD 收敛定理；收敛还需要更强的全局假设。

令

$$
F(w):=\hat R_N(w),
\qquad
\mu_t:=\nabla F(w_t),
\qquad
\xi_t:=g_t-\mu_t.
$$

$\mu_t$ 是当前点的全梯度，$\xi_t$ 是批次抽样造成的梯度误差。因此

$$
g_t=\mu_t+\xi_t.
$$

均匀抽样给出

$$
\mathbb E[\xi_t\mid w_t]=0.
$$

这只是无偏性；它没有说 $\xi_t$ 的一次取值很小。还需要假设它的二阶矩有限，才能讨论噪声大小。

#### 2.3.1 需要哪些局部假设

假设 $F$ 可微，并且存在常数 $L>0$，使任意两个参数点 $u,v$ 满足

$$
F(v)\le F(u)+\langle\nabla F(u),v-u\rangle
        +\frac L2\lVert v-u\rVert^2.
$$

这条不等式是这里使用的 $L$-光滑条件。它说的是：用一阶线性近似预测 $F(v)$ 时，二阶剩余项可以由 $L\lVert v-u\rVert^2/2$ 控制。对矩阵参数，$\lVert\cdot\rVert$ 应理解为 Frobenius 范数，$\langle A,B\rangle=\operatorname{tr}(A^{\top}B)$；对向量参数，它就是通常的欧氏范数和内积。

另一个条件是抽样梯度的条件二阶矩有限：

$$
\mathbb E[\lVert\xi_t\rVert^2\mid w_t]<\infty.
$$

#### 2.3.2 从光滑性不等式到期望下降

SGD 更新为 $w_{t+1}=w_t-\eta g_t$。在光滑性不等式中令 $u=w_t$、$v=w_{t+1}$，得到

$$
\begin{aligned}
F(w_{t+1})
&\le F(w_t)-\eta\langle\mu_t,g_t\rangle
  +\frac{L\eta^2}{2}\lVert g_t\rVert^2.
\end{aligned}
$$

现在在已知 $w_t$ 的条件下取期望。第一项中的内积满足

$$
\mathbb E\big[\langle\mu_t,g_t\rangle\mid w_t\big]
=\left\langle\mu_t,\mathbb E[g_t\mid w_t]\right\rangle
=\lVert\mu_t\rVert^2.
$$

对第二项，使用 $g_t=\mu_t+\xi_t$ 和 $\mathbb E[\xi_t\mid w_t]=0$：

$$
\begin{aligned}
\mathbb E[\lVert g_t\rVert^2\mid w_t]
&=\mathbb E[\lVert\mu_t+\xi_t\rVert^2\mid w_t]\\
&=\lVert\mu_t\rVert^2
  +\mathbb E[\lVert\xi_t\rVert^2\mid w_t].
\end{aligned}
$$

代回可得

$$
\boxed{
\mathbb E[F(w_{t+1})\mid w_t]
\le F(w_t)
-\eta\left(1-\frac{L\eta}{2}\right)\lVert\mu_t\rVert^2
 +\frac{L\eta^2}{2}\mathbb E[\lVert\xi_t\rVert^2\mid w_t]
}.
$$

这个式子中有两类项：

- 中间的负项来自真实全梯度，量级是 $O(\eta)$；当 $0<\eta<2/L$ 时，它确实提供下降方向。
- 最后的正项来自抽样噪声，量级是 $O(\eta^2)$，并且随着批次增大通常会减小。

所以“小步长下随机化仍有机会下降”的精确含义是：在条件无偏且二阶矩受控时，光滑性给出的上界中，梯度信号是一阶步长项，噪声代价是二阶步长项。靠近临界点时 $\lVert\mu_t\rVert$ 很小，噪声项可能占主导；这个不等式没有承诺每一步都下降。

#### 2.3.3 这还不是完整的收敛定理

常见的 Robbins--Monro 步长条件是

$$
\sum_{t=0}^{\infty}\eta_t=\infty,
\qquad
\sum_{t=0}^{\infty}\eta_t^2<\infty.
$$

第一条防止步长总和有限而算法过早停止，第二条让累积的零均值噪声受到控制。但仅写出这两条还不够推出任意模型的几乎必然收敛；还需要对目标函数、噪声矩、可行域和更新过程补充相应假设。

若使用恒定步长，随机更新通常不会在有限方差的情况下精确停在一个点，而是在最优点附近形成一个由步长、曲率和梯度方差共同决定的稳态邻域。这个邻域常被称为 noise ball。它是后面讨论方差与正则化时的直观入口，不应被误读为“SGD 一定收敛到一个球”。

### 2.4 偏差、方差与分布：三个不同的问题

“无偏”只回答平均位置是否正确；它不回答一次抽样离平均位置有多远，也不回答误差的形状是否接近正态。把这三个问题分开，才能正确理解 minibatch 梯度。

固定一个参数 $w$，记单样本梯度为

$$
a_i(w):=\nabla\ell_i(w),
\qquad
\bar a(w):=\frac1N\sum_{i=1}^{N}a_i(w)=\nabla\hat R_N(w).
$$

如果参数是矩阵，就把 $a_i$ 暂时展平成向量来定义协方差；计算范数时也可以直接使用对应的 Frobenius 范数。

#### 2.4.1 第一问：估计量的平均位置有没有偏

**有放回均匀抽样。** 每次从 $\{1,\ldots,N\}$ 中独立均匀抽一个索引，批次梯度是 $b$ 个单样本梯度的平均。因为每个索引的概率都是 $1/N$，

$$
\mathbb E[g_B\mid w]
=\frac1b\sum_{r=1}^{b}\mathbb E[a_{I_r}(w)]
=\frac1N\sum_{i=1}^{N}a_i(w)
=\bar a(w).
$$

**无放回均匀抽样。** 若 $B$ 是均匀随机的大小为 $b$ 的子集，每个位置的边缘分布仍然均匀，所以期望仍为 $\bar a(w)$。样本之间不再独立，但“不独立”本身不会制造偏差；它会改变方差。

**非均匀抽样。** 如果索引 $i$ 的抽样概率是 $q_i$，而仍然直接平均抽到的梯度，则

$$
\mathbb E[a_I(w)]=\sum_{i=1}^{N}q_i a_i(w).
$$

只有当 $q_i=1/N$ 时，这个式子才等于 $\bar a(w)$。如果每个 $q_i>0$ 已知，可以使用 Horvitz--Thompson（HT）校正：

$$
\mathbb E\left[\frac{a_I(w)}{Nq_I}\right]
=\sum_{i=1}^{N}q_i\frac{a_i(w)}{Nq_i}
=\frac1N\sum_{i=1}^{N}a_i(w).
$$

权重 $1/(Nq_I)$ 不是装饰；它把“抽到某个样本的概率”除掉了。如果某个 $q_i=0$，该样本永远不会出现，任何只依靠这套抽样的估计都无法恢复它的贡献。

**几种常见的偏差来源。** 下面各项需要区分“数学抽样偏差”和“数值实现误差”：

| 情形 | 发生了什么 | 偏差从哪里来 |
| --- | --- | --- |
| 非均匀抽样且不做 HT 校正 | 高频样本在平均中出现得更多 | 抽样测度与目标的均匀样本测度不一致 |
| mean of batch means | 先算每批均值，再让每批等权 | 批测度代替了样本测度；尾批被过度加权 |
| top-$k$ 压缩或确定性量化 | 某些梯度坐标被系统性删掉或舍入 | 压缩算子本身的期望不等于原向量 |
| BatchNorm 的 running statistics | 训练和推理使用不同的统计聚合 | 估计规则或所处数据分布发生变化 |
| 浮点吸收 | 小增量在有限精度累加器中被舍掉 | 理想实数加法被有方向的舍入替代 |
| 按类别排序且不打乱 | 连续批次长期只看到某些类别 | 每一步看到的样本分布不再像均匀抽样 |

最后一行需要更精确地说：如果一个 epoch 最终仍然完整遍历每个样本，整个 epoch 的样本平均可以没有抽样偏差；但每一步的梯度方向会和类别顺序相关，训练轨迹不再具有均匀随机批次的性质。

#### 2.4.2 第二问：误差有多大

先定义“有放回抽样”使用的总体协方差。令 $I$ 在 $\{1,\ldots,N\}$ 上均匀分布，且把梯度展平为 $d$ 维向量：

$$
\Sigma_N(w)
:=\frac1N\sum_{i=1}^{N}
\big(a_i(w)-\bar a(w)\big)\big(a_i(w)-\bar a(w)\big)^{\top}
\in\mathbb R^{d\times d}.
$$

它的迹是所有坐标方差之和：

$$
\operatorname{tr}\Sigma_N(w)
=\frac1N\sum_{i=1}^{N}\lVert a_i(w)-\bar a(w)\rVert^2.
$$

有放回时，$b$ 次抽样独立，因而

$$
\operatorname{Cov}(g_B\mid w)=\frac1b\Sigma_N(w).
$$

无放回时，通常定义有限总体协方差

$$
\Sigma_F(w)
:=\frac1{N-1}\sum_{i=1}^{N}
\big(a_i(w)-\bar a(w)\big)\big(a_i(w)-\bar a(w)\big)^{\top}.
$$

则均匀无放回批次满足

$$
\operatorname{Cov}(g_B\mid w)
=\left(1-\frac bN\right)\frac{\Sigma_F(w)}{b}.
$$

因子 $1-b/N$ 叫有限总体修正（finite-population correction, FPC）。当 $b=N$ 时，批次就是整个数据集，方差必须为零；当 $b\ll N$ 时，FPC 接近 $1$。如果把协方差定义成分母为 $N$ 的 $\Sigma_N$，无放回公式还会多出 $N/(N-1)$；这只是协方差归一化约定不同，不能把两个定义混用。

由 $\mathbb E\lVert Z\rVert^2=\operatorname{tr}\operatorname{Cov}(Z)$（当 $\mathbb E Z=0$）可得，有放回时

$$
\mathbb E\big[\lVert g_B-\bar a(w)\rVert^2\mid w\big]
=\frac{\operatorname{tr}\Sigma_N(w)}{b},
$$

无放回时

$$
\mathbb E\big[\lVert g_B-\bar a(w)\rVert^2\mid w\big]
=\left(1-\frac bN\right)\frac{\operatorname{tr}\Sigma_F(w)}{b}.
$$

这两个式子说明“增大批次能降方差”的精确版本是：在其他条件不变时，方差主阶按 $1/b$ 缩小；它没有说计算时间、内存和优化步数都不变。

#### 2.4.3 第三问：误差是不是正态

中心极限定理（central limit theorem, CLT）讨论的是“许多项的平均”的极限分布，而不是说每个单样本梯度必须正态。以有放回抽样为例，在独立、同分布且二阶矩有限等条件下，向量化的批次梯度满足

$$
\sqrt b\,\big(g_B-\bar a(w)\big)
\xrightarrow{d}
\mathcal N\big(0,\Sigma_N(w)\big).
$$

$\xrightarrow{d}$ 表示分布收敛：当 $b$ 足够大时，标准化后的随机向量可以用右侧的多元正态分布近似。它不是对任意小批次的精确等式。

MNIST 中许多像素坐标经常为零，因此某个坐标的单样本梯度可能在 $0$ 附近有很尖的质量，同时带有少量较大的非零值。这种分布常被称为 zero-inflated（零膨胀）；它可能偏斜、重尾。批次平均会改善分布形状，但 $b=128$ 或更小时，某些坐标仍然可能明显偏离正态。Demo 4 输出的 skewness（偏度）、kurtosis（峰度）和正态性检验，检查的是这一点，而不是检查无偏性。

梯度裁剪（gradient clipping）会把过大的梯度截断或缩放。它能控制重尾更新，但一般会改变期望，因此可能引入有意的小偏差。它解决的是“极端更新是否危险”，不是“抽样是否无偏”。

#### 2.4.4 softmax 回归中的方差如何计算

对前面得到的单样本梯度

$$
a_i=x_i(p_i-y_i)^{\top},
$$

外积的 Frobenius 范数满足

$$
\lVert x_i(p_i-y_i)^{\top}\rVert_F^2
=\lVert x_i\rVert^2\,\lVert p_i-y_i\rVert^2.
$$

因此不需要物化 $N\times7850$ 的逐样本梯度矩阵，就可以先计算全梯度 $\bar a$，再使用

$$
\operatorname{tr}\Sigma_N
=\frac1N\sum_{i=1}^{N}
  \lVert x_i\rVert^2\lVert p_i-y_i\rVert^2
 -\lVert\bar a\rVert_F^2.
$$

Demo 4 在真实 Kannada-MNIST 上比较理论值和重复抽样的经验值：$b=128$ 时比值为 $1.0079$，$b=512$ 时比值为 $0.9984$。这支持方差公式在该实验中的数值表现；它不是对所有数据集、所有参数点的普遍证明。实验还把 $\lVert\mathbb E[\hat g]-\bar a\rVert$ 换算成标准误，用来分别检查均值是否偏离和方差大小是否匹配。

#### 2.4.5 怎样降低方差

| 手段 | 它减少的量 | 代价 / 需要注意 |
| --- | --- | --- |
| 增大 $b$ | 批次均值的方差主阶按 $1/b$ 降低 | 单步计算与显存增加；步长 $\eta$ 仍受曲率和稳定性限制 |
| 步长衰减 | 更新中噪声项的影响逐渐减小 | 需要选择调度；不自动修复有偏抽样 |
| Polyak--Ruppert 平均 | 对多个迭代参数做平均，削弱零均值波动 | 对系统性偏差无效，且改变输出参数定义 |
| 分层抽样 | 固定每批类别比例，减少类别组成造成的波动 | 需要能访问标签，并设计各层配额 |
| 重要性抽样 | 把计算预算放到信息量较大的样本 | 必须使用 HT 权重，否则会把目标改成 $q_i$ 加权目标 |
| SVRG / SAGA | 用参考点梯度作为 control variate，抵消部分波动 | 适合有限和目标；需要额外存储或周期性计算快照 |
| 梯度裁剪 | 限制极端更新的影响 | 通常引入偏差；它是尾部控制而非无偏降方差 |

其中“降低方差”和“消除偏差”是两条不同的工程路线：重复平均、增大批次和控制变量主要处理零均值波动；测度对齐、HT 校正和更高精度归约才直接处理系统性偏差。

---

## 3. 代码上：如何实现这种“随机”，保住数学性质与工程表现

上一节的公式要落成代码，至少要同时守住四个契约：抽样分布要与目标一致；epoch 是否无放回要明确；随机状态要能重现且不互相污染；最后的求和与平均要尽量减少数值误差。下面每一段都先说明它在守护哪条数学性质。

### 3.1 抽样器：均匀不等于“看起来随机”

`rng.integers(0, N, size=B)` 的公共语义是从 $\{0,1,\ldots,N-1\}$ 均匀抽取整数。这里依赖的是 API 对均匀性的承诺，不需要把某个 NumPy 版本当前采用的底层有界整数算法当成 SGD 理论的一部分。

不要用 `draw % N` 代替均匀有界抽样。若原始整数在 $\{0,\ldots,255\}$ 上均匀，而 $N=100$，前 56 个余数出现 3 次，后 44 个余数出现 2 次，所以

$$
\Pr(0)=\frac3{256},
\qquad
\Pr(99)=\frac2{256}.
$$

这不是有限实验的偶然误差，而是抽样规则本身造成的概率不相等。Demo 1 用较小的 8-bit 范围把这个偏差放大到可见；换成 uint32 只会让偏差更小，不会让它在数学上消失。

### 3.2 epoch：有放回和无放回是两种不同的实验

`rng.permutation(N)` 产生一个长度为 $N$ 的排列。沿排列切片取批次，意味着一个 epoch 内每个样本恰好出现一次：这是无放回抽样，批次之间有负相关，方差对应第 2.4 节的 FPC。

相反，`rng.integers(0, N, size=B)` 允许同一个样本在同一批次中重复出现，属于有放回抽样。它的条件期望仍然可以无偏，但重复数是随机的，方差公式不同。代码必须先选择想要的统计语义，再选择 API；不能看到两者都返回索引，就把它们当成同一种 batch。

尾批也必须有明确策略。若 $N$ 不能被 $B$ 整除，最后一批大小小于 $B$；保留它时，epoch 指标要按样本数加权，丢弃它时，要明确承认有一部分样本没有参与这一轮训练。两者都可以是合法策略，沉默地把尾批当满批处理才会改变目标测度。

### 3.3 随机状态：可重复和组件隔离

`Generator` 必须由使用它的组件局部持有，不要让数据采样、数据增强和 dropout 共享一个无法追踪的全局状态（AP-01-006 也把这一点写进了数据契约）。`np.random.default_rng(seed)` 创建一个局部生成器；多个独立随机流可以由 `SeedSequence.spawn` 派生。

这样做解决两个不同问题：相同种子可以复现实验；一个组件多消耗几个随机数，不会悄悄改变另一个组件的序列。全局 `np.random.seed` 只能设置一个共享状态，不能表达这种组件边界。

### 3.4 从抽样代码到归约代码

抽样无偏并不保证浮点归约无偏。代码还要分别决定抽样权重、累加精度、累加顺序和最终的测度。

**抽样与 HT 校正示例。** 下面的 `G` 表示已经得到的逐样本梯度，`grad_norms` 表示对应的非负抽样分数；它是展示抽样契约的片段，不是独立可运行的完整训练循环：

```python
import numpy as np

seed = 42
ss = np.random.SeedSequence(seed)
rng_data, rng_aug = (np.random.default_rng(s) for s in ss.spawn(2))

N, B = 60_000, 128
perm = rng_data.permutation(N)                 # epoch 洗牌，无放回
drop_last = False
for k in range(0, N, B):
  batch_idx = perm[k:k + B]
  if drop_last and batch_idx.size < B:       # 尾批策略见 AP-01-007：保留/丢弃都须显式
    continue
  # train_on(batch_idx)                      # 保留尾批时，按实际样本数处理

idx_repl = rng_data.integers(0, N, size=B)     # 有放回：纯 Monte Carlo 估计

# 重要性抽样 + Horvitz–Thompson 校正
q = grad_norms / grad_norms.sum()              # q_i ∝ ||g^{(i)}||
idx = rng_data.choice(N, size=B, replace=True, p=q)
g_hat = (G[idx] / (N * q[idx, None])).mean(axis=0)   # 权重恢复无偏
```

**归约的数值卫生。** 这一层直接连 AP-02-001 / AP-02-010：

```python
acc = np.zeros(d, dtype=np.float64)      # 即使梯度是 float32，也在 float64 中累积（AP-01-008 约束同款）
acc += g_batch.astype(np.float64).sum(axis=0)   # np.sum 内部是 pairwise：误差 O(eps·log K)

epoch_loss = float(np.dot(sizes.astype(np.float64), means) / sizes.sum())  # 样本加权，禁止 means.mean()

def kahan_sum(xs):                       # AP-02-010：显式标量循环时的补偿求和
    s, c = 0.0, 0.0
    for v in xs:
        y = float(v) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s
```

混合精度（fp16）流水线的对应物：fp16 累加器 + loss scale $S$，更新前反缩放到 fp32 主权重（Demo 3）。

**分层 batch 草图。** 把 AP-01-006 的思想搬到 batch 构造：按类分组 → 各类内部 `rng.permutation` → 按类比例 round-robin 取数拼批。每批类比例固定，类间方差项被结构性消去。

### 3.5 分层 batch：改变方差，不要误称为改变目标

分层 batch（stratified batch）的步骤是：先按类别分组，在每个类别内部随机排列，再按预定比例从各组取样拼成一个 batch。它减少的是“这一批恰好包含哪些类别”的波动；只要各层权重和目标风险一致，就仍然可以估计原来的经验风险。

如果每批都强制使用与总体相同的类别比例，类间组成的随机波动会被结构性地压低。若数据类别比例本身不均衡，或者训练目标有意给类别不同权重，则必须重新写出目标和抽样权重，不能把“固定比例”自动称作无偏。

### 3.6 用测试把性质钉死

AP-02-013 的验收思路不是只检查一次输出，而是分别检查随机实验的几条性质：

- 相同种子时，索引序列或完整输出应按约定可复现；
- 不同种子时，不应把偶然相同的短序列误认为随机状态被复用；
- Monte Carlo 均值应落在预先说明的标准误范围内，例如 $\pm5$ 个标准误；
- 经验方差与理论方差的比值应接近 $1$，并报告批次大小和抽样方式。

随机测试不应要求一个本来有抽样波动的量“精确到小数点后 12 位”。另一方面，“结果不一样”也不能单独证明随机实现正确；均值、方差和抽样分布必须分别验证。

### 可运行验证（需要 NumPy、SciPy 和本地 Kannada-MNIST 数据）

```python
from pathlib import Path
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# --- Demo 0: ULP 阶梯与吸收（AP-02-001） ---
for dt in (np.float16, np.float32, np.float64):
    row = [float(np.nextafter(dt(x0), dt(np.inf)) - dt(x0)) for x0 in (1.0, 1024.0)]
    print(f"{dt.__name__}: ulp(1)={row[0]:.6e}  ulp(1024)={row[1]:.6e}  ratio={row[1]/row[0]:.1f}")
big = np.float32(1.0e8)
print(f"float32: 1e8 + 4.0 = {float(big + np.float32(4.0))!r}  (ulp(1e8)={float(np.spacing(big))})")

# --- Demo 1: 取模偏差的机制（8-bit 抽取 mod 100） ---
r = rng.integers(0, 256, size=4_000_000).astype(np.uint64)
counts = np.bincount(r % 100, minlength=100)
print(f"empirical P(residue 0)  = {counts[0]/len(r):.8f}   theory 3/256 = {3/256:.8f}")
print(f"empirical P(residue 99) = {counts[99]/len(r):.8f}   theory 2/256 = {2/256:.8f}")

# --- Demo 2: 求和算法误差（fp32） ---
x = np.full(10_000_000, np.float32(0.1), dtype=np.float32)
xs = x[:1_000_000]
s32 = np.float32(0.0)
for v in xs:
    s32 = np.float32(s32 + v)
ref1 = float(np.sum(xs.astype(np.float64)))
print(f"sequential fp32 loop, 1e6 adds: {float(s32):.6f}  oracle {ref1:.6f}  err {float(s32)-ref1:+.6f}")
pair, ref = float(np.sum(x)), float(np.sum(x.astype(np.float64)))
print(f"np.sum pairwise fp32, 1e7     : {pair:.6f}  oracle {ref:.6f}  err {pair-ref:+.6f}")
s, c = 0.0, 0.0
for v in xs:
    yv = float(v) - c
    t = s + yv
    c = (t - s) - yv
    s = t
print(f"Kahan fp64 loop, 1e6          : {s:.9f}  oracle {ref1:.9f}  err {s-ref1:+.2e}")

# --- Demo 3: fp16 梯度吸收与 loss scaling ---
start, inc, steps, S = np.float16(256.0), np.float16(0.1), 1000, 8
s16, absorbed = start, 0
for _ in range(steps):
    nxt = np.float16(s16 + inc)
    absorbed += int(nxt == s16)
    s16 = nxt
print(f"fp16 acc at {float(start)}, add {float(inc)} x{steps}: absorbed {absorbed}/{steps}, gained {float(s16-start)} (true 100.0)")
s16s = start
for _ in range(steps):
    s16s = np.float16(s16s + np.float16(inc * S))
print(f"loss scale S={S}: scaled gain {float(s16s-start)}, unscaled {float(s16s-start)/S:.4f} (true 100.0)")

# --- Demo 4: 真实 Kannada-MNIST 上的 minibatch 梯度噪声（AP-02-013） ---
data_dir = Path("kannada-mnist-study/data/Kannada_MNIST_npz/Kannada_MNIST")
with np.load(data_dir / "X_kannada_MNIST_train.npz") as z:
    X = z[z.files[0]]
with np.load(data_dir / "y_kannada_MNIST_train.npz") as z:
    y = z[z.files[0]]
N = X.shape[0]
Xf = X.astype(np.float32).reshape(N, -1) / 255.0
Xb = np.concatenate([Xf, np.ones((N, 1), np.float32)], axis=1)
Y = np.eye(10, dtype=np.float32)[y.astype(np.int64)]
W = np.zeros((Xb.shape[1], 10), np.float32)

def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

for _ in range(200):                            # 先用 GD 走到一个非退化的 w
    W -= np.float32(0.5) * (Xb.T @ (softmax_rows(Xb @ W) - Y)) / np.float32(N)
P = softmax_rows(Xb @ W)
R = (P - Y).astype(np.float64)
gbar = (Xb.astype(np.float64).T @ R) / N
tr_sigma = float((np.sum(Xb.astype(np.float64)**2, axis=1) * np.sum(R**2, axis=1)).mean() - (gbar**2).sum())
print(f"N={N}, d={gbar.size}, ||gbar||_F={np.linalg.norm(gbar):.6f}, tr Sigma={tr_sigma:.6f}")

reps = 4000
for B in (128, 512):
    samples = np.empty((reps,) + gbar.shape, dtype=np.float64)
    for k in range(reps):
        idx = rng.choice(N, size=B, replace=False)
        samples[k] = (Xb[idx].astype(np.float64).T @ R[idx]) / B
    err = samples - gbar
    emp_tr = float((err**2).sum(axis=(1, 2)).mean())
    theory = (1.0 - B / N) * tr_sigma / B
    se = float(np.linalg.norm(err.mean(axis=0))) / np.sqrt(theory / reps)
    print(f"B={B}: E||ghat-gbar||^2 empirical={emp_tr:.6f} theory(FPC)={theory:.6f} "
          f"ratio={emp_tr/theory:.4f}; ||E[ghat]-gbar|| in SE units = {se:.2f}")
    for coord in ((350, 3), (200, 7)):
        e = err[:, coord[0], coord[1]]
        print(f"  coord{coord}: skew={float(stats.skew(e)):+.3f} "
              f"kurtosis={float(stats.kurtosis(e)):+.3f} normaltest p={stats.normaltest(e).pvalue:.4f}")

# --- Demo 5: mean-of-batch-means 偏差（AP-01-008） ---
means = np.array([0.2, 0.4, 1.0], dtype=np.float32)
sizes = np.array([64, 64, 2], dtype=np.int64)
oracle = float(np.repeat(means.astype(np.float64), sizes).mean())
naive = float(means.astype(np.float64).mean())
print(f"fixture (64,64,2): weighted={oracle:.12f}  naive={naive:.12f}  Delta_batch={naive-oracle:+.12f}")
m = np.concatenate([np.full(468, 0.3), [0.9]])
n = np.concatenate([np.full(468, 128), [96]]).astype(np.int64)
w, na = float((m * n).sum() / n.sum()), float(m.mean())
print(f"MNIST split 468x128+96: weighted={w:.12f}  naive={na:.12f}  Delta_batch={na-w:+.12f}")
```

实测输出（本仓库，amp 环境，numpy 2.5.1 / scipy 1.18.0）：

```text
=== Demo 0 ===
float16: ulp(1)=9.765625e-04  ulp(1024)=1.000000e+00  ratio=1024.0
float32: ulp(1)=1.192093e-07  ulp(1024)=1.220703e-04  ratio=1024.0
float64: ulp(1)=2.220446e-16  ulp(1024)=2.273737e-13  ratio=1024.0
float32: 1e8 + 4.0 = 100000000.0  (ulp(1e8)=8.0)

=== Demo 1 ===
empirical P(residue 0)  = 0.01173675   theory 3/256 = 0.01171875
empirical P(residue 99) = 0.00776150   theory 2/256 = 0.00781250

=== Demo 2 ===
sequential fp32 loop, 1e6 adds: 100958.343750  oracle 100000.001490  err +958.342260
np.sum pairwise fp32, 1e7     : 1000000.125000  oracle 1000000.014901  err +0.110099
Kahan fp64 loop, 1e6          : 100000.001490116  oracle 100000.001490116  err +0.00e+00

=== Demo 3 ===
fp16 acc at 256.0, add 0.0999755859375 x1000: absorbed 1000/1000, gained 0.0 (true 100.0)
loss scale S=8: scaled gain 914.0, unscaled 114.2500 (true 100.0)

=== Demo 4 ===
N=60000, d=7850, ||gbar||_F=0.025134, tr Sigma=4.063921
B=128: E||ghat-gbar||^2 empirical=0.031932 theory(FPC)=0.031682 ratio=1.0079; ||E[ghat]-gbar|| in SE units = 1.08
  coord(350, 3): skew=-1.052 kurtosis=+2.294 normaltest p=0.0000
  coord(200, 7): skew=+17.954 kurtosis=+350.797 normaltest p=0.0000
B=512: E||ghat-gbar||^2 empirical=0.007857 theory(FPC)=0.007870 ratio=0.9984; ||E[ghat]-gbar|| in SE units = 1.00
  coord(350, 3): skew=-0.410 kurtosis=+0.455 normaltest p=0.0000
  coord(200, 7): skew=+8.656 kurtosis=+78.378 normaltest p=0.0000

=== Demo 5 ===
fixture (64,64,2): weighted=0.310769235171  naive=0.533333336314  Delta_batch=+0.222564101143
MNIST split 468x128+96: weighted=0.300960000000  naive=0.301279317697  Delta_batch=+0.000319317697
```

读法：

- Demo 2：顺序 fp32 累加在 $10^6$ 次加法后相对误差已达 $\sim10^{-2}$（单向，随 $K$ 继续涨）；pairwise 在 $10^7$ 上只有 $10^{-7}$ 相对误差；Kahan 与 float64 oracle 逐位一致。这就是"系统性 $\sim K$ vs 零均值 $\sim\sqrt K$"的实证。
- Demo 3：无缩放时 1000 个增量全部被吸收（收益 0）；$S=8$ 时增量存活，残余误差来自增量自身的 fp16 量化——所以真实 AMP 把反缩放后的更新落到 fp32 主权重上，而不是停在 fp16 累加器。
- Demo 4：trace 公式与 FPC 在真实数据上精确到 1% 内；无偏性在 1 个标准误内；但单坐标正态性被强烈拒绝（零膨胀重尾），$b$ 增大四倍后偏度/峰度显著收敛。聚合量（trace、范数）比单个坐标更早进入 CLT 区域。
- Demo 5：$\Delta_{\mathrm{batch}}$ 的大小 $\approx$（尾批占比）×（尾批均值 − 全局均值）。尾批 2/130 个样本就能制造 0.22 的偏差；60000 拆分时尾批占 0.16%，偏差 $\sim3\times10^{-4}$——小，但它是确定方向的，且出现在你拿来调超参的验证指标里。

---

## 4. aggregation bias 在第 2、3 节中出现在哪里，为什么重要

第 2 节中的无偏条件

$$
\mathbb E[g_t\mid w_t]=\nabla\hat R_N(w_t)
$$

本身就是一个聚合条件：抽样得到的平均必须和目标中的样本平均使用同一个测度。为了避免和 batch size $b$ 混淆，定义梯度估计的系统性偏差为

$$
r(w):=\mathbb E[g\mid w]-\nabla\hat R_N(w).
$$

于是

$$
r(w)=0
$$

表示无偏，$r(w)\neq0$ 表示在当前参数点上，抽样或归约规则会把平均更新方向推向某一侧。一次抽样的随机偏离 $g-\mathbb E[g\mid w]$ 是噪声；$r(w)$ 是噪声的中心相对于目标梯度的位移。两者不能用同一个“波动”概念代替。

### 4.1 偏差怎样移动解的位置

设 $w^*$ 是经验风险的一个局部驻点，满足

$$
\nabla\hat R_N(w^*)=0.
$$

在 $w^*$ 附近，假设经验风险二阶可微，并令

$$
H:=\nabla^2\hat R_N(w^*).
$$

若某种有偏更新的平均不动点为 $w^\dagger$，它近似满足

$$
\nabla\hat R_N(w^\dagger)+r(w^\dagger)=0.
$$

令 $\delta=w^\dagger-w^*$，在 $w^*$ 处对梯度做一阶 Taylor 展开：

$$
\nabla\hat R_N(w^\dagger)
\approx \nabla\hat R_N(w^*)+H\delta
=H\delta.
$$

如果进一步假设 $H$ 可逆，且 $r(w^\dagger)$ 可以用 $r(w^*)$ 近似，则

$$
H\delta+r(w^*)\approx0,
\qquad
\delta\approx-H^{-1}r(w^*).
$$

这条式子有三个前提：局部线性化有效、偏差在该邻域变化不剧烈、$H$ 可逆。若 $H$ 奇异，不能直接写 $H^{-1}$；此时某些平坦方向上的位移可能不唯一，需要单独分析可辨识子空间或使用额外的正则化假设。

因此，方差主要改变“围绕哪里散开”的半径，偏差则改变中心位置。重复运行、Polyak--Ruppert 平均和增大样本数可以削弱零均值噪声，却不会自动消除一个始终存在的 $r(w)$。

### 4.2 代码中常见的注入点

| 位置 | 具体例子 | 数学后果 |
| --- | --- | --- |
| 采样器 | 取模抽样、非均匀抽样却不加 HT 权重 | $\mathbb E[g]$ 变成 $q_i$ 加权的梯度 |
| 随机状态 | 多个组件共享一个全局 RNG | 实验序列与组件边界隐式耦合，难以复现和归因 |
| 归约器 | fp32 顺序累加、fp16 累加、错误的 mean of means | 理想实数和被有限精度或错误测度替代 |
| 有状态统计 | BatchNorm 的 running statistics、优化器的 EMA | 初始值、更新系数和 train/eval 规则共同决定估计偏差 |

Demo 1 展示抽样偏差，Demo 2 和 Demo 3 展示有限精度归约，Demo 5 展示批均值与样本均值的测度错位。它们发生在不同层，不能用“多抽几次”统一修复。

### 4.3 EMA 的启动偏差：Adam 例子

指数滑动平均（exponential moving average, EMA）写成

$$
m_t=\beta m_{t-1}+(1-\beta)g_t,
\qquad 0<\beta<1,
\qquad m_0=0.
$$

为了只说明初始化效应，暂时假设每一步的梯度期望都是同一个固定向量 $\mu$。递推展开后

$$
\mathbb E[m_t]=(1-\beta^t)\mu.
$$

前几步的期望被因子 $1-\beta^t$ 压低，这就是启动偏差（initialization bias）。Adam 使用

$$
\hat m_t=\frac{m_t}{1-\beta^t}
$$

做偏差修正。真实优化过程中 $g_t$ 的期望会随 $w_t$ 改变，所以“修正后等于当前真实梯度”不是无条件结论；上面的固定 $\mu$ 是解释修正项来源的简化模型。

### 4.4 为什么工程上不能忽略它

1. **系统性偏差不会靠重复平均消失。** 重复实验减少的是零均值部分；如果每次实验都使用同一个错误测度，平均只会更稳定地得到错误目标。
2. **累加误差可能随操作次数增长。** 浮点吸收的方向由当前累加器格点决定，不能把它当成独立、均值为零的噪声。
3. **被污染的通常是决策依据。** 验证指标、早停判据和超参数搜索目标都依赖归约；一个稳定但有偏的指标会让错误选择看起来很可信。
4. **现代训练放大了归约次数。** fp16/bf16、梯度累积和多卡 all-reduce 都增加了求和次数或改变了求和顺序，因此需要显式规定精度、顺序和可重复性要求。

**在第 2 问（数学）里**：SGD 理论的**第一条前提** $\mathbb E[g_t\mid w_t]=\nabla\hat R_N(w_t)$ 本身就是一句聚合陈述——"抽样测度必须与目标和的测度对齐"。任何错位都产生偏差项 $b(w)=\mathbb E[g_t\mid w]-\nabla\hat R_N(w)$。在最优点邻域内线性化：迭代实际求解 $\nabla\hat R_N(w)+b(w)=0$，不动点偏移
$$w^\dagger-w^\*\approx-\,H^{-1}\,b,\qquad H=\nabla^2\hat R_N(w^\*).$$
**方差决定噪声球的半径，偏差直接移动球心。** Polyak 平均、重复实验、增大重复次数只能压零均值部分；$H^{-1}b$ 一动不动。

**在第 3 问（代码）里**，注入点有三处：

1. **采样器**：取模偏差（Demo 1）、非均匀抽样不配 HT 权重、全局 RNG 被其他模块消费导致序列漂移（相关性与隐式耦合）。
2. **归约器**：fp32 顺序累加的吸收（Demo 2）、fp16 累加（Demo 3）、日志里的 mean-of-means（Demo 5）。
3. **有状态聚合**：BatchNorm 的 running stats（EMA），以及一个教科書级例子——Adam 的偏差修正 $\hat m_t=m_t/(1-\beta_1^t)$：$m_0=0$ 初始化的 EMA 是有偏聚合，$\mathbb E[m_t]=(1-\beta_1^t)\,\mathbb E[g]$，修正项就是显式地去 aggregation bias。

**为什么重要**（四条，按杀伤力排序）：

1. 它不平均消失。上面 $H^{-1}b$：偏差是定点移动，不是噪声。
2. 渐近增长率不同：零均值舍入误差 $\sim\sqrt K$，系统性吸收 $\sim K$。$K$ 大时偏差必然主导（Demo 2 的 $10^6$ 次加法已经 1% 相对误差）。
3. 它污染的是你信任的东西：验证指标、早停判据、超参搜索的目标值。方差大而置信区间宽，至少你知道自己不知道；聚合偏差让你**确信一个错的数**。
4. 现代流水线把它放大：fp16/bf16、梯度累积、多卡 all-reduce 都是更多次、更粗格点上的聚合。

---

## 5. 无偏之后，方差怎么办；正则化和标准化是否相关

这一节先把四个概念分开：

1. **抽样方差**描述批次梯度围绕其期望波动多大。
2. **显式正则化**直接修改优化目标，例如加入 $\lambda\lVert w\rVert^2/2$。
3. **输入标准化**改变输入的尺度和几何条件。
4. **BatchNorm**在每个批次内计算统计量，因此它确实把聚合噪声带进了前向计算。

它们可能互相影响，但不是同一个数学操作。

### 5.1 SGD 噪声的定量位置

无偏性只说明

$$
\mathbb E[g_t\mid w_t]=\nabla\hat R_N(w_t);
$$

它没有让 $\mathbb E\lVert g_t-\nabla\hat R_N(w_t)\rVert^2$ 自动变成零。这个二阶量仍然由批次大小、数据分布和参数点决定。

在一个局部极小点 $w^*$ 附近，令误差 $e_t=w_t-w^*$。如果目标梯度可以线性化为 $\nabla F(w_t)\approx H e_t$，其中 $H$ 是局部 Hessian，并把抽样噪声记为 $\xi_t$，则更新近似为

$$
e_{t+1}\approx(I-\eta H)e_t-\eta\xi_t.
$$

若 $H$ 的相关方向是稳定的、噪声均值为零且二阶矩有限，恒定步长下误差会形成稳态分布。令

$$
C:=\mathbb E[e_te_t^{\top}]
$$

表示这个稳态误差的协方差。在小步长和线性化近似下，$C$ 满足一个近似的离散 Lyapunov 关系：

$$
HC+CH^{\top}\approx\frac{\eta}{b}\Sigma.
$$

这里的 $\Sigma$ 是单样本梯度噪声的协方差；右侧的 $1/b$ 来自批次平均。这个式子需要局部稳定性、近似恒定的协方差和足够小的 $\eta$，不是任意非凸网络上的精确恒等式。

如果 $H$ 在某个方向上的特征值较小，恢复该方向的确定性力较弱，噪声更容易在该方向积累。因此会出现“噪声与曲率共同选择参数区域”的现象，常被概括为 SGD 的隐式正则化或 flat-minima preference。这里的 preference 是训练动力学的经验性描述，不等于一个不加条件的全局定理。

### 5.2 显式正则化不是降方差

以 L2 正则化为例，把目标改成

$$
F_{\lambda}(w)=\hat R_N(w)+\frac\lambda2\lVert w\rVert^2,
\qquad \lambda\ge0.
$$

它的梯度是

$$
\nabla F_{\lambda}(w)=\nabla\hat R_N(w)+\lambda w.
$$

$\lambda w$ 是确定性项：它把参数向较小范数方向拉动，因而有意改变了目标的最优点。它不是把 minibatch 梯度的方差变小了，也不是对抽样偏差做校正。把 weight decay 当成降方差技术，会把“改变目标”和“估计目标”两个问题混在一起。

两者存在间接耦合。例如，正则化可能改变训练过程中遇到的参数区域，从而改变 $\Sigma(w)$；但这不是 L2 正则化的定义，也不能由定义推出方差一定下降。

### 5.3 输入标准化改变的是几何条件

对每个输入坐标，可以用训练集均值和尺度做变换：

$$
\widetilde x_j=\frac{x_j-\mu_j}{\sigma_j},
\qquad \sigma_j>0.
$$

这改变了参数空间中不同方向的尺度，通常会改善 Hessian 的条件数 $\kappa(H)$，从而让相同步长下的优化更容易稳定。它也会间接改变单样本梯度分布和 $\Sigma(w)$，但“改善条件数”和“构造无偏梯度估计”是两件不同的事。

因此，输入标准化的直接问题是数值尺度和优化几何；它不是 aggregation bias 的通用修复。如果标准化统计量本身用错了数据切分或把验证集信息泄漏进训练，也会引入新的数据契约问题。

### 5.4 BatchNorm：聚合噪声进入前向计算

BatchNorm 的训练前向会在一个 batch 上计算通道或特征的均值和方差。对一维特征写成

$$
\hat\mu_B=\frac1b\sum_{i\in B}x_i,
\qquad
\hat\sigma_B^2=\frac1b\sum_{i\in B}(x_i-\hat\mu_B)^2.
$$

这两个量是总体均值和方差的 Monte Carlo 估计，所以每个 batch 的前向归一化都带有抽样波动。它可能成为 BatchNorm 正则化效果的一部分，但前提是先认识到它是噪声来源，而不是把它写成一个确定的总体统计量。

若样本独立同分布，真实方差为 $\sigma^2$，则分母为 $b$ 的方差估计满足

$$
\mathbb E[\hat\sigma_B^2]=\frac{b-1}{b}\sigma^2.
$$

它是有偏的；分母改成 $b-1$（要求 $b>1$）才得到常用的无偏样本方差。具体框架可以在训练前向和 running statistics 中选择不同的约定，必须查清 API，不能把某一框架的行为推广成 BatchNorm 的抽象定义。

running statistics 通常用 EMA 更新。例如 running mean 可以写成

$$
m_t=\beta m_{t-1}+(1-\beta)\hat\mu_{B_t}.
$$

训练时使用当前批次统计量，推理时使用累积的 running statistics，这相当于切换了聚合规则。若数据分布发生漂移，EMA 还会有滞后；因此 train/eval gap 的一部分可能来自统计量估计与切换，而不只是模型参数本身。

### 5.5 四种操作的对照

| 操作 | 直接改变什么 | 是否直接修复 minibatch 偏差 |
| --- | --- | --- |
| 增大 batch、分层抽样、控制变量 | 梯度估计的方差 | 不一定；仍需检查抽样测度 |
| HT 加权、样本加权归约 | 估计量的期望 | 是，前提是权重和抽样概率正确 |
| L2 / weight decay | 优化目标和最优点 | 否；它是有意的目标修改 |
| 输入标准化 | 输入尺度与优化几何 | 否；它不是抽样校正 |
| BatchNorm | 每批前向统计和训练/推理状态 | 它本身引入并管理一类聚合噪声 |

最短总结是：方差靠平均、分层和控制变量；偏差靠测度对齐、权重校正和数值精度；显式正则化是另一根轴；BatchNorm 则是聚合、噪声和状态切换交汇的模块。

---

## 6. 总结与仓库映射

如果只保留一张 mental model，可以按下面的顺序检查一次训练量：

1. **目标是什么？** 总体风险 $R$ 是对未知新样本分布取期望；经验风险 $\hat R_N$ 是训练集上的样本平均。
2. **估计了什么？** GD 计算经验风险的完整梯度；SGD 用批次梯度估计它，目标函数没有因此换掉。
3. **抽样测度是什么？** 均匀有放回、均匀无放回和非均匀抽样的期望与方差不同；非均匀抽样需要 HT 等权重校正才能保持原目标。
4. **归约怎样实现？** 尾批需要样本加权；浮点求和需要明确精度和顺序；同一个数学平均在不同 dtype 下可能有不同误差。
5. **状态怎样变化？** BatchNorm running statistics、EMA 和优化器状态都有初始化、更新系数和 train/eval 边界，不能只看一个瞬时 batch。

| 仓库问题 | 在本文的位置 | 它具体钉住什么 |
| --- | --- | --- |
| AP-01-006 stratified split | §2.4.5、§3.3--§3.5 | 分层抽样和局部随机状态 |
| AP-01-007 last batch | §1.1、§3.2、Demo 5 | 尾批策略与样本加权 |
| AP-01-008 sample-weighted mean | §1.1、Demo 5 | 批测度与样本测度的差异 |
| AP-02-001 ulp spacing | §1.2、§3.4、Demo 0/3 | 浮点格点、吸收和量级依赖 |
| AP-02-010 Kahan | §3.4、Demo 2 | 补偿求和和归约误差 |
| AP-02-011 Welford | §2.4.2、§3.4 | 在线均值/方差的数值卫生 |
| AP-02-013 minibatch estimator | §2.1、§2.4、Demo 4 | 无偏性、FPC、方差与验收证据 |

**来源。** 总体风险与 generalization error 的区分、softmax 回归的 logits/概率/one-hot/cross-entropy 链条，依据本地 [Dive into Deep Learning 1.0.3 的 §3.6.1、§4.1](kannada-mnist-study/references/d2l/dive-into-deep-learning.pdf) 改写。代码和验收映射来自本仓库列出的 AP-01-008、AP-02-001、AP-02-007 与 AP-02-013 题目契约；本文件没有把教材中的代码当作实现答案。
