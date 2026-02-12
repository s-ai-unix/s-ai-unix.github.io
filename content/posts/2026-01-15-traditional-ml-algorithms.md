---
title: "深度学习前夜：十大传统机器学习算法的历史与数学之美"
date: 2026-01-15T22:30:00+08:00
draft: false
description: "回顾机器学习黄金时代，详细推导十大经典算法的数学原理，从线性回归到主成分分析"
categories: ["机器学习", "算法"]
tags: ["机器学习", "算法", "数学史"]
cover:
    image: "images/covers/ml-algorithms-legacy.jpg"
    alt: "抽象几何图案"
    caption: "数学的优雅与智慧"
---

## 引言：黄金时代

想象一下 2006 年的秋天，深度学习尚未兴起。那时的机器学习领域正经历着一场静悄悄的革命。统计学习方法、核方法、集成学习层出不穷，数学家们用优雅的公式编织着智能的梦想。

那时，人们相信：只要数据足够、特征工程足够细致，我们就能教机器做任何事。这种信念催生了一批经典算法——它们或许不如今天的深度神经网络那样炫目，但每一款都凝聚着数学家的智慧，每一步推导都闪耀着逻辑的光辉。

今天，我们回顾这段黄金时代，讲述十个改变了世界的传统机器学习算法的故事。但这次，让我们放慢脚步，亲手推导每一步，感受数学的力量。

---

## 一、线性回归：回归分析的鼻祖

**时间：1795 年 - 阿德里安-马里·勒让德 (Adrien-Marie Legendre)**

### 历史的偶然

1795 年，法国天文学家勒让德正在为一个问题头疼：如何用最简单的方法拟合行星轨道数据？他需要找到一条直线，让所有数据点到这条直线的距离平方和最小。

这就是**最小二乘法**的诞生。

### 推导过程

让我们从最简单的情况开始。假设我们有 $n$ 个数据点 $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$，想要找到一条直线 $y = w_0 + w_1 x$ 来拟合这些数据。

**第一步：定义误差**

对于每个数据点 $(x_i, y_i)$，我们的预测值是 $\hat{y}_i = w_0 + w_1 x_i$，误差就是观测值和预测值的差：

$$
e_i = y_i - \hat{y}_i = y_i - (w_0 + w_1 x_i)
$$

**第二步：定义损失函数**

为什么是平方误差？勒让德选择平方误差有几个好处：
1. 非负：平方后总是非负
2. 可导：处处光滑，便于优化
3. 凸函数：只有一个最小值

损失函数定义为：

$$
L(w_0, w_1) = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} [y_i - (w_0 + w_1 x_i)]^2
$$

**第三步：求偏导**

为了找到最小值，我们对 $w_0$ 和 $w_1$ 分别求偏导：

$$
\frac{\partial L}{\partial w_0} = \sum_{i=1}^{n} 2[y_i - (w_0 + w_1 x_i)] \cdot (-1) = -2 \sum_{i=1}^{n} [y_i - (w_0 + w_1 x_i)]
$$

$$
\frac{\partial L}{\partial w_1} = \sum_{i=1}^{n} 2[y_i - (w_0 + w_1 x_i)] \cdot (-x_i) = -2 \sum_{i=1}^{n} x_i [y_i - (w_0 + w_1 x_i)]
$$

**第四步：令偏导为零**

$$
\begin{align}
\frac{\partial L}{\partial w_0} &= 0 \Rightarrow \sum_{i=1}^{n} [y_i - (w_0 + w_1 x_i)] = 0 \\
&\Rightarrow \sum_{i=1}^{n} y_i - n w_0 - w_1 \sum_{i=1}^{n} x_i = 0 \\
&\Rightarrow n w_0 + w_1 \sum_{i=1}^{n} x_i = \sum_{i=1}^{n} y_i
\end{align}
$$

$$
\begin{align}
\frac{\partial L}{\partial w_1} &= 0 \Rightarrow \sum_{i=1}^{n} x_i [y_i - (w_0 + w_1 x_i)] = 0 \\
&\Rightarrow \sum_{i=1}^{n} x_i y_i - w_0 \sum_{i=1}^{n} x_i - w_1 \sum_{i=1}^{n} x_i^2 = 0 \\
&\Rightarrow w_0 \sum_{i=1}^{n} x_i + w_1 \sum_{i=1}^{n} x_i^2 = \sum_{i=1}^{n} x_i y_i
\end{align}
$$

**第五步：解线性方程组**

记 $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$，$\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$，从第一个方程：

$$
w_0 = \bar{y} - w_1 \bar{x}
$$

代入第二个方程：

$$
\begin{align}
(\bar{y} - w_1 \bar{x}) \sum_{i=1}^{n} x_i + w_1 \sum_{i=1}^{n} x_i^2 &= \sum_{i=1}^{n} x_i y_i \\
\bar{y} n \bar{x} - w_1 n \bar{x}^2 + w_1 \sum_{i=1}^{n} x_i^2 &= \sum_{i=1}^{n} x_i y_i \\
w_1 \left(\sum_{i=1}^{n} x_i^2 - n \bar{x}^2\right) &= \sum_{i=1}^{n} x_i y_i - n \bar{x} \bar{y} \\
w_1 &= \frac{\sum_{i=1}^{n} x_i y_i - n \bar{x} \bar{y}}{\sum_{i=1}^{n} x_i^2 - n \bar{x}^2}
\end{align}
$$

这就是著名的**最小二乘估计**。

### 矩阵形式

对于多元线性回归，我们有 $d$ 个特征。设 $\mathbf{x}_i = (1, x_{i,1}, x_{i,2}, \ldots, x_{i,d})^T$ 是增广特征向量，$\mathbf{w} = (w_0, w_1, \ldots, w_d)^T$ 是参数向量。

损失函数写为：

$$
L(\mathbf{w}) = \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2
$$

其中 $\mathbf{X} = \begin{pmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_n^T \end{pmatrix}$ 是设计矩阵，$\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}$ 是响应向量。

展开损失函数：

$$
\begin{align}
L(\mathbf{w}) &= (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w}) \\
&= \mathbf{y}^T \mathbf{y} - \mathbf{y}^T \mathbf{X}\mathbf{w} - \mathbf{w}^T \mathbf{X}^T \mathbf{y} + \mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w}
\end{align}
$$

注意 $\mathbf{y}^T \mathbf{X}\mathbf{w}$ 是标量，等于其转置 $\mathbf{w}^T \mathbf{X}^T \mathbf{y}$，因此：

$$
L(\mathbf{w}) = \mathbf{y}^T \mathbf{y} - 2 \mathbf{w}^T \mathbf{X}^T \mathbf{y} + \mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w}
$$

求梯度：

$$
\begin{align}
\nabla_{\mathbf{w}} L(\mathbf{w}) &= \nabla_{\mathbf{w}} (\mathbf{y}^T \mathbf{y}) - 2 \nabla_{\mathbf{w}} (\mathbf{w}^T \mathbf{X}^T \mathbf{y}) + \nabla_{\mathbf{w}} (\mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w}) \\
&= 0 - 2 \mathbf{X}^T \mathbf{y} + 2 \mathbf{X}^T \mathbf{X} \mathbf{w}
\end{align}
$$

令梯度为零：

$$
\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{y}
$$

这就是著名的**正规方程**（Normal Equation）。如果 $\mathbf{X}^T \mathbf{X}$ 可逆，解为：

$$
\mathbf{w}^{\ast} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

### 几何直观

从几何上看，$\mathbf{y}$ 在列空间 $\mathcal{C}(\mathbf{X})$ 上的投影是：

$$
\hat{\mathbf{y}} = \mathbf{X} \mathbf{w}^{\ast} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} = \mathbf{H} \mathbf{y}
$$

其中 $\mathbf{H} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$ 是**帽子矩阵**（hat matrix）。它把 $\mathbf{y}$ "戴上了帽子"。

残差 $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\mathbf{y}$ 与列空间正交：

$$
\mathbf{X}^T \mathbf{e} = \mathbf{X}^T (\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{X}^T \mathbf{y} - \mathbf{X}^T \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} = \mathbf{X}^T \mathbf{y} - \mathbf{X}^T \mathbf{y} = \mathbf{0}
$$

这就是正交投影的数学表达！

---

## 二、逻辑回归：从天文学到生物学的跨界

**时间：1958 年 - 大卫·考克斯 (David Cox)**

### 跨界的灵感

1958 年，英国统计学家大卫·考克斯遇到了一个新问题：如何预测二元变量的概率？传统的线性回归给出的是实数值，但概率必须在 $[0, 1]$ 之间。

考克斯灵机一动，想到了 Sigmoid 函数。

### 推导过程

**第一步：理解二分类问题**

给定 $n$ 个样本 $(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)$，其中 $\mathbf{x}_i \in \mathbb{R}^d$，$y_i \in \{0, 1\}$。我们想要学习一个模型 $f: \mathbb{R}^d \to [0, 1]$，使得 $f(\mathbf{x})$ 表示 $P(y=1|\mathbf{x})$。

**第二步：为什么不能直接用线性回归？**

如果用线性回归 $y = \mathbf{w}^T \mathbf{x}$，输出可以是任意实数，但概率必须满足 $0 \leq p \leq 1$。

**第三步：引入 Sigmoid 函数**

Sigmoid 函数定义为：

$$
\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}
$$

它的性质：
- 当 $z \to -\infty$，$\sigma(z) \to 0$
- 当 $z \to +\infty$，$\sigma(z) \to 1$
- 当 $z = 0$，$\sigma(z) = 0.5$

因此，我们定义：

$$
p(\mathbf{x}) = P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}
$$

**第四步：导出似然函数**

对于单个样本 $(\mathbf{x}_i, y_i)$，其概率可以统一写成：

$$
P(y_i|\mathbf{x}_i, \mathbf{w}) = p(\mathbf{x}_i)^{y_i} (1 - p(\mathbf{x}_i))^{1 - y_i}
$$

验证：
- 若 $y_i = 1$：$P(y_i=1|\mathbf{x}_i) = p(\mathbf{x}_i) \cdot (1-p(\mathbf{x}_i))^0 = p(\mathbf{x}_i)$ ✓
- 若 $y_i = 0$：$P(y_i=0|\mathbf{x}_i) = p(\mathbf{x}_i)^0 \cdot (1-p(\mathbf{x}_i))^1 = 1 - p(\mathbf{x}_i)$ ✓

假设样本独立同分布，联合概率（似然）为：

$$
\mathcal{L}(\mathbf{w}) = \prod_{i=1}^{n} P(y_i|\mathbf{x}_i, \mathbf{w}) = \prod_{i=1}^{n} p(\mathbf{x}_i)^{y_i} (1 - p(\mathbf{x}_i))^{1 - y_i}
$$

**第五步：取对数得到对数似然**

取对数简化计算：

$$
\begin{align}
\ell(\mathbf{w}) = \log \mathcal{L}(\mathbf{w}) &= \sum_{i=1}^{n} \left[ y_i \log p(\mathbf{x}_i) + (1 - y_i) \log(1 - p(\mathbf{x}_i)) \right]
\end{align}
$$

**第六步：计算梯度**

我们需要计算 $\frac{\partial \ell}{\partial w}$。首先计算 $\frac{\partial p(x)}{\partial w}$：

$$
\begin{align}
p(x) &= \frac{1}{1 + e^{-w^T x}} \\
\frac{\partial p(x)}{\partial w} &= -\frac{1}{(1 + e^{-w^T x})^2} \cdot \frac{\partial}{\partial w} (1 + e^{-w^T x}) \\
&= -\frac{1}{(1 + e^{-w^T x})^2} \cdot e^{-w^T x} \cdot (-x) \\
&= \frac{e^{-w^T x}}{(1 + e^{-w^T x})^2} x
\end{align}
$$

注意到：

$$
p(x)(1 - p(x)) = \frac{1}{1 + e^{-w^T x}} \cdot \frac{e^{-w^T x}}{1 + e^{-w^T x}} = \frac{e^{-w^T x}}{(1 + e^{-w^T x})^2}
$$

因此：

$$
\frac{\partial p(x)}{\partial w} = p(x)(1 - p(x)) x
$$

这是一个非常优雅的结论！

**第七步：计算对数似然的梯度**

$$
\frac{\partial \ell}{\partial w} = \sum_{i=1}^{n} (y_i - p(x_i)) x_i
$$

这就是逻辑回归的梯度公式！

**第八步：梯度上升法**

由于我们要最大化对数似然，使用梯度上升：

$$
w_{t+1} = w_t + \eta \sum_{i=1}^{n} (y_i - p_t(x_i)) x_i
$$

其中 $\eta$ 是学习率。

### Logit 变换

我们也可以从另一个角度理解逻辑回归。定义 **logit 变换**：

$$
\text{logit}(p) = \ln\left(\frac{p}{1-p}\right)
$$

对于逻辑回归：

$$
\begin{align}
\text{logit}(P(y=1|\mathbf{x})) &= \ln\left(\frac{P(y=1|\mathbf{x})}{1 - P(y=1|\mathbf{x})}\right) \\
&= \ln\left(\frac{\sigma(\mathbf{w}^T \mathbf{x})}{1 - \sigma(\mathbf{w}^T \mathbf{x})}\right) \\
&= \ln\left(\frac{\frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}}{\frac{e^{-\mathbf{w}^T \mathbf{x}}}{1 + e^{-\mathbf{w}^T \mathbf{x}}}}\right) \\
&= \ln(e^{\mathbf{w}^T \mathbf{x}}) \\
&= \mathbf{w}^T \mathbf{x}
\end{align}
$$

这表明：logit 变换后，概率的对数几率（log-odds）与特征呈线性关系。

---

## 三、朴素贝叶斯：两个世纪前的概率魔法

**时间：1763 年 - 托马斯·贝叶斯（Thomas Bayes，定理发表）；1950 年代 - 机器学习应用**

### 延迟发表的天才

1763 年，英国牧师托马斯·贝叶斯去世两年后，他的朋友理查德·普赖斯整理并发表了他的一篇论文——《关于机会问题的解法》。

### 推导过程

**第一步：贝叶斯定理**

设 $D$ 为观测数据，$H$ 为假设。贝叶斯定理表述为：

$$
P(H|D) = \frac{P(D|H) P(H)}{P(D)}
$$

其中：
- $P(H)$ 是先验概率（prior）
- $P(D|H)$ 是似然（likelihood）
- $P(D)$ 是证据（evidence）
- $P(H|D)$ 是后验概率（posterior）

**第二步：应用到分类问题**

对于分类问题，我们有类别 $y \in \{1, 2, \ldots, C\}$，特征 $\mathbf{x} = (x_1, x_2, \ldots, x_d)$。根据贝叶斯定理：

$$
P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y) P(y)}{P(\mathbf{x})}
$$

决策规则：

$$
\hat{y} = \arg\max_{y} P(y|\mathbf{x}) = \arg\max_{y} \frac{P(\mathbf{x}|y) P(y)}{P(\mathbf{x})}
$$

由于 $P(\mathbf{x})$ 对所有类别都相同，可以忽略：

$$
\hat{y} = \arg\max_{y} P(\mathbf{x}|y) P(y)
$$

**第三步：朴素独立性假设**

$P(\mathbf{x}|y)$ 的计算困难在于特征之间可能存在依赖关系。朴素贝叶斯做出**条件独立性假设**：

$$
P(\mathbf{x}|y) = P(x_1, x_2, \ldots, x_d|y) = P(x_1|y) P(x_2|y) \cdots P(x_d|y) = \prod_{j=1}^{d} P(x_j|y)
$$

这个假设在现实世界中几乎从不成立，但效果出奇地好。

**第四步：分类决策**

$$
\hat{y} = \arg\max_{y} P(y) \prod_{j=1}^{d} P(x_j|y)
$$

### 高斯朴素贝叶斯

对于连续特征，常假设 $P(x_j|y)$ 服从高斯分布：

$$
P(x_j|y=c) = \frac{1}{\sqrt{2\pi \sigma_{jc}^2}} \exp\left(-\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right)
$$

参数估计（最大似然）：

$$
\hat{P}(y=c) = \frac{n_c}{n}
$$

$$
\hat{\mu}_{jc} = \frac{1}{n_c} \sum_{i: y_i=c} x_{i,j}
$$

$$
\hat{\sigma}_{jc}^2 = \frac{1}{n_c} \sum_{i: y_i=c} (x_{i,j} - \hat{\mu}_{jc})^2
$$

### 多项式朴素贝叶斯（文本分类）

对于文本分类，采用多项式分布。设词汇表大小为 $V$，词 $w$ 在类别 $c$ 中的计数为 $N_{wc}$，类别 $c$ 的总词数为 $N_c = \sum_{w=1}^{V} N_{wc}$。

使用**拉普拉斯平滑**（Laplace smoothing）：

$$
P(w|c) = \frac{N_{wc} + 1}{N_c + V}
$$

$$
P(c) = \frac{\sum_{w} N_{wc}}{\sum_{c', w} N_{w,c'}}
$$

### 为什么有效？

虽然独立性假设不成立，但朴素贝叶斯经常表现良好，原因有：
1. **优化目标不同**：我们关心的是分类准确率，而不是概率估计的精确性
2. **去相关**：即使特征相关，决策边界可能仍然正确
3. **高维特性**：在高维空间中，不同方向的特征对分类的贡献相对独立

---

## 四、K 近邻算法：最简单的记忆学习

**时间：1951 年 - 伊芙琳·菲克斯 (Evelyn Fix) 和约瑟夫·霍奇斯 (Joseph Hodges)**

### 未发表的传奇

1951 年，加州大学伯克利分校的统计学家伊芙琳·菲克斯和约瑟夫·霍奇斯写了一篇论文，提出了一个极其简单的想法：要判断一个新样本属于哪一类，就看看训练数据中离它最近的 $k$ 个样本属于哪一类。

### 推导过程

KNN 本质上不需要"推导"，但我们可以从**风险最小化**的角度理解它。

**第一步：1-NN 的渐近最优性**

假设数据分布 $P(\mathbf{x}, y)$，1-NN 的预测是：

$$
\hat{y} = \arg\max_c P(c|\text{NN}(\mathbf{x}))
$$

其中 $\text{NN}(\mathbf{x})$ 是 $\mathbf{x}$ 的最近邻。

当训练数据量 $n \to \infty$ 时，最近邻 $\text{NN}(\mathbf{x})$ 会无限接近 $\mathbf{x}$，因此：

$$
\lim_{n \to \infty} P(y|\text{NN}(\mathbf{x}) = c) = P(y=c|\mathbf{x})
$$

于是：

$$
\hat{y} = \arg\max_c P(y=c|\mathbf{x})
$$

这正是贝叶斯最优分类器！

**第二步：1-NN 的风险**

贝叶斯最优分类器的错误率是：

$$
R^{\ast} = 1 - \sum_{x} P(x) \max_c P(y=c|x)
$$

1-NN 的渐近错误率是：

$$
R_{\text{1NN}} = 2 \sum_{x} P(x) P(y=\hat{y}_{}^{\ast}|x) (1 - P(y=\hat{y}_{}^{\ast}|x))
$$

其中 $\hat{y}_{}^{\ast}$ 是贝叶斯最优预测。

可以证明：$R^{\ast} \leq R_{\text{1NN}} \leq 2R^{\ast}$。因此，1-NN 的错误率最多是贝叶斯最优的两倍。

**第三步：K 近邻的决策**

对于 K 近邻，我们投票：

$$
\hat{y} = \arg\max_{c} \sum_{i=1}^{K} \mathbb{I}(y_{(i)} = c)
$$

其中 $y_{(i)}$ 是第 $i$ 个最近邻的标签。

也可以加权投票：

$$
\hat{y} = \arg\max_{c} \sum_{i=1}^{K} w_i \cdot \mathbb{I}(y_{(i)} = c)
$$

其中权重 $w_i$ 通常是距离的倒数：$w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_{(i)})}$。

### 距离度量

**欧几里得距离**：

$$
d(\mathbf{x}, \mathbf{x}') = \|\mathbf{x} - \mathbf{x}'\|_2 = \sqrt{\sum_{j=1}^{d} (x_j - x'_j)^2}
$$

**曼哈顿距离**：

$$
d(\mathbf{x}, \mathbf{x}') = \|\mathbf{x} - \mathbf{x}'\|_1 = \sum_{j=1}^{d} |x_j - x'_j|
$$

**余弦相似度**（常用于文本）：

$$
s(\mathbf{x}, \mathbf{x}') = \frac{\mathbf{x}^T \mathbf{x}'}{\|\mathbf{x}\| \cdot \|\mathbf{x}'\|}
$$

### 维度的诅咒

KNN 的问题在于高维空间。考虑超立方体 $[0, 1]^d$，边长为 $\epsilon$ 的立方体体积是 $\epsilon^d$。随着 $d$ 增加，即使是小的 $\epsilon$，体积也趋于零。

这意味着：在高维空间中，任何点之间的距离都趋于相同，最近邻的选择变得随机。解决方法：特征选择、降维（PCA、t-SNE）。

---

## 五、决策树：从 Hunt 算法到 C4.5

**时间：1960 年代 - Hunt 的算法；1986 年 - ID3；1993 年 - C4.5**

### 简单而强大的递归

决策树的思想非常直观：就像医生诊断疾病一样，通过一系列"是/否"的问题来逐步缩小可能性。

### 推导过程

**第一步：理解划分问题**

假设当前数据集为 $D$，我们要选择一个特征 $A$ 和一个划分方式，将 $D$ 划分为子集 $\{D_1, D_2, \ldots, D_k\}$。目标是让每个子集尽可能"纯"（属于同一类）。

**第二步：纯度的度量——熵**

信息论中，香农熵定义为：

$$
H(D) = -\sum_{c=1}^{C} p_c \log_2 p_c
$$

其中 $p_c = \frac{|D_c|}{|D|}$ 是类别 $c$ 的比例。

熵的性质：
- 当所有样本属于同一类（$p_c = 1$ 对某个 $c$），$H(D) = 0$（最纯）
- 当各类均匀分布（$p_c = \frac{1}{C}$），$H(D) = \log_2 C$（最不纯）

**第三步：条件熵**

如果用特征 $A$ 将数据集划分为 $\{D_1, D_2, \ldots, D_k\}$，条件熵为：

$$
H(D|A) = \sum_{i=1}^{k} \frac{|D_i|}{|D|} H(D_i)
$$

这是划分后各子集熵的加权平均。

**第四步：信息增益**

信息增益定义为熵的减少：

$$
\text{Gain}(D, A) = H(D) - H(D|A)
$$

信息增益越大，划分越有效。这就是 ID3 算法的标准。

**第五步：ID3 算法步骤**

```
ID3(D, 特征集):
    如果 D 中所有样本属于同一类 c:
        返回叶子节点，标签为 c
    如果 特征集 为空:
        返回叶子节点，标签为 D 的多数类
    选择信息增益最大的特征 A
    根据特征的每个可能值 a，创建子节点
    对每个子节点递归调用 ID3
```

**第六步：信息增益率的问题**

ID3 倾向于选择取值较多的特征（因为这样能产生更多的划分，条件熵更小）。为了解决这个问题，C4.5 使用**信息增益率**。

首先计算特征 $A$ 的**固有熵**（intrinsic entropy）：

$$
H_A(D) = -\sum_{i=1}^{k} \frac{|D_i|}{|D|} \log_2 \frac{|D_i|}{|D|}
$$

然后计算信息增益率：

$$
\text{GainRatio}(D, A) = \frac{\text{Gain}(D, A)}{H_A(D)}
$$

这样，取值多的特征虽然增益大，但固有熵也大，增益率不会特别高。

**第七步：CART 的基尼系数**

CART（Classification and Regression Trees）算法使用**基尼系数**（Gini index）：

$$
\text{Gini}(D) = 1 - \sum_{c=1}^{C} p_c^2 = \sum_{c=1}^{C} p_c (1 - p_c)
$$

基尼系数的含义是：随机抽取两个样本，它们属于不同类的概率。

基尼系数越小，数据越纯。CART 选择使基尼系数减少最多的划分。

**第八步：剪枝**

决策树容易过拟合，需要剪枝。

**预剪枝**（pre-pruning）：
- 限制树的深度
- 限制叶子节点的最小样本数
- 限制划分所需的最小信息增益

**后剪枝**（post-pruning）：
- 先生成完全生长的树
- 自底向上评估剪枝后的验证集误差
- 如果剪枝后误差不增加，则剪掉

### 决策边界

决策树的决策边界是**轴对齐**的（axis-aligned），即与坐标轴平行。这意味着决策边界是分段常数函数。这既是优点（可解释性），也是缺点（难以拟合对角线边界）。

---

## 六、支持向量机：最大间隔的艺术

**时间：1963 年 - 弗拉基米尔·万普尼克 (Vladimir Vapnik)；1992 年 - 核技巧**

### 冷战时期的智慧

1963 年，苏联数学家弗拉基米尔·万普尼克提出了一个革命性的想法：不要只关注分类错误，要关注分类边界到最近点的距离。

### 推导过程

**第一步：线性可分的情况**

假设数据集 $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ 线性可分，其中 $y_i \in \{-1, +1\}$。我们要找一个超平面 $\mathbf{w}^T \mathbf{x} + b = 0$ 将两类分开。

超平面的间隔定义为：

$$
\gamma = \min_i \frac{|y_i (\mathbf{w}^T \mathbf{x}_i + b)|}{\|\mathbf{w}\|}
$$

我们要最大化这个间隔。

**第二步：间隔的规范化**

注意到如果 $(\mathbf{w}, b)$ 是解，那么 $(k\mathbf{w}, kb)$ 对任意 $k > 0$ 也是解，因为：

$$
k\mathbf{w}^T \mathbf{x} + kb = 0 \iff \mathbf{w}^T \mathbf{x} + b = 0
$$

且间隔为：

$$
\frac{|y_i (k\mathbf{w}^T \mathbf{x}_i + kb)|}{\|k\mathbf{w}\|} = \frac{k|y_i (\mathbf{w}^T \mathbf{x}_i + b)|}{k\|\mathbf{w}\|} = \frac{|y_i (\mathbf{w}^T \mathbf{x}_i + b)|}{\|\mathbf{w}\|}
$$

因此，我们可以选择一个特定的尺度。选择让间隔边界的点满足：

$$
y_i (\mathbf{w}^T \mathbf{x}_i + b) = 1
$$

这些点就是**支持向量**（support vectors）。于是：

$$
\gamma = \min_i \frac{|y_i (\mathbf{w}^T \mathbf{x}_i + b)|}{\|\mathbf{w}\|} = \frac{1}{\|\mathbf{w}\|}
$$

最大化间隔等价于最小化 $\|\mathbf{w}\|$。

**第三步：原始问题**

因此，SVM 的优化问题是：

$$
\begin{align}
\min_{\mathbf{w}, b} \quad & \frac{1}{2} \|\mathbf{w}\|^2 \\
\text{s.t.} \quad & y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n
\end{align}
$$

目标函数加 $\frac{1}{2}$ 是为了求导方便（$\|\mathbf{w}\|^2$ 的导数是 $2\mathbf{w}$，乘 $\frac{1}{2}$ 后导数是 $\mathbf{w}$）。

**第四步：拉格朗日对偶**

引入拉格朗日乘子 $\alpha_i \geq 0$：

$$
\mathcal{L}(\mathbf{w}, b, \alpha) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^{n} \alpha_i [y_i (\mathbf{w}^T \mathbf{x}_i + b) - 1]
$$

对偶问题的第一步是对原始变量求极值：

$$
\min_{\mathbf{w}, b} \max_{\alpha \geq 0} \mathcal{L}(\mathbf{w}, b, \alpha)
$$

先对 $\mathbf{w}$ 求导：

$$
\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{w} - \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i = \mathbf{0} \Rightarrow \mathbf{w} = \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i
$$

对 $b$ 求导：

$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^{n} \alpha_i y_i = 0 \Rightarrow \sum_{i=1}^{n} \alpha_i y_i = 0
$$

将 $w$ 和约束代入：

$$
\mathcal{L}(\alpha) = \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j x_i^T x_j - \sum_{i=1}^{n} \alpha_i
$$

对偶问题是：

$$
\begin{align}
\max_{\alpha} \quad & \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \\
\text{s.t.} \quad & \alpha_i \geq 0, \quad i = 1, \ldots, n \\
& \sum_{i=1}^{n} \alpha_i y_i = 0
\end{align}
$$

**第五步：预测**

训练完成后，预测为：

$$
f(\mathbf{x}) = \text{sign}\left(\mathbf{w}^T \mathbf{x} + b\right) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i^T \mathbf{x} + b\right)
$$

注意只有支持向量（$\alpha_i > 0$）起作用。

**第六步：软间隔**

实际数据可能不是线性可分的。引入松弛变量 $\xi_i \geq 0$：

$$
\begin{align}
\min_{\mathbf{w}, b, \xi} \quad & \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i \\
\text{s.t.} \quad & y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad i = 1, \ldots, n \\
& \xi_i \geq 0, \quad i = 1, \ldots, n
\end{align}
$$

其中 $C$ 是惩罚参数，控制对错分类的容忍度。

对偶问题形式相同，只是约束变为 $0 \leq \alpha_i \leq C$。

**第七步：核技巧**

非线性可分怎么办？将数据映射到高维空间 $\phi: \mathbb{R}^d \to \mathbb{R}^D$（$D \gg d$）。

对偶问题变为：

$$
\max_{\alpha} \quad \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)
$$

**核技巧的魔法**：我们不需要显式计算 $\phi(\mathbf{x})$，只需要知道内积。定义核函数：

$$
K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T \phi(\mathbf{x}')
$$

于是：

$$
\max_{\alpha} \quad \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
$$

预测为：

$$
f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)
$$

**常用的核函数**：

1. **线性核**：$K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$

2. **多项式核**：$K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T \mathbf{x}' + c)^d$

3. **高斯核（RBF）**：$K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$

4. **Sigmoid 核**：$K(\mathbf{x}, \mathbf{x}') = \tanh(\kappa \mathbf{x}^T \mathbf{x}' + c)$

**为什么高斯核有效？**

考虑高斯核：

$$
K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2) = \exp\left(-\gamma \sum_{j=1}^{d} (x_j - x'_j)^2\right)
$$

使用泰勒展开：

$$
\exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2) = \sum_{k=0}^{\infty} \frac{(-\gamma)^k}{k!} \|\mathbf{x} - \mathbf{x}'\|^{2k}
$$

展开 $\|\mathbf{x} - \mathbf{x}'\|^{2k}$ 后，得到无限维的特征映射。因此，高斯核对应无限维的特征空间！

---

## 七、K 均值聚类：迭代收敛的美学

**时间：1957 年 - 雨果·斯坦因豪斯 (Hugo Steinhaus)；1965 年 - 劳埃德算法 (Lloyd's Algorithm)**

### 聚类的启蒙

1957 年，波兰数学家雨果·斯坦因豪斯在研究"平面上的点的集合"问题时，提出了将点集划分为 $k$ 个簇的方法。

### 推导过程

**第一步：定义目标函数**

给定 $n$ 个数据点 $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$ 和 $k$ 个簇中心 $\{\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_k\}$，我们要最小化簇内平方误差：

$$
J = \sum_{i=1}^{n} \sum_{j=1}^{k} \mathbb{I}(z_i = j) \|\mathbf{x}_i - \mathbf{c}_j\|^2
$$

其中 $z_i \in \{1, 2, \ldots, k\}$ 是数据点 $\mathbf{x}_i$ 的簇标签。

**第二步：交替优化**

这是一个非凸优化问题，很难找到全局最优。但我们可以交替优化 $\mathbf{c}$ 和 $z$。

**E 步（期望步）**：固定簇中心 $\{\mathbf{c}_1, \ldots, \mathbf{c}_k\}$，更新分配 $z_i$。

对于每个数据点 $\mathbf{x}_i$，选择最近的簇中心：

$$
z_i = \arg\min_{j} \|\mathbf{x}_i - \mathbf{c}_j\|^2
$$

**M 步（最大化步）**：固定分配 $z$，更新簇中心 $\mathbf{c}_j$。

对于每个簇 $j$，最小化 $\sum_{i: z_i = j} \|\mathbf{x}_i - \mathbf{c}_j\|^2$。

求导：

$$
\frac{\partial}{\partial \mathbf{c}_j} \sum_{i: z_i = j} \|\mathbf{x}_i - \mathbf{c}_j\|^2 = \sum_{i: z_i = j} -2(\mathbf{x}_i - \mathbf{c}_j) = \mathbf{0}
$$

因此：

$$
\mathbf{c}_j = \frac{\sum_{i: z_i = j} \mathbf{x}_i}{|\{i: z_i = j\}|}
$$

这是簇 $j$ 中所有点的均值，因此称为"K 均值"。

**第三步：收敛性分析**

每次 E 步和 M 步后，目标函数 $J$ 都会减少（或保持不变）：

- E 步：每个点选择最近的簇中心，不会增加距离
- M 步：簇中心设为均值，使该簇内误差最小

由于簇的分配方式有限（最多 $k^n$ 种），算法必然在有限步内收敛。

**第四步：K 均值++ 初始化**

K 均值对初始化敏感。K-Means++ 使用概率初始化：

1. 随机选择第一个中心 $\mathbf{c}_1$
2. 对于未选的点 $\mathbf{x}$，计算 $D(\mathbf{x}) = \min_j \|\mathbf{x} - \mathbf{c}_j\|^2$
3. 以概率 $\frac{D(\mathbf{x})}{\sum_{\mathbf{x}'} D(\mathbf{x}')}$ 选择下一个中心
4. 重复直到选择 $k$ 个中心

这种初始化使得初始中心之间相互远离，提升了聚类质量。

**第五步：选择 K**

肘部法则（Elbow Method）：
1. 对不同的 $k$ 运行 K 均值
2. 绘制 $k$ vs. 目标函数 $J$ 的曲线
3. 选择肘部（曲线平缓的点）对应的 $k$

轮廓系数（Silhouette Coefficient）：

对于数据点 $\mathbf{x}_i$：
- $a_i = \frac{1}{|C_{z_i}| - 1} \sum_{j \neq i, z_j = z_i} \|\mathbf{x}_i - \mathbf{x}_j\|$（同簇平均距离）
- $b_i = \min_{c \neq z_i} \frac{1}{|C_c|} \sum_{j: z_j = c} \|\mathbf{x}_i - \mathbf{x}_j\|$（最近异簇平均距离）

轮廓系数：

$$
s_i = \frac{b_i - a_i}{\max(a_i, b_i)}
$$

平均轮廓系数越大，聚类效果越好。

---

## 八、随机森林：民主投票的胜利

**时间：2001 年 - 里奥·布雷曼 (Leo Breiman)**

### 从决策树到森林

2001 年，统计学家里奥·布雷曼发表了里程碑式的论文，提出了随机森林。

### 推导过程

**第一步：Bagging 的思想**

Bagging（Bootstrap Aggregating）的核心思想：对训练集进行多次 Bootstrap 采样，每次训练一个基学习器，最后聚合。

Bootstrap 采样：从原始训练集有放回地采样 $n$ 个样本（$n$ 是原始样本数）。每次约有 $63.2\%$ 的样本被采样到，称为**袋内样本**（in-bag samples）；未被采样的 $36.8\%$ 称为**袋外样本**（out-of-bag samples, OOB）。

对于回归问题，随机森林的预测是 $T$ 棵决策树预测的平均：

$$
\hat{f}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} f_t(\mathbf{x})
$$

对于分类问题，采用多数投票：

$$
\hat{y} = \arg\max_c \sum_{t=1}^{T} \mathbb{I}(f_t(\mathbf{x}) = c)
$$

**第二步：偏差-方差分解**

对于回归问题，定义均方误差：

$$
\text{MSE} = \mathbb{E}[(\hat{f}(\mathbf{x}) - y)^2]
$$

分解为偏差和方差：

$$
\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

其中：
- $\text{Bias}^2 = (\mathbb{E}[\hat{f}(\mathbf{x})] - f^{\ast}(\mathbf{x}))^2$（模型平均与真实值的差距）
- $\text{Variance} = \mathbb{E}[(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])^2]$（模型预测的不稳定性）
- $\text{Noise} = \mathbb{E}[(y - f^{\ast}(\mathbf{x}))^2]$（不可约误差）

Bagging 减少方差的推导：

设 $T$ 个基学习器 $f_1, f_2, \ldots, f_T$，满足：
- $\mathbb{E}[f_t] = \bar{f}$（无偏）
- $\text{Var}(f_t) = \sigma^2$（同方差）
- $\text{Cov}(f_i, f_j) = \rho \sigma^2$（同协方差）

Bagging 预测为 $\hat{f} = \frac{1}{T} \sum_{t=1}^{T} f_t$，其方差为：

$$
\begin{align}
\text{Var}(\hat{f}) &= \text{Var}\left(\frac{1}{T} \sum_{t=1}^{T} f_t\right) \\
&= \frac{1}{T^2} \text{Var}\left(\sum_{t=1}^{T} f_t\right) \\
&= \frac{1}{T^2} \left( \sum_{t=1}^{T} \text{Var}(f_t) + 2 \sum_{i < j} \text{Cov}(f_i, f_j) \right) \\
&= \frac{1}{T^2} \left( T \sigma^2 + 2 \cdot \frac{T(T-1)}{2} \rho \sigma^2 \right) \\
&= \frac{1}{T^2} \left( T \sigma^2 + T(T-1) \rho \sigma^2 \right) \\
&= \frac{\sigma^2}{T} + \frac{T-1}{T} \rho \sigma^2 \\
&= \rho \sigma^2 + \frac{1 - \rho}{T} \sigma^2
\end{align}
$$

当 $T \to \infty$，$\text{Var}(\hat{f}) \to \rho \sigma^2$。

如果基学习器完全相关（$\rho = 1$），$\text{Var}(\hat{f}) = \sigma^2$，Bagging 没有作用。
如果基学习器独立（$\rho = 0$），$\text{Var}(\hat{f}) = \frac{\sigma^2}{T} \to 0$，方差趋于零！

因此，Bagging 的关键是**降低基学习器之间的相关性**。

**第三步：随机森林的去相关机制**

随机森林引入两个随机化：

1. **Bootstrap 采样**：每棵树看到不同的训练数据
2. **随机特征选择**：在每个分裂点，随机选择 $m \leq d$ 个特征，从中选择最优分裂

对于分类问题，常用 $m = \lfloor \sqrt{d} \rfloor$；对于回归问题，常用 $m = \lfloor d/3 \rfloor$。

**第四步：OOB 误差估计**

对于每棵树 $t$，使用袋外样本预测。对于数据点 $\mathbf{x}_i$，只有未用于训练第 $t$ 棵树的树（即 $\mathbf{x}_i$ 是第 $t$ 棵树的 OOB 样本）才能预测 $\mathbf{x}_i$。

设 $\mathcal{T}_i = \{t : \mathbf{x}_i \text{ is OOB for tree } t\}$，则 OOB 预测为：

$$
\hat{y}_i^{\text{OOB}} = \begin{cases}
\arg\max_c \sum_{t \in \mathcal{T}_i} \mathbb{I}(f_t(\mathbf{x}_i) = c) & \text{classification} \\
\frac{1}{|\mathcal{T}_i|} \sum_{t \in \mathcal{T}_i} f_t(\mathbf{x}_i) & \text{regression}
\end{cases}
$$

OOB 误差是无偏的交叉验证估计，无需额外的验证集。

**第五步：特征重要性**

**置换重要性**（Permutation Importance）：
1. 计算原始 OOB 误差 $e_{\text{original}}$
2. 对特征 $j$，在 OOB 样本中随机置换该特征的值
3. 重新计算 OOB 误差 $e_{\text{permuted}}$
4. 特征重要性：$\text{Importance}_j = e_{\text{permuted}} - e_{\text{original}}$

置换后的误差增加越多，说明该特征越重要。

---

## 九、梯度提升机：贪婪优化之美

**时间：2001 年 - 杰罗姆·弗里德曼 (Jerome Friedman)**

### 函数空间中的梯度下降

2001 年，杰罗姆·弗里德曼提出了梯度提升机（Gradient Boosting Machine, GBM）。

### 推导过程

**第一步：理解前向分步算法**

我们想学习函数 $F: \mathcal{X} \to \mathbb{R}$，使得期望损失最小：

$$
F^{\ast} = \arg\min_{F} \mathbb{E}_{\mathbf{x}, y}[L(y, F(\mathbf{x}))]
$$

GBM 采用**前向分步算法**（Forward Stagewise Algorithm）。假设我们已经构建了 $m-1$ 轮模型 $F_{m-1}(\mathbf{x})$，第 $m$ 轮的目标是找到一个新模型 $h(\mathbf{x})$ 和步长 $\rho$：

$$
(F_m, \rho) = \arg\min_{h, \rho} \sum_{i=1}^{n} L(y_i, F_{m-1}(\mathbf{x}_i) + \rho h(\mathbf{x}_i))
$$

然后更新：

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \rho h(\mathbf{x})
$$

**第二步：函数优化的梯度下降**

在欧几里得空间中，目标函数 $f: \mathbb{R}^d \to \mathbb{R}$ 的梯度下降为：

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)
$$

其中 $\eta$ 是学习率。

在**函数空间**中，我们对函数 $F$ 进行优化。定义损失函数：

$$
\mathcal{L}(F) = \sum_{i=1}^{n} L(y_i, F(\mathbf{x}_i))
$$

$\mathcal{L}(F)$ 在函数空间中的"梯度"是：

$$
\nabla_F \mathcal{L}(F) = \left( \frac{\partial \mathcal{L}(F)}{\partial F(\mathbf{x}_1)}, \frac{\partial \mathcal{L}(F)}{\partial F(\mathbf{x}_2)}, \ldots, \frac{\partial \mathcal{L}(F)}{\partial F(\mathbf{x}_n)} \right)
$$

计算：

$$
\frac{\partial \mathcal{L}(F)}{\partial F(\mathbf{x}_i)} = \frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} = \left. \frac{\partial L(y, F)}{\partial F} \right|_{y=y_i, F=F(\mathbf{x}_i)}
$$

因此，"梯度下降"更新为：

$$
F_{m}(\mathbf{x}) = F_{m-1}(\mathbf{x}) - \eta \left. \frac{\partial L(y, F)}{\partial F} \right|_{y=y, F=F_{m-1}(\mathbf{x})}
$$

**第三步：拟合负梯度**

对于每个样本 $\mathbf{x}_i$，负梯度为：

$$
r_i = -\left. \frac{\partial L(y, F)}{\partial F} \right|_{y=y_i, F=F_{m-1}(\mathbf{x}_i)}
$$

我们要用一个弱学习器 $h(\mathbf{x})$ 拟合这些负梯度：

$$
h = \arg\min_{h} \sum_{i=1}^{n} (r_i - h(\mathbf{x}_i))^2
$$

然后更新：

$$
F_{m}(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta h(\mathbf{x})
$$

**第四步：平方损失的情况**

对于平方损失 $L(y, F) = \frac{1}{2}(y - F)^2$：

$$
\frac{\partial L}{\partial F} = -(y - F)
$$

负梯度为：

$$
r_i = -(y_i - F_{m-1}(\mathbf{x}_i)) = F_{m-1}(\mathbf{x}_i) - y_i
$$

这正是残差！因此，GBM 的每一步都是在拟合残差。

**第五步：逻辑损失的情况**

对于逻辑损失（二元分类）：

$$
L(y, F) = \log(1 + e^{-y F}), \quad y \in \{-1, +1\}
$$

计算梯度：

$$
\frac{\partial L}{\partial F} = \frac{-y e^{-y F}}{1 + e^{-y F}} = -y \cdot \sigma(-y F)
$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 Sigmoid 函数。

负梯度为：

$$
r_i = y_i \cdot \sigma(-y_i F_{m-1}(\mathbf{x}_i))
$$

拟合这个负梯度后，更新：

$$
F_{m}(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta h(\mathbf{x})
$$

预测概率为 $\sigma(F(\mathbf{x}))$。

**第六步：行搜索优化步长**

拟合 $h(\mathbf{x})$ 后，我们可以用行搜索优化步长 $\rho$：

$$
\rho = \arg\min_{\rho} \sum_{i=1}^{n} L(y_i, F_{m-1}(\mathbf{x}_i) + \rho h(\mathbf{x}_i))
$$

对于平方损失：

$$
\frac{\partial}{\partial \rho} \sum_{i=1}^{n} \frac{1}{2}(y_i - F_{m-1}(\mathbf{x}_i) - \rho h(\mathbf{x}_i))^2 = 0
$$

$$
\sum_{i=1}^{n} (y_i - F_{m-1}(\mathbf{x}_i) - \rho h(\mathbf{x}_i)) \cdot (-h(\mathbf{x}_i)) = 0
$$

$$
\sum_{i=1}^{n} (F_{m-1}(\mathbf{x}_i) - y_i) h(\mathbf{x}_i) = \rho \sum_{i=1}^{n} h(\mathbf{x}_i)^2
$$

$$
\rho = \frac{\sum_{i=1}^{n} (F_{m-1}(\mathbf{x}_i) - y_i) h(\mathbf{x}_i)}{\sum_{i=1}^{n} h(\mathbf{x}_i)^2}
$$

**第七步：正则化**

GBM 容易过拟合，常用的正则化方法：

1. **学习率**：使用较小的学习率 $\eta$（如 0.01 或 0.1），配合更多的迭代次数。

2. **树深限制**：限制决策树的深度（如 max_depth = 3）。

3. **叶子节点最小样本数**：限制叶子节点的最小样本数（如 min_samples_leaf = 10）。

4. **子采样**：每次迭代只使用部分样本（如 subsample = 0.8），类似随机森林。

**第八步：XGBoost 的二阶近似**

XGBoost 使用泰勒展开到二阶：

$$
L(y, F + \Delta F) \approx L(y, F) + \frac{\partial L}{\partial F} \Delta F + \frac{1}{2} \frac{\partial^2 L}{\partial F^2} (\Delta F)^2
$$

定义：
- 一阶导数：$g = \frac{\partial L}{\partial F}$
- 二阶导数：$h = \frac{\partial^2 L}{\partial F^2}$

目标函数近似为：

$$
\mathcal{L}(F + \Delta F) \approx \sum_{i=1}^{n} [L(y_i, F(\mathbf{x}_i)) + g_i \Delta F(\mathbf{x}_i) + \frac{1}{2} h_i (\Delta F(\mathbf{x}_i))^2] + \Omega(f)
$$

其中 $\Omega(f)$ 是正则项：

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

这里 $T$ 是叶子节点数，$w_j$ 是叶子节点的输出值，$\gamma$ 和 $\lambda$ 是正则化系数。

通过推导，叶子节点 $j$ 的最优值为：

$$
w_j^{\ast} = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

其中 $I_j$ 是叶子节点 $j$ 中的样本集合。

---

## 十、主成分分析：降维的艺术

**时间：1901 年 - 卡尔·皮尔逊 (Karl Pearson)**

### 最早的降维方法

1901 年，英国数学家卡尔·皮尔逊提出了主成分分析（PCA）。这是数学史上最早的降维方法之一。

### 推导过程

**第一步：数据标准化**

给定 $n$ 个 $d$ 维数据点 $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$，首先中心化：

$$
\tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}
$$

其中 $\bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i$ 是均值。

如果特征量纲不同，还需要标准化：

$$
z_{i,j} = \frac{x_{i,j} - \bar{x}_j}{\sigma_j}
$$

其中 $\sigma_j$ 是特征 $j$ 的标准差。

**第二步：最大化投影方差**

我们要找到一个单位向量 $\mathbf{v}$（$\|\mathbf{v}\| = 1$），使得数据在 $\mathbf{v}$ 方向上的投影方差最大。

投影为：

$$
y_i = \mathbf{v}^T \tilde{\mathbf{x}}_i
$$

投影方差为：

$$
\text{Var}(y) = \frac{1}{n} \sum_{i=1}^{n} y_i^2 = \frac{1}{n} \sum_{i=1}^{n} (\mathbf{v}^T \tilde{\mathbf{x}}_i)^2 = \frac{1}{n} \mathbf{v}^T \left( \sum_{i=1}^{n} \tilde{\mathbf{x}}_i \tilde{\mathbf{x}}_i^T \right) \mathbf{v}
$$

定义协方差矩阵：

$$
\mathbf{C} = \frac{1}{n} \sum_{i=1}^{n} \tilde{\mathbf{x}}_i \tilde{\mathbf{x}}_i^T = \frac{1}{n} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}
$$

其中 $\tilde{\mathbf{X}}$ 是中心化后的数据矩阵。

因此，优化问题为：

$$
\max_{\mathbf{v}} \mathbf{v}^T \mathbf{C} \mathbf{v} \quad \text{s.t.} \quad \mathbf{v}^T \mathbf{v} = 1
$$

**第三步：使用拉格朗日乘子法**

引入拉格朗日乘子 $\lambda$：

$$
\mathcal{L}(\mathbf{v}, \lambda) = \mathbf{v}^T \mathbf{C} \mathbf{v} - \lambda (\mathbf{v}^T \mathbf{v} - 1)
$$

对 $\mathbf{v}$ 求导：

$$
\nabla_{\mathbf{v}} \mathcal{L} = 2\mathbf{C} \mathbf{v} - 2\lambda \mathbf{v} = \mathbf{0}
$$

因此：

$$
\mathbf{C} \mathbf{v} = \lambda \mathbf{v}
$$

这正是**特征值问题**！$\mathbf{v}$ 是 $\mathbf{C}$ 的特征向量，$\lambda$ 是对应的特征值。

投影方差为：

$$
\mathbf{v}^T \mathbf{C} \mathbf{v} = \mathbf{v}^T (\lambda \mathbf{v}) = \lambda \mathbf{v}^T \mathbf{v} = \lambda
$$

因此，**投影方差等于对应的特征值**。最大化方差等价于选择最大特征值对应的特征向量。

**第四步：多个主成分**

第一个主成分 $\mathbf{v}_1$ 是 $\mathbf{C}$ 的最大特征值对应的特征向量。

第二个主成分 $\mathbf{v}_2$ 在与 $\mathbf{v}_1$ 正交的单位向量中最大化投影方差：

$$
\max_{\mathbf{v}} \mathbf{v}^T \mathbf{C} \mathbf{v} \quad \text{s.t.} \quad \mathbf{v}^T \mathbf{v} = 1, \mathbf{v}^T \mathbf{v}_1 = 0
$$

使用拉格朗日乘子法：

$$
\mathcal{L}(\mathbf{v}, \lambda, \mu) = \mathbf{v}^T \mathbf{C} \mathbf{v} - \lambda (\mathbf{v}^T \mathbf{v} - 1) - \mu \mathbf{v}^T \mathbf{v}_1
$$

对 $\mathbf{v}$ 求导：

$$
2\mathbf{C} \mathbf{v} - 2\lambda \mathbf{v} - \mu \mathbf{v}_1 = \mathbf{0}
$$

左乘 $\mathbf{v}_1^T$：

$$
2\mathbf{v}_1^T \mathbf{C} \mathbf{v} - 2\lambda \mathbf{v}_1^T \mathbf{v} - \mu \mathbf{v}_1^T \mathbf{v}_1 = 0
$$

由于 $\mathbf{v}_1^T \mathbf{v} = 0$ 且 $\mathbf{v}_1^T \mathbf{v}_1 = 1$：

$$
\mathbf{v}_1^T \mathbf{C} \mathbf{v} = \frac{\mu}{2}
$$

又因为 $\mathbf{C} \mathbf{v}_1 = \lambda_1 \mathbf{v}_1$：

$$
\mathbf{v}_1^T \mathbf{C} \mathbf{v} = (\mathbf{C} \mathbf{v}_1)^T \mathbf{v} = \lambda_1 \mathbf{v}_1^T \mathbf{v} = 0
$$

因此 $\mu = 0$，回到特征值问题 $\mathbf{C} \mathbf{v} = \lambda \mathbf{v}$。

$\mathbf{v}_2$ 是 $\mathbf{C}$ 的第二大特征值对应的特征向量。

一般地，第 $k$ 个主成分是 $\mathbf{C}$ 的第 $k$ 大特征值对应的特征向量。

**第五步：奇异值分解（SVD）**

协方差矩阵 $\mathbf{C} = \frac{1}{n} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$ 的特征值分解等价于 $\tilde{\mathbf{X}}$ 的奇异值分解。

设 $\tilde{\mathbf{X}}$ 的 SVD 为：

$$
\tilde{\mathbf{X}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

其中：
- $\mathbf{U} \in \mathbb{R}^{n \times d}$ 是左奇异矩阵（$\mathbf{U}^T \mathbf{U} = \mathbf{I}$）
- $\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$ 是对角矩阵，对角元素是奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_d \geq 0$
- $\mathbf{V} \in \mathbb{R}^{d \times d}$ 是右奇异矩阵（$\mathbf{V}^T \mathbf{V} = \mathbf{I}$）

协方差矩阵为：

$$
\mathbf{C} = \frac{1}{n} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}} = \frac{1}{n} \mathbf{V} \mathbf{\Sigma}^T \mathbf{U}^T \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \frac{1}{n} \mathbf{V} \mathbf{\Sigma}^2 \mathbf{V}^T
$$

因此，$\mathbf{C}$ 的特征向量是 $\mathbf{V}$ 的列，特征值是 $\frac{\sigma_i^2}{n}$。

**第六步：降维与重构**

保留前 $k$ 个主成分：

$$
\tilde{\mathbf{X}}' = \tilde{\mathbf{X}} \mathbf{V}_{[:, 1:k]}
$$

重构为：

$$
\tilde{\mathbf{X}}'' = \tilde{\mathbf{X}}' \mathbf{V}_{[:, 1:k]}^T = \tilde{\mathbf{X}} \mathbf{V}_{[:, 1:k]} \mathbf{V}_{[:, 1:k]}^T
$$

重构误差为：

$$
\|\tilde{\mathbf{X}} - \tilde{\mathbf{X}}''\|_F^2 = \sum_{j=k+1}^{d} \sigma_j^2
$$

其中 $\|\cdot\|_F$ 是 Frobenius 范数。

**第七步：选择主成分数量**

保留前 $k$ 个主成分的解释方差比为：

$$
\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i} = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{d} \sigma_i^2}
$$

通常选择 $k$ 使得解释方差比达到 80%-95%。

肘部法则：绘制 $k$ vs. 累积解释方差比，选择曲线趋于平缓的点。

---

## 结语：黄金时代的遗产

2006 年，深度学习在 Hinton 等人的推动下开始兴起。2012 年，AlexNet 在 ImageNet 上的横空出世标志着新纪元的开始。

但请不要忘记，那些传统的机器学习算法并没有消失。它们在各自的领域依然发挥着重要作用：
- 线性回归和逻辑回归仍然是解释性模型的首选
- 朴素贝叶斯在文本分类中不可替代
- 随机森林和梯度提升机在结构化数据上屡战屡胜
- PCA 是数据可视化和特征工程的基础工具

更重要的是，这些算法背后蕴含的数学思想——最小二乘、最大似然、贝叶斯推断、拉格朗日对偶、梯度优化——构成了现代机器学习的基础。即便是今天的深度神经网络，也离不开这些古老的智慧。

黄金时代或许已经过去，但它留给我们的遗产将永存。让我们怀着敬意，回顾那些用公式编织智能梦想的年代。

---

**参考文献**：

1. Legendre, A. M. (1805). *Nouvelles méthodes pour la détermination des orbites des comètes*.
2. Cox, D. R. (1958). "The regression analysis of binary sequences". *Journal of the Royal Statistical Society*.
3. Fix, E., & Hodges, J. L. (1951). *Discriminatory analysis, nonparametric discrimination: Consistency properties*.
4. Quinlan, J. R. (1986). "Induction of decision trees". *Machine Learning*.
5. Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*.
6. Breiman, L. (2001). "Random Forests". *Machine Learning*.
7. Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine". *Annals of Statistics*.
8. Pearson, K. (1901). "On lines and planes of closest fit to systems of points in space". *Philosophical Magazine*.
