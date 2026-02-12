---
title: "传统机器学习与统计学习算法：从理论到实践的完整指南"
date: 2026-01-14T08:18:25+08:00
draft: false
description: "本文全面回顾传统机器学习和统计学习算法的发展历程、数学原理、应用场景及未来前景，涵盖从线性回归到深度学习之前的关键算法。"
categories: ["机器学习", "数学"]
tags: ["机器学习", "算法", "数学史", "综述"]
cover:
    image: "images/covers/photo-1509228468518-180dd4864904.jpg"
    alt: "抽象几何图形"
    caption: "数学之美"
---

## 引言：从统计学到机器学习

1956年，达特茅斯会议上正式提出了"人工智能"这个词。但在那之前的一百年里，统计学家们已经在用数学工具从数据中提取规律。高斯在1809年就用最小二乘法解决了天文学中的观测数据拟合问题，这可以看作是最早的机器学习算法。

机器学习和统计学习，本质上是一回事：从数据中学习规律，并用这些规律做出预测。只是出发点略有不同——统计学家关注估计的可靠性和显著性检验，而计算机科学家更关心算法的计算效率和泛化能力。

当我们说"传统机器学习"时，指的是深度学习时代之前的那些经典算法。这些算法虽然不像神经网络那样"万能"，但在数据量有限、需要可解释性的场景下，依然发挥着不可替代的作用。

## 第一章：统计学习的理论基础

### 1.1 学习问题的数学框架

假设我们有一个数据集 $D = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$，其中 $x_i \in \mathcal{X}$ 是输入（特征），$y_i \in \mathcal{Y}$ 是输出（标签）。我们的目标是找到一个函数 $f: \mathcal{X} \to \mathcal{Y}$，使得对于新的输入 $x$，$f(x)$ 能准确预测对应的 $y$。

但在统计学习的框架下，我们还需要引入概率论的概念。假设数据是按照某个未知的联合分布 $P(X,Y)$ 生成的，我们的目标是学习一个决策函数 $f$，使得期望风险最小化：

$$R(f) = \mathbb{E}_{(X,Y) \sim P}[L(Y, f(X))]$$

其中 $L$ 是损失函数。对于回归问题，常用平方损失；对于分类问题，常用0-1损失或交叉熵损失。

问题在于：我们不知道 $P(X,Y)$，无法直接计算 $R(f)$。我们只能用经验风险（Empirical Risk）来近似：

$$\hat{R}(f) = \frac{1}{n}\sum_{i=1}^n L(y_i, f(x_i))$$

这就是经验风险最小化（ERM）的基本思想。但直接最小化经验风险会导致过拟合（overfitting）。

### 1.2 偏差-方差权衡

这是统计学习中最重要的概念之一。模型的预测误差可以分解为三个部分：

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

其中：
- $\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f^{\ast}(x)$：模型预测的期望与真实值的差距
- $\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$：模型预测的方差
- $\sigma^2$：不可约误差（数据本身的噪声）

**偏差**反映了模型的"假设强度"。如果模型过于简单（比如用线性模型拟合高度非线性的数据），会产生高偏差，导致欠拟合。

**方差**反映了模型对数据波动的敏感程度。如果模型过于复杂（比如高阶多项式拟合），会记住训练数据的噪声，产生高方差，导致过拟合。

偏差-方差权衡的核心思想是：我们需要在模型复杂度之间找到一个平衡点。

### 1.3 正则化：控制模型复杂度的数学工具

为了防止过拟合，我们在目标函数中加入正则化项。最常见的形式是：

$$\min_f \frac{1}{n}\sum_{i=1}^n L(y_i, f(x_i)) + \lambda \Omega(f)$$

其中 $\Omega(f)$ 是正则化项，$\lambda \geq 0$ 是超参数。

**L2正则化**（岭回归）：
$$\Omega(f) = \|w\|_2^2 = \sum_{j=1}^d w_j^2$$

L2正则化倾向于让权重变小但不为零，相当于对权重施加了高斯先验。

**L1正则化**（Lasso）：
$$\Omega(f) = \|w\|_1 = \sum_{j=1}^d |w_j|$$

L1正则化倾向于产生稀疏解（很多权重为零），相当于对权重施加了拉普拉斯先验。

### 1.4 泛化误差与PAC学习框架

一个关键问题是：经验风险最小化是否能保证泛化能力？PAC（Probably Approximately Correct）学习框架给出了理论保证。

设 $\mathcal{F}$ 是一个假设类，如果对于任意 $\epsilon, \delta > 0$，存在样本量 $n(\epsilon, \delta)$，使得当 $n \geq n(\epsilon, \delta)$ 时，经验风险最小化算法以至少 $1-\delta$ 的概率找到一个假设 $f$，满足 $R(f) - R(f^{\ast}) \leq \epsilon$，则称 $\mathcal{F}$ 是PAC可学习的。

根据VC维理论，经验风险与期望风险的差距有如下界限：

$$R(f) \leq \hat{R}(f) + \mathcal{O}\left(\sqrt{\frac{d \log(n/d) + \log(1/\delta)}{n}}\right)$$

其中 $d$ 是VC维。这告诉我们：模型复杂度越高，需要的样本量就越大。

## 第二章：经典监督学习算法

### 2.1 线性回归：统计学习的起点

#### 2.1.1 基本模型

线性回归是最简单的回归模型，假设输出 $y$ 是输入 $x$ 的线性函数：

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon$$

其中 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 是噪声项。用矩阵表示：

$$Y = X\beta + \epsilon$$

其中 $X$ 是 $n \times (p+1)$ 的设计矩阵，$\beta = (\beta_0, \beta_1, \ldots, \beta_p)^T$。

#### 2.1.2 最小二乘估计

最小二乘法的目标是最小化残差平方和：

$$\min_\beta \|Y - X\beta\|_2^2$$

这是一个凸优化问题。对 $\beta$ 求导并令导数为零：

$$\frac{\partial}{\partial \beta} \|Y - X\beta\|_2^2 = -2X^T(Y - X\beta) = 0$$

解这个方程，得到正规方程（Normal Equation）：

$$X^TX\beta = X^TY$$

如果 $X^TX$ 可逆，则唯一解为：

$$\hat{\beta} = (X^TX)^{-1}X^TY$$

这就是**普通最小二乘估计（OLS）**。

**几何解释**：$\hat{Y} = X\hat{\beta}$ 是 $Y$ 在 $X$ 的列空间上的正交投影。残差 $e = Y - \hat{Y}$ 与 $X$ 的每一列都正交，即 $X^Te = 0$。

#### 2.1.3 统计性质

如果误差项满足高斯-马尔可夫假设（$\mathbb{E}[\epsilon] = 0$，$\text{Cov}(\epsilon) = \sigma^2 I_n$），那么OLS估计量具有以下性质：

**无偏性**：
$$\mathbb{E}[\hat{\beta}] = \beta$$

证明：
$$\hat{\beta} = (X^TX)^{-1}X^TY = (X^TX)^{-1}X^T(X\beta + \epsilon) = \beta + (X^TX)^{-1}X^T\epsilon$$
$$\mathbb{E}[\hat{\beta}] = \beta + (X^TX)^{-1}X^T\mathbb{E}[\epsilon] = \beta$$

**有效性（BLUE）**：在所有线性无偏估计中，OLS的方差最小。

**协方差矩阵**：
$$\text{Cov}(\hat{\beta}) = \sigma^2 (X^TX)^{-1}$$

#### 2.1.4 岭回归与Lasso

当 $X^TX$ 接近奇异矩阵时（多重共线性），OLS估计会变得不稳定。正则化是解决方案。

**岭回归（Ridge Regression）**：
$$\min_\beta \|Y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$$

解为：
$$\hat{\beta}_{\text{ridge}} = (X^TX + \lambda I)^{-1}X^TY$$

添加 $\lambda I$ 确保矩阵可逆。

**Lasso**：
$$\min_\beta \frac{1}{2n}\|Y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

Lasso的优化问题是非光滑的（由于L1范数的绝对值），没有解析解，需要用坐标下降法（Coordinate Descent）求解。

**重要差异**：Lasso可以进行变量选择（稀疏性），而岭回归不能。这是因为L1范数的几何形状是菱形，更容易与等值线在坐标轴上相交。

#### 2.1.5 应用场景

- **房价预测**：根据房屋面积、房间数、地段等特征预测房价
- **金融分析**：根据公司财务指标预测股票收益率
- **医疗研究**：根据患者生理指标预测疾病风险

**案例**：波士顿房价数据集
```
特征： crim（犯罪率）, zn（住宅用地比例）, indus（非零售商业用地比例）,
      chas（是否临河）, nox（氮氧化物浓度）, rm（平均房间数）, age（房龄）,
      dis（到就业中心距离）, rad（高速可达性）, tax（房产税）,
      ptratio（师生比）, black（黑人比例）, lstat（低收入人群比例）

目标： medv（房屋中位价，单位千美元）
```

岭回归可以帮助处理多重共线性问题（比如zn和indus高度相关）。

### 2.2 逻辑回归：分类问题的经典方法

#### 2.2.1 从线性回归到逻辑回归

为什么不能直接用线性回归做分类？如果 $y \in \{0, 1\}$，线性回归会预测任意实数，而我们需要概率输出。

逻辑回归引入了Sigmoid函数（又称Logistic函数）：
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

模型假设：
$$P(y=1|x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}$$

$$P(y=0|x) = 1 - P(y=1|x) = \frac{e^{-(w^Tx + b)}}{1 + e^{-(w^Tx + b)}}$$

Sigmoid函数的性质：
- $\sigma(0) = 0.5$
- $\sigma(+\infty) \to 1$，$\sigma(-\infty) \to 0$
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

#### 2.2.2 似然函数与极大似然估计

给定数据集 $\{(x_i, y_i)\}_{i=1}^n$，其中 $y_i \in \{0, 1\}$，似然函数为：

$$L(w, b) = \prod_{i=1}^n P(y_i|x_i) = \prod_{i=1}^n [\sigma(w^Tx_i + b)]^{y_i} [1 - \sigma(w^Tx_i + b)]^{1 - y_i}$$

取对数，得到对数似然：

$$\ell(w, b) = \sum_{i=1}^n [y_i \log \sigma(z_i) + (1 - y_i) \log(1 - \sigma(z_i))]$$

其中 $z_i = w^Tx_i + b$。

极大似然估计：
$$\max_{w, b} \ell(w, b)$$

等价于最小化负对数似然（也是交叉熵损失）：
$$\min_{w, b} J(w, b) = -\sum_{i=1}^n [y_i \log \sigma(z_i) + (1 - y_i) \log(1 - \sigma(z_i))]$$

#### 2.2.3 梯度下降法

对 $J$ 求梯度：
$$\frac{\partial J}{\partial w_j} = -\sum_{i=1}^n [y_i - \sigma(z_i)] x_{i,j}$$

$$\frac{\partial J}{\partial b} = -\sum_{i=1}^n [y_i - \sigma(z_i)]$$

梯度下降更新规则：
$$w_j := w_j - \eta \frac{\partial J}{\partial w_j} = w_j + \eta \sum_{i=1}^n [y_i - \sigma(z_i)] x_{i,j}$$

$$b := b - \eta \frac{\partial J}{\partial b} = b + \eta \sum_{i=1}^n [y_i - \sigma(z_i)]$$

其中 $\eta$ 是学习率。

**随机梯度下降（SGD）**：每次只使用一个样本更新参数，计算更快但方差更大。

**小批量梯度下降（Mini-batch GD）**：每次使用一批样本，介于全量和单个样本之间。

#### 2.2.4 正则化逻辑回归

为防止过拟合，加入L2正则化：

$$J(w, b) = -\sum_{i=1}^n [y_i \log \sigma(z_i) + (1 - y_i) \log(1 - \sigma(z_i))] + \frac{\lambda}{2} \|w\|_2^2$$

梯度变为：
$$\frac{\partial J}{\partial w_j} = -\sum_{i=1}^n [y_i - \sigma(z_i)] x_{i,j} + \lambda w_j$$

这相当于在权重上施加了一个"拉回"的力，防止权重过大。

#### 2.2.5 应用场景

- **垃圾邮件检测**：根据邮件内容、发件人等特征判断是否为垃圾邮件
- **信用评分**：根据用户收入、信用历史等预测违约概率
- **医疗诊断**：根据症状、检验结果预测疾病概率
- **广告点击率（CTR）预测**：预测用户是否点击广告

### 2.3 支持向量机：最大间隔分类器

#### 2.3.1 几何直觉：寻找最大间隔

支持向量机（SVM）的核心思想是：找到一个超平面，不仅能正确分类，而且离两类数据点的距离都尽可能大。

在二维空间中，超平面就是一条直线：$w_1 x_1 + w_2 x_2 + b = 0$

点 $(x_1, x_2)$ 到超平面的距离：
$$d = \frac{|w_1 x_1 + w_2 x_2 + b|}{\sqrt{w_1^2 + w_2^2}}$$

SVM的目标是：最大化最小距离。

#### 2.3.2 硬间隔SVM

对于线性可分的数据，假设分类标签 $y_i \in \{-1, +1\}$，约束条件为：

$$y_i (w^T x_i + b) \geq 1, \quad i = 1, \ldots, n$$

这个约束保证了所有分类正确的点距离超平面至少为 $1/\|w\|$。

优化问题：
$$\min_{w, b} \frac{1}{2} \|w\|_2^2$$
$$\text{s.t. } y_i (w^T x_i + b) \geq 1, \quad i = 1, \ldots, n$$

**为什么最小化 $\|w\|_2^2$？**

因为间隔大小为 $2/\|w\|$，最大化间隔等价于最小化 $\|w\|$。

#### 2.3.3 对偶问题与拉格朗日乘子法

引入拉格朗日乘子 $\alpha_i \geq 0$，构造拉格朗日函数：

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i [y_i (w^T x_i + b) - 1]$$

原问题的对偶问题：
$$\max_{\alpha \geq 0} \min_{w, b} \mathcal{L}(w, b, \alpha)$$

先对 $w, b$ 求导并令导数为零：
$$\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i=1}^n \alpha_i y_i x_i = 0 \Rightarrow w = \sum_{i=1}^n \alpha_i y_i x_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^n \alpha_i y_i = 0 \Rightarrow \sum_{i=1}^n \alpha_i y_i = 0$$

代回拉格朗日函数，得到对偶问题：
$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j$$

约束：
$$\alpha_i \geq 0, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

这是一个凸二次规划问题，可以用SMO（Sequential Minimal Optimization）算法高效求解。

#### 2.3.4 支持向量与核技巧

**支持向量（Support Vectors）**：$\alpha_i > 0$ 的样本点。这些点位于间隔边界上。

KKT条件告诉我们：
$$\alpha_i [y_i (w^T x_i + b) - 1] = 0$$

如果 $\alpha_i > 0$，则 $y_i (w^T x_i + b) = 1$，即该点在边界上。

决策函数可以表示为：
$$f(x) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i x_i^T x + b\right)$$

**核技巧（Kernel Trick）**：将特征映射到高维空间 $\phi(x)$，使得线性不可分的问题变得可分。

只关注内积 $x_i^T x_j$，替换为核函数 $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$。

常用核函数：
- **线性核**：$K(x_i, x_j) = x_i^T x_j$
- **多项式核**：$K(x_i, x_j) = (x_i^T x_j + c)^d$
- **高斯核（RBF）**：$K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$
- **Sigmoid核**：$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)$

#### 2.3.5 软间隔SVM

对于线性不可分的数据，引入松弛变量 $\xi_i \geq 0$，允许部分点被误分类：

$$y_i (w^T x_i + b) \geq 1 - \xi_i, \quad i = 1, \ldots, n$$

优化问题：
$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \xi_i$$
$$\text{s.t. } y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

$C$ 是正则化参数，控制间隔与误分类之间的权衡。$C \to \infty$ 时，退化为硬间隔SVM。

#### 2.3.6 应用场景

- **文本分类**：文档分类、情感分析
- **图像识别**：手写数字识别（MNIST）
- **生物信息学**：蛋白质分类、基因表达数据分析
- **异常检测**：利用单分类SVM检测异常数据

### 2.4 决策树：基于规则的分类与回归

#### 2.4.1 基本思想

决策树通过一系列"如果-那么"规则进行预测。树由节点和边组成：
- **根节点**：整个数据集
- **内部节点**：对某个特征进行测试
- **叶子节点**：预测值（分类中的类别，回归中的数值）

#### 2.4.2 ID3算法：基于信息增益

ID3算法（Iterative Dichotomiser 3）使用信息增益选择分裂特征。

**熵（Entropy）**：衡量数据集的不纯度

对于分类问题，假设有 $K$ 个类别，第 $k$ 类的比例为 $p_k$：

$$H(D) = -\sum_{k=1}^K p_k \log_2 p_k$$

当所有样本属于同一类时，$H(D) = 0$（最纯）；当各类均匀分布时，$H(D)$ 最大（最不纯）。

**条件熵**：给定特征 $A$ 后的熵

$$H(D|A) = \sum_{v \in \text{Values}(A)} \frac{|D_v|}{|D|} H(D_v)$$

其中 $D_v$ 是特征 $A$ 取值为 $v$ 的子集。

**信息增益**：分裂前后的熵减少量

$$\text{Gain}(D, A) = H(D) - H(D|A)$$

ID3选择使信息增益最大的特征进行分裂。

#### 2.4.3 C4.5算法：基于信息增益率

ID3的缺点：倾向于选择取值多的特征（如ID）。C4.5用**信息增益率（Information Gain Ratio）**修正：

$$\text{GainRatio}(D, A) = \frac{\text{Gain}(D, A)}{\text{SplitInfo}(D, A)}$$

其中分裂信息：
$$\text{SplitInfo}(D, A) = -\sum_{v \in \text{Values}(A)} \frac{|D_v|}{|D|} \log_2 \frac{|D_v|}{|D|}$$

#### 2.4.4 CART算法：基尼指数与回归树

CART（Classification and Regression Trees）可以处理分类和回归问题。

**分类树**：使用基尼指数（Gini Index）

$$\text{Gini}(D) = 1 - \sum_{k=1}^K p_k^2$$

基尼指数越小，数据集越纯。选择使基尼指数下降最大的分裂。

**回归树**：预测值是叶子节点中样本的均值

假设叶子节点 $R_m$ 中有 $n_m$ 个样本，预测值为：
$$\hat{c}_m = \frac{1}{n_m}\sum_{x_i \in R_m} y_i$$

分裂准则：最小化平方误差
$$\sum_{x_i \in R_m} (y_i - \hat{c}_m)^2$$

#### 2.4.5 剪枝：防止过拟合

决策树容易过拟合，需要剪枝。

**预剪枝（Pre-pruning）**：
- 限制树的最大深度
- 限制每个节点的最小样本数
- 如果信息增益小于阈值，停止分裂

**后剪枝（Post-pruning）**：
- 从完全生长的树开始，自底向上剪枝
- 用验证集评估剪枝效果
- 代价复杂度剪枝（Cost-Complexity Pruning）

定义树 $T$ 的代价复杂度：
$$C_\alpha(T) = \frac{1}{N} \sum_{x_i \in \text{Training}} L(y_i, \hat{y}_i) + \alpha |T|$$

其中 $|T|$ 是叶子节点数，$\alpha$ 是正则化参数。选择使 $C_\alpha(T)$ 最小的子树。

#### 2.4.6 特征重要性

决策树可以提供特征重要性（Feature Importance）：

$$\text{Importance}_j = \sum_{t \in \text{Splits using } j} \frac{n_t}{N} \times \Delta \text{Impurity}(t)$$

其中 $\Delta \text{Impurity}(t)$ 是节点 $t$ 分裂前后的不纯度减少量。

#### 2.4.7 应用场景

- **医疗诊断**：根据症状和检查结果诊断疾病
- **金融风控**：评估贷款申请人的信用风险
- **推荐系统**：基于用户行为推荐商品
- **客户细分**：根据消费行为对客户分类

## 第三章：集成学习：集众智之长

### 3.1 偏差-方差分解与集成学习

集成学习通过组合多个模型来提升性能。基本原理：

- **Bagging（Bootstrap Aggregating）**：降低方差
  - 通过 bootstrap 采样创建多个训练集
  - 每个模型独立训练
  - 预测时取平均（回归）或投票（分类）
  - 代表：随机森林

- **Boosting**：降低偏差
  - 顺序训练模型，每个模型专注于前一个模型的错误
  - 加权组合模型
  - 代表：AdaBoost、Gradient Boosting、XGBoost、LightGBM

- **Stacking**：结合多个不同类型模型的预测
  - 基模型：不同算法（如逻辑回归、SVM、决策树）
  - 元模型：学习如何组合基模型的预测

### 3.2 随机森林

#### 3.2.1 算法原理

随机森林是Bagging与决策树的结合，通过引入随机性减少相关性。

**训练过程**：
1. Bootstrap采样：从训练集有放回地抽取 $n$ 个样本，创建 $B$ 个训练集
2. 对每个训练集训练一棵决策树
3. 每次分裂时，从所有特征中随机选择 $m$ 个特征（通常 $m = \sqrt{p}$）
4. 从这 $m$ 个特征中选择最优分裂特征

**预测**：
- 分类：多数投票
- 回归：平均

#### 3.2.2 为什么有效？

**Bagging的作用**：减少方差
- 单棵决策树方差大（易过拟合）
- 取平均后，方差降低为 $\sigma^2/B$（假设独立）

**特征随机性的作用**：减少相关性
- 如果使用全部特征，树之间高度相关
- 随机选择特征子集，增加多样性
- 不相关模型的平均更有效（根据方差公式：$\text{Var}(\bar{X}) = \frac{\sigma^2}{B} + \frac{B-1}{B}\rho\sigma^2$，其中 $\rho$ 是相关性）

#### 3.2.3 超参数

- `n_estimators`：树的数量（越多越好，但计算成本增加）
- `max_depth`：树的最大深度（控制过拟合）
- `min_samples_split`：节点分裂的最小样本数
- `min_samples_leaf`：叶子节点的最小样本数
- `max_features`：每次分裂考虑的特征数（`"sqrt"`、`"log2"`或整数）

#### 3.2.4 特征重要性（Out-of-Bag）

随机森林的OOB（Out-of-Bag）样本（bootstrap中未被选中的样本）可以用于：
- 估计泛化误差（无需验证集）
- 计算特征重要性

特征重要性的计算：打乱特征 $j$ 的值，观察OOB误差的增加量。

### 3.3 梯度提升树（GBDT）

#### 3.3.1 核心思想

梯度提升树通过拟合负梯度来逐步改进模型。

给定损失函数 $L(y, F(x))$，目标是最小化期望损失：

$$\min_F \mathbb{E}[L(y, F(x))]$$

用贪心算法：逐步添加弱学习器 $h_m(x)$：

$$F_m(x) = F_{m-1}(x) + h_m(x)$$

选择 $h_m$ 使损失下降最大：

$$h_m = \arg\min_h \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + h(x_i))$$

#### 3.3.2 负梯度拟合

对 $L$ 在 $F_{m-1}(x_i)$ 处做泰勒展开（一阶）：

$$L(y_i, F_{m-1}(x_i) + h(x_i)) \approx L(y_i, F_{m-1}(x_i)) + \frac{\partial L(y_i, F(x))}{\partial F(x)}\bigg|_{F = F_{m-1}} h(x_i)$$

负梯度：
$$r_{im} = -\frac{\partial L(y_i, F(x))}{\partial F(x)}\bigg|_{F = F_{m-1}}$$

因此，$h_m(x)$ 应该拟合负梯度 $r_{im}$。

**平方损失**：$L = \frac{1}{2}(y - F)^2$
$$\frac{\partial L}{\partial F} = -(y - F)$$
负梯度：$r_{im} = y_i - F_{m-1}(x_i)$（残差）

**逻辑损失**：$L = \log(1 + e^{-yF})$，其中 $y \in \{-1, 1\}$
$$\frac{\partial L}{\partial F} = \frac{-y e^{-yF}}{1 + e^{-yF}} = -\frac{y}{1 + e^{yF}}$$
负梯度：$r_{im} = \frac{y_i}{1 + e^{y_i F_{m-1}(x_i)}}$

#### 3.3.3 算法流程

**输入**：训练集 $\{(x_i, y_i)\}_{i=1}^n$，损失函数 $L$，学习率 $\eta$，树的数量 $M$

**步骤**：
1. 初始化：$F_0(x) = \arg\min_c \sum_{i=1}^n L(y_i, c)$（对回归，取均值；对分类，取对数几率）
2. 对于 $m = 1, 2, \ldots, M$：
   - 计算负梯度：$r_{im} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F(x)}$
   - 用决策树拟合 $(x_i, r_{im})$，得到区域 $R_{jm}$（$j = 1, \ldots, J_m$）
   - 计算叶子节点预测值：$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$
   - 更新模型：$F_m(x) = F_{m-1}(x) + \eta \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$

#### 3.3.4 正则化

GBDT通过多种方式防止过拟合：

1. **学习率（Learning Rate）**：$\eta \in (0, 1]$，控制每一步的步长
2. **树的数量（n_estimators）**：使用早停法（Early Stopping）选择最佳数量
3. **树的复杂度**：
   - `max_depth`：限制树深度
   - `min_samples_split`、`min_samples_leaf`：控制叶子节点
4. **子采样（Subsampling）**：每棵树只使用部分数据（类似Bagging）

#### 3.3.5 XGBoost与LightGBM

**XGBoost（eXtreme Gradient Boosting）**的改进：
- 二阶泰勒展开（同时使用一阶和二阶导数）
- 正则化项（叶子节点数和L2正则）
- 稀疏感知（处理缺失值）
- 并行化（特征级别）
- 列块设计（缓存优化）

**LightGBM**的改进：
- 基于直方图（Histogram）的算法（将连续值离散化）
- GOSS（Gradient-based One-Side Sampling）：只保留高梯度和随机低梯度样本
- EFB（Exclusive Feature Bundling）：合并稀疏特征
- Leaf-wise生长（优先分裂增益最大的叶子节点）

#### 3.3.6 应用场景

- **搜索排名**：LambdaMART（基于GBDT的学习排序算法）
- **欺诈检测**：信用卡欺诈、税务欺诈
- **点击率预测**：广告CTR预测
- **时间序列预测**：销量预测、流量预测

### 3.4 AdaBoost：自适应提升

#### 3.4.1 算法原理

AdaBoost（Adaptive Boosting）通过加权训练样本，逐步关注难以分类的样本。

**输入**：训练集 $\{(x_i, y_i)\}_{i=1}^n$，$y_i \in \{-1, 1\}$，迭代次数 $T$

**步骤**：
1. 初始化样本权重：$w_i^{(1)} = 1/n$
2. 对于 $t = 1, 2, \ldots, T$：
   - 用权重 $w_i^{(t)}$ 训练弱分类器 $h_t(x)$
   - 计算分类误差：$\epsilon_t = \sum_{i=1}^n w_i^{(t)} I(y_i \neq h_t(x_i))$
   - 计算分类器权重：$\alpha_t = \frac{1}{2}\log\frac{1 - \epsilon_t}{\epsilon_t}$
   - 更新样本权重：$w_i^{(t+1)} = w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i))$
   - 归一化权重：$w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^n w_j^{(t+1)}}$
3. 最终分类器：$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$

#### 3.4.2 为什么有效？

AdaBoost通过指数损失最小化：

$$L = \sum_{i=1}^n \exp(-y_i H(x_i))$$

其中 $H(x) = \sum_t \alpha_t h_t(x)$。

可以证明：每一步选择使误差最小的 $h_t$，等价于使指数损失下降最多。

**权重更新的含义**：
- 如果 $y_i = h_t(x_i)$（分类正确），权重减小
- 如果 $y_i \neq h_t(x_i)$（分类错误），权重增大
- 难以分类的样本权重越来越大，模型更关注这些样本

#### 3.4.3 理论保证

AdaBoost的泛化误差有如下界限：

$$P[H(x) \neq y] \leq P\left[\sum_{t=1}^T \alpha_t h_t(x) y \leq 0\right] \leq \prod_{t=1}^T \sqrt{1 - 4\gamma_t^2} \leq \exp\left(-2\sum_{t=1}^T \gamma_t^2\right)$$

其中 $\gamma_t = \frac{1}{2} - \epsilon_t$（边缘）。

如果每个弱分类器比随机猜测好（$\epsilon_t < 0.5$），则AdaBoost会收敛到零训练误差。

## 第四章：无监督学习：从数据中发现结构

### 4.1 主成分分析（PCA）

#### 4.1.1 降维问题

给定数据 $X \in \mathbb{R}^{n \times p}$，我们想找到一个低维表示 $Z \in \mathbb{R}^{n \times k}$（$k < p$），保留尽可能多的信息。

#### 4.1.2 几何视角：最大化投影方差

投影矩阵 $W \in \mathbb{R}^{p \times k}$，满足 $W^TW = I_k$（正交矩阵）。

投影后的数据：$Z = XW$

最大化投影方差：
$$\max_W \text{tr}(Z^TZ) = \max_W \text{tr}(W^TX^TXW)$$

约束：$W^TW = I_k$

#### 4.1.3 求解：特征值分解

数据协方差矩阵：$\Sigma = \frac{1}{n-1}X^TX$

优化问题等价于：
$$\max_W \text{tr}(W^T\Sigma W), \quad W^TW = I_k$$

根据瑞利商理论，最优解是 $\Sigma$ 的前 $k$ 个最大特征值对应的特征向量。

设 $\Sigma$ 的特征值分解：$\Sigma = U \Lambda U^T$，其中 $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_p)$，$\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_p \geq 0$

主成分：$W = [u_1, u_2, \ldots, u_k]$

**方差解释比例**：
$$\frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^p \lambda_i}$$

#### 4.1.4 代数视角：最小化重构误差

重构：$\hat{X} = ZW^T = XWW^T$

最小化重构误差：
$$\min_W \|X - XWW^T\|_F^2, \quad W^TW = I_k$$

可以证明这与最大化方差等价。

#### 4.1.5 奇异值分解（SVD）

对于大数据，直接计算 $\Sigma = \frac{1}{n-1}X^TX$ 计算量大（$p^3$）。

使用SVD：$X = U \Sigma V^T$

其中 $U \in \mathbb{R}^{n \times n}$，$\Sigma \in \mathbb{R}^{n \times p}$，$V \in \mathbb{R}^{p \times p}$

主成分：$W = V_{:, 1:k}$

投影：$Z = XW = U \Sigma V^T V_{:, 1:k} = U \Sigma_{:, 1:k}$

#### 4.1.6 应用场景

- **数据可视化**：将高维数据投影到2D或3D进行可视化
- **图像压缩**：Eigenfaces（人脸识别）
- **噪声减少**：保留主成分，去除噪声
- **特征提取**：为后续算法提供低维特征

### 4.2 聚类算法

#### 4.2.1 K-means算法

**目标**：将 $n$ 个样本分成 $K$ 个簇，使簇内距离最小。

**算法流程**：
1. 随机初始化 $K$ 个质心：$\mu_1, \mu_2, \ldots, \mu_K$
2. 重复直到收敛：
   - 分配：$c_i = \arg\min_j \|x_i - \mu_j\|^2$
   - 更新：$\mu_j = \frac{1}{|C_j|}\sum_{i \in C_j} x_i$

**目标函数（误差平方和，SSE）**：
$$J = \sum_{j=1}^K \sum_{i \in C_j} \|x_i - \mu_j\|^2$$

#### 4.2.2 收敛性

K-means单调下降目标函数 $J$：
- 分配步骤：每个点分配到最近的质心，$J$ 不增
- 更新步骤：质心移动到簇内均值，$J$ 不减

但可能收敛到局部最优（依赖初始化）。

#### 4.2.3 K-means++：改进初始化

初始化质心：
1. 第一个质心：随机选择
2. 对于 $k = 2, \ldots, K$：
   - 计算每个点到最近质心的距离：$d(x_i) = \min_j \|x_i - \mu_j\|$
   - 按概率 $\frac{d(x_i)^2}{\sum_{j=1}^n d(x_j)^2}$ 选择下一个质心

理论保证：$O(\log K)$ 近似最优。

#### 4.2.4 层次聚类

层次聚类创建聚类树（Dendrogram），无需指定簇数。

**凝聚（Agglomerative）**：自底向上，逐步合并最近的簇

距离度量：
- 单链接（Single Linkage）：$\min_{x \in C_i, y \in C_j} \|x - y\|$
- 完全链接（Complete Linkage）：$\max_{x \in C_i, y \in C_j} \|x - y\|$
- 平均链接（Average Linkage）：$\frac{1}{|C_i||C_j|}\sum_{x \in C_i} \sum_{y \in C_j} \|x - y\|$

**分裂（Divisive）**：自顶向下，逐步分裂簇

#### 4.2.5 DBSCAN：基于密度的聚类

K-means无法发现非凸簇和异常点。DBSCAN（Density-Based Spatial Clustering of Applications with Noise）基于密度聚类。

**定义**：
- $\epsilon$-邻域：$N_\epsilon(x) = \{y : \|y - x\| \leq \epsilon\}$
- 核心点：$|N_\epsilon(x)| \geq \text{minPts}$
- 边界点：邻域内点数少于minPts，但与某个核心点相邻
- 噪声点：既不是核心点也不是边界点

**算法流程**：
1. 标记所有点为未访问
2. 对于每个未访问点 $x$：
   - 标记为已访问
   - 如果 $|N_\epsilon(x)| < \text{minPts}$：标记为噪声
   - 否则：创建新簇，通过密度连接添加点

**优点**：
- 自动发现簇数
- 发现任意形状的簇
- 识别异常点

#### 4.2.6 应用场景

- **客户细分**：根据购买行为分群
- **文档聚类**：主题发现
- **图像分割**：像素聚类
- **异常检测**：发现离群点

## 第五章：传统机器学习的应用场景

### 5.1 金融领域

#### 5.1.1 信用评分

**问题**：根据借款人的历史数据，预测违约概率。

**常用算法**：
- 逻辑回归：可解释性强，易于满足监管要求
- 随机森林：处理非线性关系，特征重要性分析
- XGBoost：在Kaggle竞赛中表现优异

**特征**：
- 收入、负债收入比
- 信用历史（逾期次数、信用卡使用率）
- 贷款金额、期限
- 职业稳定性、教育程度

**评估指标**：
- AUC-ROC
- KS统计量
- 提升度（Lift）

#### 5.1.2 欺诈检测

**问题**：识别信用卡交易、保险理赔中的欺诈行为。

**挑战**：
- 类别极度不平衡（欺诈样本极少）
- 欺诈模式不断变化

**常用算法**：
- 异常检测：Isolation Forest、One-Class SVM
- 不平衡学习：SMOTE、代价敏感学习
- 集成方法：XGBoost（scale_pos_weight参数）

### 5.2 医疗健康

#### 5.2.1 疾病诊断

**问题**：根据症状、检验结果诊断疾病。

**示例**：
- 乳腺癌检测：决策树、逻辑回归（可解释性重要）
- 糖尿病预测：SVM、随机森林
- 心脏病风险评估：逻辑回归（计算风险评分）

**特征**：
- 患者年龄、性别、家族史
- 症状、检验指标（血压、血糖、胆固醇）
- 影像学特征（从医学图像提取）

#### 5.2.2 药物发现

**问题**：预测化合物的生物活性、毒性。

**挑战**：
- 数据量有限（实验成本高）
- 分子表示复杂

**常用算法**：
- 随机森林：处理分子描述符
- 深度学习：图神经网络（GNN）处理分子结构
- 迁移学习：从大规模数据预训练

### 5.3 推荐系统

#### 5.3.1 协同过滤

**问题**：根据用户历史行为预测偏好。

**用户-物品矩阵**：$R \in \mathbb{R}^{n \times m}$，$R_{ij}$ 表示用户 $i$ 对物品 $j$ 的评分

**矩阵分解**：
$$R \approx UV^T$$

其中 $U \in \mathbb{R}^{n \times k}$（用户隐因子），$V \in \mathbb{R}^{m \times k}$（物品隐因子）

**优化**：
$$\min_{U, V} \sum_{(i,j) \in \Omega} (R_{ij} - u_i^T v_j)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2)$$

$\Omega$ 是已知评分的索引集合。

#### 5.3.2 内容推荐

**问题**：基于物品内容相似性推荐。

**方法**：
- TF-IDF + 余弦相似度（文本）
- LSA（潜在语义分析）：降维后计算相似度
- 内容特征 + 协同过滤：混合模型

#### 5.3.3 排序学习

**问题**：对搜索结果排序。

**算法**：
- LambdaMART：基于GBDT的学习排序
- ListNet、ListMLE：基于列表的学习排序

### 5.4 自然语言处理（NLP）

#### 5.4.1 文本分类

**传统方法**：
- 特征提取：TF-IDF、N-grams、Word2Vec
- 分类器：朴素贝叶斯、SVM、逻辑回归

**示例**：
- 垃圾邮件分类：朴素贝叶斯
- 情感分析：SVM
- 新闻分类：逻辑回归

#### 5.4.2 主题模型

**LDA（Latent Dirichlet Allocation）**：

生成模型：每篇文档包含多个主题，每个主题包含多个词。

推断：Gibbs采样、变分推断

#### 5.4.3 命名实体识别（NER）

**传统方法**：
- HMM（隐马尔可夫模型）
- CRF（条件随机场）

特征：词性标注、上下文窗口、词形特征

### 5.5 计算机视觉

#### 5.5.1 图像分类（传统方法）

**特征提取**：
- SIFT（尺度不变特征变换）
- HOG（方向梯度直方图）
- LBP（局部二值模式）

**分类器**：
- SVM：ImageNet竞赛中表现优异（2012年之前）
- 随机森林：处理高维特征

#### 5.5.2 目标检测

**传统方法**：
- Viola-Jones框架（Haar特征 + AdaBoost）：人脸检测
- HOG + SVM：行人检测
- DPM（可变形部件模型）：多类别目标检测

## 第六章：未来展望

### 6.1 传统机器学习的现状

尽管深度学习在图像、语音等感知任务上取得了巨大成功，传统机器学习依然在以下场景中不可替代：

1. **数据量有限**：当样本量在几千到几万时，传统算法（特别是集成方法）往往表现更好
2. **可解释性要求高**：金融风控、医疗诊断等需要解释决策依据的场景
3. **计算资源受限**：传统算法计算量小，适合边缘计算、实时推理
4. **结构化数据**：表格数据是传统算法的主场，深度学习在这方面没有明显优势

### 6.2 未来发展趋势

#### 6.2.1 自动机器学习（AutoML）

**现状**：传统机器学习模型调参复杂，需要大量领域知识。

**未来**：
- 自动特征工程：自动选择、构造特征
- 超参数优化：贝叶斯优化、进化算法
- 模型选择：自动选择最优算法
- 神经架构搜索（NAS）：为深度学习设计架构

**工具**：
- Auto-sklearn
- H2O AutoML
- Google AutoML
- Microsoft AutoML

#### 6.2.2 可解释性AI（XAI）

**挑战**：传统算法（如随机森林、XGBoost）虽然可解释，但深度学习是黑箱。

**方法**：
- LIME（Local Interpretable Model-agnostic Explanations）：局部解释
- SHAP（SHapley Additive exPlanations）：基于博弈论的全局/局部解释
- 反事实解释：说明"如果特征X改变，结果会如何变化"
- 注意力机制：深度学习中的可解释性

**应用**：
- 医疗诊断：解释为什么预测某种疾病
- 金融风控：解释为什么拒绝贷款申请
- 公平性：检测和消除算法偏见

#### 6.2.3 因果推断

**传统机器学习**：关联性（Correlation）

**未来**：因果性（Causality）

**方法**：
- 结构因果模型（SCM）
- do-算子（Pearl的因果演算）
- 双重机器学习（DML）：结合因果推断与机器学习

**应用**：
- 营销：计算广告的因果效应（提升度）
- 政策评估：估计政策的因果影响
- 推荐系统：用户行为归因

#### 6.2.4 迁移学习与小样本学习

**挑战**：传统机器学习需要大量标注数据。

**未来**：
- 预训练模型：从大规模无标注数据预训练
- 微调：在目标任务上少量标注数据微调
- 元学习（Meta-Learning）：学习如何学习

**示例**：
- NLP：BERT、GPT（预训练+微调）
- 表格数据：预训练的表格数据模型（如TabNet、SAINT）
- 跨领域迁移：从源域知识迁移到目标域

#### 6.2.5 在线学习与强化学习

**传统机器学习**：离线训练，静态数据

**未来**：
- 在线学习：实时更新模型，适应数据分布变化
- 增量学习：增量式学习新任务，不遗忘旧知识
- 强化学习：通过与环境交互学习最优策略

**应用**：
- 实时推荐：根据用户实时行为更新推荐
- 自适应系统：自动驾驶、机器人控制
- 序列决策：游戏AI、资源调度

#### 6.2.6 传统算法与深度学习的融合

**融合方向**：

1. **深度嵌入传统算法**
   - Deep Forest：用深度森林替代神经网络
   - 树神经网络（TreeNN）：决策树的神经网络化

2. **传统算法作为组件**
   - GBDT的叶子编码作为特征，输入神经网络
   - 注意力机制结合树模型

3. **混合模型**
   - 结构化数据：传统算法（XGBoost）
   - 非结构化数据：深度学习（CNN、RNN、Transformer）
   - 多模态融合：结合两种模型的输出

#### 6.2.7 鲁棒性与安全性

**挑战**：传统算法和深度学习都易受对抗攻击。

**研究方向**：
- 对抗训练：提升模型鲁棒性
- 防御性蒸馏
- 神经网络验证

**公平性**：
- 消除算法偏见（种族、性别等）
- 公平性约束优化

### 6.3 传统机器学习的长期价值

尽管深度学习风头正劲，传统机器学习算法在以下方面依然具有重要价值：

1. **理论基础扎实**：统计学、凸优化等数学理论支撑，理论保证完善
2. **工程实践成熟**：Scikit-learn等工具库成熟，部署简单
3. **计算效率高**：适合实时应用、边缘计算
4. **可解释性好**：决策规则清晰，易于理解和调试
5. **适用范围广**：结构化数据分析的主场

### 6.4 结论

传统机器学习与统计学习算法经历了半个多世纪的发展，从高斯的线性回归到XGBoost的集成学习，形成了完整、成熟的理论体系和工程实践。

在深度学习时代，传统算法并未过时。相反，它们在特定场景下依然不可替代。未来的发展方向不是相互替代，而是相互融合：传统算法提供坚实的理论基础和工程实践，深度学习拓展了感知能力的边界，AutoML、可解释性、因果推断等技术将进一步释放机器学习的潜力。

正如统计学家George Box所说："All models are wrong, but some are useful."（所有模型都是错的，但有些是有用的）。传统机器学习算法的价值在于它们在"有用"这个维度上做到了极致。

## 参考文献

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
3. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
5. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189-1232.
6. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD*.
7. Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.
8. Schapire, R. E., & Freund, Y. (2012). *Boosting: Foundations and Algorithms*. MIT Press.
9. Pearl, J., & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.
10. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
