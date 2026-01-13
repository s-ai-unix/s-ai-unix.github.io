---
title: "从 Gauss-Bonnet 到 Gauss-Bonnet-Chern：微分几何中的经典定理"
date: 2025-01-12T20:30:00+08:00
draft: false
tags:
  - 微分几何
  - Gauss-Bonnet定理
  - Gauss-Bonnet-Chern定理
  - 陈省身
  - 数学
categories:
  - 数学
comments: true
mathjax: true
description: "深入探讨Gauss-Bonnet定理及其高维推广Gauss-Bonnet-Chern定理，介绍这两个经典公式及其证明思路，展示微分几何中局部与全局性质之间的深刻联系。"
cover:
  image: "/images/covers/1632981824734-43c34a45e816.jpg"
  alt: "微分几何曲面"
  caption: "Gauss-Bonnet定理：几何与拓扑的对话"
---

## 引言

在微分几何的宏伟殿堂中，Gauss-Bonnet 定理和它的推广形式 Gauss-Bonnet-Chern 定理堪称璀璨的明珠。它们建立了曲面（及更一般的紧致 Riemann 流形）的局部几何性质（曲率）与全局拓扑性质（Euler 示性数）之间的深刻联系。这种局部与全局之间的桥梁，正是现代几何学的核心思想之一。

本文将从经典的二维 Gauss-Bonnet 定理出发，逐步介绍其高维推广——Gauss-Bonnet-Chern 定理，并探讨这些定理的证明思路。

## 一、Gauss-Bonnet 定理

### 1.1 二维情形

**经典 Gauss-Bonnet 定理**是关于曲面的最基本也是最重要的定理之一。对于**紧致定向 Riemann 曲面** $M$，我们有：

$$
\int_M K \, dA = 2\pi \chi(M)
$$

其中：
- $K$ 是曲面的**Gauss 曲率**
- $dA$ 是面积元素
- $\chi(M)$ 是曲面的**Euler 示性数**

这个定理之所以重要，是因为它告诉我们：**曲面的总曲率是一个拓扑不变量**！无论你如何弯曲曲面（保持拓扑结构不变），曲率的积分永远等于 $2\pi$ 乘以 Euler 示性数。

#### 一些经典例子

**球面 $S^2$**：
- Euler 示性数 $\chi(S^2) = 2$
- Gauss 曲率 $K = \frac{1}{R^2}$（$R$ 为球面半径）
- 总面积 $A = 4\pi R^2$

$$
\int_{S^2} K \, dA = \frac{1}{R^2} \cdot 4\pi R^2 = 4\pi = 2\pi \chi(S^2) ✓
$$

**环面 $T^2$**：
- Euler 示性数 $\chi(T^2) = 0$
- 环面是平直的，Gauss 曲率 $K = 0$

$$
\int_{T^2} K \, dA = 0 = 2\pi \chi(T^2) ✓
$$

### 1.2 带边界的 Gauss-Bonnet 定理

对于**有边界的定向紧致曲面** $M$，Gauss-Bonnet 定理的形式为：

$$
\int_M K \, dA + \int_{\partial M} k_g \, ds = 2\pi \chi(M)
$$

其中：
- $\partial M$ 是 $M$ 的边界
- $k_g$ 是边界的**测地曲率**
- $ds$ 是边界的弧长元素

如果边界由分段光滑曲线组成，还需要加上顶点处的**外角**：

$$
\int_M K \, dA + \sum_i \int_{C_i} k_g \, ds + \sum_i \theta_i = 2\pi \chi(M)
$$

其中 $\theta_i$ 是第 $i$ 个顶点的外角。

#### 几何直观

这个定理有一个非常优美的几何解释：想象你在曲面上沿着边界行走，当你完成一圈时，你的**转向角度总和**（曲率积分 + 测地曲率积分 + 外角）恰好等于 $2\pi$ 乘以曲面的"洞数"（Euler 示性数）。

## 二、Gauss-Bonnet-Chern 定理

### 2.1 从二维到高维

一个自然的问题是：**Gauss-Bonnet 定理能否推广到高维流形？**

答案是肯定的！这就是著名的 **Gauss-Bonnet-Chern 定理**。这个定理由伟大的数学家**陈省身**（Shiing-Shen Chern）在 1944 年给出证明，是现代微分几何的奠基性工作之一。

对于 **$n$ 维紧致定向 Riemann 流形** $M$，我们有：

$$
\int_M \Omega = (2\pi)^{n/2} \chi(M)
$$

其中 $\Omega$ 是**Gauss-Bonnet-Chern 形式**，它是由 Riemann 曲率张量构造的**示性类**（Pfaffian）的微分形式。

### 2.2 Gauss-Bonnet-Chern 形式

为了精确表达 Gauss-Bonnet-Chern 定理，我们需要引入一些记号。

设 $R$ 是 Riemann 流形 $M$ 的**曲率 2-形式**，它可以用矩阵表示为：

$$
R = \left[ \begin{matrix}
R_{11} & R_{12} & \cdots & R_{1n} \\
R_{21} & R_{22} & \cdots & R_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
R_{n1} & R_{n2} & \cdots & R_{nn}
\end{matrix} \right]
$$

其中 $R_{ij}$ 是 2-形式，满足反对称性 $R_{ij} = -R_{ji}$。

**Gauss-Bonnet-Chern 形式**定义为：

$$
\Omega = \frac{1}{(2\pi)^{n/2}} \text{Pf}\left(\frac{R}{2\pi}\right)
$$

其中 $\text{Pf}(\cdot)$ 是**Pfaffian**算子。对于偶数阶反对称矩阵 $A$，Pfaffian 定义为：

$$
\text{Pf}(A) = \frac{1}{2^n n!} \sum_{\sigma \in S_{2n}} \text{sgn}(\sigma) \prod_{i=1}^n A_{\sigma(2i-1), \sigma(2i)}
$$

### 2.3 具体例子

**四维情形**（$n=4$）：

$$
\Omega = \frac{1}{32\pi^2} \left( |R|^2 - 4|Ric|^2 + R^2 \right) dV
$$

其中：
- $|R|^2$ 是曲率张量的模长平方
- $|Ric|^2$ 是 Ricci 曲率的模长平方
- $R$ 是标量曲率
- $dV$ 是体积形式

## 三、证明思路

### 3.1 Gauss-Bonnet 定理的证明思路

Gauss-Bonnet 定理有多种证明方法，这里介绍最具启发性的两种。

#### 方法一：三角剖分法

**主要思想**：用测地三角形将曲面剖分，然后局部验证公式。

1. **测地三角形的 Gauss-Bonnet 公式**：

对于测地三角形 $T$（边都是测地线，即 $k_g = 0$），我们有：

$$
\int_T K \, dA = \alpha + \beta + \gamma - \pi
$$

其中 $\alpha, \beta, \gamma$ 是三角形的内角。

这个结果的直观理解是：曲面弯曲使三角形内角和偏离 $\pi$，偏离量正好等于三角形内的总曲率。

2. **推广到多边形**：

对于测地多边形 $P$，将公式推广：

$$
\int_P K \, dA = \sum_{i=1}^n \theta_i - (n-2)\pi
$$

其中 $\theta_i$ 是内角。

3. **剖分求和**：

将曲面剖分成若干测地三角形 $T_1, T_2, \ldots, T_k$：

$$
\int_M K \, dA = \sum_{i=1}^k \int_{T_i} K \, dA = \sum_{i=1}^k \left(\sum_{j=1}^3 \alpha_{ij} - \pi\right)
$$

内部顶点的角度和为 $2\pi$，边界顶点为 $\pi$。经过整理可得：

$$
\int_M K \, dA = 2\pi (V - E + F) = 2\pi \chi(M)
$$

这正是 Euler 示性数的定义 $\chi(M) = V - E + F$。

#### 方法二：联络论方法

**主要思想**：利用单位球丛的联络计算。

1. 考虑曲面 $M$ 的**单位切丛** $SM$，这是一个三维流形。

2. 构造 SM 上的**Sasaki 度量**和对应的 Levi-Civita 联络。

3. 计算这个联络的曲率，证明其 Gauss-Bonnet 积分等于 $2\pi$ 乘以 Euler 示性数。

这种方法虽然抽象，但它为高维推广提供了清晰的框架。

### 3.2 Gauss-Bonnet-Chern 定理的证明思路

陈省身的证明采用了**活动标架法**（moving frames）和**外微分**的语言，这是他的标志性方法。

#### 步骤概要

1. **活动标架与联络形式**：

在局部选取标架场 $\{e_1, e_2, \ldots, e_n\}$，计算对偶 1-形式 $\{\theta^1, \theta^2, \ldots, \theta^n\}$ 和联络 1-形式 $\{\omega^i_j\}$。

2. **曲率形式**：

曲率 2-形式定义为：

$$
\Omega^i_j = d\omega^i_j - \sum_{k=1}^n \omega^i_k \wedge \omega^k_j
$$

3. **示性类**：

Gauss-Bonnet-Chern 形式可以通过曲率形式的 Pfaffian 表达：

$$
\text{Pf}\left(\frac{\Omega}{2\pi}\right) = \frac{1}{(2\pi)^{n/2}} \text{Pf}(\Omega)
$$

4. **关键观察**：

Gauss-Bonnet-Chern 形式是**闭形式**：

$$
d\left[\text{Pf}\left(\frac{\Omega}{2\pi}\right)\right] = 0
$$

这保证了它在整个流形上的积分只依赖于**上同调类**，即是一个拓扑不变量。

5. **归约到欧氏空间**：

对于任何紧致流形，可以将其嵌入到足够高维的欧氏空间。利用单位球丛的纤维积分，将问题归约为计算欧氏空间单位球上的积分，最终得到 $(2\pi)^{n/2} \chi(M)$。

#### 陈省身的创新

陈省身的证明有几个关键创新点：

1. **内蕴方法**：不需要将流形嵌入到欧氏空间，完全内蕴地处理。

2. **活动标架**：使用活动标架和微分形式，使计算简洁优雅。

3. **示性类**：证明 Gauss-Bonnet-Chern 形式代表 Euler 示性类，建立了局部几何与全局拓扑之间的桥梁。

## 四、应用与意义

### 4.1 数学意义

1. **局部-全局联系**：建立了局部曲率与全局拓扑之间的精确关系。

2. **示性类理论**：为特征类（Chern 类、Pontryagin 类等）的几何理解奠定了基础。

3. **微分几何的里程碑**：标志着现代微分几何的诞生，影响了后续几十年的研究。

### 4.2 物理应用

1. **广义相对论**：四维时空的 Einstein-Hilbert 作用量与 Gauss-Bonnet 项密切相关。

2. **弦理论**：在超弦理论中，Gauss-Bonnet 形式出现在高阶引力修正项中。

3. **规范场论**：规范场的拓扑荷（如瞬子数）可以用类似的 Chern-Weil 理论描述。

### 4.3 几何不等式

利用 Gauss-Bonnet 定理，可以推导许多有趣的几何不等式。例如，对于亏格为 $g$ 的紧致曲面：

$$
\int_M K \, dA = 4\pi(1-g)
$$

这告诉我们：
- $g = 0$（球面拓扑）：$\int_M K \, dA > 0$，必须存在正曲率区域
- $g = 1$（环面拓扑）：$\int_M K \, dA = 0$，曲率必须变号
- $g \geq 2$（高亏格）：$\int_M K \, dA < 0$，负曲率占主导

## 五、结语

Gauss-Bonnet 定理和 Gauss-Bonnet-Chern 定理堪称微分几何中最深刻、最美丽的定理之一。它们不仅建立了曲率与 Euler 示性数之间的精确关系，更重要的是揭示了**局部几何量与全局拓扑量之间深刻的内蕴联系**。

陈省身在 1944 年证明 Gauss-Bonnet-Chern 定理的工作，为现代微分几何奠定了基础。他的方法——活动标架法和外微分——至今仍是研究微分几何的标准工具。

从 Gauss 最先在二维情形发现这个定理，到陈省身将其推广到任意偶数维，历经一个多世纪。这段历史展现了数学发展的经典模式：从特殊到一般，从具体到抽象，从技巧到结构。

正如陈省身所说：**"几何学是研究空间形式的科学，而微分几何则是用微积分作为工具来研究几何学。"** Gauss-Bonnet-Chern 定理完美诠释了这一理念，将微积分的精确计算与拓扑学的大局观念结合得天衣无缝。

---

## 参考文献

1. Chern, S. S. (1944). "A simple intrinsic proof of the Gauss-Bonnet formula for closed Riemannian manifolds." *Annals of Mathematics*, 45(4), 747-752.

2. Do Carmo, M. P. (1992). *Riemannian Geometry*. Birkhäuser.

3. Kobayashi, S., & Nomizu, K. (1963). *Foundations of Differential Geometry*, Vol. II. Wiley.

4. Lee, J. M. (2018). *Introduction to Riemannian Manifolds*. Springer.

5. 陈省身. (1993). 《陈省身文集》. 科学出版社.
