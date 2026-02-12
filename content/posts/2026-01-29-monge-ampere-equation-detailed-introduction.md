---
title: "蒙日-安培方程：从经典几何到现代分析的系统综述"
date: 2026-01-29T19:30:00+08:00
draft: false
description: "本文系统综述蒙日-安培方程的理论体系，从18世纪的几何起源到现代分析理论，深入剖析其数学结构、解理论及跨学科应用，展现这一完全非线性偏微分方程的深刻内涵。"
categories: ["数学", "偏微分方程"]
tags: ["偏微分方程", "微分几何", "综述", "最优传输", "凸几何"]
cover:
    image: "images/covers/1555255707-c07966088b7b.jpg"
    alt: "Monge-Ampere方程与最优传输"
    caption: "蒙日-安培方程：连接几何与分析的桥梁"
math: true
---

## 引言：一个跨越两个半世纪的数学传奇

1771年，法国数学家加斯帕尔·蒙日（Gaspard Monge）在研究曲面和曲线理论时，写下了一个看似简单的方程。他或许不会想到，这个方程将在接下来的两个半世纪里，成为连接微分几何、偏微分方程、变分法和概率论的深刻纽带，并最终在2018年帮助阿莱西奥·菲加利（Alessio Figalli）获得菲尔兹奖。

这个方程就是**蒙日-安培方程**（Monge-Ampère Equation）。

![历史发展时间线](/images/plots/ma_history_timeline.png)

**图1**：蒙日-安培方程从18世纪到现代的发展历程，涵盖了几何、分析和应用数学的多个里程碑。

蒙日-安培方程的特殊之处在于它的**完全非线性**特性。与拉普拉斯方程或热方程这类线性方程不同，蒙日-安培方程涉及未知函数二阶导数的行列式——这是所有二阶导数的非线性组合。这种结构既带来了深刻的数学挑战，也赋予了它独特的几何意义。

在本文中，我们将从三个维度深入探索这一优美的数学对象：
- **历史维度**：从蒙日的几何洞察到现代正则性理论
- **理论维度**：方程的结构、椭圆性理论和解的适定性
- **应用维度**：从凸几何到最优传输，从气象学到机器学习

---

## 第一章：历史渊源——从蒙日到现代

### 1.1 蒙日的几何洞察（1771-1807）

加斯帕尔·蒙日（1746-1818）是法国大革命时期的杰出数学家，被誉为**画法几何**的奠基人。他对曲面的研究源于工程学的实际问题：如何在二维平面上精确表示三维物体？

1771年，蒙日在论文《Memoire sur les developpées, les rayons de courbure et les différens genres d'inflexions des courbes à double courbure》中首次研究了一类涉及曲面曲率的偏微分方程。他考虑的核心问题是：**给定曲面的曲率信息，能否重建曲面本身？**

蒙日的洞察在于认识到曲面的**高斯曲率**与函数二阶导数之间的深刻联系。对于一个由 $z = u(x, y)$ 给出的曲面，其高斯曲率 $K$ 可以表示为：

<div class="math">
$$
K = \frac{u_{xx}u_{yy} - u_{xy}^2}{(1 + u_x^2 + u_y^2)^2}
$$
</div>

分子中的 $u_{xx}u_{yy} - u_{xy}^2$ 正是函数 $u$ 的**Hessian行列式**——蒙日-安培方程的核心结构。

### 1.2 安培的分析贡献（1820s）

安德烈-玛丽·安培（André-Marie Ampère，1775-1836）更为人熟知的是他在电磁学方面的贡献（电流单位"安培"即以他命名）。但在1820年代，安培对蒙日的方程进行了系统的分析研究，将其推广到更一般的形式。

安培考虑了方程的**一般二阶形式**：

<div class="math">
$$
A(u_{xx}u_{yy} - u_{xy}^2) + Bu_{xx} + Cu_{xy} + Du_{yy} + E = 0
$$
</div>

其中系数 $A, B, C, D, E$ 可以依赖于 $(x, y, u, u_x, u_y)$。当 $A \neq 0$ 时，方程具有典型的蒙日-安培结构。

安培的工作确立了对这类方程进行系统分析的可能性，为后来的偏微分方程理论奠定了基础。

### 1.3 闵可夫斯基与凸几何（1903）

赫尔曼·闵可夫斯基（Hermann Minkowski，1864-1909）的工作为蒙日-安培方程带来了新的几何视角。1903年，他在论文《体积与表面积》（Volumen und Oberfläche）中提出了著名的**闵可夫斯基问题**：

> **闵可夫斯基问题**：给定球面上的一个正函数 $K(n)$，是否存在一个严格凸的闭曲面，使其在法向量为 $n$ 的点处具有高斯曲率 $K(n)$？

这一问题与蒙日-安培方程的联系在于：如果曲面表示为径向函数 $r = r(n)$，则高斯曲率条件恰好转化为一个蒙日-安培方程。

欧金尼奥·卡拉比（Eugenio Calabi）曾评价道："从几何观点看，闵可夫斯基问题是罗塞塔石碑，许多相关问题都可以从中得到解答。"

### 1.4 Alexandrov的弱解理论（1950s）

20世纪中期，苏联数学家亚历山大·亚历山德罗夫（Alexander Alexandrov，1912-1999）为蒙日-安培方程引入了革命性的**弱解概念**。传统的经典解要求函数二阶可微，但Alexandrov意识到这对非线性方程过于严格。

Alexandrov的洞见是利用**凸函数的性质**：凸函数虽然可能不可微，但具有**次微分**（subdifferential）。基于这一思想，他定义了蒙日-安培测度：

<div class="math">
$$
\mu_u(E) = |\partial u(E)|
$$
</div>

其中 $\partial u(E)$ 表示集合 $E$ 上次梯度的像。这使得蒙日-安培方程可以在测度意义下求解，即使解本身可能高度奇异。

### 1.5 丘成桐与程秀耀：高维闵可夫斯基问题（1976）

1976年，程秀耀（Shiu-Yuen Cheng）和丘成桐（Shing-Tung Yau）在论文《On the regularity of the solution of the n-dimensional Minkowski problem》中完全解决了高维闵可夫斯基问题。他们证明了给定曲率条件下凸解的存在性和正则性，这一工作被认为是现代几何分析的重要里程碑。

丘成桐因此在1982年获得菲尔兹奖，表彰他在微分几何和偏微分方程方面的杰出贡献，特别是对卡拉比猜想和闵可夫斯基问题的解决。

### 1.6 Caffarelli的正则性革命（1990s）

20世纪90年代，路易斯·卡法雷利（Luis Caffarelli）的工作彻底改变了人们对蒙日-安培方程解光滑性的理解。在此之前，即使知道弱解存在，也很难判断它们是否具有足够的正则性（光滑性）。

Caffarelli的开创性贡献包括：
- **严格凸性**：证明了在适当条件下，解必须是严格凸的
- **内部正则性**：证明了严格凸解具有 Hölder 连续的二阶导数（$C^{2,\alpha}$）
- **边界正则性**：当边界数据足够光滑时，解在边界附近也有良好正则性

这些结果使得蒙日-安培方程从一个理论优美的方程变成了可以实际计算和应用的数学工具。

### 1.7 最优传输的复兴（1987-2018）

1987年，伊夫·布伦尼耶（Yves Brenier）发现了蒙日-安培方程与**最优传输问题**（Optimal Transport）之间的深刻联系。他证明了在二次代价下，最优传输映射可以表示为一个凸函数的梯度 $\nabla u$，且该凸函数满足蒙日-安培方程。

这一发现引发了蒙日-安培方程研究的复兴。2010年，塞德里克·维拉尼（Cédric Villani）因其在最优传输和相关领域的工作获得菲尔兹奖。2018年，阿莱西奥·菲加利（Alessio Figalli）同样因为在最优传输和蒙日-安培方程方面的贡献获得菲尔兹奖。

---

## 第二章：方程的数学结构

### 2.1 基本形式与分类

**实蒙日-安培方程**最简形式为：

<div class="math">
$$
\det(D^2 u) = f(x, u, \nabla u), \quad x \in \Omega \subset \mathbb{R}^n
$$
</div>

其中：
- $u: \Omega \to \mathbb{R}$ 是未知函数
- $D^2 u = (\partial_{ij} u)$ 是 $u$ 的 **Hessian矩阵**
- $\det(D^2 u)$ 表示 Hessian 矩阵的行列式
- $f$ 是给定的正函数

在二维情形（$n=2$），Hessian行列式有显式表达：

<div class="math">
$$
\det(D^2 u) = u_{xx}u_{yy} - u_{xy}^2
$$
</div>

这就是二阶导数的"判别式"，在经典微分几何中起着核心作用。

![凸函数与Hessian矩阵](/images/plots/ma_convex_function.png)

**图2**：凸函数的几何特性。左图展示了一个典型的凸函数（$z = x^2 + y^2$），右图显示其等高线和梯度向量场。凸性是蒙日-安培方程解的关键条件。

### 2.2 椭圆性与凸性

蒙日-安培方程的**类型**（椭圆、双曲、抛物）取决于解的性质。为了定义类型，我们需要考察方程在线性化后的**主符号**。

对于方程 $\det(D^2 u) = f(x)$，在解 $u$ 处的线性化具有主符号：

<div class="math">
$$
a^{ij}(x) \xi_i \xi_j = \frac{\partial \det(D^2 u)}{\partial u_{ij}} \xi_i \xi_j
$$
</div>

利用Jacobi公式 $\frac{\partial \det A}{\partial A_{ij}} = (A^{-1})_{ji} \det A$，我们得到：

<div class="math">
$$
a^{ij} = (D^2 u)^{ij} \det(D^2 u) = f(x) (D^2 u)^{ij}
$$
</div>

其中 $(D^2 u)^{ij}$ 表示Hessian矩阵的逆矩阵的 $(i,j)$ 元素。

**关键观察**：如果 $u$ 是**严格凸函数**，则 $D^2 u$ 是**正定矩阵**，因此 $(D^2 u)^{ij}$ 也是正定的。这意味着：

<div class="math">
**椭圆性条件**：当 $u$ 严格凸时，蒙日-安培方程是**一致椭圆**的。
</div>

这一观察至关重要，因为它意味着：
1. 在严格凸解的框架下，蒙日-安培方程享有椭圆型方程的良好性质
2. 最大原理适用，保证了唯一性
3. 正则性理论得以建立

![Hessian行列式与椭圆性](/images/plots/ma_determinant_ellipticity.png)

**图3**：Hessian行列式与椭圆性的关系。左图展示了不同正定矩阵对应的椭圆；右图显示行列式（特征值的乘积）随特征值变化的曲线。

### 2.3 与曲率的几何联系

蒙日-安培方程与**高斯曲率**之间的深刻联系是其几何应用的核心。

设曲面由函数图像 $z = u(x)$，$x \in \mathbb{R}^n$ 给出。则曲面的**高斯曲率**为：

<div class="math">
$$
K = \frac{\det(D^2 u)}{(1 + |\nabla u|^2)^{(n+2)/2}}
$$
</div>

因此，**给定高斯曲率问题**等价于求解：

<div class="math">
$$
\det(D^2 u) = K(x)(1 + |\nabla u|^2)^{(n+2)/2}
$$
</div>

这是**非齐次**蒙日-安培方程的一个典型例子。

![高斯曲率与曲面类型](/images/plots/ma_gaussian_curvature.png)

**图4**：不同曲面的高斯曲率。左图展示球面（$K > 0$ 处处为正），右图展示双曲抛物面（$K < 0$ 处处为负）。蒙日-安培方程与曲率问题密切相关。

---

## 第三章：解的理论——从弱解到正则性

### 3.1 Alexandrov弱解

对于凸函数（不必光滑），Alexandrov引入了基于**次梯度**（subgradient）概念的弱解定义。

**定义**（凸函数的次微分）：对于凸函数 $u$，在点 $x$ 处的次微分定义为：

<div class="math">
$$
\partial u(x) = \{ p \in \mathbb{R}^n : u(y) \geq u(x) + p \cdot (y-x), \forall y \}
$$
</div>

几何上，$\partial u(x)$ 包含所有在 $(x, u(x))$ 处支撑 $u$ 的图的超平面的斜率。

**定义**（Alexandrov弱解）：设 $u$ 是凸函数，定义**蒙日-安培测度**：

<div class="math">
$$
\mu_u(E) = |\partial u(E)|
$$
</div>

其中 $|\cdot|$ 表示Lebesgue测度。称 $u$ 是方程 $\det(D^2 u) = f$ 的**Alexandrov解**，如果：

<div class="math">
$$
\mu_u(E) = \int_E f(x) dx
$$
</div>

对所有Borel集 $E$ 成立。

Alexandrov解的存在性可以通过**Perron方法**或**连续性方法**证明。这种方法的关键优势是**不依赖于解的正则性**，允许高度奇异的解。

### 3.2 唯一性与比较原理

在Alexandrov解的框架下，唯一性由**比较原理**保证：

<div class="math">
**比较原理**：设 $u, v$ 是有界域 $\Omega$ 上的凸函数，满足
$$
\mu_u \geq \mu_v \text{ 在 } \Omega \text{ 内}, \quad u \leq v \text{  在 } \partial\Omega \text{ 上}
$$
则 $u \leq v$ 在 $\Omega$ 内成立。
</div>

比较原理的一个重要推论是**Dirichlet问题解的唯一性**：如果 $u, v$ 都是 $\det(D^2 u) = f$ 的解，且 $u = v$ 在边界上，则 $u \equiv v$。

### 3.3 Caffarelli的正则性理论

弱解的存在性只是第一步。真正使蒙日-安培方程成为实用工具的是**正则性理论**——证明在适当条件下，弱解实际上是光滑的。

Caffarelli的正则性理论建立在两个关键观察之上：

**观察1：严格凸性的传播**

如果 $f$ 远离零有界（即 $0 < \lambda \leq f \leq \Lambda < \infty$），则Alexandrov解必须是**严格凸**的。这是因为如果 $u$ 在某处有平坦部分（非严格凸），则在该处 $\det(D^2 u) = 0$，与 $f > 0$ 矛盾。

**观察2：内部正则性**

一旦解严格凸，方程就成为一致椭圆的，经典的**Schauder估计**适用。Caffarelli证明了：

<div class="math">
**内部正则性定理**：设 $u$ 是 $B_1$ 上的Alexandrov解，满足
$$
\lambda \leq \det(D^2 u) \leq \Lambda
$$
且 $u$ 严格凸。则存在 $\alpha = \alpha(n, \lambda, \Lambda) \in (0, 1)$，使得 $u \in C^{1,\alpha}_{loc}$。进一步，如果 $f \in C^{\alpha}$，则 $u \in C^{2,\alpha}_{loc}$。
</div>

![正则性理论](/images/plots/ma_regularity_theory.png)

**图5**：Caffarelli正则性理论示意。左上：严格凸的光滑解；右上：非严格凸解（存在角点）；左下：边界正则性；右下：内部正则性的局部性质。

### 3.4 边界正则性

边界正则性比内部正则性更复杂，需要额外的几何条件。关键条件是边界的**严格凸性**：

<div class="math">
**边界正则性定理**：设 $\Omega$ 是严格凸域，边界 $\partial\Omega \in C^{2,\alpha}$，边界数据 $g \in C^{2,\alpha}(\partial\Omega)$，且 $f \in C^{\alpha}(\overline{\Omega})$，$f > 0$。则Dirichlet问题
$$
\begin{cases}
\det(D^2 u) = f & \text{在 } \Omega \\
u = g & \text{在 } \partial\Omega
\end{cases}
$$
的解 $u \in C^{2,\alpha}(\overline{\Omega})$。
</div>

---

## 第四章：几何应用

### 4.1 Minkowski问题

Minkowski问题是凸几何中与蒙日-安培方程联系最紧密的问题之一。

**问题陈述**：给定单位球面 $S^{n-1}$ 上的正函数 $K(n)$，寻找一个严格凸的紧集（凸体）$\Omega \subset \mathbb{R}^n$，使得其在边界点处（外法向为 $n$）的高斯曲率为 $K(n)$。

当凸体由支撑函数 $h(n)$ 参数化时，高斯曲率条件转化为：

<div class="math">
$$
\det(\nabla^2 h + h \cdot \text{Id}) = K(n)^{-1}
$$
</div>

这是球面上的蒙日-安培方程。

![Minkowski问题](/images/plots/ma_minkowski_problem.png)

**图6**：Minkowski问题示意图。给定球面上每个方向 $n$ 的曲率 $K(n)$，求对应的凸曲面。虚线表示参考球面，实线表示求解的目标曲面。

### 4.2 Weyl问题

**Weyl问题**是另一个经典的几何问题：给定球面上的一个度量，能否将其等距嵌入到 $\mathbb{R}^3$ 中作为凸曲面的第一基本形式？

这个问题同样归结为蒙日-安培方程的求解。1953年，路易斯·尼伦伯格（Louis Nirenberg）成功解决了三维空间中的Weyl问题，这一工作被认为是他在2010年获得陈省身奖的主要原因之一。

### 4.3 仿射几何与仿射球面

在仿射微分几何中，**仿射球面**（Affine Spheres）由特定的蒙日-安培方程描述。特别地，**仿射极大曲面**（Affine Maximal Surfaces）满足：

<div class="math">
$$
\Delta \left( (\det D^2 u)^{-\frac{n+1}{n+2}} \right) = 0
$$
</div>

这是蒙日-安培型方程的一个变体，在仿射几何中扮演类似极小曲面在欧氏几何中的角色。

### 4.4 反射器设计

在**几何光学**中，蒙日-安培方程出现在反射器/折射器设计问题中：设计一个反射面，使得点光源发出的光线经反射后产生预定的照度分布。

这类问题可以表述为最优传输问题，因此归结为蒙日-安培方程的求解。实际应用包括：
- 汽车前灯设计
- 太阳能聚光器
- 激光束整形

---

## 第五章：最优传输与蒙日-安培方程

### 5.1 最优传输问题

**最优传输问题**（Optimal Transport）可以追溯到你蒙日1781年的论文。问题陈述如下：

> 给定两个概率密度 $f$ 和 $g$（分别定义在域 $\Omega$ 和 $\Omega'$ 上），寻找一个传输映射 $T: \Omega \to \Omega'$，将 $f$ "推送"到 $g$（即满足质量守恒），并最小化传输代价。

在**二次代价**（Quadratic Cost）情形下，传输代价为：

<div class="math">
$$
\min_T \int_{\Omega} |x - T(x)|^2 f(x) dx
$$
</div>

### 5.2 Brenier定理

1987年，布伦尼耶证明了最优传输理论中最重要的结果之一：

<div class="math">
**Brenier定理**：在二次代价下，存在唯一的最优传输映射 $T$，且 $T$ 可以表示为一个凸函数的梯度：
$$
T = \nabla u
$$
其中 $u$ 是某个凸函数（称为**Brenier势**）。
</div>

![最优传输问题](/images/plots/ma_optimal_transport.png)

**图7**：最优传输问题示意图。左图：源分布 $f(x)$；中图：传输映射 $T = \nabla u$（凸函数的梯度）；右图：目标分布 $g(y)$。质量守恒条件要求 $f(x) = g(\nabla u(x)) \det(D^2 u(x))$。

### 5.3 Monge-Ampere方程的出现

利用Brenier定理和质量守恒条件，可以导出Brenier势满足的方程。

**质量守恒条件**：对于任意可测集 $A$，

<div class="math">
$$
\int_A f(x) dx = \int_{T(A)} g(y) dy = \int_A g(T(x)) |\det DT(x)| dx
$$
</div>

由于 $T = \nabla u$，有 $DT = D^2 u$，因此：

<div class="math">
$$
f(x) = g(\nabla u(x)) \det(D^2 u(x))
$$
</div>

整理得到**Brenier-Monge-Ampere方程**：

<div class="math">
$$
\det(D^2 u(x)) = \frac{f(x)}{g(\nabla u(x))}
$$
</div>

这是蒙日-安培方程在最优传输中的核心形式。

### 5.4 Wasserstein距离与梯度流

最优传输理论定义了**Wasserstein距离**（也称Earth Mover's Distance），这是概率测度空间上的度量：

<div class="math">
$$
W_2(\mu, \nu) = \left( \inf_{T_{\#}\mu = \nu} \int |x - T(x)|^2 d\mu \right)^{1/2}
$$
</div>

在Wasserstein度量下，可以定义**蒙日-安培流**（Monge-Ampère Flow），这是一类重要的梯度流，在图像处理和机器学习中有所应用。

---

## 第六章：其他应用领域

### 6.1 气象学与半地转流

在**大气科学**中，**半地转流方程**（Semigeostrophic Equations）描述了大尺度大气运动的准平衡状态。通过**Legendre变换**，这些方程可以转化为蒙日-安培方程。

这一转化的重要性在于：原本复杂的流体动力学问题，通过几何变换变成了结构良好的蒙日-安培方程，可以应用成熟的正则性理论和数值方法。

### 6.2 经济学与机制设计

在**经济学**中，最优传输理论应用于：
- **匹配理论**：劳动力市场匹配、婚姻市场匹配
- **拍卖设计**：最优拍卖机制
- **定价模型**：基于运输成本的定价策略

这些应用通常涉及将一类经济主体（如买家）"匹配"到另一类（如卖家），在满足约束条件的同时最大化某种社会福利函数——这正是最优传输问题的核心。

### 6.3 图像处理与计算机视觉

蒙日-安培方程和最优传输在**图像处理**中的应用包括：
- **图像配准**：将一幅图像"变形"以匹配另一幅
- **直方图均衡化**：将图像的灰度分布变换为指定分布
- **纹理合成与迁移**：保持纹理特征的同时改变外观

### 6.4 机器学习与生成模型

近年来，蒙日-安培方程在**机器学习**领域获得了新的关注：

**正规化流**（Normalizing Flows）：一类生成模型，通过可逆变换学习数据分布。某些架构（如凸势流）直接基于蒙日-安培方程的结构。

**Wasserstein GAN**：使用Wasserstein距离替代JS散度的生成对抗网络，训练更加稳定，生成的样本质量更高。

**蒙日-安培流采样**：利用蒙日-安培流进行概率分布采样，在贝叶斯推断中有所应用。

---

## 第七章：数值方法与前沿问题

### 7.1 经典数值方法

蒙日-安培方程的**数值求解**是一个具有挑战性的问题，主要原因包括：
- **非线性**：Hessian行列式是非线性算子
- **凸性约束**：解必须保持凸性
- **奇异性**：在退化点附近数值困难

**常见数值方法**包括：
- **有限差分法**：在结构化网格上离散，需要特别处理凸性约束
- **有限元法**：使用分段多项式逼近，适合复杂几何
- **谱方法**：对于光滑解具有指数收敛速度
- **半离散方法**：基于Alexandrov弱解的几何解释

### 7.2 深度学习与神经PDE求解器

近年来，**物理信息神经网络**（Physics-Informed Neural Networks, PINNs）被应用于求解蒙日-安培方程。基本思想是用神经网络 $u_\theta$ 参数化解，通过最小化PDE残差来训练：

<div class="math">
$$
\mathcal{L}(\theta) = \int_{\Omega} |\det(D^2 u_\theta) - f|^2 dx + \text{边界项}
$$
</div>

神经网络方法的优势在于：
- 无需结构化网格
- 可以处理复杂几何
- 易于与现代深度学习框架集成

挑战包括确保凸性约束和在高维情形下的收敛性。

### 7.3 前沿问题

蒙日-安培方程研究领域仍有许多未解决的问题：

**退化情形**：当 $f$ 可以取零值时（即允许 $\det(D^2 u) = 0$），解的正则性理论仍不完善。

**边界奇性**：当边界非严格凸时，解在边界附近的行为尚不完全清楚。

**高维数值**：在高维情形（$n \geq 4$），数值方法面临"维度灾难"。

**随机蒙日-安培方程**：带有随机源或随机系数的蒙日-安培方程的理论和数值方法。

**离散蒙日-安培方程**：在图或离散点集上定义蒙日-安培算子，连接离散几何。

---

## 结语：一座连接数学世界的桥梁

蒙日-安培方程从18世纪的一个几何问题出发，历经两个半世纪的发展，已经成为连接多个数学分支的深刻桥梁。

**从几何角度**，它是**凸几何**的核心方程，决定了曲面的曲率和形状。闵可夫斯基问题、Weyl问题等经典几何问题都围绕它展开。

**从分析角度**，它是**完全非线性椭圆型偏微分方程**的典型代表。Alexandrov的弱解理论、Caffarelli的正则性理论为这类方程建立了坚实的理论基础。

**从应用角度**，它是**最优传输问题**的数学核心，连接概率论、经济学、图像处理和机器学习。

回顾这一发展历程，我们可以看到数学思想的深刻连贯性：从蒙日的几何洞察，到安培的分析研究，再到Alexandrov的弱解理论、Caffarelli的正则性革命，以及Brenier-Figalli的最优传输联系——每一代数学家都在前人的基础上推进，将这一方程的理解推向新的高度。

正如阿莱西奥·菲加利在其菲尔兹奖获奖演说中所言："**蒙日-安培方程不仅是一个优美的数学对象，它更是理解几何、分析和概率之间深刻联系的窗口。**"

在这个数据驱动的时代，蒙日-安培方程的理论和应用价值愈发凸显。从气候模型的改进到生成式AI的发展，这一古老的方程正在21世纪焕发新的生命力。数学之美，正在于这种跨越时空的永恒性和应用的不断更新。

---

## 参考文献

1. Figalli, A. (2017). *The Monge-Ampère Equation and Its Applications*. Zurich Lectures in Advanced Mathematics, European Mathematical Society.

2. Gutiérrez, C. E. (2016). *The Monge-Ampère Equation* (2nd ed.). Progress in Nonlinear Differential Equations and Their Applications, Birkhäuser.

3. Caffarelli, L. A. (1990). "Interior $W^{2,p}$ estimates for solutions of the Monge-Ampère equation". *Annals of Mathematics*, 131(1), 135-150.

4. De Philippis, G., & Figalli, A. (2014). "The Monge-Ampère Equation and Its Link to Optimal Transportation". *Bulletin of the American Mathematical Society*, 51(4), 527-580.

5. Brenier, Y. (1991). "Polar factorization and monotone rearrangement of vector-valued functions". *Communications on Pure and Applied Mathematics*, 44(4), 375-417.

6. Cheng, S. Y., & Yau, S. T. (1976). "On the regularity of the solution of the n-dimensional Minkowski problem". *Communications on Pure and Applied Mathematics*, 29(5), 495-516.

7. Nirenberg, L. (1953). "The Weyl and Minkowski problems in differential geometry in the large". *Communications on Pure and Applied Mathematics*, 6(3), 337-394.

8. Villani, C. (2009). *Optimal Transport: Old and New*. Grundlehren der mathematischen Wissenschaften, Vol. 338, Springer.

9. Le, N. Q. (2024). *Analysis of Monge-Ampère Equations*. Graduate Studies in Mathematics, Vol. 240, American Mathematical Society.

10. Trudinger, N. S., & Wang, X. J. (2008). "The Monge-Ampère equation and its geometric applications". *Handbook of Geometric Analysis*, Vol. I, 467-524.
