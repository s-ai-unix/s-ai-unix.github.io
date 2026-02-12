---
title: "黎曼张量度量：弯曲空间的距离语言"
date: 2026-01-14T21:35:00+08:00
draft: false
description: "从欧几里得到黎曼，探索度量张量如何描述弯曲空间的距离和角度"
categories: ["数学", "微分几何"]
tags: ["微分几何", "广义相对论", "数学史"]
cover:
    image: "images/covers/photo-1620641788421-7a1c342ea42e.jpg"
    alt: "抽象几何空间"
    caption: "度量的几何"
---

## 引言：如何测量弯曲的世界？

想象一下，你生活在一个球面上。如果你想测量两点之间的距离，或者两条线之间的夹角，你会怎么做？

在平坦的欧几里得平面上，这很简单：距离用勾股定理计算，角度用点积定义。但在球面上，直线变成了大圆弧，勾股定理不再成立，角度的计算也变得更加复杂。

问题的关键在于：**我们需要一个通用的方法来定义任意空间中的距离和角度。**

这个方法就是**黎曼度量**（Riemannian Metric），或者更准确地说，**度量张量**（Metric Tensor）。它是黎曼几何的基础，也是广义相对论中描述时空的核心工具。

## 第一章：从勾股定理到度量张量

### 欧几里得距离

在二维欧几里得平面上，两点 $(x_1, y_1)$ 和 $(x_2, y_2)$ 之间的距离是：

$$ d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} $$

这个公式源自勾股定理。更一般地，如果我们考虑一个微小的位移 $(dx, dy)$，那么对应的距离是：

$$ ds^2 = dx^2 + dy^2 $$

这个表达式被称为**线元素**（line element）。它告诉我们：沿 $x$ 方向移动 $dx$，沿 $y$ 方向移动 $dy$，总距离的平方是 $dx^2 + dy^2$。

### 三维欧几里得空间

在三维欧几里得空间中，线元素是：

$$ ds^2 = dx^2 + dy^2 + dz^2 $$

我们可以把它写成矩阵形式：

$$ ds^2 = \begin{pmatrix} dx & dy & dz \end{pmatrix} \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} dx \\ dy \\ dz \end{pmatrix} $$

这个对角矩阵，就是欧几里得空间的**度量张量**。记作：

$$ g_{ij} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} $$

### 一般的度量张量

现在，我们不再局限于直角坐标系。考虑一个任意的坐标系 $(x^1, x^2, x^3)$（注意：这里使用上标表示坐标，这是张量分析的习惯）。

一个微小的位移可以用切向量 $d\mathbf{r} = dx^i \frac{\partial}{\partial x^i}$ 表示。这个向量的长度（或者说，距离的平方）是：

$$ ds^2 = d\mathbf{r} \cdot d\mathbf{r} = g_{ij} dx^i dx^j $$

这里，$g_{ij}$ 就是**度量张量**（Metric Tensor）。它是一个对称的二阶张量：

$$ g_{ij} = g_{ji} $$

度量张量告诉我们：在坐标 $(x^1, x^2, x^3)$ 处，沿方向 $dx^i$ 移动的距离平方是多少。

### 向量内积

度量张量不仅可以用来计算距离，还可以用来计算向量的内积（点积）。

给定两个切向量 $X = X^i \frac{\partial}{\partial x^i}$ 和 $Y = Y^j \frac{\partial}{\partial x^j}$，它们的内积是：

$$ \langle X, Y \rangle = g_{ij} X^i Y^j $$

特别地，向量的长度是：

$$ \|X\| = \sqrt{g_{ij} X^i X^j} $$

两个向量之间的夹角是：

$$ \cos \theta = \frac{\langle X, Y \rangle}{\|X\| \|Y\|} = \frac{g_{ij} X^i Y^j}{\sqrt{g_{kl} X^k X^l} \sqrt{g_{mn} Y^m Y^n}} $$

## 第二章：黎曼的远见——1854年的演讲

### 伯恩哈德·黎曼的突破

在1854年6月10日，黎曼在哥廷根大学做了他的**教授就职演讲**（Habilitationsschrift），题为**《论几何基础的假设》**（Über die Hypothesen, welche der Geometrie zu Grunde liegen）。

这篇演讲是数学史上最重要的文献之一，它开创了**黎曼几何**（Riemannian Geometry）。

### 黎曼的基本思想

黎曼提出了一个革命性的想法：**几何学不应该局限于三维欧几里得空间，而应该研究任意维度的"流形"（manifold）。**

黎曼的定义：
- **流形**：一个局部看起来像欧几里得空间的几何对象。例如，球面的任何一个小区域都可以近似地看作平面。
- **度量**：定义流形上两点之间的距离和角度。
- **曲率**：描述流形的弯曲程度。

黎曼意识到：如果我们有一个度量 $g_{ij}$，我们就可以计算各种几何量，包括长度、角度、面积、曲率等。

### 度量的自由

黎曼的一个重要洞察是：**度量不是唯一的。** 我们可以定义任意合理的度量（只要满足一定的条件，如正定性），每种度量对应一种不同的几何。

在平坦的欧几里得空间中，度量是：

$$ g_{ij} = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases} $$

但在球面上，度量是：

$$ ds^2 = R^2 (d\theta^2 + \sin^2 \theta \, d\phi^2) $$

因此，度量张量是：

$$ g_{ij} = \begin{pmatrix} R^2 & 0 \\ 0 & R^2 \sin^2 \theta \end{pmatrix} $$

## 第三章：度量张量的性质

### 正定性

一个合理的度量张量必须是**正定**的（positive definite）。这意味着对于任何非零向量 $X$：

$$ \langle X, X \rangle = g_{ij} X^i X^j > 0 $$

这个条件保证：任何非零向量都有正的长度。

**注**：在广义相对论中，使用的是**洛伦兹度量**（Lorentzian metric），它不是正定的，而是不定定的。这种度量被称为**伪黎曼度量**（pseudo-Riemannian metric）。

### 对称性

度量张量是对称的：

$$ g_{ij} = g_{ji} $$

这个条件来源于向量内积的对称性：$\langle X, Y \rangle = \langle Y, X \rangle$。

### 坐标变换

当我们从坐标系 $(x^i)$ 变换到坐标系 $(x'^i)$ 时，度量张量如何变化？

如果坐标变换是 $x'^i = x'^i(x^1, x^2, \ldots, x^n)$，那么切向量变换为：

$$ \frac{\partial}{\partial x^i} = \frac{\partial x'^j}{\partial x^i} \frac{\partial}{\partial x'^j} $$

因此，度量张量变换为：

$$ g'_{ij} = \frac{\partial x^k}{\partial x'^i} \frac{\partial x^l}{\partial x'^j} g_{kl} $$

这是**张量变换法则**的一个例子：度量张量是一个二阶协变张量（covariant tensor of rank 2）。

### 逆度量张量

由于度量张量 $g_{ij}$ 是正定对称矩阵，它总是可逆的。我们定义**逆度量张量**（inverse metric tensor）为：

$$ g^{ij} = (g_{ij})^{-1} $$

逆度量张量满足：

$$ g_{ij} g^{jk} = \delta_i^k $$

其中 $\delta_i^k$ 是**克罗内克符号**（Kronecker delta）：

$$ \delta_i^k = \begin{cases} 1 & \text{if } i = k \\ 0 & \text{if } i \neq k \end{cases} $$

逆度量张量用于升高指标的运算：给定一个协变向量 $v_i$，我们可以定义对应的逆变向量 $v^i$：

$$ v^i = g^{ij} v_j $$

### 体积元

度量张量还可以用来定义流形的**体积元**（volume element）。

在欧几里得空间中，体积元是 $dV = dx^1 dx^2 \cdots dx^n$。在一般的黎曼流形中，体积元是：

$$ dV = \sqrt{|g|} \, dx^1 dx^2 \cdots dx^n $$

其中 $|g| = |\det(g_{ij})|$ 是度量张量行列式的绝对值。

在二维情况下，这给出面积元；在三维情况下，这给出体积元；在四维情况下，这给出四维体积元。

## 第四章：具体计算实例

### 例1：极坐标下的平面

在极坐标 $(r, \theta)$ 中，欧几里得平面的线元素是：

$$ ds^2 = dr^2 + r^2 d\theta^2 $$

因此，度量张量是：

$$ g_{ij} = \begin{pmatrix} 1 & 0 \\ 0 & r^2 \end{pmatrix} $$

行列式：

$$ |g| = \det(g_{ij}) = r^2 $$

面积元：

$$ dA = \sqrt{|g|} \, dr d\theta = r \, dr d\theta $$

这与我们在微积分中学习的极坐标面积元一致。

### 例2：球面

考虑半径为 $R$ 的球面，用球坐标 $(\theta, \phi)$ 参数化，其中 $\theta \in (0, \pi)$ 是极角，$\phi \in (0, 2\pi)$ 是方位角。

球面的线元素是：

$$ ds^2 = R^2 d\theta^2 + R^2 \sin^2 \theta \, d\phi^2 $$

因此，度量张量是：

$$ g_{ij} = \begin{pmatrix} R^2 & 0 \\ 0 & R^2 \sin^2 \theta\end{pmatrix} $$

行列式：

$$ |g| = \det(g_{ij}) = R^4 \sin^2 \theta $$

面积元：

$$ dA = \sqrt{|g|} \, d\theta d\phi = R^2 \sin \theta \, d\theta d\phi $$

这给出球面的面积：

$$ A = \int dA = \int_0^{2\pi} \int_0^\pi R^2 \sin \theta \, d\theta d\phi = 4\pi R^2 $$

这正是我们熟悉的球面面积公式。

### 例3：柱面坐标

在柱面坐标 $(r, \phi, z)$ 中，欧几里得空间的线元素是：

$$ ds^2 = dr^2 + r^2 d\phi^2 + dz^2 $$

因此，度量张量是：

$$ g_{ij} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & r^2 & 0 \\ 0 & 0 & 1 \end{pmatrix} $$

行列式：

$$ |g| = \det(g_{ij}) = r^2 $$

体积元：

$$ dV = \sqrt{|g|} \, dr d\phi dz = r \, dr d\phi dz $$

这与我们在微积分中学习的柱面坐标体积元一致。

### 例4：双曲面

考虑**双曲面**（hyperboloid）$x^2 + y^2 - z^2 = -1$ 的上半部分。

用双曲坐标 $(r, \theta)$ 参数化，其中 $r > 0$，$\theta \in (0, 2\pi)$。

双曲面的线元素是：

$$ ds^2 = \frac{dr^2}{1 + r^2} + r^2 d\theta^2 $$

因此，度量张量是：

$$ g_{ij} = \begin{pmatrix} \frac{1}{1 + r^2} & 0 \\ 0 & r^2 \end{pmatrix} $$

行列式：

$$ |g| = \det(g_{ij}) = \frac{r^2}{1 + r^2} $$

面积元：

$$ dA = \sqrt{|g|} \, dr d\theta = \frac{r}{\sqrt{1 + r^2}} \, dr d\theta $$

## 第五章：度量张量与曲率

### 从度量到克里斯托费尔符号

度量张量是黎曼几何的出发点。从度量张量出发，我们可以定义**列维-奇维塔联络**（Levi-Civita connection），它由**克里斯托费尔符号**（Christoffel symbols）给出：

$$ \Gamma_{ij}^k = \frac{1}{2} g^{kl} \left( \frac{\partial g_{il}}{\partial x^j} + \frac{\partial g_{jl}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^l} \right) $$

克里斯托费尔符号告诉我们：如何在流形上平行移动向量。

### 从克里斯托费尔符号到曲率

从克里斯托费尔符号出发，我们可以定义**黎曼曲率张量**（Riemann curvature tensor）：

$$ R_{ijk}^{\quad l} = \frac{\partial \Gamma_{ij}^l}{\partial x^k} - \frac{\partial \Gamma_{ik}^l}{\partial x^j} + \Gamma_{ij}^m \Gamma_{km}^l - \Gamma_{ik}^m \Gamma_{jm}^l $$

黎曼曲率张量描述了流形的弯曲程度。

### 从曲率到爱因斯坦

在广义相对论中，时空的度规（metric）是**洛伦兹度规**（Lorentzian metric）：

$$ ds^2 = g_{\mu\nu} dx^\mu dx^\nu = -(1 - \frac{2GM}{c^2 r})c^2 dt^2 + (1 - \frac{2GM}{c^2 r})^{-1} dr^2 + r^2 (d\theta^2 + \sin^2 \theta d\phi^2) $$

这是**史瓦西度规**（Schwarzschild metric），描述了质量为 $M$ 的球对称物体周围的时空几何。

从这个度规出发，我们可以计算曲率，进而构造**爱因斯坦张量**（Einstein tensor）：

$$ G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu} $$

其中 $R_{\mu\nu}$ 是里奇曲率张量（Ricci curvature tensor），$R$ 是标量曲率（scalar curvature）。

爱因斯坦场方程将时空的曲率与物质的分布联系起来：

$$ G_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu} $$

其中 $T_{\mu\nu}$ 是应力-能量张量（stress-energy tensor），描述物质的分布和运动。

## 第六章：度量张量的应用

### 广义相对论

广义相对论是度量张量最著名的应用。在广义相对论中，**引力不是一种"力"，而是时空的弯曲。**

物质的分布决定了时空的度规，而时空的度规决定了物质的运动。具体来说：
1. 通过爱因斯坦场方程，从物质分布计算出度规 $g_{\mu\nu}$
2. 从度规计算出克里斯托费尔符号 $\Gamma_{\mu\nu}^\lambda$
3. 物体沿着**测地线**（geodesics）运动，测地线由克里斯托费尔符号决定

测地线方程是：

$$ \frac{d^2 x^\mu}{d\tau^2} + \Gamma_{\alpha\beta}^\mu \frac{dx^\alpha}{d\tau} \frac{dx^\beta}{d\tau} = 0 $$

其中 $\tau$ 是固有时间（proper time）。

### 计算机图形学

在计算机图形学中，度量张量用于：
- **曲面参数化**（Surface Parameterization）：将三维曲面映射到二维平面，同时保持距离和角度的关系。
- **网格处理**（Mesh Processing）：定义网格上的几何运算，如平滑、简化、变形。
- **纹理映射**（Texture Mapping）：将二维纹理贴图映射到三维曲面上。

### 机器学习

在机器学习中，度量张量用于：
- **流形学习**（Manifold Learning）：假设高维数据"生活"在低维流形上，度量张量帮助学习流形的几何结构。
- **信息几何**（Information Geometry）：将统计模型看作黎曼流形，度量张量由费雪信息矩阵（Fisher information matrix）给出。
- **度量学习**（Metric Learning）：学习数据空间中的距离度量，使得相似的数据点距离更近，不相似的数据点距离更远。

### 计算机视觉

在计算机视觉中，度量张量用于：
- **形状分析**（Shape Analysis）：比较和分类三维形状。
- **图像处理**（Image Processing）：定义图像上的几何运算，如各向异性扩散（anisotropic diffusion）。
- **3D重建**（3D Reconstruction）：从二维图像重建三维场景。

## 第七章：度量的分类

### 常曲率度量

如果曲率处处相同，这种度量称为**常曲率度量**（constant curvature metric）。

- **正曲率**（$K > 0$）：例如，球面。
- **零曲率**（$K = 0$）：例如，欧几里得空间。
- **负曲率**（$K < 0$）：例如，双曲空间。

### 共形度量

两个度量 $g$ 和 $\tilde{g}$ 称为**共形**（conformal），如果存在一个正函数 $\lambda$，使得：

$$ \tilde{g} = \lambda^2 g $$

共形变换保持角度不变，但不保持长度不变。

在地图投影中，常用共形变换来保持地图上的角度与真实地球上的角度一致。

### 乘积度量

给定两个黎曼流形 $(M_1, g_1)$ 和 $(M_2, g_2)$，它们的**乘积流形**（product manifold）$M_1 \times M_2$ 上的乘积度量是：

$$ g = g_1 \oplus g_2 $$

例如，圆柱面是直线（$R$）和圆（$S^1$）的乘积：$R \times S^1$。因此，圆柱面的度量是：

$$ ds^2 = dz^2 + R^2 d\phi^2 $$

这与我们之前计算的柱面坐标下的度量一致。

## 第八章：测地线——最短路径

### 测地线的定义

**测地线**（geodesic）是黎曼流形上的"最短路径"（在局部意义上）。

给定一个曲线 $\gamma(t)$，其长度是：

$$ L(\gamma) = \int_a^b \sqrt{g_{ij}(\gamma(t)) \frac{d\gamma^i}{dt} \frac{d\gamma^j}{dt}} \, dt $$

测地线是使长度达到极小的曲线。通过变分法，我们可以得到测地线方程：

$$ \frac{d^2 x^i}{dt^2} + \Gamma_{jk}^i \frac{dx^j}{dt} \frac{dx^k}{dt} = 0 $$

### 测地线的例子

#### 例1：欧几里得空间的直线

在欧几里得空间中，克里斯托费尔符号处处为零，因此测地线方程简化为：

$$ \frac{d^2 x^i}{dt^2} = 0 $$

解是：

$$ x^i(t) = A^i t + B^i $$

这正是直线的参数方程。

#### 例2：球面上的大圆

在球面上，测地线是**大圆**（great circles），即通过球心的平面与球面的交线。

例如，赤道是一条测地线，经线也是测地线。从赤道上的一个点出发，沿大圆弧移动，可以到达赤道上的任何其他点，而这是"最短"路径。

#### 例3：双曲面上的测地线

在双曲面上，测地线是"直线"的推广。在庞加莱圆盘模型（Poincaré disk model）中，测地线是与圆周正交的圆弧。

### 测地距离

给定两点 $P$ 和 $Q$，它们之间的**测地距离**（geodesic distance）定义为连接它们的测地线的长度：

$$ d(P, Q) = \inf_\gamma L(\gamma) $$

其中 $\gamma$ 是所有从 $P$ 到 $Q$ 的曲线。

测地距离定义了黎曼流形上的一个**度量空间**（metric space），满足：
1. $d(P, Q) \geq 0$，且 $d(P, Q) = 0$ 当且仅当 $P = Q$（正定性）
2. $d(P, Q) = d(Q, P)$（对称性）
3. $d(P, R) \leq d(P, Q) + d(Q, R)$（三角不等式）

## 结语：度量的哲学

黎曼度量不仅仅是一个数学对象，它代表了我们对空间的理解方式。

### 从绝对到相对

在牛顿时代，空间被认为是绝对的、固定的。欧几里得几何被认为是唯一的几何学。

黎曼改变了这一切。他提出：**空间本身可以有几何结构，这种结构由度量定义。**

更深远的是，黎曼暗示：**几何学可能不是先验的，而是经验的。** 也就是说，空间的几何结构可能需要通过实验和观察来确定，而不是通过纯粹的推理得出。

### 从数学到物理

爱因斯坦将黎曼的远见变成了现实。在广义相对论中，时空的几何结构不是固定的，而是由物质的分布决定的。

这意味着：**空间和时间不是绝对的和不变的，而是弯曲的和动态的。**

当我们观察星光经过太阳时弯曲（引力透镜效应），我们实际上是在见证时空的几何结构被物质改变了。当我们探测到引力波时，我们实际上是在聆听时空的涟漪。

### 从抽象到应用

黎曼度量不仅在纯数学中有重要地位，在应用数学和物理学中也有广泛应用：

- **计算机图形学**：定义曲面的几何运算
- **机器学习**：学习数据的几何结构
- **物理学**：描述时空的弯曲
- **工程学**：分析结构和流体的动力学

### 数学之美

黎曼度量体现了数学的统一性和美学：

- 它统一了欧几里得几何和非欧几里得几何
- 它连接了微分几何、拓扑学和物理学
- 它将"距离"这个直观概念抽象为严格的数学对象

正如黎曼在1854年的演讲中引用高斯的话："空间是否有度规，这是我们应当追问的问题。"（Ob die den Raum eine Maßbestimmung zukommt, das ist eine Frage, zu deren Beantwortung wir berufen sind.）

今天，我们仍在继续探索这个问题，通过实验和理论，不断地深化我们对空间、时间和几何的理解。

黎曼度量告诉我们：**世界是复杂的，但我们可以用数学来理解它。**

---

## 参考文献

1. Riemann, B. (1854). *Über die Hypothesen, welche der Geometrie zu Grunde liegen*
2. do Carmo, M. P. (1992). *Riemannian Geometry*
3. Lee, J. M. (2018). *Introduction to Riemannian Manifolds*
4. Einstein, A. (1915). *Die Feldgleichungen der Gravitation*
5. Misner, C. W., Thorne, K. S., & Wheeler, J. A. (1973). *Gravitation*
6. Wald, R. M. (1984). *General Relativity*
7. O'Neill, B. (1983). *Semi-Riemannian Geometry with Applications to Relativity*
