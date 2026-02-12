---
title: "广义相对论入门：从微分几何到爱因斯坦场方程"
date: 2026-01-12T21:30:00+08:00
draft: false
description: "从零开始详细推导广义相对论，包括张量分析、测地线方程和爱因斯坦场方程，适合有微积分和线性代数基础的读者"
categories: ["物理学", "数学"]
tags: ["广义相对论", "微分几何", "黎曼几何"]
cover:
    image: "images/covers/1462331940025-496dfbfc7564.jpg"
    alt: "广义相对论"
    caption: "广义相对论：时空的弯曲与引力"
---

## 引言：为什么我们需要新理论？

### 从牛顿到爱因斯坦

1905年，爱因斯坦发表了狭义相对论，彻底改变了我们对时空的认知。在这个理论中，他告诉我们：光速是恒定的，物理定律在所有惯性参考系中都是相同的。然而，这个理论有一个明显的局限性——它无法将引力纳入框架。

在牛顿的经典力学中，引力是一种超距作用力，瞬间传播，不需要任何媒介。太阳和地球之间的引力似乎可以"穿越"真空，瞬间作用于对方。这在直觉上很难接受，但更重要的是，这与狭义相对论的基本假设相矛盾——任何信号或相互作用的传播速度都不能超过光速。

爱因斯坦花了整整十年时间来解决这个问题。1907年，他提出了著名的"等效原理"（Equivalence Principle）的雏形：在足够小的时空区域内，引力场无法与加速参考系区分开来。这个看似简单的洞见，开启了通向广义相对论的大门。

### 核心思想：时空是弯曲的

想象一下这个场景：一个小球在光滑的表面上滚动。如果表面是平的，小球会沿直线运动。但如果表面是弯曲的——比如一个马鞍形或者球面——小球的轨迹就会弯曲。在牛顿力学中，我们会说这是因为有一个"力"作用在小球上。

但爱因斯坦有一个更深刻的想法：也许根本没有什么"引力"，小球只是沿着弯曲表面上的"直线"运动。在四维时空中，自由下落的物体沿测地线（geodesic）运动——这是弯曲空间中最直的曲线。

这就是广义相对论的核心思想：**引力不是一种力，而是时空弯曲的几何表现**。物质告诉时空如何弯曲，时空告诉物质如何运动。

### 这篇文章的目标

在接下来的篇幅中，我将带领大家从最基本的概念开始，一步一步地构建广义相对论的数学框架。我们会学到：

1. **张量分析**：描述物理规律的语言
2. **黎曼几何**：弯曲时空的数学描述
3. **测地线方程**：自由粒子在弯曲时空中的运动
4. **爱因斯坦场方程**：物质如何弯曲时空
5. **史瓦西解**：最简单的黑洞解

让我们开始这段旅程。

---

## 第一章：曲线坐标系与张量

### 1.1 为什么要用曲线坐标系？

在欧几里得空间中，我们通常使用直角坐标系。直线就是坐标轴平行的线，角度可以用点积来计算。然而，在弯曲空间或研究广义坐标变换时，直角坐标系往往不是最方便的选择。

想象一个球面。球面上没有"直线"（大圆除外），也没有全局的直角坐标系。任何尝试在球面上定义坐标网格的努力都会在某些地方遇到奇点（比如经线的汇聚点）。这迫使我们使用曲线坐标系。

设我们在 $n$ 维空间中有一个曲线坐标系 $\{x^1, x^2, \dots, x^n\}$。空间中的每个点可以用这 $n$ 个坐标值来表示。反过来，每个坐标值 $\{x^i\}$ 对应空间中的一个点。

### 1.2 基向量与坐标变换

在曲线坐标系中，我们需要引入**局部基向量**的概念。考虑一个从原点出发的位移向量：

$$\mathbf{r} = x^1 \mathbf{e}_1 + x^2 \mathbf{e}_2 + \dots + x^n \mathbf{e}_n$$

在直角坐标系中，基向量 $\mathbf{e}_i$ 是常向量。但在曲线坐标系中，基向量会随位置变化。

**切向量**（tangent vector）定义为坐标线的切向：

$$\mathbf{e}_i = \frac{\partial \mathbf{r}}{\partial x^i}$$

这 $n$ 个向量 $\{\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_n\}$ 构成了该点的**协变基**（covariant basis）或**自然基**。

它们的**对偶基**（dual basis）$\{\mathbf{e}^1, \mathbf{e}^2, \dots, \mathbf{e}^n\}$ 满足：

$$\mathbf{e}^i \cdot \mathbf{e}_j = \delta^i_j = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

### 1.3 度规张量：测量距离的工具

在黎曼几何中，**度规张量**（metric tensor）是最基本的对象。它告诉我们如何在给定坐标系中测量距离和角度。

无穷小位移 $d\mathbf{r}$ 的长度为：

$$ds^2 = d\mathbf{r} \cdot d\mathbf{r}$$

展开这个表达式：

$$ds^2 = \left(\sum_i \mathbf{e}_i dx^i\right) \cdot \left(\sum_j \mathbf{e}_j dx^j\right) = \sum_{i,j} (\mathbf{e}_i \cdot \mathbf{e}_j) dx^i dx^j$$

定义**度规张量的分量**：

$$g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j$$

于是我们得到：

$$ds^2 = g_{ij} dx^i dx^j$$

这就是著名的**线元**（line element）表达式。注意这里使用了**爱因斯坦求和约定**：重复指标自动求和。

度规张量是一个对称的 $(0,2)$ 型张量：

$$g_{ij} = g_{ji}$$

它决定了空间的全部几何性质。

### 1.4 张量的定义与运算

在广义相对论中，物理定律必须用**张量方程**来表述，因为张量在坐标变换下具有确定的变换规律。

**张量的定义**：一个 $(k, l)$ 型张量 $T$ 是一个多重线性映射：

$$T: \underbrace{\mathcal{V}^* \times \cdots \times \mathcal{V}^*}_{k \text{ 个}} \times \underbrace{\mathcal{V} \times \cdots \times \mathcal{V}}_{l \text{ 个}} \to \mathbb{R}$$

其中 $\mathcal{V}$ 是切空间，$\mathcal{V}^*$ 是余切空间。

在分量形式中，张量有 $k$ 个上指标（逆变指标）和 $l$ 个下指标（协变指标）：

$$T^{i_1 i_2 \dots i_k}_{j_1 j_2 \dots j_l}$$

**张量的基本运算**：

1. **张量积**（Tensor Product）：将两个张量组合成更高阶的张量

   $$(A \otimes B)^{i_1 \dots i_k j_1 \dots j_m}_{l_1 \dots l_n} = A^{i_1 \dots i_k}_{l_1 \dots l_n} B^{j_1 \dots j_m}$$

2. **缩并**（Contraction）：将一个上指标和一个下指标求和

   $$(T)^{i_1 \dots i_{k-1}}_{j_1 \dots j_{l-1}} = T^{i_1 \dots i_{k-1} m}_{j_1 \dots j_{l-1} m}$$

3. **提升与下降指标**（Raising and Lowering Indices）：使用度规张量

   $$T^i = g^{ij} T_j, \quad T_i = g_{ij} T^j$$

### 1.5 克里斯托费尔符号

当我们对张量进行微分时，会遇到一个微妙的问题。在欧几里得空间中，偏导数 $\partial_i V^j = \frac{\partial V^j}{\partial x^i}$ 是一个张量。但在弯曲空间中，普通偏导数不再具有张量的变换性质。

**克里斯托费尔符号**（Christoffel symbols）$\Gamma^k_{ij}$ 是解决这个问题工具。它们描述了坐标系基向量随位置的变化率。

**定义**（从度规导出）：

$$\Gamma^k_{ij} = \frac{1}{2} g^{kl} \left( \frac{\partial g_{il}}{\partial x^j} + \frac{\partial g_{jl}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^l} \right)$$

这个公式称为**第一类克里斯托费尔符号**的变体。也可以写成：

$$\Gamma^k_{ij} = \Gamma^k_{ji}$$

克里斯托费尔符号不是张量！它们在坐标变换下的行为比较复杂。

**对称性**：

$$\Gamma^k_{ij} = \Gamma^k_{ji}$$

这个对称性源于度规张量的对称性 $g_{ij} = g_{ji}$。

### 1.6 协变导数：张量的微分

现在我们终于可以定义**协变导数**（covariant derivative），这是张量分析中最重要的运算。

对于一个逆变向量 $V^i$，协变导数为：

$$\nabla_j V^i = \frac{\partial V^i}{\partial x^j} + \Gamma^i_{jk} V^k$$

对于一个协变向量 $V_i$：

$$\nabla_j V_i = \frac{\partial V_i}{\partial x^j} - \Gamma^k_{ij} V_k$$

对于一般的 $(k, l)$ 型张量 $T^{i_1 \dots i_k}_{j_1 \dots j_l}$，协变导数为：

$$\nabla_m T^{i_1 \dots i_k}_{j_1 \dots j_l} = \frac{\partial T^{i_1 \dots i_k}_{j_1 \dots j_l}}{\partial x^m} + \sum_{p=1}^k \Gamma^{i_p}_{mn} T^{i_1 \dots n \dots i_k}_{j_1 \dots j_l} - \sum_{q=1}^l \Gamma^n_{jm} T^{i_1 \dots i_k}_{j_1 \dots n \dots j_l}$$

**协变导数的重要性质**：

1. 协变导数是张量
2. $\nabla_i g_{jk} = 0$（度规兼容性）
3. $\nabla_i \delta^j_k = 0$（克罗内克δ的协变导数为零）

---

## 第二章：黎曼曲率张量

### 2.1 什么是曲率？

曲率是描述空间"弯曲程度"的量。在二维曲面中，我们可以用高斯曲率来描述。但对于四维时空，我们需要更一般的数学工具——**黎曼曲率张量**（Riemann curvature tensor）。

考虑一个向量 $V^i$ 沿一个闭合路径平移。当它回到起点时，可能会指向不同的方向。这正是曲率的表现。

### 2.2 曲率张量的定义

黎曼曲率张量是 $(1, 3)$ 型张量，定义为：

$$R^i_{jkl} = \frac{\partial \Gamma^i_{jl}}{\partial x^k} - \frac{\partial \Gamma^i_{jk}}{\partial x^l} + \Gamma^i_{km} \Gamma^m_{jl} - \Gamma^i_{lm} \Gamma^m_{jk}$$

这个定义可能看起来有些复杂，让我们理解它的几何意义。

**另一种写法**（更直观）：

$$R(V, W)U = \nabla_V \nabla_W U - \nabla_W \nabla_V U - \nabla_{[V, W]} U$$

其中 $R(V, W)U$ 是一个算子，它衡量当我们沿着两个向量场 $V$ 和 $W$ 依次进行平行移动后，向量 $U$ 的变化。

### 2.3 对称性与比安基恒等式

黎曼曲率张量具有丰富的对称性：

**对称性**：

$$R_{ijkl} = -R_{jikl} = -R_{ijlk} = R_{klij}$$

第一对指标和第二对指标内部是反对称的，两对之间是对称的。

**比安基恒等式**（Bianchi Identity）：

$$\nabla_{[i} R_{jk]l}^m = 0$$

这是微分几何中最重要的恒等式之一，稍后我们会看到它在推导爱因斯坦方程中的作用。

### 2.4 里奇张量与标量曲率

从黎曼曲率张量，我们可以收缩指标得到几个重要的几何量。

**里奇张量**（Ricci Tensor）是 $(0, 2)$ 型张量：

$$R_{jl} = R^i_{jil} = R^i_{jli}$$

或者明确写出：

$$R_{jl} = \frac{\partial \Gamma^i_{jl}}{\partial x^i} - \frac{\partial \Gamma^i_{jj}}{\partial x^l} + \Gamma^i_{ik} \Gamma^k_{jl} - \Gamma^i_{lk} \Gamma^k_{ji}$$

**标量曲率**（Scalar Curvature）是里奇张量的缩并：

$$R = g^{jl} R_{jl}$$

它是一个标量（不变量），代表时空的"总曲率"。

### 2.5 爱因斯坦张量

**爱因斯坦张量**（Einstein Tensor）定义为：

$$G_{jl} = R_{jl} - \frac{1}{2} g_{jl} R$$

它是最重要的几何张量之一，原因稍后会清晰。

**重要性质**：爱因斯坦张量满足**散度为零**的条件：

$$\nabla^j G_{jl} = 0$$

这个性质可以通过比安基恒等式严格证明。它是推导爱因斯坦场方程的关键。

---

## 第三章：测地线方程——自由粒子的运动

### 3.1 什么是测地线？

在日常生活中，"直线"是两点之间最短的路径。在弯曲空间中，这个概念需要推广——**测地线**（geodesic）是弯曲空间中的"最直的曲线"。

从物理角度看，测地线是自由粒子在引力场中的运动轨迹。在广义相对论中，这正是我们描述行星运动、光线偏折等现象的基础。

### 3.2 变分原理

我们用**变分原理**来推导测enodesic方程。这是物理学中最强大的方法之一。

考虑粒子从时空点 $A$ 运动到 $B$。定义**世界线**的参数化：

$$x^i = x^i(\lambda), \quad \lambda_1 \leq \lambda \leq \lambda_2$$

粒子的**固有时**（proper time）$\tau$ 定义为：

$$d\tau = \sqrt{-g_{ij} dx^i dx^j} = \sqrt{-g_{ij} \frac{dx^i}{d\lambda} \frac{dx^j}{d\lambda}} d\lambda$$

对于类时世界线，$g_{ij} dx^i dx^j < 0$，所以我们取负号使 $\tau$ 为正。

**作用量**（action）定义为固有时：

$$S = \int_A^B d\tau = \int_{\lambda_1}^{\lambda_2} \sqrt{-g_{ij} \dot{x}^i \dot{x}^j} \, d\lambda$$

其中 $\dot{x}^i = \frac{dx^i}{d\lambda}$。

根据**最小作用量原理**，真实的世界线使作用量取极值：

$$\delta S = 0$$

### 3.3 推导测地线方程

现在我们来计算变分 $\delta S = 0$ 导致的运动方程。

被积函数是：

$$L = \sqrt{-g_{ij} \dot{x}^i \dot{x}^j}$$

**简化技巧**：由于被积函数是 $\dot{x}^i$ 的齐次函数，我们可以使用另一个等价的作用量：

$$S' = \frac{1}{2} \int g_{ij} \dot{x}^i \dot{x}^j \, d\lambda$$

这个作用量与原作用量有相同的极值曲线（测地线）。方便之处在于它避免了平方根。

现在应用**欧拉-拉格朗日方程**：

$$\frac{d}{d\lambda} \left( \frac{\partial L}{\partial \dot{x}^k} \right) - \frac{\partial L}{\partial x^k} = 0$$

对于 $L' = \frac{1}{2} g_{ij} \dot{x}^i \dot{x}^j$：

$$\frac{\partial L'}{\partial \dot{x}^k} = g_{kj} \dot{x}^j$$

$$\frac{\partial L'}{\partial x^k} = \frac{1}{2} \frac{\partial g_{ij}}{\partial x^k} \dot{x}^i \dot{x}^j$$

代入欧拉-拉格朗日方程：

$$\frac{d}{d\lambda} (g_{kj} \dot{x}^j) - \frac{1}{2} \frac{\partial g_{ij}}{\partial x^k} \dot{x}^i \dot{x}^j = 0$$

展开第一项：

$$g_{kj} \ddot{x}^j + \frac{\partial g_{kj}}{\partial x^l} \dot{x}^l \dot{x}^j - \frac{1}{2} \frac{\partial g_{ij}}{\partial x^k} \dot{x}^i \dot{x}^j = 0$$

重新整理指标（将 $l$ 换成 $i$）：

$$g_{kj} \ddot{x}^j + \left( \frac{\partial g_{ki}}{\partial x^j} + \frac{\partial g_{kj}}{\partial x^i} - \frac{1}{2} \frac{\partial g_{ij}}{\partial x^k} \right) \dot{x}^i \dot{x}^j = 0$$

用度规张量 $g^{km}$ 乘以两边以"提升"指标：

$$\ddot{x}^m + \Gamma^m_{ij} \dot{x}^i \dot{x}^j = 0$$

其中：

$$\Gamma^m_{ij} = \frac{1}{2} g^{km} \left( \frac{\partial g_{ki}}{\partial x^j} + \frac{\partial g_{kj}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^k} \right)$$

**这就是测地线方程！**

它告诉我们：自由粒子在弯曲时空中沿测地线运动，其轨迹由克里斯托费尔符号决定。

### 3.4 牛顿极限：恢复经典引力

为了验证我们的理论是否正确，让我们看看在弱场、低速极限下，测地线方程如何退化为牛顿的运动方程。

**弱场近似**：度规张量接近闵可夫斯基度规 $\eta_{\mu\nu}$：

$$g_{\mu\nu} = \eta_{\mu\nu} + h_{\mu\nu}, \quad |h_{\mu\nu}| \ll 1$$

**低速近似**：$v \ll c$，选择参数 $\lambda = t$（坐标时）。

我们只考虑时间-空间分量 $i=0$（时间）和 $i=1,2,3$（空间）。

对于静态引力场，度规与时间无关，且 $\dot{x}^0 = 1$。

测地线方程的空间分量变为：

$$\frac{d^2 x^i}{dt^2} + \Gamma^i_{00} = 0$$

计算 $\Gamma^i_{00}$（使用 $g_{00} \approx -(1 + 2\phi)$，其中 $\phi$ 是牛顿引力势）：

$$\Gamma^i_{00} \approx -\frac{1}{2} g^{ii} \frac{\partial g_{00}}{\partial x^i} = \frac{\partial \phi}{\partial x^i}$$

因此：

$$\frac{d^2 \mathbf{x}}{dt^2} = -\nabla \phi$$

**这正是牛顿引力场中的运动方程！**

从几何角度看，引力势 $\phi$ 与度规分量 $g_{00}$ 直接相关：

$$g_{00} \approx -(1 + 2\phi)$$

这建立了广义相对论与牛顿引力理论之间的对应关系。

---

## 第四章：爱因斯坦场方程——物质如何弯曲时空

### 4.1 从直觉到方程

现在我们面临一个根本性的问题：**物质（能量-动量）如何决定时空的弯曲？**

爱因斯坦花了数年时间探索这个问题的答案。1915年，他终于找到了正确的方程。核心思想可以概括为：

> 时空的弯曲由物质-能量分布决定。

我们需要找到一个几何量（描述弯曲）和一个物理量（描述物质）之间的等式。

### 4.2 能量-动量张量

**能量-动量张量**（energy-momentum tensor）$T_{\mu\nu}$ 描述了物质-能量的分布和流动。

它的物理意义：
- $T_{00}$：能量密度
- $T_{0i}$：能量流密度（动量密度）
- $T_{i0}$：动量流密度（能流）
- $T_{ij}$：动量流密度（应力）

**守恒定律**：能量-动量必须守恒：

$$\nabla^\mu T_{\mu\nu} = 0$$

这对应于经典物理中的连续性方程和动量守恒方程。

### 4.3 候选几何量

我们需要一个几何张量来与 $T_{\mu\nu}$ 匹配。可能的选择：

1. **黎曼曲率张量** $R_{\mu\nu\rho\sigma}$：太复杂，有20个独立分量
2. **里奇张量** $R_{\mu\nu}$：有10个独立分量
3. **爱因斯坦张量** $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} g_{\mu\nu}R$：有10个独立分量，且散度为零

爱因斯坦张量是最好的选择，因为它的散度为零：

$$\nabla^\mu G_{\mu\nu} = 0$$

这与能量-动量守恒 $\nabla^\mu T_{\mu\nu} = 0$ 完全匹配！

### 4.4 爱因斯坦场方程的推导

**基本假设**：场方程应该是：

$$G_{\mu\nu} = \kappa T_{\mu\nu}$$

其中 $\kappa$ 是某个常数。

为了确定 $\kappa$，我们考虑**弱场极限**和**低速极限**。

在牛顿引力理论中，牛泊松方程是：

$$\nabla^2 \phi = 4\pi G \rho$$

其中 $G$ 是牛顿引力常数，$\rho$ 是质量密度。

在狭义相对论中，能量密度 $\rho c^2$ 对应于 $T_{00}$。

考虑静态、弱场近似下的爱因斯坦场方程。度规近似为：

$$g_{00} \approx -(1 + 2\phi), \quad g_{ij} \approx \delta_{ij}$$

里奇张量的时间-时间分量：

$$R_{00} \approx -\frac{1}{2} \nabla^2 g_{00} = \nabla^2 \phi$$

爱因斯坦张量：

$$G_{00} = R_{00} - \frac{1}{2} g_{00} R \approx \nabla^2 \phi - \frac{1}{2}(-1)(-\nabla^2 g_{00}) \approx \nabla^2 \phi$$

场方程 $G_{00} = \kappa T_{00}$ 变为：

$$\nabla^2 \phi = \kappa \frac{1}{2} \rho c^2$$

（这里 $T_{00} = \frac{1}{2} \rho c^2$，因子取决于具体定义）

与牛顿方程 $\nabla^2 \phi = 4\pi G \rho$ 比较，得到：

$$\kappa = \frac{8\pi G}{c^4}$$

### 4.5 最终形式

**爱因斯坦场方程**：

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$$

或者写成更紧凑的形式：

$$R_{\mu\nu} - \frac{1}{2} g_{\mu\nu} R + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$$

其中：
- $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} g_{\mu\nu}R$ 是爱因斯坦张量
- $\Lambda$ 是**宇宙学常数**（爱因斯坦最初引入，后来称其为"最大的错误"，但现在我们知道它可能代表暗能量）
- $G$ 是牛顿引力常数
- $c$ 是光速

**物理意义**：
- 左边描述时空的几何性质（曲率）
- 右边描述物质-能量的分布
- 方程表明：**物质告诉时空如何弯曲，时空告诉物质如何运动**

### 4.6 真空场方程

在没有物质的区域，$T_{\mu\nu} = 0$，场方程变为：

$$R_{\mu\nu} = 0$$

这个方程描述了没有物质时的真空时空几何。在下一章中，我们会看到它的解——史瓦西解。

---

## 第五章：史瓦西解——最简单的黑洞

### 5.1 问题的设定

现在我们要求解爱因斯坦场方程的最简单解：一个静态、球对称、无旋转、不带电的星体外部的时空。

**对称性假设**：
1. **静态**：不随时间变化
2. **球对称**：在空间旋转下不变

这些假设极大地限制了度规的形式。

### 5.2 球对称度规的最一般形式

在球对称坐标系中，最一般的静态球对称度规为：

$$ds^2 = -e^{2\alpha(r)} c^2 dt^2 + e^{2\beta(r)} dr^2 + r^2 (d\theta^2 + \sin^2\theta \, d\phi^2)$$

其中 $\alpha(r)$ 和 $\beta(r)$ 是待确定的径向函数。

通过坐标变换，我们可以将度规简化为更标准的形式。设：

$$e^{2\beta(r)} = \frac{1}{1 - \frac{r_s}{r}}$$

其中 $r_s$ 是一个常数（稍后会证明它就是史瓦西半径）。

最终的标准形式是：

$$ds^2 = -\left(1 - \frac{r_s}{r}\right) c^2 dt^2 + \frac{dr^2}{1 - \frac{r_s}{r}} + r^2 (d\theta^2 + \sin^2\theta \, d\phi^2)$$

这就是**史瓦西度规**（Schwarzschild metric）。

### 5.3 计算曲率张量

为了验证这个度规满足爱因斯坦场方程，我们需要计算里奇张量。

首先计算非零的克里斯托费尔符号。由于度规的对称性，我们只需要计算一部分。

**步骤1：计算度规分量**

$$g_{tt} = -\left(1 - \frac{r_s}{r}\right), \quad g_{rr} = \left(1 - \frac{r_s}{r}\right)^{-1}, \quad g_{\theta\theta} = r^2, \quad g_{\phi\phi} = r^2 \sin^2\theta$$

**步骤2：计算克里斯托费尔符号**

$$\Gamma^t_{tr} = \Gamma^t_{rt} = \frac{1}{2} g^{tt} \frac{\partial g_{tt}}{\partial r} = \frac{r_s}{2r(r - r_s)}$$

$$\Gamma^r_{tt} = \frac{1}{2} g^{rr} \frac{\partial g_{tt}}{\partial r} = \frac{c^2 r_s}{2r^2} \left(1 - \frac{r_s}{r}\right)$$

$$\Gamma^r_{rr} = -\frac{1}{2} g^{rr} \frac{\partial g_{rr}}{\partial r} = -\frac{r_s}{2r(r - r_s)}$$

$$\Gamma^r_{\theta\theta} = -r \left(1 - \frac{r_s}{r}\right)$$

$$\Gamma^r_{\phi\phi} = -r \left(1 - \frac{r_s}{r}\right) \sin^2\theta$$

$$\Gamma^\theta_{r\theta} = \Gamma^\theta_{\theta r} = \frac{1}{r}$$

$$\Gamma^\theta_{\phi\phi} = -\sin\theta \cos\theta$$

$$\Gamma^\phi_{r\phi} = \Gamma^\phi_{\phi r} = \frac{1}{r}$$

$$\Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta} = \cot\theta$$

**步骤3：计算里奇张量**

对于真空场方程 $R_{\mu\nu} = 0$，我们只需要验证里奇张量的所有分量都为零。

以 $R_{tt}$ 为例：

$$R_{tt} = \frac{\partial \Gamma^t_{tt}}{\partial x^t} - \frac{\partial \Gamma^t_{t\lambda}}{\partial x^\lambda} + \Gamma^t_{\lambda\sigma} \Gamma^\lambda_{tt} - \Gamma^t_{t\lambda} \Gamma^\lambda_{t\sigma}$$

由于度规是静态的，$\Gamma^t_{tt} = 0$。经过计算：

$$R_{tt} = \frac{c^2 r_s}{2r^3} \left(1 - \frac{r_s}{r}\right) - \frac{c^2 r_s^2}{2r^4} = 0$$

其他分量的计算类似。最终我们发现：

$$R_{\mu\nu} = 0$$

**史瓦西度规确实是真空爱因斯坦场方程的解！**

### 5.4 史瓦西半径与黑洞

在史瓦西度规中，有一个特殊的半径：

$$r_s = \frac{2GM}{c^2}$$

这就是**史瓦西半径**（Schwarzschild radius）。

在这个半径处，度规分量出现奇异性：
- $g_{tt} = 0$：时间坐标"停止"
- $g_{rr} \to \infty$：径向坐标"拉伸到无穷"

这就是**事件视界**（event horizon）。任何物质（包括光）一旦越过这个半径，就无法逃脱——这就是黑洞。

### 5.5 经典验证：水星近日点进动

史瓦西解的一个著名验证是解释水星近日点的进动。

**观测事实**：水星的轨道并不是闭合的椭圆。每转一圈，近日点会移动大约43角秒/世纪。这个数值用牛顿力学无法完全解释。

**广义相对论的解释**：由于太阳的引力场（用史瓦西度规描述），行星的轨道会发生微小的偏移。

计算近日点进动率：

$$\Delta \phi = \frac{6\pi GM}{c^2 a(1 - e^2)}$$

其中：
- $a$ 是半长轴
- $e$ 是离心率
- $M$ 是太阳质量

对于水星：
- $a = 5.79 \times 10^{10}$ m
- $e = 0.2056$
- $M = 1.989 \times 10^{30}$ kg

代入计算：

$$\Delta \phi \approx 42.98 \text{ 角秒/世纪}$$

**与观测值43角秒/世纪完美吻合！**

这是广义相对论最早和最精确的实验验证之一。

### 5.6 光线偏折

另一个著名预言是：光线经过太阳附近时会发生偏折。

**计算**：考虑光子的测地线。对于史瓦西度规，光线的偏折角为：

$$\Delta \theta = \frac{4GM}{c^2 b}$$

其中 $b$ 是光线到太阳中心的最近距离（瞄准参数）。

当光线刚好掠过太阳表面时（$b = R_\odot$）：

$$\Delta \theta \approx 1.75 \text{ 弧秒}$$

1919年，爱丁顿率领的日全食观测队测量到的偏折角约为 $1.98 \pm 0.12$ 弧秒（第一次观测）和 $1.61 \pm 0.30$ 弧秒（第二次观测）。考虑到实验误差，这与理论预言基本一致。

这一发现使爱因斯坦一夜成名。

---

## 第六章：进一步探索

### 6.1 克尔度规与旋转黑洞

史瓦西解描述的是静态、球对称的黑洞。但真实的天体（如恒星、星系中心）通常是旋转的。

1963年，克尔（Roy Kerr）找到了旋转黑洞的精确解——**克尔度规**（Kerr metric）。它的形式更加复杂：

$$ds^2 = -\left(1 - \frac{2Mr - a^2}{\Sigma}\right) dt^2 - \frac{2a(2Mr - a^2)\sin^2\theta}{\Sigma} dt d\phi + \frac{\Sigma}{\Delta} dr^2 + \Sigma d\theta^2 + \frac{(2Mr - a^2)^2 - a^2\Delta\sin^2\theta}{\Sigma} \sin^2\theta d\phi^2$$

其中：

$$\Sigma = r^2 + a^2 \cos^2\theta, \quad \Delta = r^2 - 2Mr + a^2, \quad a = \frac{J}{Mc}$$

克尔黑洞有两个重要特征：
- **事件视界**：$r = M \pm \sqrt{M^2 - a^2}$
- **能层**（ergosphere）：一个可以提取旋转能量的区域

### 6.2 引力波

爱因斯坦场方程的另一个预言是**引力波**——时空涟漪以光速传播。

1916年，爱因斯坦预言了引力波的存在。2015年9月14日，LIGO首次直接探测到两个黑洞合并产生的引力波（GW150914），标志着引力波天文学的诞生。

引力波携带的能量可以用下面的公式描述（近似）：

$$P \approx \frac{G}{5c^5} \left\langle \dddot{Q}_{ij} \dddot{Q}^{ij} \right\rangle$$

其中 $Q_{ij}$ 是四极矩张量。

### 6.3 宇宙学解

将爱因斯坦场方程应用到整个宇宙，我们得到**弗里德曼-勒梅特-罗伯逊-沃尔克度规**（FLRW度规）：

$$ds^2 = -c^2 dt^2 + a^2(t) \left[ \frac{dr^2}{1 - kr^2} + r^2(d\theta^2 + \sin^2\theta d\phi^2) \right]$$

其中 $a(t)$ 是宇宙标度因子，$k$ 是空间曲率（$k = -1, 0, 1$）。

结合宇宙学常数，FLRW度规的解描述了宇宙的膨胀，包括大爆炸和可能的最终命运。

---

## 结语：理论的美丽与局限

### 广义相对论的美

回顾我们走过的旅程，从张量分析到测地线方程，再到爱因斯坦场方程，最后到史瓦西解，我们见证了一个完整、优雅、自洽的理论体系的诞生。

广义相对论的美体现在多个层面：

1. **几何之美**：引力不再是"力"，而是时空的几何性质。物质与时空相互依存，构成一个统一的整体。

2. **数学之美**：从变分原理导出的测地线方程，从张量分析构建的爱因斯坦场方程，每一个公式都是数学之美的体现。

3. **预言之美**：黑洞、引力波、宇宙膨胀——这些在当时看似疯狂的预言，一个接一个被实验验证。

### 理论的局限

然而，广义相对论并非终极理论：

1. **与量子力学的矛盾**：在普朗克尺度（$10^{-35}$ m），广义相对论预言的时空奇点可能是理论失效的信号。量子引力理论（如弦论、圈量子引力）仍在发展中。

2. **暗物质与暗能量**：宇宙学观测表明，普通物质只占宇宙总能量的大约5%。暗物质和暗能量的本质仍是未解之谜。

3. **奇点定理**：霍金和彭罗斯证明，在非常一般的条件下，时空奇点是不可避免的。但奇点处的物理定律是什么？我们还不知道。

### 给读者的话

如果你读到这里，恭喜你！你已经完成了一段非凡的旅程——从欧几里得空间的概念出发，到理解宇宙中最神秘的天体之一：黑洞。

广义相对论不仅仅是一门物理学分支，它是一种看待世界的方式。它告诉我们：空间不是固定的舞台，而是可以被物质弯曲的动态实体；时间不是均匀流逝的河流，而是与空间交织在一起的织物。

在20世纪初，爱因斯坦用他深刻的物理直觉和精湛的数学技巧，开创了这门革命性的理论。一个世纪后，我们仍然在探索它的边界，验证它的预言，发现它的美。

也许，下一个改变人类对宇宙认知的发现，就在你的手中。

---

## 附录：重要公式汇总

### 基本定义

**度规张量**：
$$ds^2 = g_{\mu\nu} dx^\mu dx^\nu$$

**克里斯托费尔符号**：
$$\Gamma^\lambda_{\mu\nu} = \frac{1}{2} g^{\lambda\sigma} (\partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma} - \partial_\sigma g_{\mu\nu})$$

**黎曼曲率张量**：
$$R^\rho_{\sigma\mu\nu} = \partial_\mu \Gamma^\rho_{\nu\sigma} - \partial_\nu \Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$$

### 核心方程

**测地线方程**：
$$\frac{d^2 x^\mu}{d\tau^2} + \Gamma^\mu_{\nu\lambda} \frac{dx^\nu}{d\tau} \frac{dx^\lambda}{d\tau} = 0$$

**爱因斯坦场方程**：
$$R_{\mu\nu} - \frac{1}{2} g_{\mu\nu} R + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$$

**史瓦西度规**：
$$ds^2 = -\left(1 - \frac{2GM}{c^2 r}\right) c^2 dt^2 + \frac{dr^2}{1 - \frac{2GM}{c^2 r}} + r^2 d\Omega^2$$

---

*本文旨在为有一定数学基础的读者提供广义相对论的入门导引。更深入的学习建议参考专业教材，如 Sean Carroll 的《Spacetime and Geometry》、Landau 和 Lifshitz 的《The Classical Theory of Fields》等。*
