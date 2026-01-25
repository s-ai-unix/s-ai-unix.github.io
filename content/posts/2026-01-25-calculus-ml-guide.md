---
title: "微积分：从变化率到神经网络梯度的完整旅程"
date: 2026-01-25T18:00:00+08:00
draft: false
description: "这是一篇关于微积分的系统综述，从变化率的几何直观到反向传播的数学推导，全面阐述微积分在现代人工智能中的核心作用。"
categories: ["数学", "机器学习", "深度学习"]
tags: ["综述", "机器学习", "深度学习", "数��", "算法"]
cover:
    image: "images/covers/calculus-ml-journey.jpg"
    alt: "微积分的几何美感"
    caption: "微积分：描述变化的数学语言"
math: true
---

## 引言：微积分如何驱动现代人工智能

想象你站在山顶,想要找到下山的路径。你会观察周围的地形,选择最陡峭的方向迈出一步,然后重复这个过程。这就是微积分最核心的思想——用局部变化来理解全局行为。

17世纪,牛顿和莱布尼茨独立发明了微积分,用一种统一的语言描述了变化和累积。三个世纪后,当我们训练神经网络时,本质上也是在用同样的思想:计算损失函数在某一点的变化率(梯度),然后沿着负梯度方向调整参数,让损失函数的值不断降低。

在这篇文章中,我们将踏上一段从基础理论到前沿应用的完整旅程。我们会从导数的几何直观出发,理解微分如何线性化复杂问题,然后逐步深入到机器学习的梯度下降、深度学习的反向传播,以及变分法和随机微积分等高级主题。

我们不只学习"怎么做",更重要的是理解"为什么"——为什么梯度下降能找到最小值?为什么反向传播要这样计算?为什么 ReLU 能解决梯度消失问题?

让我们开始这段旅程。

## 第一部分:微积分基础理论

### 1. 导数的本质

#### 1.1 从变化率到瞬时变化��

人类很早就意识到一个问题:如何描述物体运动的快慢?如果一辆车2小时行驶了100公里,我们说它的平均速度是50公里/小时。但这只是平均值——它在某个时刻可能加速,在另一时刻可能减速。

**核心问题**:如何描述物体在某一瞬间的变化率?

几何直观给了我们启示。考虑函数 $f(x) = x^2$ 的图像。如果我们想计算函数在 $x = 1$ 处的"瞬时变化率",可以用一条直线逼近曲线在这一点的形态。这条直线的斜率,就是导数。

![导数的几何意义：切线的斜率](/images/math/derivative-tangent.png)

*图1：导数的几何直观。蓝色曲线是 $f(x) = x^2$,橙色虚线是在 $x=1$ 处的切线,切线的斜率就是导数 $f'(1) = 2$。*

#### 1.2 严格定义:极限与 $\epsilon$-$\delta$

19世纪,数学家柯西和魏尔斯特拉斯给出了导数的严格定义。函数 $f(x)$ 在点 $x_0$ 处的导数为:
$$
f'(x_0) = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}
$$

这个定义告诉我们:导数是**差商的极限**。当 $\Delta x$ 趋近于0时,割线逐渐变成切线,平均变化率逐渐变成瞬时变化率。

**直观理解**:
- $\Delta x$ 是自变量的微小变化
- $f(x_0 + \Delta x) - f(x_0)$ 是函数值的相应变化
- 它们的比值是平均变化率
- 极限过程让"平均"变成"瞬时"

#### 1.3 导数的计算规则

从定义出发,我们可以推导出常用的求导法则:

**和的导数**:
$$
(f + g)' = f' + g'
$$

**积的导数**:
$$
(fg)' = f'g + fg'
$$

**链式法则**(最重要):
$$
\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)
$$

链式法则告诉我们:复合函数的导数是外函数的导数乘以内函数的导数。这个简单的规则将在神经网络的反向传播中发挥核心作用。

---

### 2. 微分与线性化

#### 2.1 微分的几何意义

导数描述的是函数在某一点的变化率,而微分则提供了一种**线性近似**的工具。

对于函数 $y = f(x)$,当 $x$ 从 $x_0$ 变化到 $x_0 + \Delta x$ 时,$y$ 的变化可以近似为:
$$
\Delta y \approx f'(x_0) \cdot \Delta x
$$

这被称为**一阶泰勒公式**或**线性近似**。它的几何意义是:在 $x_0$ 附近,我们用切线来代替曲线。

**为什么重要?**因为直线是最简单的函数。如果我们能把复杂函数局部线性化,就能用线性代数的工具处理非线性问题。

#### 2.2 多元函数的微分:梯度与方向导数

对于二元函数 $f(x, y)$,它在不同方向上的变化率可能不同。例如,在山上向北走和向东走,坡度会不一样。

**偏导数**是沿坐标轴方向的变化率:
$$
\frac{\partial f}{\partial x} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x, y) - f(x, y)}{\Delta x}
$$

**梯度**是将所有偏导数组合成向量:
$$
\nabla f = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{pmatrix}
$$

**关键性质**:梯度指向函数增长最快的方向,梯度的大小是该方向的变化率。

**方向导数**:沿单位向量 $\mathbf{u}$ 的变化率为:
$$
D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = \lVert \nabla f \rVert \cos\theta
$$

其中 $\theta$ 是梯度方向与 $\mathbf{u}$ 的夹角。当 $\theta = 0$ 时,方向导数最大,即梯度方向是**最速上升方向**。

#### 2.3 雅可比矩阵与链式法则

对于向量值函数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$,我们需要用**雅可比矩阵**来描述所有偏导数:
$$
J_{\mathbf{f}}(\mathbf{x}) = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
$$

**多元链式法则**:如果 $\mathbf{z} = \mathbf{f}(\mathbf{y})$ 且 $\mathbf{y} = \mathbf{g}(\mathbf{x})$,则:
$$
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}
$$

在矩阵形式中:
$$
J_{\mathbf{z}}(\mathbf{x}) = J_{\mathbf{z}}(\mathbf{y}) \cdot J_{\mathbf{y}}(\mathbf{x})
$$

这就是**反向传播算法**的数学基础!

---

### 3. 积分与累积

#### 3.1 从求和到黎曼积分

如果说导数研究的是变化,那么积分研究的就是累积。

考虑计算曲线下的面积。我们可以把区间 $[a, b]$ 分成 $n$ 个子区间,在每个子区间上用矩形面积近似曲线下面积,然后求和:
$$
S_n = \sum_{i=1}^{n} f(\xi_i) \Delta x_i
$$

当分割越来越细(最大的子区间长度趋于0)时,这个和的极限就是**定积分**:
$$
\int_a^b f(x) dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(\xi_i) \Delta x_i
$$

**几何意义**:定积分是曲线与 $x$ 轴之间的有向面积(在 $x$ 轴上方为正,下方为负)。

#### 3.2 微积分基本定理

牛顿和莱布尼茨的伟大发现是:微分和积分互为逆运算!

**微积分基本定理**:
$$
\int_a^b f'(x) dx = f(b) - f(a)
$$

这个定理告诉我们:要计算定积分,只需要找到原函数,然后计算端点值之差。它把复杂的求和极限问题变成了求原函数的问题,大大简化了计算。

#### 3.3 多重积分与变量替换

在机器学习中,我们经常遇到多元函数的积分。例如,计算两个随机变量的联合概率。

**二重积分**:
$$
\iint_D f(x, y) dx dy
$$

**变量替换公式**(重要!):
$$
\iint_D f(x, y) dx dy = \iint_{D'} f(x(u, v), y(u, v)) \left| \frac{\partial(x, y)}{\partial(u, v)} \right| du dv
$$

其中 $\left| \frac{\partial(x, y)}{\partial(u, v)} \right|$ 是**雅可比行列式**,它描述了变量替换时的"面积缩放因子"。

**应用**:在概率密度函数的变换中,雅可比行列式确保概率总和为1。

---

### 4. 级数与逼近

#### 4.1 泰勒展开:用多项式逼近函数

微积分最强大的工具之一是**泰勒展开**。它告诉我们:任何光滑函数都可以在某点附近用多项式逼近。

**一元泰勒公式**:
$$
f(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x - x_0)^2 + \cdots + \frac{f^{(n)}(x_0)}{n!}(x - x_0)^n + R_n
$$

其中 $R_n$ 是余项,描述了近似误差。

**几何直观**:
- 零阶近似:$f(x) \approx f(x_0)$(常数)
- 一阶近似:$f(x) \approx f(x_0) + f'(x_0)(x - x_0)$(切线)
- 二阶近似:考虑曲率,更精确
- 高阶近似:多项式越来越接近原函数

![泰勒级数逼近：多项式逐渐接近原函数](/images/math/taylor-approximation.png)

*图2:正弦函数的泰勒级数逼近。蓝色是原函数 $\sin(x)$,橙色、绿色、紫色分别是1阶、3阶、5阶泰勒展开。可以看到,阶数越高,多项式越接近原函数。*

#### 4.2 泰勒级数的收敛性

泰勒级数在什么情况下收敛到原函数?这取决于函数的性质:

- **解析函数**:在整个定义域上处处可微,泰勒级数收敛
- **光滑函数**:可能只在某个半径内收敛(收敛半径)
- **举例**:$e^x, \sin x, \cos x$ 的泰勒级数对所有 $x$ 都收敛

#### 4.3 在优化中的应用:牛顿法

泰勒展开为优化算法提供了理论基础。对于无约束优化问题 $\min f(x)$,我们可以用二阶泰勒展开:
$$
f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{f''(x)}{2}(\Delta x)^2
$$

对 $\Delta x$ 求导并令导数为零:
$$
f'(x) + f''(x)\Delta x = 0 \implies \Delta x = -\frac{f'(x)}{f''(x)}
$$

这就是**牛顿法**的迭代公式:
$$
x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}
$$

**优点**:
- 二阶收敛(接近最优点时非常快)
- 利用了函数的曲率信息

**缺点**:
- 需要计算二阶导数
- Hessian 矩阵可能不可逆

---

## 第二部分:机器学习中的微积分

### 1. 梯度下降法

#### 1.1 梯度的几何意义:最速下降方向

对于无约束优化问题:
$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$

**关键问题**:如何找到使 $f(\mathbf{x})$ 最小的 $\mathbf{x}$?

微积分告诉我们:梯度 $\nabla f(\mathbf{x})$ 指向函数增长最快的方向,因此**负梯度方向** $-\nabla f(\mathbf{x})$ 就是**最速下降方向**。

#### 1.2 梯度下降的推导

**直观推导**:
1. 在当前点 $\mathbf{x}_k$,计算梯度 $\nabla f(\mathbf{x}_k)$
2. 沿负梯度方向走一小步:
   $$
   \mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)
   $$
3. 重复直到收敛

其中 $\eta$ 是**学习率**(learning rate),控制步长大小。

**数学推导**:

考虑函数在 $\mathbf{x}_k$ 处的一阶泰勒展开:
$$
f(\mathbf{x}_k + \Delta \mathbf{x}) \approx f(\mathbf{x}_k) + \nabla f(\mathbf{x}_k)^	op \Delta \mathbf{x}
$$

我们想找到 $\Delta \mathbf{x}$ 使 $f$ 减小最多。设 $\lVert \Delta \mathbf{x} \rVert = \epsilon$(步长固定),则:
$$
\min_{\lVert \Delta \mathbf{x} \rVert = \epsilon} \nabla f(\mathbf{x}_k)^	op \Delta \mathbf{x}
$$

由柯西-施瓦茨不等式:
$$
\nabla f^T \Delta \mathbf{x} \geq -\lVert \nabla f \rVert \cdot \lVert \Delta \mathbf{x} \rVert = -\epsilon \lVert \nabla f \rVert
$$

当且仅当 $\Delta \mathbf{x}$ 与 $-\nabla f$ 同向时取等号。因此,**负梯度方向是最速下降方向**。

![梯度下降：沿着负梯度方向走到最低点](/images/math/gradient-descent-3d.png)

*图3:梯度下降在等高线地形中的轨迹。从橙色起点出发,沿着最陡峭的方向(垂直于等高线)一步步走到绿色的最低点。*

#### 1.3 学习率的选择

学习率 $\eta$ 是梯度下降中最重要的超参数:

- **太小**:收敛慢,需要很多步才能到达最优点
- **太大**:可能"冲过"最优点,甚至发散

**学习率衰减策略**:
$$
\eta_t = \frac{\eta_0}{1 + \lambda t}
$$

或指数衰减:
$$
\eta_t = \eta_0 e^{-\lambda t}
$$

#### 1.4 收敛性分析:强凸情况

假设 $f$ 是 $\mu$-强凸的,即:
$$
f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\lVert \mathbf{y} - \mathbf{x} \rVert^2
$$

且 $L$-光滑(梯度 Lipschitz 连续):
$$
\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert \leq L \lVert \mathbf{x} - \mathbf{y} \rVert
$$

**收敛速度**:
$$
f(\mathbf{x}_k) - f(\mathbf{x}^*) \leq \left(1 - \frac{\mu}{L}\right)^k [f(\mathbf{x}_0) - f(\mathbf{x}^*)]
$$

其中 $\kappa = L/\mu$ 是**条件数**。条件数越小,收敛越快。

#### 1.5 动量方法与自适应学习率

**问题**:梯度下降在"山谷"中会震荡,收敛慢。

**动量方法**(Momentum):
$$
\mathbf{v}_{k+1} = \beta \mathbf{v}_k + (1 - \beta)\nabla f(\mathbf{x}_k)
$$

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \mathbf{v}_{k+1}
$$

**几何解释**:动量累积了之前的梯度方向,有助于"冲过"平坦区域,减少震荡。

**AdaGrad**(自适应学习率):
$$
\mathbf{x}_{k+1, i} = \mathbf{x}_{k, i} - \frac{\eta}{\sqrt{G_{k, ii} + \epsilon}} \nabla f(\mathbf{x}_k)_i
$$

其中 $G_{k, ii} = \sum_{j=1}^k (\nabla f(\mathbf{x}_j)_i)^2$ 是历史梯度的平方和。

**思想**:对频繁更新的参数使用较小的学习率,对稀疏更新的参数使用较大的学习率。

---

### 2. 拉格朗日乘数法

#### 2.1 约束优化问题

许多机器学习问题带有约束,例如:
$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g(\mathbf{x}) = 0
$$

**拉格朗日乘数法**将约束优化转化为无约束优化。

#### 2.2 拉格朗日乘数的几何解释

考虑约束 $g(x, y) = 0$ 下优化 $f(x, y)$。

在最优解处,目标函数的等高线与约束曲线相切。这意味着它们的梯度平行:
$$
\nabla f = -\lambda \nabla g
$$

其中 $\lambda$ 是**拉格朗日乘数**。

**拉格朗日函数**:
$$
\mathcal{L}(x, y, \lambda) = f(x, y) + \lambda g(x, y)
$$

最优解满足:
$$
\frac{\partial \mathcal{L}}{\partial x} = 0, \quad \frac{\partial \mathcal{L}}{\partial y} = 0, \quad \frac{\partial \mathcal{L}}{\partial \lambda} = 0
$$

#### 2.3 KKT 条件

对于不等式约束:
$$
\min f(\mathbf{x}) \quad \text{s.t.} \quad g(\mathbf{x}) \leq 0
$$

**KKT 条件**(Karush-Kuhn-Tucker):
1. **平稳性**: $\nabla f(\mathbf{x}^*) + \sum_i \lambda_i \nabla g_i(\mathbf{x}^*) = 0$
2. **原始可行性**: $g_i(\mathbf{x}^*) \leq 0$
3. **对偶可行性**: $\lambda_i \geq 0$
4. **互补松弛**: $\lambda_i g_i(\mathbf{x}^*) = 0$

**互补松弛**告诉我们:如果约束不起作用($g_i(\mathbf{x}^*) < 0$),则 $\lambda_i = 0$;如果 $\lambda_i > 0$,则 $g_i(\mathbf{x}^*) = 0$(约束起作用)。

#### 2.4 在 SVM 中的应用

**硬间隔 SVM**:
$$
\min_{\mathbf{w}, b} \frac{1}{2}\lVert \mathbf{w} \rVert^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \forall i
$$

拉格朗日函数:
$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\lVert \mathbf{w} \rVert^2 - \sum_{i=1}^n \alpha_i [y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1]
$$

对 $\mathbf{w}$ 和 $b$ 求导并令为零:
$$
\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i, \quad \sum_{i=1}^n \alpha_i y_i = 0
$$

代回拉格朗日函数,得到**对偶问题**:
$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j
$$

这是一个二次规划问题,可以用标准优化工具求解。

---

### 3. 信息论中的微积分

#### 3.1 熵的定义与微分

**香农熵**:
$$
H(X) = -\sum_{i=1}^n p_i \log p_i
$$

其中 $p_i$ 是随机变量 $X$ 取第 $i$ 个值的概率。

**直观理解**:熵是**不确定性**的度量。熵越大,我们越不确定随机变量的取值。

**微分熵**(连续随机变量):
$$
h(X) = -\int_{-\infty}^{\infty} f(x) \log f(x) dx
$$

其中 $f(x)$ 是概率密度函数。

#### 3.2 交叉熵与 KL 散度

**交叉熵**:
$$
H(p, q) = -\sum_{i=1}^n p_i \log q_i
$$

其中 $p$ 是真实分布,$q$ 是预测分布。

**KL 散度**(Kullback-Leibler divergence):
$$
D_{KL}(p \parallel q) = \sum_{i=1}^n p_i \log \frac{p_i}{q_i}
$$

**关系**:
$$
H(p, q) = H(p) + D_{KL}(p \parallel q)
$$

最小化交叉熵等价于最小化 KL 散度,因为 $H(p)$ 是常数。

**为什么 KL 散度不能是对称的?**

考虑 $p = [1, 0]$,$q = [0.5, 0.5]$:
$$
D_{KL}(p \parallel q) = 1 \cdot \log \frac{1}{0.5} = \log 2
$$

但 $D_{KL}(q \parallel p)$ 是无穷大,因为 $\log(0/1)$ 无定义。

#### 3.3 最大熵原理

**思想**:在所有满足约束的概率分布中,选择熵最大的分布。

**应用**:语言模型。在满足历史观测约束下,选择最不确定的预测。

#### 3.4 在分类损失函数中的应用

**交叉熵损失**:
$$
L = -\sum_{i=1}^n \sum_{c=1}^C y_{ic} \log \hat{y}_{ic}
$$

其中 $y_{ic}$ 是真实标签,$\hat{y}_{ic}$ 是预测概率。

**梯度**:
$$
\frac{\partial L}{\partial z_j} = \hat{y}_j - y_j
$$

这个简洁的形式使得反向传播非常高效!

---

## 第三部分:深度学习中的微积分

### 1. 反向传播算法

#### 1.1 链式法则的矩阵形式

考虑一个简单的神经网络:
$$
\mathbf{z}^{(1)} = W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}
$$

$$
\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)})
$$

$$
\mathbf{z}^{(2)} = W^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}
$$

$$
\hat{\mathbf{y}} = \sigma(\mathbf{z}^{(2)})
$$

损失函数: $L = \frac{1}{2}\lVert \hat{\mathbf{y}} - \mathbf{y} \rVert^2$

**反向传播**:

计算输出层梯度:
$$
\frac{\partial L}{\partial \mathbf{z}^{(2)}} = (\hat{\mathbf{y}} - \mathbf{y}) \odot \sigma'(\mathbf{z}^{(2)})
$$

计算隐藏层梯度:
$$
\frac{\partial L}{\partial \mathbf{z}^{(1)}} = \left[(W^{(2)})^	op \frac{\partial L}{\partial \mathbf{z}^{(2)}}\right] \odot \sigma'(\mathbf{z}^{(1)})
$$

计算权重梯度:
$$
\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} (\mathbf{a}^{(1)})^	op $$

$$
\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} \mathbf{x}^T
$$

**关键观察**:梯度的计算遵循链式法则,从输出层反向传播到输入层。

#### 1.2 计算图与自动微分

现代深度学习框架使用**计算图**(computational graph)和**自动微分**(automatic differentiation)来计算梯度。

**计算图**:将计算分解为基本操作(加法、乘法、指数等),每个操作都是一个节点。

**前向传播**:从输入到输出,计算中间值。
**反向传播**:从输出到输入,使用链式法则计算梯度。

**两种自动微分模式**:
- **前向模式**:计算 $\frac{\partial f}{\partial x_i}$ 对于所有 $i$,适合输入少输出多
- **反向模式**:计算 $\frac{\partial f}{\partial y_i}$ 对于所有 $i$,适合输入多输出少(神经网络!)

**为什么反向模式适合神经网络?**
神经网络的损失是标量,参数可能有数百万个。反向模式只需一次反向传播就能计算所有参数的梯度!

#### 1.3 梯度消失与梯度爆炸

**问题**:在深层网络中,梯度可能变得极小或极大。

**分析**:考虑 $L$ 层线性网络(忽略激活):
$$
\mathbf{h}^{(k)} = W^{(k)} \mathbf{h}^{(k-1)}
$$

输出对输入的梯度:
$$
\frac{\partial L}{\partial \mathbf{h}^{(0)}} = (W^{(L)})^	op \cdots (W^{(1)})^	op \frac{\partial L}{\partial \mathbf{h}^{(L)}}
$$

令 $\sigma_{\max}^{(k)}$ 为 $W^{(k)}$ 的最大奇异值。梯度范数为:
$$
\left\lVert \frac{\partial L}{\partial \mathbf{h}^{(0)}} \right\rVert \approx \left(\prod_{k=1}^L \sigma_{\max}^{(k)}\right) \left\lVert \frac{\partial L}{\partial \mathbf{h}^{(L)}} \right\rVert
$$

- 如果 $\sigma_{\max}^{(k)} < 1$: 指数级衰减 \to **梯度消失**
- 如果 $\sigma_{\max}^{(k)} > 1$: 指数级增长 \to **梯度爆炸**

**解决方案**:
- **Xavier 初始化**: $W_{ij} \sim \mathcal{N}(0, \frac{2}{n_{\text{in}} + n_{\text{out}}})$
- **ReLU 激活**:导数为0或1,不改变梯度幅度
- **残差连接**:提供"梯度高速公路"
- **层归一化**:规范化激活值分布

#### 1.4 现代框架的实现

PyTorch 和 TensorFlow 使用**动态计算图**:
- 每次前向传播构建计算图
- 反向传播时自动计算梯度
- 图在反向传播后销毁,释放内存

**示例**(PyTorch):
```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
z = y.sum()

z.backward()  # 自动计算梯度
print(x.grad)  # tensor([2., 4.])
```

框架使用**自动微分**而非符号微分,既高效又灵活。

---

### 2. 激活函数的导数

#### 2.1 Sigmoid、Tanh、ReLU 及其导数

**Sigmoid**:
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

导数:
$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

**问题**:
- 当 $x$ 很大或很小时, $\sigma'(x) \approx 0$ \to **梯度消失**
- 输出不是零中心的

**Tanh**:
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

导数:
$$
\tanh'(x) = 1 - \tanh^2(x)
$$

**优点**:输出是零中心的。
**缺点**:仍然有梯度消失问题。

**ReLU**(Rectified Linear Unit):
$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

导数:
$$
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

![常用激活函数对比](/images/math/activation-functions.png)

*图4:三种常用激活函数。蓝色是 Sigmoid,绿色是 Tanh,橙色是 ReLU。ReLU 是最简单的,但也是现代神经网络中最常用的。*

#### 2.2 为什么需要非线性激活

**问题**:如果没有非线性激活,多层神经网络退化为单层。

**证明**:考虑两层线性网络:
$$
\mathbf{y} = W^{(2)} (W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)} = W^{(2)} W^{(1)} \mathbf{x} + (W^{(2)} \mathbf{b}^{(1)} + \mathbf{b}^{(2)})
$$

这等价于单层网络: $\mathbf{y} = W \mathbf{x} + \mathbf{b}$,其中 $W = W^{(2)} W^{(1)}$。

**结论**:非线性激活赋予了神经网络**通用近似能力**(universal approximation)。

#### 2.3 导数消失问题

**Sigmoid 的导数**:
$$
\sigma'(x) \in (0, 0.25]
$$

在深层网络中,多个小于1的数相乘会导致指数级衰减:
$$
\prod_{k=1}^L \sigma'(z_k) \to 0
$$

**ReLU 的优势**:
$$
\text{ReLU}'(x) \in \{0, 1\}
$$

不会导致梯度消失(要么是0,要么是1)。

**缺点**:"Dead ReLU"问题:如果输入总是负的,ReLU 永远不会被激活,梯度也是0,参数无法更新。

#### 2.4 ReLU 的优势

1. **计算简单**:只需要比较运算
2. **缓解梯度消失**:导数是0或1
3. **稀疏激活**:约50%的神经元被激活(正输入)
4. **生物学合理性**:脑神经的激活机制

---

### 3. 正则化的微积分视角

#### 3.1 L1/L2 正则的几何意义

**L2 正则**:
$$
L = \frac{1}{n}\sum_{i=1}^n (y_i - f(\mathbf{x}_i))^2 + \lambda \lVert \mathbf{w} \rVert^2
$$

**梯度**:
$$
\frac{\partial L}{\partial \mathbf{w}} = -\frac{2}{n}\sum_{i=1}^n (y_i - f(\mathbf{x}_i))\frac{\partial f}{\partial \mathbf{w}} + 2\lambda \mathbf{w}
$$

**几何解释**:L2 正则倾向于让权重"均匀地小",防止过拟合。

**L1 正则**:
$$
L = \frac{1}{n}\sum_{i=1}^n (y_i - f(\mathbf{x}_i))^2 + \lambda \lVert \mathbf{w} \rVert_1
$$

**次梯度**:
$$
\frac{\partial L}{\partial w_j} = \text{数据项} + \lambda \cdot \text{sign}(w_j)
$$

**几何解释**:L1 正则倾向于产生**稀疏解**(许多权重为0),因为 L1 球的"角点"在坐标轴上。

#### 3.2 梯度流与正则化

**梯度流**(gradient flow)是学习率为无穷小时的梯度下降连续版本:
$$
\frac{d\mathbf{w}}{dt} = -\nabla L(\mathbf{w})
$$

**L2 正则的梯度流**:
$$
\frac{d\mathbf{w}}{dt} = -\nabla \text{数据项} - 2\lambda \mathbf{w}
$$

这是一个**阻尼振荡**系统,会收敛到平衡点。

#### 3.3 Dropout 的效果

Dropout 在训练时随机"丢弃"神经元:
$$
\mathbf{h} = \mathbf{m} \odot \sigma(W \mathbf{x})
$$

其中 $\mathbf{m} \sim \text{Bernoulli}(p)$,$\odot$ 是逐元素乘法。

**期望**:
$$
\mathbb{E}[\mathbf{h}] = p \cdot \sigma(W \mathbf{x})
$$

**测试时**:使用 $p \sigma(W \mathbf{x})$ 或 $\sigma(p W \mathbf{x})$(反向缩放)。

**正则化效果**:
1. 防止共适应(co-adaptation):神经元不能依赖特定其他神经元
2. 等价于**集成学习**(bagging):每次训练不同的子网络
3. 近似贝叶斯推断:高斯过程

#### 3.4 Batch Normalization 的微分

**BatchNorm**:
$$
\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}
$$

$$
\mathbf{y} = \gamma \hat{\mathbf{x}} + \beta
$$

其中 $\mu_\mathcal{B}$ 和 $\sigma_\mathcal{B}^2$ 是小批量的均值和方差。

**梯度**:
$$
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^m \frac{\partial L}{\partial y_i} \hat{x}_i
$$

$$
\frac{\partial L}{\partial \beta} = \sum_{i=1}^m \frac{\partial L}{\partial y_i}
$$

对于 $\mathbf{x}$ 的梯度更复杂,需要考虑对 $\mu_\mathcal{B}$ 和 $\sigma_\mathcal{B}^2$ 的影响。

**效果**:
1. 稳定训练:减少内部协变量偏移(internal covariate shift)
2. 允许更大的学习率
3. 有轻微的正则化效果(通过引入噪声)

---

### 4. 优化算法的演进

#### 4.1 从 SGD 到 Adam

**SGD**:
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)
$$

**Momentum**:
$$
\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \nabla L(\mathbf{w}_t)
$$

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_t
$$

**AdaGrad**:
$$
\mathbf{w}_{t+1, i} = \mathbf{w}_{t, i} - \frac{\eta}{\sqrt{G_{t, ii} + \epsilon}} \nabla L(\mathbf{w}_t)_i
$$

**RMSprop**:
$$
\mathbf{s}_t = \rho \mathbf{s}_{t-1} + (1 - \rho) (\nabla L(\mathbf{w}_t))^2
$$

$$
\mathbf{w}_{t+1} i = \mathbf{w}_{t, i} - \frac{\eta}{\sqrt{\mathbf{s}_{t, i} + \epsilon}} \nabla L(\mathbf{w}_t)_i
$$

**Adam**(Adaptive Moment Estimation):
$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla L(\mathbf{w}_t)
$$

$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla L(\mathbf{w}_t))^2
$$

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}} \hat{\mathbf{m}}_t
$$

Adam 结合了动量和自适应学习率的优点,是现代深度学习的**默认选择**。

#### 4.2 自适应学习率的数学原理

**核心思想**:对不同参数使用不同的学习率。

**直觉**:如果某个参数的梯度总是很大,说明它很敏感,应该用较小的学习率;如果梯度总是很小,说明它不敏感,应该用较大的学习率。

**数学推导**:考虑对角二阶近似:
$$
L(\mathbf{w} + \Delta \mathbf{w}) \approx L(\mathbf{w}) + \nabla L(\mathbf{w})^	op \Delta \mathbf{w} + \frac{1}{2} \Delta \mathbf{w}^T \text{diag}(\mathbf{g}) \Delta \mathbf{w}
$$

其中 $\mathbf{g}$ 包含历史梯度平方和。对 $\Delta \mathbf{w}$ 求导并令为零:
$$
\nabla L(\mathbf{w}) + \text{diag}(\mathbf{g}) \Delta \mathbf{w} = 0 \implies \Delta \mathbf{w} = -\text{diag}(\mathbf{g})^{-1} \nabla L(\mathbf{w})
$$

这就是 AdaGrad 的更新公式!

#### 4.3 二阶优化方法

**牛顿法**(Newton's method):
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - H^{-1} \nabla L(\mathbf{w}_t)
$$

其中 $H$ 是 Hessian 矩阵(二阶导数矩阵):
$$
H_{ij} = \frac{\partial^2 L}{\partial w_i \partial w_j}
$$

**优点**:
- 二阶收敛(接近最优点时非常快)
- 自动适应曲率

**缺点**:
- 计算 Hessian: $O(d^2)$
- 求解线性系统: $O(d^3)$
- Hessian 可能不正定

**拟牛顿法**(Quasi-Newton,如 L-BFGS):用一阶信息近似 Hessian,避免显式计算 Hessian。

**收敛速度比较**:
- SGD: $O(1/\sqrt{t})$(次线性)
- SGD + 动量: $O(1/t)$
- 牛顿法: $O(1/k^2)$ 或更快(二次收敛)

---

## 第四部分:高级主题

### 1. 变分法

#### 1.1 泛函的极值问题

微积分研究函数的极值,**变分法**(calculus of variations)研究**泛函**的极值。

**泛函**:函数的函数,例如:
$$
J[y] = \int_a^b F(x, y(x), y'(x)) dx
$$

**问题**:找到函数 $y(x)$ 使 $J[y]$ 最小化。

**示例**:最速降线问题(brachistochrone)。在重力作用下,物体沿什么曲线从 A 点滑到 B 点时间最短?

#### 1.2 欧拉-拉格朗日方程

**推导**:考虑 $y(x)$ 的微小变分 $y(x) + \epsilon \eta(x)$,其中 $\eta(a) = \eta(b) = 0$。

$$
\frac{d}{d\epsilon} J[y + \epsilon \eta] \bigg|_{\epsilon=0} = 0
$$

分部积分后得到**欧拉-拉格朗日方程**:
$$
\frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) = 0
$$

这是一个二阶微分方程,边界条件 $y(a) = y_a$, $y(b) = y_b$。

#### 1.3 在变分自编码器(VAE)中的应用

**VAE 目标**:
$$
\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))
$$

**变分下界**(ELBO):
$$
\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}, \mathbf{z}) - \log q(\mathbf{z}|\mathbf{x})]
$$

**训练**:最大化 ELBO,等价于最小化 KL 散度。

**重参数化技巧**(reparameterization trick):
$$
\mathbf{z} = \mu + \sigma \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

这使得梯度可以通过采样传播!

---

### 2. 矩阵微积分

#### 2.1 矩阵求导法则

**标量对向量求导**:
$$
\frac{\partial}{\partial \mathbf{x}} (\mathbf{a}^T \mathbf{x}) = \mathbf{a}
$$

$$
\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}
$$

如果 $A$ 对称:
$$
\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T A \mathbf{x}) = 2A\mathbf{x}
$$

**矩阵对矩阵求导**:
$$
\frac{\partial}{\partial X} \lVert X \rVert_F^2 = \frac{\partial}{\partial X} \text{tr}(X^T X) = 2X
$$

#### 2.2 Kronecker 乘积

**定义**:
$$
A \otimes B = \begin{pmatrix} a_{11}B & \cdots & a_{1n}B \\ \vdots & \ddots & \vdots \\ a_{m1}B & \cdots & a_{mn}B \end{pmatrix}
$$

**性质**:
- $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$
- $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$
- $\text{vec}(AXB) = (B^T \otimes A)\text{vec}(X)$

其中 $\text{vec}(X)$ 是将矩阵按列拉直成向量。

#### 2.3 向量化技术

在神经网络中,经常需要并行处理多个样本。**向量化**利用矩阵乘法的高效实现(BLAS 库)。

**未向量化**:
```python
for i in range(batch_size):
    for j in range(output_dim):
        z[i, j] = np.dot(x[i], w[j]) + b[j]
```

**向量化**:
```python
Z = X @ W.T + b
```

速度差异:向量化通常快 10-100 倍!

#### 2.4 在神经网络训练中的应用

**批量梯度计算**:
$$
\frac{\partial L}{\partial W} = \sum_{i=1}^n \frac{\partial L_i}{\partial W}
$$

矩阵形式:
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Z} X^T
$$

其中 $Z = XW^T + \mathbf{1}\mathbf{b}^T$。

---

### 3. 微分几何初步

#### 3.1 流形与切空间

**流形**(manifold):局部像欧氏空间的拓扑空间。

**例子**:
- 球面 $S^2 = \{(x, y, z) : x^2 + y^2 + z^2 = 1\}$
- 环面 $T^2 = S^1 \times S^1$

**切空间**(tangent space):流形在某一点的线性近似。

**例子**:球面上点 $(x_0, y_0, z_0)$ 处的切平面:
$$
x_0 x + y_0 y + z_0 z = 1
$$

#### 3.2 黎曼度量

**度量**(metric):定义长度和角度。

在欧氏空间中,度量是单位矩阵 $I$。在流形上,度量 $g$ 是位置相关的:
$$
ds^2 = g_{ij} dx^i dx^j
$$

**球面上的度量**:
$$
ds^2 = d\theta^2 + \sin^2\theta d\phi^2
$$

#### 3.3 测地线与梯度流

**测地线**(geodesic):流形上的"直线",即长度最短的曲线。

**欧拉-拉格朗日方程**(对于弧长):
$$
\frac{d}{dt}\left(g_{ij}\frac{dx^j}{dt}\right) - \frac{1}{2}\frac{\partial g_{jk}}{\partial x^i}\frac{dx^j}{dt}\frac{dx^k}{dt} = 0
$$

**梯度流**(gradient flow):沿梯度方向积分:
$$
\frac{d\mathbf{x}}{dt} = -\nabla f(\mathbf{x})
$$

在黎曼流形上:
$$
\frac{dx^i}{dt} = -g^{ij} \frac{\partial f}{\partial x^j}
$$

其中 $g^{ij}$ 是度量张量的逆。

#### 3.4 在流形学习中的应用

**假设**:高维数据实际位于低维流形上。

**Isomap**:
1. 构建邻域图
2. 计算图上最短路径(测地线距离)
3. 在测地线距离矩阵上做 MDS

**UMAP**(Uniform Manifold Approximation and Projection):
1. 在高维空间和低维空间都构建加权 k-NN 图
2. 优化两个图的相似度
3. 使用黎曼几何和代数拓扑理论

---

### 4. 随机微积分

#### 4.1 随机过程的微分

**布朗运动**(Brownian motion)$W_t$:连续时间随机过程,满足:
- $W_0 = 0$
- 独立增量: $W_t - W_s$ 独立于过去
- 高斯增量: $W_t - W_s \sim \mathcal{N}(0, t-s)$

**性质**:
- **连续但几乎处处不可导**:路径很粗糙
- **自相似**: $W_{ct} \sim \sqrt{c} W_t$
- **马尔可夫性**:未来只依赖现在,不依赖过去

#### 4.2 伊藤积分(Itô integral)

**伊藤积分**:
$$
I_T = \int_0^T f(t, W_t) dW_t
$$

**定义**:黎曼和的极限:
$$
I_T = \lim_{n \to \infty} \sum_{i=1}^n f(t_{i-1}, W_{i-1})(W_i - W_{i-1})
$$

注意:被积函数在**左端点**求值!

**伊藤公式**(Itô's lemma):

对于 $Y_t = f(t, W_t)$:
$$
dY_t = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial W_t} dW_t + \frac{1}{2}\frac{\partial^2 f}{\partial W_t^2} dt
$$

与普通微积分相比,多了一项 $\frac{1}{2}\frac{\partial^2 f}{\partial W_t^2} dt$,这是布朗运动的二次变差导致的!

#### 4.3 在扩散模型中的应用

**扩散过程**(SDE):
$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(\mathbf{x}, t) d\mathbf{W}_t
$$

**前向扩散**:逐渐向数据添加噪声:
$$
d\mathbf{x} = -\frac{1}{2}\beta(t) \mathbf{x} dt + \sqrt{\beta(t)} d\mathbf{W}_t
$$

**反向扩散**:去噪恢复数据:
$$
d\mathbf{x} = \left(-\frac{1}{2}\beta(t) \mathbf{x} + \nabla_\mathbf{x} \log p(\mathbf{x} | t)\right) dt + \sqrt{\beta(t)} d\mathbf{W}_t
$$

**Score-based models**:
$$
\nabla_\mathbf{x} \log p(\mathbf{x}) \approx -\frac{1}{\sqrt{2\pi}} \mathbf{s}_\theta(\mathbf{x}, t)
$$

其中 $\mathbf{s}_\theta$ 是神经网络,学习噪声的分数(score)。

**DDPM、DDPM++、Stable Diffusion** 等都基于这个理论!

---

## 结语:微积分的生命力

回顾这段旅程,我们从变化率的几何直观出发,经历了微分的线性化之美、积分的累积之效、泰勒展开的逼近之力,最终抵达了神经网络的反向传播和扩散模型的随机微积分。

微积分之所以成为现代人工智能的基石,正是因为它提供了一种**分析变化**的统一语言:

- **一阶微积分**(导数、微分、梯度):告诉我们如何沿着最陡方向下降
- **二阶微积分**(泰勒展开、牛顿法):利用曲率信息加速收敛
- **多元微积分**(雅可比矩阵、链式法则):处理复杂函数的复合关系
- **变分法**:优化函数的函数
- **随机微积分**:描述带有随机性的演化过程

更重要的是,微积分培养了一种**思维方式**:
1. **线性化思维**:将复杂问题局部线性化
2. **迭代思维**:用局部逼近逐步逼近全局最优
3. **权衡思维**:偏差与方差的权衡、欠拟合与过拟合的权衡
4. **抽象思维**:从具体问题中提炼数学本质

在未来,随着深度学习、神经符号AI、因果推断等新领域的发展,微积分将继续扮演关键角色。理解微积分,不仅是掌握一门数学工具,更是培养一种分析问题、解决问题、创新思考的能力。

正如伟大的数学家希尔伯特所说:"无限!再也没有其他问题如此深刻地打动过人类的心灵。"微积分,正是人类理解无限和变化的伟大发明。

## 参考文献

1. Spivak, M. (2008). *Calculus On Manifolds*. Westview Press.
2. Hubbard, J. H., & Hubbard, B. B. (2015). *Vector Calculus, Linear Algebra, and Differential Forms: A Unified Approach* (5th ed.). Matrix Editions.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
5. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
6. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.
7. Oksendal, B. (2003). *Stochastic Differential Equations: An Introduction with Applications* (6th ed.). Springer.
8. Gelfand, I. M., & Fomin, S. V. (2000). *Calculus of Variations*. Dover Publications.
9. Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *Journal of Machine Learning Research*, 6, 695-709.
10. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR*.
