---
title: "麦克斯韦方程组：从库仑到电磁波"
date: 2026-01-12T22:00:00+08:00
draft: false
description: "从零开始详细推导麦克斯韦方程组，包括向量微积分基础、静电场、静磁场、法拉第感应定律和位移电流，适合有微积分和线性代数基础的读者"
categories: ["物理学", "电磁学"]
tags: ["数学史", "广义相对论"]
cover:
    image: "images/covers/1567427017947-545c5f8d16ad.jpg"
    alt: "麦克斯韦方程"
    caption: "麦克斯韦方程：电磁统一的完美方程组"
---

## 引言：电与磁的统一

### 从孤立到统一

19世纪初期，电和磁被认为是两种完全独立的现象。电荷产生电场，磁荷（假想的）产生磁场，它们之间似乎没有任何联系。

然而，一系列令人惊叹的发现彻底改变了这个观点。1820年，丹麦物理学家奥斯特德（Hans Christian Ørsted）意外地发现，电流可以使指南针偏转——电可以产生磁。1831年，英国物理学家法拉第（Michael Faraday）发现变化的磁场可以产生电流——磁可以产生电。

这些发现暗示着电和磁之间存在深刻的联系。最终，这个谜团被苏格兰物理学家詹姆斯·克拉克·麦克斯韦（James Clerk Maxwell）在1860年代解开。他不仅统一了电和磁，还预言了电磁波的存在——而光正是一种电磁波。

### 麦克斯韦方程组的美

麦克斯韦方程组是经典电磁学的基石，也是物理学中最优美的方程组之一。它仅用四个方程就描述了所有经典电磁现象：

1. **高斯定律**：电荷如何产生电场
2. **高斯磁定律**：不存在磁单极子
3. **法拉第电磁感应定律**：变化的磁场如何产生电场
4. **安培-麦克斯韦定律**：电流和变化的电场如何产生磁场

在接下来的篇幅中，我们将从最基本的概念开始，一步一步地推导出这四个方程。让我们开始这段电磁学的旅程。

---

## 第一章：向量微积分的语言

### 1.1 为什么要用向量？

在描述电磁场时，我们需要同时描述电场和磁场在空间中的分布和变化。场是空间的函数——每一点都有一个值（可能是标量或向量）。

**标量场**：温度场 $T(x, y, z)$，每点一个数值
**向量场**：电场 $\mathbf{E}(x, y, z)$，每点一个向量（有大小和方向）

向量是描述电磁场的完美语言，因为电场和磁场都有方向。

### 1.2 向量的基本运算

设 $\mathbf{A}$ 和 $\mathbf{B}$ 是三维向量：

$$\mathbf{A} = (A_x, A_y, A_z), \quad \mathbf{B} = (B_x, B_y, B_z)$$

**点积**（标量积）：

$$\mathbf{A} \cdot \mathbf{B} = A_x B_x + A_y B_y + A_z B_z = |\mathbf{A}| |\mathbf{B}| \cos\theta$$

**叉积**（向量积）：

$$\mathbf{A} \times \mathbf{B} = \begin{pmatrix} A_y B_z - A_z B_y \\ A_z B_x - A_x B_z \\ A_x B_y - A_y B_x \end{pmatrix} = (A_y B_z - A_z B_y, A_z B_x - A_x B_z, A_x B_y - A_y B_x)$$

### 1.3 梯度：标量场的变化率

对于一个标量场 $\phi(x, y, z)$，**梯度**（gradient）是一个向量，指向函数增长最快的方向：

$$\nabla \phi = \left( \frac{\partial \phi}{\partial x}, \frac{\partial \phi}{\partial y}, \frac{\partial \phi}{\partial z} \right)$$

其中 $\nabla$ 是**微分算子**（读作"del"或"nabla"）：

$$\nabla = \left( \frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{\partial}{\partial z} \right)$$

**物理意义**：梯度指向方向导数最大的方向，其大小等于该方向的方向导数。

想象山坡上的每一点，梯度指向最陡峭的上坡方向，梯度的大小就是那个方向的陡峭程度。

### 1.4 散度：向量场的"源头"强度

**散度**（divergence）是向量场的标量函数，描述该点是场的"源"还是"汇"：

$$\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}$$

**物理意义**：
- $\nabla \cdot \mathbf{F} > 0$：该点有"源"，向量向外发散
- $\nabla \cdot \mathbf{F} < 0$：该点有"汇"，向量向内汇聚
- $\nabla \cdot \mathbf{F} = 0$：向量既不产生也不消失

想象一个水源（正散度）和一个排水口（负散度），散度描述了这些"源头"的强度。

### 1.5 旋度：向量场的"旋转"程度

**旋度**（curl）是向量场的向量函数，描述该点附近向量场的"旋转"程度：

$$\nabla \times \mathbf{F} = \left( \frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z}, \frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x}, \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y} \right)$$

**物理意义**：旋度的方向是旋转轴的方向（右手螺旋定则），旋度的大小是旋转的剧烈程度。

想象一个漩涡，旋度指向漩涡的中心轴方向。

### 1.6 高斯定理：体积分与面积分的关系

**高斯定理**（散度定理）是向量微积分中最重要的定理之一：

$$\iiint_V (\nabla \cdot \mathbf{F}) \, dV = ∯_S \mathbf{F} \cdot d\mathbf{S}$$

**含义**：向量场穿过闭合曲面的通量等于该曲面所围体积内场的散度的积分。

**直观理解**：想象一个气球，内部有气体不断产生（散度为正），那么气球表面的气体流出速率（通量）就等于内部产生气体的速率。

### 1.7 斯托克斯定理：面积分与线积分的关系

**斯托克斯定理**（旋度定理）描述了曲面积分与曲线积分的关系：

$$\iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S} = \oint_C \mathbf{F} \cdot d\mathbf{r}$$

**含义**：向量场沿闭合曲线的环流等于该曲线所围曲面上旋度的通量。

**直观理解**：想象水流中的漩涡，旋度越大，水流沿漩涡边缘流动得越快。

---

## 第二章：静电场与高斯定律

### 2.1 库仑定律：电荷之间的相互作用

1785年，法国物理学家库仑（Charles-Augustin de Coulomb）通过扭秤实验发现了电荷之间相互作用的规律。

**库仑定律**：两个点电荷 $q_1$ 和 $q_2$ 之间的力与它们电量的乘积成正比，与距离的平方成反比：

$$\mathbf{F}_{12} = k_e \frac{q_1 q_2}{r^2} \hat{\mathbf{r}}_{12}$$

其中 $k_e = \frac{1}{4\pi\varepsilon_0} \approx 8.99 \times 10^9 \, \text{N}\cdot\text{m}^2/\text{C}^2$ 是库仑常数，$\varepsilon_0 \approx 8.85 \times 10^{-12} \, \text{F/m}$ 是真空介电常数。

**方向**：
- 同性电荷相斥，异性电荷相吸
- $\hat{\mathbf{r}}_{12}$ 是从 $q_1$ 指向 $q_2$ 的单位向量

### 2.2 电场的定义

电场是电荷在周围空间激发的场。当我们把一个试探电荷 $q_0$ 放在电场中时，它会受到电力：

$$\mathbf{F} = q_0 \mathbf{E}$$

因此，**电场强度**定义为：

$$\mathbf{E} = \frac{\mathbf{F}}{q_0}$$

对于点电荷 $q$ 在空间中产生的电场：

$$\mathbf{E}(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0} \frac{q}{r^2} \hat{\mathbf{r}}$$

### 2.3 电场叠加原理

电场满足**叠加原理**：多个电荷产生的电场等于各电荷单独产生电场的矢量和。

对于连续分布的电荷：

$$\mathbf{E}(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0} \int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|^2} \hat{\mathbf{r}} \, d\tau'$$

其中 $\rho(\mathbf{r}')$ 是电荷密度，$d\tau'$ 是体积元。

### 2.4 电场线的概念

电场线是帮助我们可视化电场的工具：
- 电场线的切线方向是电场的方向
- 电场线的密度（垂直于方向）是电场强度的大小

**性质**：
- 电场线从正电荷出发，终止于负电荷
- 电场线不会闭合（静电场是保守场）
- 电场线不会相交（每点电场方向唯一）

### 2.5 高斯定律的推导

现在我们从库仑定律推导出高斯定律。

**第一步：计算点电荷的电场通量**

考虑一个点电荷 $q$，计算它通过任意闭合曲面 $S$ 的电通量。

以点电荷为中心画一个半径为 $r$ 的球面 $S_0$，电场通量为：

$$\Phi_E = ∯_{S_0} \mathbf{E} \cdot d\mathbf{S} = ∯_{S_0} E \, dS = \frac{q}{4\pi\varepsilon_0 r^2} \cdot 4\pi r^2 = \frac{q}{\varepsilon_0}$$

**第二步：任意闭合曲面**

对于任意闭合曲面 $S$，利用电场线的概念：
- 如果 $S$ 包围点电荷 $q$，电场线全部穿出，通量等于 $\frac{q}{\varepsilon_0}$
- 如果 $S$ 不包围点电荷，进入的电场线数等于穿出的电场线数，通量为零

**第三步：电介质中的情况**

如果曲面内有体电荷密度 $\rho$，则总电量为：

$$Q_{\text{enc}} = \iiint_V \rho \, dV$$

因此，**高斯定律**为：

$$∯_S \mathbf{E} \cdot d\mathbf{S} = \frac{Q_{\text{enc}}}{\varepsilon_0} = \frac{1}{\varepsilon_0} \iiint_V \rho \, dV$$

**第四步：微分形式**

使用高斯定理（散度定理）：

$$\iiint_V (\nabla \cdot \mathbf{E}) \, dV = \frac{1}{\varepsilon_0} \iiint_V \rho \, dV$$

由于这个等式对任意体积 $V$ 成立，被积函数必须相等：

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$$

**这就是高斯定律的微分形式！**

### 2.6 电势与电场的关系

由于静电场是保守场，可以定义**电势** $V$（或 $\phi$）：

$$\mathbf{E} = -\nabla V$$

积分形式：

$$V(\mathbf{r}) = -\int_{\infty}^{\mathbf{r}} \mathbf{E} \cdot d\mathbf{l}$$

对于点电荷：

$$V(r) = \frac{1}{4\pi\varepsilon_0} \frac{q}{r}$$

---

## 第三章：静磁场与安培定律

### 3.1 磁场的发现

人们很早就知道磁石可以吸引铁器。1820年，奥斯特德发现电流可以使磁针偏转，这是电产生磁的第一个证据。

安培（André-Marie Ampère）进一步研究发现，通有电流的导线之间也存在相互作用力——这建立了电流与磁场之间的定量关系。

### 3.2 毕奥-萨伐尔定律

毕奥-萨伐尔定律（Biot-Savart Law）描述了电流元在空间中产生的磁场：

$$d\mathbf{B} = \frac{\mu_0}{4\pi} \frac{I \, d\mathbf{l} \times \hat{\mathbf{r}}}{r^2}$$

其中：
- $I \, d\mathbf{l}$ 是电流元（$d\mathbf{l}$ 是导线长度元矢量）
- $\hat{\mathbf{r}}$ 是从电流元指向场点的单位向量
- $\mu_0 = 4\pi \times 10^{-7} \, \text{N/A}^2$ 是真空磁导率

**叉积的意义**：电流元的磁场方向垂直于电流方向和连线方向（右手定则）。

对于一段导线产生的磁场：

$$\mathbf{B} = \frac{\mu_0}{4\pi} \int \frac{I \, d\mathbf{l} \times \hat{\mathbf{r}}}{r^2}$$

### 3.3 安培力

电流在磁场中会受到力。对于电流元 $I \, d\mathbf{l}$ 在磁场 $\mathbf{B}$ 中：

$$d\mathbf{F} = I \, d\mathbf{l} \times \mathbf{B}$$

这就是**洛伦兹力**在磁场中的形式（洛伦兹力的完整形式还包括电力 $q\mathbf{E}$）。

### 3.4 安培环路定律的推导

**安培环路定律**描述了磁场与产生它的电流之间的关系。

**实验观察**：长直导线的磁场

对于无限长直导线，距离导线 $r$ 处的磁场大小为：

$$B = \frac{\mu_0 I}{2\pi r}$$

方向沿圆周切线方向（右手握住导线，大拇指指向电流方向，四指环绕方向为磁场方向）。

**计算沿闭合环路的线积分**

考虑以导线为中心、半径为 $r$ 的圆周环路 $C$：

$$\oint_C \mathbf{B} \cdot d\mathbf{l} = \oint_C B \, dl = \frac{\mu_0 I}{2\pi r} \cdot 2\pi r = \mu_0 I$$

**推广到任意环路**

安培环路定律的完整形式：

$$\oint_C \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}}$$

其中 $I_{\text{enc}}$ 是环路所包围的净电流。

**微分形式**

使用斯托克斯定理：

$$\iint_S (\nabla \times \mathbf{B}) \cdot d\mathbf{S} = \oint_C \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}} = \mu_0 \iint_S \mathbf{J} \cdot d\mathbf{S}$$

由于对任意曲面 $S$ 成立：

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$$

**这就是安培定律的微分形式！**

### 3.5 静磁场的性质

从安培定律可以推出静磁场的重要性质：

1. **散度为零**：$\nabla \cdot \mathbf{B} = 0$
   - 表明不存在"磁单极子"
   - 磁感线是闭合曲线

2. **旋度不为零**：$\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$
   - 磁场是非保守场
   - 电流是磁场的"旋涡源"

---

## 第四章：法拉第电磁感应定律

### 4.1 电磁感应的发现

1831年，法拉第发现了电磁感应现象：变化的磁场可以在导体中产生电流。

**实验现象**：
- 磁铁插入线圈时，线圈中产生电流
- 磁铁拔出时，线圈中产生反向电流
- 电流变化越快，感应电动势越大

### 4.2 磁通量的定义

**磁通量** $\Phi_B$ 是磁场穿过某一面积的量度：

$$\Phi_B = \iint_S \mathbf{B} \cdot d\mathbf{S}$$

其中 $d\mathbf{S}$ 是面积元矢量，方向沿曲面的法向。

### 4.3 法拉第定律的推导

**法拉第定律**描述了感应电动势与磁通量变化率的关系：

$$\mathcal{E} = -\frac{d\Phi_B}{dt}$$

负号表示**楞次定律**：感应电流的效果总是反抗引起感应电流的原因。

**线圈的感应电动势**

对于 $N$ 匝线圈：

$$\mathcal{E} = -N \frac{d\Phi_B}{dt}$$

**积分形式**

感应电动势是电场沿闭合回路的线积分：

$$\mathcal{E} = \oint_C \mathbf{E} \cdot d\mathbf{l}$$

因此：

$$\oint_C \mathbf{E} \cdot d\mathbf{l} = -\frac{d}{dt} \iint_S \mathbf{B} \cdot d\mathbf{S}$$

**微分形式**

使用斯托克斯定理：

$$\iint_S (\nabla \times \mathbf{E}) \cdot d\mathbf{S} = -\frac{d}{dt} \iint_S \mathbf{B} \cdot d\mathbf{S}$$

由于曲面 $S$ 固定，可以将时间导数移入积分：

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

**这就是法拉第定律的微分形式！**

### 4.4 电磁感应的物理意义

法拉第定律揭示了一个深刻的物理事实：

**变化的磁场会产生电场**

这与静电场（由电荷产生，电场线从正电荷出发终止于负电荷）不同，感生电场的电场线是闭合的。

---

## 第五章：位移电流与麦克斯韦方程组

### 5.1 安培定律的疑难

麦克斯韦注意到安培定律 $\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$ 存在一个问题。

**考虑一个电容器充电的电路**：
- 电荷从电源流向电容器极板
- 电容器两极板之间的区域没有传导电流

如果我们取一个穿过电容器极板间的环路，安培定律给出：

$$\oint_C \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}}$$

但当我们选择的曲面与导线相交时，$I_{\text{enc}} = I$；当我们选择的曲面穿过电容器两极板之间时，$I_{\text{enc}} = 0$。

**矛盾**：同一个闭合环路的线积分不可能同时等于 $\mu_0 I$ 和 0！

### 5.2 麦克斯韦的解决方案

麦克斯韦意识到，问题在于电流是不稳定的——电荷正在积累。他引入了**位移电流**的概念。

**位移电流密度**定义为：

$$\mathbf{J}_d = \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$

**总电流密度**：

$$\mathbf{J}_{\text{total}} = \mathbf{J} + \mathbf{J}_d = \mathbf{J} + \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$

### 5.3 位移电流的物理意义

位移电流不是真实的电荷流动，而是电场变化的等效效应。

**考虑电容器充电过程**：
- 导线中有传导电流 $I$
- 电容器极板上电荷积累：$Q = CV$
- 电场随时间变化：$\frac{dE}{dt}$

电容器中的等效电流等于导线中的传导电流，保证了电流的连续性。

### 5.4 修正后的安培定律

将位移电流加入安培定律：

$$\nabla \times \mathbf{B} = \mu_0 \left( \mathbf{J} + \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t} \right)$$

或者写成：

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$

**这就是安培-麦克斯韦定律！**

### 5.5 完整的麦克斯韦方程组

现在，我们拥有了完整的麦克斯韦方程组。

**积分形式**：

1. **高斯定律（电场）**：
   $$∯_S \mathbf{E} \cdot d\mathbf{S} = \frac{Q_{\text{enc}}}{\varepsilon_0}$$

2. **高斯磁定律**：
   $$∯_S \mathbf{B} \cdot d\mathbf{S} = 0$$

3. **法拉第电磁感应定律**：
   $$\oint_C \mathbf{E} \cdot d\mathbf{l} = -\frac{d\Phi_B}{dt} = -\frac{d}{dt} \iint_S \mathbf{B} \cdot d\mathbf{S}$$

4. **安培-麦克斯韦定律**：
   $$\oint_C \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}} + \mu_0 \varepsilon_0 \frac{d\Phi_E}{dt}$$

**微分形式**（更简洁的形式）：

1. **高斯定律**：
   $$\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$$

2. **高斯磁定律**：
   $$\nabla \cdot \mathbf{B} = 0$$

3. **法拉第定律**：
   $$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

4. **安培-麦克斯韦定律**：
   $$\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$

---

## 第六章：从麦克斯韦方程到电磁波

### 6.1 麦克斯韦的预言

1865年，麦克斯韦从他的方程组中推导出一个惊人的预言：**电磁波的存在**。

让我们从麦克斯韦方程组出发，看看如何推导出电磁波方程。

### 6.2 真空中的麦克斯韦方程

在没有电荷和电流的区域（$\rho = 0$, $\mathbf{J} = 0$），麦克斯韦方程组简化为：

$$\nabla \cdot \mathbf{E} = 0$$
$$\nabla \cdot \mathbf{B} = 0$$
$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$
$$\nabla \times \mathbf{B} = \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$

### 6.3 推导电磁波方程

对方程3两边取旋度：

$$\nabla \times (\nabla \times \mathbf{E}) = \nabla \times \left(-\frac{\partial \mathbf{B}}{\partial t}\right)$$

利用向量恒等式 $\nabla \times (\nabla \times \mathbf{A}) = \nabla(\nabla \cdot \mathbf{A}) - \nabla^2 \mathbf{A}$，以及 $\nabla \cdot \mathbf{E} = 0$：

$$-\nabla^2 \mathbf{E} = -\frac{\partial}{\partial t} (\nabla \times \mathbf{B})$$

代入方程4：

$$-\nabla^2 \mathbf{E} = -\frac{\partial}{\partial t} \left(\mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}\right)$$

整理得：

$$\nabla^2 \mathbf{E} = \mu_0 \varepsilon_0 \frac{\partial^2 \mathbf{E}}{\partial t^2}$$

**这就是电场满足的波动方程！**

同理对磁场有：

$$\nabla^2 \mathbf{B} = \mu_0 \varepsilon_0 \frac{\partial^2 \mathbf{B}}{\partial t^2}$$

### 6.4 波速的计算

标准波动方程的形式为：

$$\nabla^2 \mathbf{E} = \frac{1}{v^2} \frac{\partial^2 \mathbf{E}}{\partial t^2}$$

比较得：

$$v = \frac{1}{\sqrt{\mu_0 \varepsilon_0}}$$

代入数值：

$$\mu_0 = 4\pi \times 10^{-7} \, \text{N/A}^2$$
$$\varepsilon_0 = 8.85 \times 10^{-12} \, \text{F/m}$$

$$v = \frac{1}{\sqrt{(4\pi \times 10^{-7}) \times (8.85 \times 10^{-12})}} \approx 3.00 \times 10^8 \, \text{m/s}$$

**这个速度恰好等于光速！**

### 6.5 麦克斯韦的革命性结论

麦克斯韦得出结论：

**光是一种电磁波**

这个结论统一了光学和电磁学——光学的规律可以从电磁学的基本方程推导出来。

### 6.6 电磁波的性质

从麦克斯韦方程组可以推出电磁波的重要性质：

1. **横波**：电场和磁场都垂直于传播方向
2. **互相垂直**：$\mathbf{E} \perp \mathbf{B}$，且 $\mathbf{E} \perp \mathbf{k}$，$\mathbf{B} \perp \mathbf{k}$（$\mathbf{k}$ 是波矢）
3. **同相位**：$\mathbf{E}$ 和 $\mathbf{B}$ 同时达到最大值和最小值
4. **固定比例**：$|\mathbf{E}| = c|\mathbf{B}|$
5. **能量密度**：$u = \frac{1}{2}\varepsilon_0 E^2 + \frac{1}{2\mu_0} B^2$
6. **坡印廷矢量**：$\mathbf{S} = \frac{1}{\mu_0} \mathbf{E} \times \mathbf{B}$（能量流密度）

---

## 第七章：边界条件与介质的麦克斯韦方程

### 7.1 为什么需要边界条件？

麦克斯韦方程组的微分形式在空间中的每一点都成立，但在两种不同介质的分界面上，场可能出现不连续（电场在导体表面垂直分量很大，磁场可能有切向分量跳跃）。

边界条件描述了场在穿过界面时的变化。

### 7.2 电场和磁场的边界条件

**电场的边界条件**：

$$(\mathbf{E}_2 - \mathbf{E}_1) \cdot \hat{\mathbf{n}} = \frac{\sigma}{\varepsilon_0}$$
$$(\mathbf{E}_2 - \mathbf{E}_1) \times \hat{\mathbf{n}} = 0$$

其中 $\hat{\mathbf{n}}$ 是从介质1指向介质2的单位法向量，$\sigma$ 是表面电荷密度。

**磁场的边界条件**：

$$(\mathbf{B}_2 - \mathbf{B}_1) \cdot \hat{\mathbf{n}} = 0$$
$$(\mathbf{B}_2 - \mathbf{B}_1) \times \hat{\mathbf{n}} = \mu_0 (\mathbf{K} \times \hat{\mathbf{n}})$$

其中 $\mathbf{K}$ 是表面电流密度。

### 7.3 介质中的麦克斯韦方程

在介质中，我们需要引入**电位移** $\mathbf{D}$ 和**磁场强度** $\mathbf{H}$：

$$\mathbf{D} = \varepsilon_0 \mathbf{E} + \mathbf{P}$$
$$\mathbf{H} = \frac{\mathbf{B}}{\mu_0} - \mathbf{M}$$

其中 $\mathbf{P}$ 是极化强度，$\mathbf{M}$ 是磁化强度。

**介质中的麦克斯韦方程**：

$$\nabla \cdot \mathbf{D} = \rho_{\text{free}}$$
$$\nabla \cdot \mathbf{B} = 0$$
$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$
$$\nabla \times \mathbf{H} = \mathbf{J}_{\text{free}} + \frac{\partial \mathbf{D}}{\partial t}$$

### 7.4 线性各向同性介质

对于线性各向同性介质：

$$\mathbf{D} = \varepsilon \mathbf{E} = \varepsilon_r \varepsilon_0 \mathbf{E}$$
$$\mathbf{H} = \frac{\mathbf{B}}{\mu} = \frac{\mathbf{B}}{\mu_r \mu_0}$$

---

## 第八章：电磁学的应用

### 8.1 静电学的应用

**电容器**：储存电能的器件

$$C = \frac{Q}{V} = \frac{\varepsilon A}{d}$$

**电场的能量**：

$$U = \frac{1}{2} C V^2 = \frac{1}{2} \frac{Q^2}{C}$$

### 8.2 磁学的应用

**电感器**：储存磁能的器件

$$L = \frac{\Phi}{I}$$

**磁场的能量**：

$$U = \frac{1}{2} L I^2$$

### 8.3 电磁感应的应用

**变压器**：利用互感改变交流电压

$$V_1/V_2 = N_1/N_2$$

**发电机**：机械能转化为电能

**电动机**：电能转化为机械能

### 8.4 电磁波的应用

**无线电通信**：利用电磁波传输信息
**雷达**：利用电磁波探测距离
**光纤通信**：利用光（电磁波）传输信息
**微波炉**：利用电磁波加热食物

---

## 结语：方程组的完美与力量

### 麦克斯韦方程组的意义

回顾我们走过的旅程，从库仑定律到法拉第感应，从安培环路定律到位移电流，我们最终得到了麦克斯韦方程组——四个简洁而深刻的方程。

麦克斯韦方程组的美体现在：

1. **统一性**：电、磁、光原本被认为是三种独立的现象，现在被统一在同一个理论框架下。

2. **预言性**：麦克斯韦方程组预言了电磁波的存在，并计算出它的速度。这是对理论力量的最好证明。

3. **简洁性**：自然界复杂的电磁现象可以用四个方程完美描述。

4. **对称性**：电场和磁场在方程中表现出优美的对称性。

### 狭义相对论的诞生

麦克斯韦方程组还带来了一个意想不到的惊喜。在1887年，迈克尔逊-莫雷实验发现光速是各向同性的，这与经典物理学（伽利略变换）矛盾。

这个问题最终由爱因斯坦在1905年解决——他提出了**狭义相对论**。爱因斯坦发现，麦克斯韦方程组在洛伦兹变换下保持不变，而伽利略变换需要被抛弃。

事实上，狭义相对论正是从麦克斯韦方程组中"长"出来的。

### 给读者的话

如果你读到这里，恭喜你！你已经完成了从库仑定律到麦克斯韦方程组的完整旅程。

麦克斯韦方程组是物理学中最伟大的成就之一。它不仅统一了电、磁、光三种现象，还预言了无线电、电视、手机等现代技术的理论基础。

每当我们使用手机、打开电视、连接WiFi时，我们都在享受麦克斯韦方程组的成果。这个19世纪推导出的方程组，至今仍在塑造我们的日常生活。

---

## 附录：重要公式汇总

### 向量微分算子

**梯度**：
$$\nabla \phi = \left( \frac{\partial \phi}{\partial x}, \frac{\partial \phi}{\partial y}, \frac{\partial \phi}{\partial z} \right)$$

**散度**：
$$\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}$$

**旋度**：
$$\nabla \times \mathbf{F} = \left( \frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z}, \frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x}, \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y} \right)$$

### 基本定律

**库仑定律**：
$$\mathbf{F} = \frac{1}{4\pi\varepsilon_0} \frac{q_1 q_2}{r^2} \hat{\mathbf{r}}$$

**毕奥-萨伐尔定律**：
$$d\mathbf{B} = \frac{\mu_0}{4\pi} \frac{I \, d\mathbf{l} \times \hat{\mathbf{r}}}{r^2}$$

**洛伦兹力**：
$$\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

### 麦克斯韦方程组（微分形式）

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$$

$$\nabla \cdot \mathbf{B} = 0$$

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$

### 麦克斯韦方程组（积分形式）

$$∯_S \mathbf{E} \cdot d\mathbf{S} = \frac{Q_{\text{enc}}}{\varepsilon_0}$$

$$∯_S \mathbf{B} \cdot d\mathbf{S} = 0$$

$$\oint_C \mathbf{E} \cdot d\mathbf{l} = -\frac{d\Phi_B}{dt}$$

$$\oint_C \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}} + \mu_0 \varepsilon_0 \frac{d\Phi_E}{dt}$$

### 重要常数

| 符号 | 名称 | 数值 |
|------|------|------|
| $\varepsilon_0$ | 真空介电常数 | $8.85 \times 10^{-12} \, \text{F/m}$ |
| $\mu_0$ | 真空磁导率 | $4\pi \times 10^{-7} \, \text{N/A}^2$ |
| $c$ | 光速 | $3.00 \times 10^8 \, \text{m/s}$ |
| $k_e$ | 库仑常数 | $8.99 \times 10^9 \, \text{N}\cdot\text{m}^2/\text{C}^2$ |

---

*本文旨在为有一定数学基础的读者提供电磁学的入门导引。更深入的学习建议参考专业教材，如David J. Griffiths的《Introduction to Electrodynamics》、Jackson的《Classical Electrodynamics》等。*
