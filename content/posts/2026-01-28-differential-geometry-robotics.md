---
title: "微分几何在机器人学中的系统综述：从李群到现代应用"
date: 2026-01-28T14:00:00+08:00
draft: false
description: "系统综述微分几何在机器人学中的核心应用，涵盖李群李代数、运动学、动力学、轨迹规划、SLAM和现代深度学习，适合有微积分和线性代数基础的读者"
categories: ["数学", "机器人"]
tags: ["微分几何", "综述", "机器人", "李群"]
cover:
    image: "images/covers/robotics-geometry-cover.jpg"
    alt: "机器人与几何"
    caption: "微分几何：机器人学的数学基础"
math: true
---

## 引言：当机器人遇上几何

想象你正在操控一台工业机械臂。你输入一个目标位置，机械臂的末端执行器精准地移动到那里。这看似简单的动作背后，蕴含着深刻的数学原理。

**一个基本问题**：如何描述机械臂的姿态？

如果你说"用坐标 $(x, y, z)$ 表示位置，用三个角度表示方向"，这没错。但当你尝试在两个姿态之间插值时，问题出现了——简单的线性插值可能导致中间姿态根本不是有效的旋转！

这就是**流形约束**的体现：机器人的姿态空间不是一个简单的欧几里得空间，而是一个弯曲的流形。

### 从欧几里得到黎曼

古希腊人认为空间是平坦的。欧几里得几何告诉我们：平行线永不相交，三角形内角和恒为 $180^{\circ}$。

但 $19$ 世纪的数学家们发现，空间可以是弯曲的。高斯研究曲面，黎曼将这一理论推广到任意维度——**黎曼几何**诞生了。

$20$ 世纪，这些抽象理论找到了惊人应用：
- 爱因斯坦用黎曼几何描述引力（广义相对论）
- 工程师用微分几何控制机器人
- 计算机科学家用流形学习理解高维数据

本文将系统梳理微分几何在机器人学中的应用，从理论基础到现代实践，带你领略这门数学如何赋能智能机器。

---

## 第一章：李群与李代数——描述运动的数学语言

### 1.1 刚体运动的困境

在三维空间中，刚体的**位姿**（位置和方向）需要几个参数描述？

- 位置：$3$ 个参数 $(x, y, z)$
- 方向：至少需要 $3$ 个参数（如欧拉角）

**欧拉角的陷阱**：经典的万向节锁（Gimbal Lock）问题——当俯仰角为 $90^{\circ}$ 时，偏航和滚转失去独立意义。这说明用欧拉角表示旋转存在本质缺陷。

更优雅的选择是**旋转矩阵**：一个 $3 \times 3$ 的正交矩阵 $R$，满足 $R^T R = I$ 且 $\det(R) = 1$。

所有这样的矩阵构成**特殊正交群** $\text{SO}(3)$（Special Orthogonal Group）。

### 1.2 李群的引入

**李群**（Lie Group）是一种特殊的数学结构，它同时具有两种性质：
1. **群结构**：可以定义乘法（旋转的合成）和逆元（反向旋转）
2. **流形结构**：局部看起来像欧几里得空间，可以定义微积分

$\text{SO}(3)$ 就是一个李群。类似的，描述刚体完整位姿（旋转 $+$ 平移）的**特殊欧几里得群** $\text{SE}(3)$ 也是李群。

$$T = \begin{pmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix} \in \text{SE}(3)$$

其中 $R \in \text{SO}(3)$ 是旋转矩阵，$\mathbf{t} \in \mathbb{R}^3$ 是平移向量。

![李群 SE(3) 与李代数 se(3)](/images/plots/robotics-se3-lie-group.png)

*图1：李代数 se(3) 中的螺旋运动（左）与李群 SE(3) 中的刚体变换（右）。指数映射将切空间中的速度映射为流形上的姿态。*

### 1.3 李代数：李群的切空间

李群是弯曲的流形，但它在每一点都有一个**切空间**——局部看是平坦的。

在李群的单位元（恒等变换）处的切空间称为**李代数**（Lie Algebra）：
- $\text{SO}(3)$ 的李代数是 $\mathfrak{so}(3)$
- $\text{SE}(3)$ 的李代数是 $\mathfrak{se}(3)$

**$\mathfrak{so}(3)$ 的具体形式**：

$$\Omega = \begin{pmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{pmatrix}$$

这是一个反对称矩阵，由三维向量 $\mathbf{\omega} = (\omega_x, \omega_y, \omega_z)$ 唯一确定。物理上，$\mathbf{\omega}$ 就是**角速度**。

### 1.4 指数映射与对数映射

李代数和李群之间通过**指数映射**（Exponential Map）连接：

$$\exp: \mathfrak{so}(3) \to \text{SO}(3)$$

对于 $3 \times 3$ 反对称矩阵，这就是著名的**罗德里格斯公式**（Rodrigues' Formula）：

$$R = \exp(\Omega) = I + \frac{\sin\theta}{\theta}\Omega + \frac{1-\cos\theta}{\theta^2}\Omega^2$$

其中 $\theta = \|\mathbf{\omega}\|$ 是旋转角度。

**直观理解**：
- 李代数：速度空间（角速度 $\mathbf{\omega}$）
- 李群：姿态空间（旋转矩阵 $R$）
- 指数映射：积分速度，得到姿态变化

![指数映射](/images/plots/robotics-exponential-map.png)

*图2：指数映射将李代数中的角速度映射为李群中的旋转角度。小角度时近似线性，大角度时呈现周期性。*

逆映射称为**对数映射**：

$$\Omega = \log(R)$$

给定旋转矩阵 $R$，可以提取旋转轴和角度：

$$\theta = \arccos\left(\frac{\text{tr}(R) - 1}{2}\right)$$

$$\mathbf{\omega} = \frac{1}{2\sin\theta} \begin{pmatrix} R_{32} - R_{23} \\ R_{13} - R_{31} \\ R_{21} - R_{12} \end{pmatrix}$$

### 1.5 为什么这很重要？

**关键洞察**：在李代数空间中进行线性运算（如加法、插值），然后通过指数映射回到李群，这样得到的变换始终在流形上。

对比两种旋转插值方法：

**错误方法（线性插值）**：
$$R(t) = (1-t)R_1 + tR_2$$

问题：$R(t)$ 可能不是旋转矩阵（不正交，或行列式不为 $1$）。

**正确方法（测地线插值）**：
1. 计算相对旋转：$R_{12} = R_1^T R_2$
2. 取对数：$\mathbf{\omega}_{12} = \log(R_{12})$
3. 插值：$\mathbf{\omega}(t) = t \cdot \mathbf{\omega}_{12}$
4. 指数映射：$R(t) = R_1 \exp(\mathbf{\omega}(t))$

结果：$R(t)$ 始终是有效的旋转矩阵，且路径是"最短"的（测地线）。

![测地线插值对比](/images/plots/robotics-geodesic-interpolation.png)

*图3：SO(2) 上的插值对比。线性插值（左，红色）偏离流形（单位圆），而测地线插值（右，绿色）始终保持在流形上。*

---

## 第二章：机器人运动学——从关节空间到任务空间

### 2.1 运动学问题

机器人运动学研究几何关系，不考虑力和质量。核心问题有两个：

**前向运动学**（Forward Kinematics）：给定关节角度 $\mathbf{\theta}$，求末端执行器位姿 $T$。

**逆向运动学**（Inverse Kinematics）：给定目标位姿 $T$，求关节角度 $\mathbf{\theta}$。

对于串联机械臂，前向运动学是李群元素的乘积：

$$T_{0n}(\mathbf{\theta}) = e^{\mathbf{\xi}_1 \theta_1} e^{\mathbf{\xi}_2 \theta_2} \cdots e^{\mathbf{\xi}_n \theta_n} T_{0n}(0)$$

其中 $\mathbf{\xi}_i$ 是第 $i$ 个关节的螺旋轴（在李代数中）。

### 2.2 雅可比矩阵

当机器人运动时，关节角速度如何映射为末端执行器的速度？这就是**雅可比矩阵**（Jacobian）的作用。

$$\mathbf{V} = J(\mathbf{\theta}) \dot{\mathbf{\theta}}$$

其中：
- $\dot{\mathbf{\theta}} \in \mathbb{R}^n$ 是关节空间速度
- $\mathbf{V} \in \mathbb{R}^6$ 是任务空间速度（线速度 $+$ 角速度）
- $J(\mathbf{\theta}) \in \mathbb{R}^{6 \times n}$ 是雅可比矩阵

![雅可比矩阵映射](/images/plots/robotics-jacobian-mapping.png)

*图4：雅可比矩阵将关节空间速度映射到任务空间速度，并定义了可操作性椭球，反映机器人在各方向上的运动能力。*

**几何视角**：雅可比矩阵的列向量是各关节轴在末端执行器处的**螺旋运动**：

$$J = \begin{pmatrix} | & | & & | \\ \mathbf{\xi}_1' & \mathbf{\xi}_2' & \cdots & \mathbf{\xi}_n' \\ | & | & & | \end{pmatrix}$$

### 2.3 可操作性与速度椭球

雅可比矩阵的**奇异值分解**（SVD）揭示了机器人的运动能力：

$$J = U \Sigma V^T$$

其中 $\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_6)$ 包含奇异值。

**速度椭球**：单位关节速度 $\|\dot{\mathbf{\theta}}\| = 1$ 映射到任务空间形成一个椭球：

$$\{\mathbf{V} : \mathbf{V} = J\dot{\mathbf{\theta}}, \|\dot{\mathbf{\theta}}\| = 1\}$$

椭球的半轴长度就是奇异值 $\sigma_i$，方向由 $U$ 的列向量给出。

**可操作性度量**：

$$\mu = \sqrt{\det(J J^T)} = \sigma_1 \sigma_2 \cdots \sigma_6$$

当机器人处于奇异位形时，至少一个奇异值为零，可操作性 $\mu = 0$。

### 2.4 关节空间与任务空间的映射

![关节空间到任务空间](/images/plots/robotics-joint-task-space.png)

*图5：2关节机械臂的关节空间（左）与任务空间（右）映射。网格线在映射下发生扭曲，反映了非线性变换的几何特性。*

从图中可以看到：
- 关节空间是简单的矩形区域
- 任务空间呈现出复杂的边界
- 某些区域的映射是一一对应的，而另一些区域（如奇异点附近）是多对一的

---

## 第三章：机器人动力学——黎曼几何视角

### 3.1 从运动学到动力学

运动学回答"在哪里"，动力学回答"如何运动"。考虑质量、惯性和外力，我们需要**动力学方程**。

经典的欧拉-拉格朗日方程：

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{\mathbf{q}}} - \frac{\partial L}{\partial \mathbf{q}} = \mathbf{\tau}$$

其中 $L = T - V$ 是拉格朗日量，$T$ 是动能，$V$ 是势能，$\mathbf{\tau}$ 是广义力。

对于机器人，这可以写成标准的**操作空间动力学**形式：

$$M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \mathbf{\tau}$$

### 3.2 质量矩阵作为度量张量

观察质量矩阵 $M(\mathbf{q})$，它具有以下性质：
- **对称**：$M = M^T$
- **正定**：$\mathbf{v}^T M \mathbf{v} > 0$ 对所有非零 $\mathbf{v}$
- **依赖于构型**：$M = M(\mathbf{q})$

这不正是**黎曼度量**的定义吗？

**关键洞见**：机器人的构型空间配备了自然的黎曼度量 $M(\mathbf{q})$。在这个几何框架下：
- **动能**：$T = \frac{1}{2}\dot{\mathbf{q}}^T M(\mathbf{q}) \dot{\mathbf{q}}$
- **测地线**：无外力时的自然运动轨迹
- **克里斯托费尔符号**：由 $M$ 的导数定义，出现在科里奥利力项中

![关节空间势能](/images/plots/robotics-dynamics-energy.png)

*图6：关节空间的势能等高线（双势阱模型）。星号标记稳定平衡点，叉号标记不稳定平衡点。红色曲线展示了一个阻尼振荡轨迹。*

### 3.3 测地方程与动力学

在黎曼流形上，测地线满足**测地方程**：

$$\ddot{q}^i + \Gamma^i_{jk}\dot{q}^j\dot{q}^k = 0$$

其中 $\Gamma^i_{jk}$ 是**克里斯托费尔符号**：

$$\Gamma^i_{jk} = \frac{1}{2}g^{il}\left(\frac{\partial g_{lj}}{\partial q^k} + \frac{\partial g_{lk}}{\partial q^j} - \frac{\partial g_{jk}}{\partial q^l}\right)$$

对比机器人动力学方程：

$$M\ddot{\mathbf{q}} + C\dot{\mathbf{q}} = 0$$

（假设无重力和外力）

可以发现，科里奥利力项 $C\dot{\mathbf{q}}$ 正好对应克里斯托费尔符号的贡献！

### 3.4 力控制与阻抗控制

**问题**：如何让机器人与环境交互时表现柔顺？

**阻抗控制**：
$$\mathbf{\tau} = M(\mathbf{\theta})\ddot{\mathbf{\theta}}_d + C(\mathbf{\theta}, \dot{\mathbf{\theta}})\dot{\mathbf{\theta}}_d + \mathbf{g}(\mathbf{\theta}) + K_p(\mathbf{\theta}_d - \mathbf{\theta}) + K_d(\dot{\mathbf{\theta}}_d - \dot{\mathbf{\theta}})$$

**几何视角**：在李群框架下，误差计算必须在李代数中进行：
$$\mathbf{e} = \log(T_d^{-1} T)$$

而不是简单的欧几里得减法。

---

## 第四章：轨迹规划与优化——测地线与测地流

### 4.1 轨迹规划问题

给定起点 $\mathbf{q}_{\text{start}}$ 和终点 $\mathbf{q}_{\text{goal}}$，寻找一条可行的轨迹：
- 满足运动学约束
- 避障
- 优化某些性能指标（如时间、能量）

### 4.2 测地线作为最优路径

在黎曼流形上，**测地线**是局部最短路径。如果构型空间没有障碍物，测地线就是最优轨迹。

对于旋转，这意味着**球面线性插值**（SLERP）：

$$R(t) = R_0 \exp(t \log(R_0^T R_1))$$

而不是简单的线性插值：
$$R(t) = (1-t)R_0 + t R_1 \quad \text{（错误！）}$$

后者产生的矩阵甚至不是有效的旋转（不正交）。

### 4.3 RRT 与基于采样的规划

**在流形上的 RRT**：
1. 在切空间中采样
2. 使用指数映射投影回流形
3. 确保边是测地线段

**扩展 RRT\***：
- 利用黎曼度量计算路径代价
- 重新布线以优化总成本

### 4.4 最优控制与泛函极值

最小化能量泛函：
$$J = \int_0^T \frac{1}{2} \dot{\mathbf{\theta}}^T M(\mathbf{\theta}) \dot{\mathbf{\theta}} \, dt$$

**欧拉-拉格朗日方程**给出最优轨迹，这导出了测地线方程！

---

## 第五章：SLAM与状态估计——流形上的概率分布

### 5.1 SLAM问题

**同步定位与地图构建**（SLAM）是机器人学的核心问题：
- 机器人在未知环境中移动
- 同时估计自身位姿和构建环境地图
- 利用传感器（相机、激光雷达、IMU等）观测

### 5.2 流形上的状态估计

传统卡尔曼滤波假设状态在欧几里得空间中。但机器人的位姿 $T \in \text{SE}(3)$ 位于流形上！

**错误做法**：
$$T_{k+1} = T_k + \Delta T$$

问题：$T_{k+1}$ 可能不在 $\text{SE}(3)$ 上。

**正确做法**：
$$T_{k+1} = T_k \cdot \exp(\Delta \mathbf{\xi})$$

其中 $\Delta \mathbf{\xi} \in \mathfrak{se}(3)$ 是李代数增量，$\exp$ 是指数映射。

### 5.3 流形上的概率分布

如何在流形上定义高斯分布？

**方法：在切空间中定义**

对于点 $\bar{T} \in \text{SE}(3)$，在其切空间中定义高斯分布：

$$\mathbf{\xi} \sim \mathcal{N}(\mathbf{0}, \Sigma)$$

然后通过指数映射得到流形上的分布：

$$T = \bar{T} \cdot \exp(\mathbf{\xi})$$

这就是**聚焦高斯**（Concentrated Gaussian）。

### 5.4 李群上的卡尔曼滤波

**预测步骤**：
$$\hat{T}_{k|k-1} = \hat{T}_{k-1|k-1} \cdot \exp(\Delta t \cdot \mathbf{v}_k)$$
$$\Sigma_{k|k-1} = F_k \Sigma_{k-1|k-1} F_k^T + Q_k$$

**更新步骤**：
$$K_k = \Sigma_{k|k-1} H_k^T (H_k \Sigma_{k|k-1} H_k^T + R_k)^{-1}$$
$$\mathbf{\xi}_k = K_k \mathbf{y}_k$$
$$\hat{T}_{k|k} = \hat{T}_{k|k-1} \cdot \exp(\mathbf{\xi}_k)$$

### 5.5 Bundle Adjustment

视觉 SLAM 中的**光束法平差**（Bundle Adjustment）是一个非线性最小二乘问题：

$$\min_{T, X} \sum_{i,j} \rho\left(\|\mathbf{u}_{ij} - \pi(T_i, X_j)\|^2\right)$$

**关键**：优化在 $\text{SE}(3)$ 的切空间进行，使用李代数参数化位姿增量。

---

## 第六章：机器人实例分析

### 6.1 工业机械臂——UR5

UR5 是一个典型的 6 自由度串联机械臂。

**运动学结构**：
- 6 个旋转关节
- 腕部球关节（3 个相交轴）
- 满足 Pieper 准则：逆运动学有解析解

使用李群方法：
1. 分解位置与方向问题
2. 利用腕部几何特性
3. 最多 8 个解析解

### 6.2 并联机器人——Delta 机器人

Delta 机器人使用并联机构实现高速拾取。

**结构特点**：
- 3 个主动臂 + 3 个从动臂
- 末端执行器始终保持水平
- 闭环约束

雅可比矩阵 $J$ 将关节速度与末端速度关联：
$$\mathbf{V} = J \dot{\mathbf{q}}$$

但由于闭环约束，$J$ 不是方阵。

### 6.3 足式机器人——四足机器人

四足机器人（如 Spot、ANYmal）的运动涉及复杂的接触动力学。

**浮动基座**：
- 6 个自由度的浮动基座（位置和方向）
- 每个腿 3 个关节
- 总位形空间：$\mathbb{R}^3 \times \text{SO}(3) \times (S^1)^9$

**零空间运动**：
雅可比矩阵 $J_c$ 描述接触约束：
$$J_c \dot{\mathbf{q}} = 0$$

零空间 $\mathcal{N}(J_c)$ 中的运动保持接触，用于姿态调整。

---

## 第七章：现代发展——深度学习与机器人几何

### 7.1 几何深度学习

传统深度学习处理欧几里得数据。但机器人数据常定义在非欧几里得空间：
- 点云（$3$D 坐标集合）
- 网格（图结构）
- 位姿序列（流形上的轨迹）

**SE(3) 等变网络**：设计网络结构，使其对输入的 $\text{SE}(3)$ 变换保持等变。

### 7.2 视觉-惯性里程计

VIO 融合相机和 IMU 数据进行状态估计。

**状态空间**：
$$\mathbf{x} = (\mathbf{p}, \mathbf{v}, \mathbf{q}, \mathbf{b}_a, \mathbf{b}_g) \in \mathbb{R}^3 \times \mathbb{R}^3 \times \text{SO}(3) \times \mathbb{R}^3 \times \mathbb{R}^3$$

**扩展卡尔曼滤波**：
1. 状态传播：$\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)$
2. 误差在李代数中计算
3. 协方差在切空间中传播

### 7.3 学习黎曼度量

能否让机器人自己学习适合任务的度量？

**度量学习**（Metric Learning）从数据中学习距离函数：

$$d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^T M (\mathbf{x}-\mathbf{y})}$$

对于机器人，可以学习：
- 构型空间度量 $M(\mathbf{q})$
- 任务相关的距离
- 考虑人类演示的偏好

---

## 第八章：实践工具与实现

### 8.1 常用的数学库

**Sophus**（C++）：
- 实现 SO(3)、SE(3) 及其李代数
- 指数/对数映射
- 方便的运算符重载

**manif**（C++）：
- 轻量级流形库
- 支持多种李群
- 自动微分兼容

**GeoTorch**（Python/PyTorch）：
- 神经网络中的约束优化
- 流形上的梯度下降
- 适用于学习算法

### 8.2 数值注意事项

**旋转表示的选择**：

| 表示 | 优点 | 缺点 |
|-----|------|------|
| 旋转矩阵 | 无奇异，直接复合 | 9个参数，约束多 |
| 四元数 | 4个参数，插值方便 | 双覆盖，$q$ 和 $-q$ 等价 |
| 欧拉角 | 直观 | 万向节锁 |
| 李代数 | 优化友好 | 需要指数映射 |

**推荐**：内部计算用旋转矩阵或四元数，优化问题用李代数。

**积分旋转**：

更新旋转：
$$R_{k+1} = R_k \exp([\mathbf{\omega}]_\times \Delta t)$$

而不是：
$$R_{k+1} = R_k + \dot{R} \Delta t \quad \text{（不正交！）}$$

---

## 结语：几何之美，实用之真

回顾我们的旅程：

1. **李群与李代数**为描述刚体运动提供了优雅的数学框架
2. **雅可比矩阵**揭示了几何映射的局部线性结构
3. **黎曼度量**赋予了构型空间丰富的几何结构
4. **测地线**连接了最优轨迹与微分几何
5. **流形上的概率**让状态估计在正确的空间进行
6. **几何深度学习**将现代 AI 与经典几何结合

### 为什么微分几何如此重要？

**本质原因**：机器人系统的状态空间天然是流形。忽视这一点会导致：
- 无效的插值结果
- 数值不稳定
- 次优的规划
- 错误的概率模型

而拥抱几何，我们获得：
- 全局有效的表示
- 数值稳定的算法
- 几何直观的设计
- 理论保证的正确性

### 给读者的建议

如果你希望深入这个领域：

**数学基础**：
- 线性代数（矩阵分解、特征值）
- 多元微积分（链式法则、梯度）
- 微分几何（流形、张量、联络）

**实践技能**：
- 学习李群库（如 Sophus、Manif）
- 实现简单的运动学和动力学
- 在真实或仿真机器人上测试

**前沿方向**：
- 几何深度学习
- 学习-based 控制
- 多机器人协同

微分几何不仅仅是抽象的数学，它是理解机器人世界、构建智能系统的基石。从刚体姿态的描述，到复杂环境的导航，从单机器人的控制，到群体机器人的协调——几何无处不在。

希望这篇综述为你打开了通往机器人几何世界的大门。

---

## 附录：重要公式速查

### 李群与李代数

**SO(3) 的指数映射（罗德里格斯公式）**：
$$R = \exp(\Omega) = I + \frac{\sin\theta}{\theta}\Omega + \frac{1-\cos\theta}{\theta^2}\Omega^2$$

**SE(3) 的指数映射**：
$$T = \exp(\mathbf{\xi}^\wedge) = \begin{pmatrix} \exp(\Omega) & V\mathbf{v} \\ \mathbf{0}^T & 1 \end{pmatrix}$$

其中 $V = I + \frac{1-\cos\theta}{\theta^2}\Omega + \frac{\theta-\sin\theta}{\theta^3}\Omega^2$。

### 雅可比矩阵

**速度映射**：
$$\mathbf{V} = J(\mathbf{\theta})\dot{\mathbf{\theta}}$$

**可操作性**：
$$\mu = \sqrt{\det(J J^T)}$$

### 动力学

**欧拉-拉格朗日方程**：
$$M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \mathbf{\tau}$$

---

**延伸阅读**：
- Murray, Li, and Sastry. *A Mathematical Introduction to Robotic Manipulation*. CRC Press, 1994.
- Bullo and Lewis. *Geometric Control of Mechanical Systems*. Springer, 2005.
- Sola et al. "A micro Lie theory for state estimation in robotics." *arXiv:1812.01537*, 2018.
- Absil et al. *Optimization Algorithms on Matrix Manifolds*. Princeton University Press, 2008.

*愿你的机器人在弯曲的空间中，走出优美的轨迹。*
