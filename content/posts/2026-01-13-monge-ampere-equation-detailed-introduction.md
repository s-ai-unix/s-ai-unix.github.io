---
title: "蒙日-安培方程详解：历史、演进、推导与应用"
date: 2026-01-13T16:00:00+08:00
draft: false
description: "从历史脉络到核心公式推导，系统梳理蒙日-安培方程的理论演进与跨学科应用。"
categories: ["数学", "偏微分方程"]
tags: ["几何"]
mathjax: true
cover:
  image: "images/covers/1555255707-c07966088b7b.jpg"
  alt: "Monge–Ampère equation"
  caption: "Monge–Ampère 方程与最优传输"
---

$

\det(D^2 u) = f(x, u, \nabla u), \quad x \in \Omega \subset \mathbb{R}^n
$

其中 $u$ 通常为凸函数，$D^2 u$ 是 Hessian 矩阵，$\det(D^2 u)$ 表示 Hessian 的行列式。它是所有二阶偏导的“体积型”组合，与线性椭圆方程（如拉普拉斯方程）相比高度非线性。

### 2. 二维一般形式
$
A(u_{xx}u_{yy}-u_{xy}^2)+B u_{xx}+2C u_{xy}+D u_{yy}+E=0
$

其中 $A,B,C,D,E$ 可依赖于 $(x,y,u,u_x,u_y)$。当 $A \neq 0$ 时，方程具有典型的 Monge–Ampère 结构。

## 公式推导（核心思路）

### 1. 曲率处方式推导
设曲面由函数 $z = u(x)$ 给出，其高斯曲率为
$
K = \frac{\det(D^2 u)}{(1+|\nabla u|^2)^{(n+2)/2}}
$

因此，如果希望曲面具有给定曲率 $K(x)$，则必须满足
$
\det(D^2 u) = K(x)\,(1+|\nabla u|^2)^{(n+2)/2}
$

这正是 Monge–Ampère 方程的几何起源之一，也解释了其在凸几何问题（如 Minkowski 问题）中的核心地位。

### 2. 最优传输与雅可比行列式推导
设 $T: \Omega \to \Omega'$ 为传输映射，将密度 $f_\Omega$ 传输到 $f_{\Omega'}$，满足质量守恒：
$
\int_A f_\Omega(x)\,dx = \int_{T(A)} f_{\Omega'}(y)\,dy
$

若 $T = \nabla u$（Brenier 定理：二次代价下成立），则利用变换公式得到
$
f_{\Omega}(x) = f_{\Omega'}(\nabla u(x))\,\det(D^2 u(x))
$

因此
$
\det(D^2 u(x)) = \frac{f_{\Omega}(x)}{f_{\Omega'}(\nabla u(x))}
$

这被称为 Brenier–Monge–Ampère 方程，是最优传输的核心 PDE。

### 3. 椭圆性与凸性
若 $u$ 是凸函数，则 $D^2u$ 半正定，$\det(D^2u) > 0$。此时 Monge–Ampère 方程是退化椭圆型。若缺乏凸性，椭圆性失效，解理论会出现不适定。

## 解的类型与理论结构

### 1. Alexandrov 弱解
对于非光滑凸函数，定义 Monge–Ampère 测度：
$
\mu_u(E) = |\partial u(E)|
$

并用
$
\mu_u(E) = \int_E f(x)\,dx
$

作为弱解的定义基础。这一框架使得凸几何与 PDE 理论深度融合。

### 2. 正则性理论
Caffarelli 的工作表明在适当条件下（如 $f$ 有界且正、边界严格凸），解具备 $C^{1,\alpha}$、$W^{2,p}$ 乃至 $C^{2,\alpha}$ 正则性，是 Monge–Ampère 方程理论成熟的重要标志。

## 典型应用

### 1. 凸几何与曲率处方
- Minkowski 问题：给定面积测度，求凸体
- Weyl 问题：给定度量，嵌入曲面到 $\mathbb{R}^3$
- 仿射几何：仿射球面、仿射最大曲面

### 2. 最优传输与经济学
- 资源分配、匹配理论
- 图像配准与形状匹配
- 运输成本最小化与定价模型

### 3. 气象学与流体力学
半地转流方程（semigeostrophic equations）在变换变量下转化为 Monge–Ampère 方程，描述大气锋面形成与输运现象。

### 4. 几何光学与反射器设计
设计反射面或折射面，使得光能分布满足指定照度分布，本质上是最优传输问题。

### 5. 机器学习与生成模型
- Monge–Ampère flow 与生成模型
- 基于最优传输的密度映射与对齐
- 近年神经网络与 PDE 解法结合的数值研究

## 小结

Monge–Ampère 方程以“行列式约束”为核心，汇聚了几何、变分、最优传输与数值分析等多条理论线索。从 18 世纪的工程问题出发，它在 20 世纪建立起完善的弱解与正则性理论，在 21 世纪进一步扩展到数据科学与计算应用。

若用一句话概括：**Monge–Ampère 方程是“把几何与优化联系在一起”的非线性 PDE 桥梁**。

## 参考阅读（精选）

- [De Philippis & Figalli (2014), *The Monge–Ampère Equation and Its Link to Optimal Transportation*](https://www.ams.org/journals/bull/2014-51-04/S0273-0979-2014-01459-4/)
- [C. Mooney, *The Monge–Ampère Equation* 讲义](https://www.math.uci.edu/~mooneycr/MongeAmpere_Notes.pdf)
- [Trudinger & Wang (2008), *The Monge–Ampère Equation and Its Geometric Applications*](https://maths-people.anu.edu.au/~wang/publications/MA.pdf)
- [Nam Q. Le (2024), *Analysis of Monge–Ampère Equations*](https://www.ams.org/books/gsm/240/)
- [L. C. Evans (2001), *Partial Differential Equations and Monge–Kantorovich Mass Transfer*](https://math.berkeley.edu/~evans/Monge-Kantorovich.survey.pdf)
- [Monge 生平与历史背景（St Andrews）](https://mathshistory.st-andrews.ac.uk/Biographies/Monge/)
