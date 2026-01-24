---
title: "变分自编码器：从概率建模到深度生成的优雅桥梁"
date: 2026-01-24T18:30:00+08:00
draft: false
description: "深入解析变分自编码器（VAE）的数学原理与推导，从变分推断到 ELBO 优化，从重参数化到生成应用，完整呈现 VAE 的理论框架与实践价值"
categories: ["深度学习", "综述", "算法"]
tags: ["机器学习", "深度学习", "综述", "神经网络"]
cover:
    image: "images/covers/vae-network.jpg"
    alt: "变分自编码器网络结构示意图"
    caption: "VAE 编码器-解码器架构"
math: true
---

## 引言：概率与生成的交响曲

想象你在创作一幅肖像画。你观察模特的面容，记住她的眼睛形状、嘴角弧度、颧骨位置——这些是你观察到的具体特征。但当你拿起画笔时，你不仅仅是在复制这些特征，而是在大脑中提取出某种"风格特征"：一种抽象的、压缩的表示。然后，基于这个压缩表示，你重新生成一幅作品。

这就是**自编码器（Autoencoder）**的基本思想：将高维数据压缩到低维潜在空间，然后再从潜在空间重建原始数据。但传统的自编码器有一个致命缺陷：它学习的潜在空间是**确定性**的，这意味着我们无法从潜在空间中生成新的样本——我们只能重建已有的数据。

2013 年，Kingma 和 Welling 提出了**变分自编码器（Variational Autoencoder，VAE）**，它将变分推断的思想引入深度学习，通过将潜在变量建模为概率分布，使得我们能够：
1. 学习数据生成模型
2. 从潜在空间采样生成新的、从未见过的样本
3. 控制生成过程（通过操控潜在变量）

这不仅仅是一个算法，更是**概率图模型**与**深度学习**的完美结合。让我们一同踏上这段从变分推断到深度生成的优雅之旅。

## 第一章：自编码器基础

### 1.1 自编码器的直观理解

自编码器是一个神经网络，由两部分组成：
- **编码器（Encoder）**：$z = f_{\text{enc}}(x)$，将输入 $x$ 映射到潜在表示 $z$
- **解码器（Decoder）**：$\hat{x} = f_{\text{dec}}(z)$，从潜在表示重建输入

训练目标是让重建误差最小化：

$$\mathcal{L}_{\text{AE}} = \| x - \hat{x} \|^2$$

### 1.2 标准自编码器的局限性

标准自编码器的编码器学习的是一个**确定性映射**：对于每个输入 $x$，潜在变量 $z$ 是一个固定的向量。这带来两个问题：

1. **无法生成新样本**：因为我们不知道潜在空间的概率分布，无法采样新的 $z$ 来生成 $\hat{x}$
2. **潜在空间不连续**：即使输入 $x_1$ 和 $x_2$ 很相似，它们的潜在表示 $z_1$ 和 $z_2$ 可能相距很远

这些局限性推动我们思考：如果将潜在变量建模为**概率分布**，情况会怎样？

## 第二章：变分推断的核心思想

### 2.1 生成模型的框架

假设我们有一组观测数据 $\mathbf{x} = \{x^{(1)}, x^{(2)}, \ldots, x^{(N)}\}$，我们想要学习一个**生成模型**，其过程如下：

1. 从某个先验分布 $p(z)$ 中采样潜在变量 $z$
2. 通过概率分布 $p(x|z)$ 生成观测数据 $x$

这背后的**概率图模型**可以表示为：

$$z \rightarrow x$$

联合概率分布为：
$$p(x, z) = p(x|z) p(z)$$

### 2.2 困难所在：后验推断不可解

如果我们想要进行生成，关键在于计算**后验分布** $p(z|x)$：

$$p(z|x) = \frac{p(x|z) p(z)}{p(x)}$$

其中边缘似然（证据）$p(x)$ 通过积分得到：

$$p(x) = \int p(x|z) p(z) \, dz$$

**问题**：当 $z$ 是高维变量时，这个积分是**不可解**的（intractable）。这意味着我们无法精确计算后验分布 $p(z|x)$。

### 2.3 变分推断的解决方案

变分推断的核心思想是：用**可处理的近似分布** $q_{\phi}(z|x)$ 来逼近真实的后验 $p(z|x)$。这里的 $q_{\phi}(z|x)$ 是一个参数为 $\phi$ 的分布族，我们通过优化 $\phi$ 使其尽可能接近真实后验。

如何衡量两个分布的接近程度？我们使用**KL 散度（Kullback-Leibler Divergence）**：

$$D_{\text{KL}}(q_{\phi}(z|x) \| p(z|x)) = \mathbb{E}_{z \sim q} \left[ \log \frac{q_{\phi}(z|x)}{p(z|x)} \right]$$

KL 散度有两个重要性质：
1. $D_{\text{KL}}(q \| p) \geq 0$，等号成立当且仅当 $q = p$
2. KL 散度不是对称的，$D_{\text{KL}}(q \| p) \neq D_{\text{KL}}(p \| q)$

### 2.4 推导 ELBO（Evidence Lower Bound）

现在我们开始变分推断最关键的推导。我们的目标是让 $q_{\phi}(z|x)$ 逼近 $p(z|x)$，即最小化 $D_{\text{KL}}(q_{\phi}(z|x) \| p(z|x))$。

**第一步：展开 KL 散度**

$$\begin{aligned}
D_{\text{KL}}(q_{\phi}(z|x) \| p(z|x)) &= \mathbb{E}_{z \sim q} \left[ \log \frac{q_{\phi}(z|x)}{p(z|x)} \right] \\
&= \mathbb{E}_{z \sim q} \left[ \log \frac{q_{\phi}(z|x) p(x)}{p(x, z)} \right] \\
&= \mathbb{E}_{z \sim q} \left[ \log q_{\phi}(z|x) + \log p(x) - \log p(x, z) \right] \\
&= \log p(x) + \mathbb{E}_{z \sim q} [\log q_{\phi}(z|x) - \log p(x|z) - \log p(z)]
\end{aligned}$$

**第二步：重新整理**

$$\log p(x) = D_{\text{KL}}(q_{\phi}(z|x) \| p(z|x)) - \mathbb{E}_{z \sim q} [\log q_{\phi}(z|x)] + \mathbb{E}_{z \sim q} [\log p(x|z)] + \mathbb{E}_{z \sim q} [\log p(z)]$$

**第三步：定义 ELBO**

将右边的期望项合并，我们定义**证据下界（Evidence Lower Bound，ELBO）**：

$$\text{ELBO} = \mathbb{E}_{z \sim q} [\log p(x|z) + \log p(z) - \log q_{\phi}(z|x)]$$

于是我们有：

$$\log p(x) = D_{\text{KL}}(q_{\phi}(z|x) \| p(z|x)) + \text{ELBO}$$

**第四步：理解这个等式**

这个等式是 VAE 的核心。它的物理直觉是：
- $\log p(x)$ 是**常数**（它由数据决定，与 $q_{\phi}$ 无关）
- $D_{\text{KL}}(q_{\phi}(z|x) \| p(z|x)) \geq 0$
- 因此，**最大化 ELBO 等价于最小化 KL 散度**

换句话说，通过优化 ELBO，我们实际上是在让近似后验 $q_{\phi}(z|x)$ 接近真实后验 $p(z|x)$。

## 第三章：VAE 的数学推导

### 3.1 VAE 的概率模型设定

在 VAE 中，我们做出以下概率假设：

1. **先验分布**：潜在变量 $z$ 服从标准正态分布
   $$p(z) = \mathcal{N}(z; 0, I)$$

2. **似然（解码器）**：给定 $z$，$x$ 的条件分布为正态分布
   $$p_{\theta}(x|z) = \mathcal{N}(x; \mu_{\theta}(z), \sigma_{\theta}^2(z) I)$$
   
   其中 $\mu_{\theta}(z)$ 和 $\sigma_{\theta}(z)$ 是神经网络输出的均值和方差。

3. **近似后验（编码器）**：给定 $x$，$z$ 的条件分布为正态分布
   $$q_{\phi}(z|x) = \mathcal{N}(z; \mu_{\phi}(x), \text{diag}(\sigma_{\phi}^2(x)))$$
   
   其中 $\mu_{\phi}(x)$ 和 $\sigma_{\phi}(x)$ 是编码器网络的输出。

### 3.2 ELBO 的具体形式

对于高斯分布，ELBO 可以展开为两项：

$$\begin{aligned}
\text{ELBO} &= \mathbb{E}_{z \sim q} [\log p(x|z) + \log p(z) - \log q_{\phi}(z|x)] \\
&= \mathbb{E}_{z \sim q} [\log p(x|z)] - \mathbb{E}_{z \sim q} \left[ \log \frac{q_{\phi}(z|x)}{p(z)} \right] \\
&= \underbrace{\mathbb{E}_{z \sim q} [\log p(x|z)]}_{\text{重建误差项}} - \underbrace{D_{\text{KL}}(q_{\phi}(z|x) \| p(z))}_{\text{正则化项}}
\end{aligned}$$

**第一项：重建误差项**

$$\mathbb{E}_{z \sim q} [\log p(x|z)] = \mathbb{E}_{z \sim q} \left[ -\frac{1}{2\sigma^2} \| x - \mu_{\theta}(z) \|^2 - \frac{d}{2} \log(2\pi\sigma^2) \right]$$

如果我们假设 $\sigma^2$ 是常数，优化这一项等价于最小化重建误差 $\| x - \hat{x} \|^2$。

**第二项：KL 散度项**

对于两个高斯分布：
- $q_{\phi}(z|x) = \mathcal{N}(z; \mu_{\phi}, \text{diag}(\sigma_{\phi}^2))$
- $p(z) = \mathcal{N}(z; 0, I)$

KL 散度有解析解：

$$D_{\text{KL}}(\mathcal{N}(\mu, \Sigma) \| \mathcal{N}(0, I)) = \frac{1}{2} \left[ \text{tr}(\Sigma) + \mu^T \mu - d - \log \det(\Sigma) \right]$$

对于对角协方差矩阵，简化为：

$$D_{\text{KL}} = \frac{1}{2} \sum_{j=1}^{d} \left[ \sigma_{\phi,j}^2 + \mu_{\phi,j}^2 - 1 - \log \sigma_{\phi,j}^2 \right]$$

![2D 高斯分布的 KL 散度](/images/vae/kl_divergence_2d.png)

上图展示了两个高斯分布（先验 $p(z)$ 和近似后验 $q(z|x)$）之间的 KL 散度。蓝色等高线是标准正态分布，红色虚线是偏移后的近似后验。KL 散度衡量了两个分布的差异。

### 3.3 完整的 VAE 损失函数

VAE 的损失函数是 ELBO 的负数（最小化损失 = 最大化 ELBO）：

$$\mathcal{L}_{\text{VAE}}(\theta, \phi; x) = -\text{ELBO} = \mathbb{E}_{z \sim q} [-\log p_{\theta}(x|z)] + D_{\text{KL}}(q_{\phi}(z|x) \| p(z))$$

在实现中，我们通常使用**单个样本估计**期望：

$$\mathcal{L}_{\text{VAE}}(\theta, \phi; x) \approx -\log p_{\theta}(x|z^{(l)}) + D_{\text{KL}}(q_{\phi}(z|x) \| p(z))$$

其中 $z^{(l)}$ 是从 $q_{\phi}(z|x)$ 采样得到的单个样本。

## 第四章：重参数化技巧（Reparameterization Trick）

### 4.1 采样阻碍了梯度反向传播

现在我们面临一个关键问题：如何训练编码器 $q_{\phi}(z|x)$？

在损失函数中，$z$ 是从 $q_{\phi}(z|x)$ 采样的。这意味着：

$$z \sim \mathcal{N}(\mu_{\phi}(x), \sigma_{\phi}^2(x))$$

采样是一个**随机操作**，梯度无法通过采样过程反向传播。这就像我们试图对"掷骰子"求梯度——这是不可微的。

### 4.2 重参数化的天才之处

重参数化技巧的核心思想是：将随机性从参数中分离出来。

对于高斯分布采样：

$$z = \mu + \sigma \odot \epsilon$$

其中 $\epsilon \sim \mathcal{N}(0, I)$ 是从标准正态分布采样的噪声，$\odot$ 表示逐元素乘法。

**关键洞察**：
- $\mu$ 和 $\sigma$ 是神经网络的可学习参数
- $\epsilon$ 是随机噪声，但与 $\mu$ 和 $\sigma$ 无关
- 采样只对 $\epsilon$ 进行，不涉及 $\mu$ 和 $\sigma$

因此，梯度可以通过 $\mu$ 和 $\sigma$ 反向传播！

![重参数化技巧可视化](/images/vae/reparameterization.png)

上图展示了重参数化技巧的效果。通过将随机性分离为独立的噪声 $\epsilon$，我们可以对确定性参数 $\mu$ 和 $\sigma$ 进行梯度优化。

### 4.3 梯度流向的可视化

在重参数化后，计算图的梯度流向为：

$$\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mu} &= \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z} \\
\frac{\partial \mathcal{L}}{\partial \sigma} &= \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \sigma} = \frac{\partial \mathcal{L}}{\partial z} \odot \epsilon
\end{aligned}$$

因为 $\epsilon$ 是独立于 $\mu$ 和 $\sigma$ 的随机变量，梯度可以顺利传播。

### 4.4 网络架构

结合重参数化技巧，VAE 的完整架构如下：

1. **编码器**：$x \rightarrow \mu_{\phi}(x), \log \sigma_{\phi}^2(x)$
2. **采样**：$z = \mu_{\phi}(x) + \sigma_{\phi}(x) \odot \epsilon$
3. **解码器**：$z \rightarrow \hat{x} = \mu_{\theta}(z)$

在实现中，我们通常输出 $\log \sigma_{\phi}^2$ 而非 $\sigma_{\phi}^2$，以确保方差始终为正。

![ELBO 分解可视化](/images/vae/elbo_decomposition.png)

上图展示了 ELBO 的两个组成部分：证据 $\log p(x)$ 和负 KL 散度。ELBO 是这两者之差，最大化 ELBO 等价于最小化 KL 散度，同时保持足够的重建能力。

## 第五章：VAE 的网络结构

```mermaid
flowchart TB
    subgraph Encoder["编码器 qφ z|x"]
        A["输入 x"] --> B["隐藏层 1<br/>h1 = ReLU W1 x + b1"]
        B --> C["隐藏层 2<br/>h2 = ReLU W2 h1 + b2"]
        C --> D["均值 μφ x"]
        C --> E["对数方差 log σ²φ x"]
    end

    subgraph Sampling["重参数化采样"]
        F["噪声 ε ∼ N 0 I"] --> G["z = μφ x + σφ x ⊙ ε"]
        D --> G
        E --> G
    end

    subgraph Decoder["解码器 pθ x|z"]
        G --> H["隐藏层 1<br/>h'1 = ReLU W3 z + b3"]
        H --> I["隐藏层 2<br/>h'2 = ReLU W4 h'1 + b4"]
        I --> J["重建输出 x̂ = μθ z"]
    end

    subgraph Loss["损失计算"]
        K["重建误差<br/>-log pθ x|z"]
        L["KL 散度<br/>DKL qφ z|x || p z"]
        K --> M["总损失 ℒ = K + L"]
        L --> M
    end

    J --> K
    E --> L
    D --> L

    style "输入 x" fill:#007AFF,stroke:#007AFF,stroke-width:3px,color:#ffffff
    style "z = μφ x + σφ x ⊙ ε" fill:#FF9500,stroke:#FF9500,stroke-width:2px,color:#ffffff
    style "重建输出 x̂ = μθ z" fill:#34C759,stroke:#34C759,stroke-width:2px,color:#ffffff
    style "总损失 ℒ = K + L" fill:#FF3B30,stroke:#FF3B30,stroke-width:2px,color:#ffffff
```

这个架构图展示了 VAE 的完整前向传播过程：
1. 编码器将输入映射到潜在空间的均值和方差
2. 通过重参数化技巧采样潜在变量
3. 解码器重建输入
4. 计算重建误差和 KL 散度，形成总损失

## 第六章：具体应用

### 6.1 图像生成

VAE 最直观的应用是图像生成。训练完成后，我们可以：
1. 从先验 $p(z) = \mathcal{N}(0, I)$ 采样 $z$
2. 通过解码器 $p_{\theta}(x|z)$ 生成图像

例如，在 MNIST 数据集上训练的 VAE 可以生成各种手写数字；在人脸数据集上训练的 VAE 可以生成不同姿态、表情的人脸。

### 6.2 潜在空间的可视化与探索

VAE 的潜在空间具有良好的结构。我们可以：
1. **插值**：在两个潜在向量 $z_1$ 和 $z_2$ 之间进行线性插值，观察生成的图像如何平滑过渡
2. **操控**：找到控制特定属性的潜在维度（如旋转、光照），通过修改这个维度来控制生成图像

![潜在空间插值](/images/vae/latent_interpolation.png)

上图展示了在潜在空间中从点 $z_1$ 到 $z_2$ 的线性插值路径。绿色等高线表示先验分布，蓝色路径是插值线。这种平滑插值是 VAE 生成质量的重要指标。

### 6.3 异常检测

VAE 的重建误差可以用于异常检测：
- 训练数据：正常样本，VAE 能很好地重建
- 测试数据：如果样本偏离训练分布，VAE 重建误差会很大

这常用于：
- 工业缺陷检测
- 医疗影像异常识别
- 网络入侵检测

### 6.4 半监督学习

当只有部分数据有标签时，VAE 可以结合标签信息：
1. 有标签数据：使用分类损失
2. 无标签数据：使用 VAE 重建损失
3. 潜在变量同时包含内容和标签信息

### 6.5 文本生成

虽然 VAE 在文本生成中面临一些挑战（离散输入的梯度问题），但通过一些变体（如 Categorical-VAE），仍可用于：
- 文本风格转换
- 句子生成
- 机器翻译

## 第七章：VAE 的变体与扩展

### 7.1 条件 VAE（Conditional VAE，CVAE）

标准 VAE 生成时完全随机，而 CVAE 允许我们控制生成过程：

$$p(z|x, y) = \mathcal{N}(z; \mu_{\phi}(x, y), \text{diag}(\sigma_{\phi}^2(x, y)))$$

其中 $y$ 是条件变量（如类别标签、文本描述）。这允许我们：
- 生成特定类别的图像（如"生成数字 5"）
- 根据文本描述生成图像

### 7.2 β-VAE：解耦潜在变量

标准 VAE 的 KL 散度项权重固定为 1，而 β-VAE 引入超参数 $\beta$：

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{z \sim q} [-\log p_{\theta}(x|z)] + \beta \cdot D_{\text{KL}}(q_{\phi}(z|x) \| p(z))$$

- $\beta > 1$：更强的正则化，潜在变量更解耦（每个维度对应一个语义因子）
- $\beta < 1$：更好的重建质量，但潜在空间可能纠缠

![β-VAE 损失权衡](/images/vae/beta_vae_tradeoff.png)

上图展示了不同 $\beta$ 值对重建误差和 KL 散度的影响。绿色菱形标记了标准 VAE（$\beta=1$）。通过调整 $\beta$，我们可以在重建质量和潜在变量解耦之间进行权衡。

### 7.3 VAE-GAN 混合模型

VAE 生成的图像有时会模糊（因为损失函数是对数似然的变分下界，而非真实似然）。GAN 生成的图像清晰但难以训练。结合两者：

- **VAE 部分**：编码器-解码器结构，提供可解释的潜在空间
- **GAN 部分**：判别器判断图像真伪，提供对抗损失

混合损失：

$$\mathcal{L} = \mathcal{L}_{\text{VAE}} + \lambda \mathcal{L}_{\text{GAN}}$$

### 7.4 VQ-VAE：离散潜在空间

VQ-VAE（Vector Quantized-VAE）将连续潜在空间离散化：

1. 学习一个码本（codebook）$E = \{e_1, e_2, \ldots, e_K\}$
2. 对每个潜在向量 $z_e$，找到最近的码字 $e_k$：$z_q = e_k$
3. 使用 $z_q$ 进行重建

这带来两个优势：
- 潜在表示更紧凑
- 可以与自回归模型（如 PixelCNN、Transformer）结合

## 第八章：VAE 与其他生成模型的对比

### 8.1 VAE vs GAN

| 特性 | VAE | GAN |
|------|-----|-----|
| 训练稳定性 | 稳定 | 不稳定（模式崩溃） |
| 生成质量 | 较模糊 | 清晰锐利 |
| 潜在空间 | 良好的结构 | 难以解释 |
| 可控性 | 高 | 低 |
| 训练目标 | 明确（最大化 ELBO） | 博弈对抗 |

### 8.2 VAE vs Flow-based Models

- **Normalizing Flows**：可精确计算 $p(x)$，通过可逆变换建模复杂分布
- **优势**：精确的似然估计
- **劣势**：计算成本高，难以处理高维数据

VAE 提供了一个**近似**但**高效**的框架。

### 8.3 VAE vs Diffusion Models

- **Diffusion Models**：通过逐步添加噪声然后反转过程生成样本
- **优势**：生成质量极高（SOTA）
- **劣势**：生成速度慢（需要多次扩散步骤）

有趣的是，Diffusion Models 可以看作是 VAE 的极限情况（潜在空间无限维，扩散过程无限步）。

![VAE 训练过程曲线](/images/vae/training_curves.png)

上图展示了典型的 VAE 训练曲线。重建误差（上图）随训练逐渐下降，而 KL 散度（下图）逐渐增加，达到平衡。这反映了 VAE 在重建质量和潜在空间正则化之间的动态平衡。

## 第九章：数学深入：为什么 VAE 有效

### 9.1 信息论视角

ELBO 的两项有深刻的信息论含义：

$$\text{ELBO} = \mathbb{E}_{z \sim q} [\log p(x|z)] - D_{\text{KL}}(q_{\phi}(z|x) \| p(z))$$

- **重建项**：最大化 $I(x; z)$（互信息），即 $z$ 对 $x$ 的信息量
- **KL 项**：约束 $H(z)$（$z$ 的熵），防止 $q$ 偏离先验太远

这实际上是在做**率失真权衡（Rate-Distortion Tradeoff）**：
- 增加 $z$ 的维度（更多信息）→ 更好的重建
- 减小 $z$ 的维度（压缩）→ 更高的 KL 散度惩罚

### 9.2 几何视角：潜在流形学习

数据通常位于高维空间中的低维流形上。VAE 试图：
1. 将流形"压平"到潜在空间（编码器）
2. 通过先验 $p(z)$ 约束潜在空间的结构

KL 散度项确保不同数据点的潜在表示不会"聚集"在一起，而是覆盖整个潜在空间。

### 9.3 VAE 与 EM 算法的关系

VAE 的训练可以看作是 **EM 算法**的随机梯度版本：
- **E 步**：近似后验 $q_{\phi}(z|x)$
- **M 步**：优化生成模型 $p_{\theta}(x|z)$

与传统 EM 不同，VAE 通过神经网络参数化 $q_{\phi}$ 和 $p_{\theta}$，并使用随机梯度下降进行端到端训练。

## 结语：概率与确定性的优雅舞蹈

变分自编码器是深度学习中一个真正的杰作。它不仅仅是一个算法，更是一种思维方式——一种在**概率不确定性**与**深度学习的表达能力**之间找到完美平衡的方式。

回顾这段旅程，我们看到了：

1. **从确定性到概率**：将自编码器的确定性映射推广为概率分布
2. **从精确到近似**：接受后验推断的困难，采用变分近似
3. **从不可微到可微**：通过重参数化技巧，让梯度能够通过采样传播
4. **从重建到生成**：不仅学会重建，更学会创造

VAE 的优雅之处在于：
- **理论基础扎实**：建立在变分推断、信息论、概率图模型等成熟理论之上
- **实践价值丰富**：应用于图像生成、异常检测、半监督学习等多个领域
- **可解释性强**：潜在空间有明确的概率解释，易于分析和控制
- **扩展性强**：衍生出 CVAE、β-VAE、VQ-VAE 等众多变体

在深度学习的浪潮中，VAE 始终保持着独特的地位。它不是最"炫酷"的算法，却是最"经典"的算法之一；它不是生成质量最高的模型，却是最有理论保障的模型之一。

当我们站在 VAE 的基础上继续探索——无论是扩散模型、流模型，还是其他未知的生成范式——我们会发现，VAE 教给我们的关于**概率建模**和**变分优化**的智慧，始终是前行的指路明灯。

---

**参考文献**：
1. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.
2. Doersch, C. (2016). Tutorial on variational autoencoders. *arXiv preprint arXiv:1606.05908*.
3. Higgins, I., et al. (2017). beta-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR*.
4. Oord, A. van den, et al. (2017). Neural discrete representation learning. *NeurIPS*.
5. Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press. Chapter 20: Deep Generative Models.
