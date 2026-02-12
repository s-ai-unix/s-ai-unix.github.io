---
title: "神经网络算法演进：从感知机到 Transformer 的七十年征程"
date: 2026-01-15T23:55:00+08:00
draft: false
description: "回顾神经网络七十年发展历程，从感知机到 Transformer，详解核心算法的数学原理"
categories: ["深度学习", "神经网络", "算法"]
tags: ["神经网络", "深度学习", "机器学习"]
cover:
    image: "images/covers/neural-network-evolution.jpg"
    alt: "抽象神经网络连接图"
    caption: "智能网络的演进"
---

## 引言：智慧的萌芽

想象一下 1957 年的夏天，康奈尔大学的弗兰克·罗森布拉特（Frank Rosenblatt）在实验室里调试着一台早期的电子计算机。他正在实现一个大胆的想法——能否用数学模型模拟人类的大脑神经元？

这个想法在当时看起来近乎荒谬。人类大脑由数百亿个神经元组成，神经元之间通过突触连接，形成了一个令人眩晕的复杂网络。但罗森布拉特相信，如果我们能理解单个神经元的基本工作原理，就能一步步构建出能够学习的智能系统。

那时的学术界对机器学习充满怀疑。"机器怎么可能思考？"——这是当时的主流声音。但罗森布拉特和他的同道们坚持了下来，用数学公式编织着最初的神经之梦。

今天，当我们面对能够写出论文、创作艺术、驾驶汽车的深度学习系统时，很容易忘记这一切都始于一个简单的线性分类器。让我们放慢脚步，回顾这七十年的征程，感受数学的力量与思想的演进。

---

## 一、感知机：神经网络的起点（1957）

**时间：1957 年 - 弗兰克·罗森布拉特 (Frank Rosenblatt)**

### 历史的起点

1957 年，弗兰克·罗森布拉特在康奈尔航空实验室发明了感知机（Perceptron）。这是第一个能够学习的神经网络模型，被誉为"机器学习的开端"。

1962 年的《纽约客》杂志甚至专门报道了这个发明，称它为"会思考的机器"。那时的媒体兴奋中充满了对人工智能未来的无限遐想。

### 数学形式

#### 单个神经元的工作原理

一个感知机神经元接收 $d$ 维输入 $\mathbf{x} = (x_1, x_2, \ldots, x_d)^T$，每个输入对应一个权重 $w_i$，还有一个偏置 $b$。

神经元的输出是输入的加权和，然后通过**激活函数**：

$$
y = f(z) = f\left(\sum_{i=1}^{d} w_i x_i + b\right) = f(w^T x + b)
$$

其中 $z = \mathbf{w}^T \mathbf{x} + b$ 是净输入（net input）。

#### 激活函数

在最初的感知机中，激活函数是**符号函数**（sign function）：

$$
f(z) = \begin{cases}
1 & \text{if } z \geq 0 \\
-1 & \text{if } z < 0
\end{cases}
$$

因此，感知机是一个**二元分类器**。

#### 感知机的学习规则：Rosenblatt 规则

感知机的学习非常直观。给定一个训练样本 $(\mathbf{x}_i, y_i)$，其中 $y_i \in \{-1, +1\}$。

预测值为：

$$
\hat{y}_i = \text{sign}(\mathbf{w}^T \mathbf{x}_i + b)
$$

如果预测正确（$\hat{y}_i = y_i$），不更新权重。

如果预测错误，按以下规则更新：

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i
$$

$$
b \leftarrow b + \eta y_i
$$

其中 $\eta$ 是学习率。

这个规则被称为**Rosenblatt 规则**，是梯度下降的一个简化形式。

#### 感知机的局限性：异或问题

1969 年，明斯基和佩伯特在《感知机》一书中证明了感知机的致命弱点：它无法解决**非线性可分**的问题，最著名的例子就是**异或**（XOR）问题。

异或问题的真值表：

| $x_1$ | $x_2$ | $x_1 \oplus x_2$ |
|------|------|--------------|
| $-1$   | $-1$   | $-1$           |
| $-1$   | $+1$   | $+1$           |
| $+1$   | $-1$   | $+1$           |
| $+1$   | $+1$   | $-1$           |

如果我们尝试用一条直线（决策边界）来分类这四个点，你会发现这是不可能的。因为单层感知机只能产生线性决策边界。

这个发现一度让神经网络研究进入寒冬。直到 1980 年代，多层感知机和非线性激活函数的引入才打破了僵局。

---

## 二、多层感知机与反向传播：深度学习的复兴（1986）

**时间：1986 年 - 大卫·鲁梅尔哈特 (David Rumelhart) 等**

### 寒冬后的复苏

在感知机被证明无法解决 XOR 问题后，神经网络研究沉寂了近二十年。直到 1986 年，大卫·鲁梅尔哈特、杰弗里·辛顿（Geoffrey Hinton）和罗纳德·威廉姆斯（Ronald Williams）在《Nature》上发表了题为《通过误差反向传播学习表征》的论文，提出了**反向传播算法**（Backpropagation）。

这篇论文开启了现代深度学习的大门。它解决的核心问题是：当网络有多层时，如何高效地计算梯度？

### 数学推导：反向传播的核心思想

#### 前向传播（Forward Propagation）

考虑一个多层感知机（MLP），包含：
- 输入层：d 个神经元
- 隐藏层：m 个神经元
- 输出层：c 个神经元（c 个类别）

设 $W^{(1)} \in \mathbb{R}^{m \times d}$ 是输入层到隐藏层的权重矩阵，$b^{(1)} \in \mathbb{R}^m$ 是隐藏层的偏置向量。

设 $W^{(2)} \in \mathbb{R}^{c \times m}$ 是隐藏层到输出层的权重矩阵，$b^{(2)} \in \mathbb{R}^c$ 是输出层的偏置向量。

**隐藏层的净输入**：

$$
z^{(1)} = W^{(1)} x + b^{(1)}
$$

**隐藏层的输出**（使用非线性激活函数，如 sigmoid）：

$$
a^{(1)} = \sigma(z^{(1)})
$$

其中 $\sigma$ 逐元素应用 sigmoid 函数：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**输出层的净输入**：

$$
z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}
$$

**输出层的输出**（使用 softmax，用于多分类）：

$$
\hat{y} = \text{softmax}(z^{(2)})
$$

其中 softmax 函数将 $c$ 个实数转换为概率分布：

$$
\text{softmax}(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{c} e^{z_k}}
$$

#### 损失函数：交叉熵

使用交叉熵损失：

$$
L = -\sum_{i=1}^{c} y_i \log(\hat{y}_i)
$$

其中 $y$ 是 one-hot 编码的真实标签。

#### 反向传播：链式法则

核心思想：使用**链式法则**（Chain Rule）计算损失函数对每个参数的梯度。

首先计算输出层的误差：

$$
\delta^{(2)} = \hat{y} - y
$$

这是 softmax 交叉熵损失对净输入的导数（一个优雅的简化）。

输出层权重的梯度：

$$
\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} (a^{(1)})^T
$$

输出层偏置的梯度：

$$
\frac{\partial L}{\partial b^{(2)}} = \delta^{(2)}
$$

然后，将误差**反向传播**到隐藏层。隐藏层的误差是：

$$
\delta^{(1)} = (W^{(2)})^T \delta^{(2)} \odot \sigma'(z^{(1)})
$$

其中 $\odot$ 是逐元素乘法，$\sigma'$ 是 sigmoid 的导数：

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

隐藏层权重的梯度：

$$
\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} x^T
$$

隐藏层偏置的梯度：

$$
\frac{\partial L}{\partial b^{(1)}} = \delta^{(1)}
$$

#### 为什么称为"反向传播"？

前向传播是从输入层到输出层：输入 $\to$ 隐藏层 $\to$ 输出层。

反向传播是从输出层到输入层：输出误差 $\to$ 隐藏层误差 $\to$ 输入层误差。

这就像在计算器中"反向"流动，因此得名。

#### 参数更新

使用梯度下降更新参数：

$$
W^{(1)} \leftarrow W^{(1)} - \eta \frac{\partial L}{\partial W^{(1)}}
$$

$$
b^{(1)} \leftarrow b^{(1)} - \eta \frac{\partial L}{\partial b^{(1)}}
$$

$$
W^{(2)} \leftarrow W^{(2)} - \eta \frac{\partial L}{\partial W^{(2)}}
$$

$$
b^{(2)} \leftarrow b^{(2)} - \eta \frac{\partial L}{\partial b^{(2)}}
$$

其中 $\eta$ 是学习率。

---

## 三、卷积神经网络：感受野的智慧（1998-2012）

**时间：1998 年 - LeNet-5；2012 年 - AlexNet**

### 从视觉感知到卷积

1998 年，杨·勒昆（Yann LeCun）和他的团队提出了 LeNet-5，这是第一个成功的卷积神经网络（Convolutional Neural Network, CNN）。它在 MNIST 手写数字识别任务上达到了当时的最先进水平。

卷积神经网络的核心洞察来自对生物视觉系统的研究：人类视觉皮层的神经元具有**局部感受野**（receptive field），即每个神经元只响应视野中的一小部分区域，而不是整个视野。

### 数学形式

#### 卷积操作（Convolution）

卷积神经网络的核心是**卷积层**（Convolutional Layer）。给定输入特征图 $X \in \mathbb{R}^{H \times W \times C_{\text{in}}}$（高 $H$、宽 $W$、输入通道数 $C_{\text{in}}$），卷积核 $K \in \mathbb{R}^{k \times k \times C_{\text{in}} \times C_{\text{out}}}$（高 $k$、宽 $k$、$C_{\text{out}}$ 个输出通道），卷积操作定义为：

$$
(X * K)_{i,j,o} = \sum_{c=1}^{C_{\text{in}}} \sum_{p=1}^{k} \sum_{q=1}^{k} X_{i+p-1, j+q-1, c} \cdot K_{p,q,c,o}
$$

其中 $*$ 表示卷积运算，$i, j$ 是输出特征图的空间坐标，$o$ 是输出通道索引。

更简洁的矩阵表示：

$$
Y = X * K
$$

其中 $Y \in \mathbb{R}^{H' \times W' \times C_{\text{out}}}$ 是输出特征图（空间大小为 $H' = H - k + 1$, $W' = W - k + 1$）。

#### 池化层（Pooling Layer）

为了减少计算量和参数数量，同时增加平移不变性，CNN 引入了**池化层**（Pooling Layer）。最常用的是**最大池化**（Max Pooling）：

$$
Y_{i,j,o} = \max_{p,q ∈ N_{i,j}} X_{p,q,o}
$$

其中 $N_{i,j}$ 是位置 $(i, j)$ 附近的窗口（如 $2 \times 2$）。

#### LeNet-5 架构

LeNet-5 的架构包括：

1. 输入层：$32 	imes 32 灰度图像
2. 卷积层 C1：6 个 $5 	imes 5 卷积核，输出 $28 	imes 28 	imes 6
3. 池化层 S2：$2 	imes 2 最大池化，输出 $14 	imes 14 	imes 6
4. 卷积层 C3：16 个 5×5 卷积核，输出 $10 	imes 10 	imes 16
5. 池化层 S4：2×2 最大池化，输出 $5 	imes 5 	imes 16
6. 全连接层 F5：120 个神经元
7. 全连接层 F6：84 个神经元
8. 输出层 F7：10 个神经元（对应 10 个数字）

#### AlexNet 的革命（2012）

2012 年，Alex Krizhevsky、Ilya Sutskever 和 Geoffrey Hinton 提出了 AlexNet，在 ImageNet 大规模视觉识别挑战赛（ILSVRC-2012）上以压倒性优势夺冠。

AlexNet 的创新点：
1. 使用 **ReLU**（Rectified Linear Unit）激活函数，加速收敛：
   $$f(z) = \max(0, z)$$
2. 使用 **Dropout** 随机失活，防止过拟合：
   - 训练时以概率 p 随机将神经元的输出设为 0
   - 测试时使用所有神经元，但输出乘以 p
3. 使用 **数据增强**（Data Augmentation）：随机裁剪、水平翻转等
4. 使用 **GPU** 并行计算

#### ReLU 的导数

ReLU 函数的导数是：

$$
\frac{\partial f(z)}{\partial z} = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

ReLU 的优点：
- 计算效率高（无指数运算）
- 缓解梯度消失问题（正值梯度恒为 1）

---

## 四、循环神经网络：记忆的艺术（1990s）

**时间：1990 年 - 1997 年 LSTM**

### 序列数据的挑战

前馈神经网络（如 MLP 和 CNN）假设输入和输出之间是独立的映射关系。但对于序列数据（如语言、语音、时间序列），当前时刻的输入依赖于历史信息。

循环神经网络（Recurrent Neural Network, RNN）的核心思想是：神经网络的输出不仅取决于当前输入，还取决于**隐藏状态**（hidden state），后者记忆了过去的信息。

### 数学形式

#### RNN 的基本结构

考虑一个时间序列 $x_1, x_2, \ldots, x_T$，每个时间步 $t$ 的输入是 $x_t \in \mathbb{R}^d$。

RNN 维护一个隐藏状态 $h_t \in \mathbb{R}^m$，按时间递归更新：

$$
h_t = f(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

其中：
- $W_{xh} \in \mathbb{R}^{m \times d}$ 是输入到隐藏的权重矩阵
- $W_{hh} \in \mathbb{R}^{m \times m}$ 是隐藏到隐藏的权重矩阵
- $b_h \in \mathbb{R}^m$ 是隐藏层的偏置
- $f$ 是激活函数（如 $\tanh$ 或 ReLU）

每个时间步的输出是：

$$
y_t = g(W_{hy} h_t + b_y)
$$

其中：
- $W_{hy} \in \mathbb{R}^{c \times m}$ 是隐藏到输出的权重矩阵（$c$ 是输出维度）
- $b_y \in \mathbb{R}^c$ 是输出的偏置
- $g$ 是输出激活函数（如 softmax）

#### 展开的时间图

将 RNN 按时间展开，可以看到信息如何从 $t=1$ 传递到 $t=T$：

$$
h_1 = f(W_{xh} x_1 + b_h)
$$

$$
h_2 = f(W_{xh} x_2 + W_{hh} h_1 + b_h)
$$

$$
h_3 = f(W_{xh} x_3 + W_{hh} h_2 + b_h)
$$

$$
\vdots
$$

$$
h_T = f(W_{xh} x_T + W_{hh} h_{T-1} + b_h)
$$

可以看到，$h_T$ 依赖于所有之前的输入 $x_1, x_2, \ldots, x_T$，这就是 RNN 的记忆机制。

#### 反向传播通过时间（BPTT）

RNN 的训练需要考虑时间依赖性，梯度需要**反向传播通过时间**（Backpropagation Through Time, BPTT）。

损失函数：

$$
L = \sum_{t=1}^{T} \ell(y_t, \hat{y}_t)
$$

其中 ℓ 是单个时间步的损失（如交叉熵）。

通过链式法则计算梯度：

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}}
$$

其中 $\partial h_t/\partial W_{hh}$ 是递归的：

$$
\frac{\partial h_t}{\partial W_{hh}} = \sum_{k=t}^{T} \prod_{j=k+1}^{t} f'(z_j) W_{hh}
$$

这个求和表明：$W_{hh}$ 的梯度依赖于所有时间步，导致**梯度消失**（vanishing gradient）或**梯度爆炸**（exploding gradient）问题。

#### tanh 的导数

tanh 激活函数的导数：

$$
\frac{\partial \tanh(z)}{\partial z} = 1 - \tanh^2(z) \leq 1
$$

如果 $W_{hh}$ 的特征值都小于 1，乘积会趋于 0，导致梯度消失。

---

## 五、LSTM：长记忆的解决方案（1997）

**时间：1997 年 - Sepp Hochreiter 和 Jürgen Schmidhuber**

### 梯度消失的问题

在长序列中，RNN 的梯度会呈指数衰减或增长。考虑 tanh 激活函数的导数：

$$
\tanh'(z) = 1 - \tanh^2(z) \leq 1
$$

如果 $W_{hh}$ 的特征值都小于 1，乘积会趋于 0，导致梯度消失。

### LSTM 的核心创新

1997 年，Sepp Hochreiter 和 Jürgen Schmidhuber 提出了**长短期记忆网络**（Long Short-Term Memory, LSTM），通过引入**门控机制**（gating mechanism）解决梯度消失问题。

LSTM 的细胞状态（cell state）和隐藏状态（hidden state）分离：

$$
c_t = \text{遗忘门} \odot c_{t-1} + \text{输入门} \odot \tilde{c}_t
$$

$$
h_t = \text{输出门} \odot \tanh(c_t)
$$

其中 $\odot$ 是逐元素乘法，$\tilde{c}_t$ 是候选细胞状态。

#### 遗忘门（Forget Gate）

遗忘门决定保留多少旧信息：

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

其中 $\sigma$ 是 sigmoid 函数，输出在 $[0, 1]$ 之间。

#### 输入门（Input Gate）

输入门决定写入多少新信息：

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

#### 候选细胞状态（Candidate Cell State）

$$
\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
$$

#### 输出门（Output Gate）

输出门决定输出多少信息到隐藏状态：

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

#### 细胞状态更新

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

#### 隐藏状态更新

$$
h_t = o_t \odot \tanh(c_t)
$$

### 为什么 LSTM 解决了梯度消失问题？

关键在于细胞状态的更新：

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t
$$

如果遗忘门 $f_t$ 接近 1，梯度几乎无损地传播。门控机制让网络学会**何时保留**和**何时遗忘**信息。

---

## 六、注意力机制：打破序列依赖（2017）

**时间：2017 年 - Vaswani 等**

### 从循环到注意力

在 Transformer 出现之前，序列建模主要依赖 RNN 及其变体（LSTM、GRU）。但 RNN 有两个根本限制：
1. **顺序计算**：必须从 $t=1$ 计算到 $t=T$，无法并行
2. **长距离依赖**：即使有 LSTM，信息仍难以从 $t=1$ 传递到 $t=T$

2017 年，Vaswani 等人在《Attention Is All You Need》中提出了**Transformer**，完全摒弃了循环结构，只用**注意力机制**（Attention Mechanism）。

### 数学形式：自注意力（Self-Attention）

考虑一个序列 $X = [x_1, x_2, \ldots, x_T] \in \mathbb{R}^{T \times d}$，其中 $T$ 是序列长度，$d$ 是嵌入维度。

#### Query、Key、Value

Transformer 的核心是**自注意力**（Self-Attention）。对于每个位置，我们计算三组向量：

$$
Q_i = x_i W^Q, \quad K_i = x_i W^K, \quad V_i = x_i W^V
$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ 是可学习的参数矩阵，$d_k$ 是查询/键/值的维度。

#### 注意力分数

对于位置 $i$ 和 $j$，计算注意力分数（缩放点积）：

$$
\text{score}_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_k}}
$$

#### 注意力权重

将分数转换为概率分布（使用 softmax）：

$$
\alpha_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_{k=1}^{T} \exp(\text{score}_{ik})}
$$

#### 加权求和

对值向量加权求和，得到输出：

$$
z_i = \sum_{j=1}^{T} \alpha_{ij} V_j
$$

#### 矩阵形式

可以写成矩阵形式：

$$
Z = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中：
- $Q = X W^Q \in \mathbb{R}^{T \times d_k}$
- $K = X W^K \in \mathbb{R}^{T \times d_k}$
- $V = X W^V \in \mathbb{R}^{T \times d_v}$

#### 为什么称为"注意力"？

$\alpha_{ij}$ 表示位置 $i$ 对位置 $j$ 的"关注程度"。如果 $\alpha_{ij}$ 接近 1，说明位置 $i$ 通常关注位置 $j$。

#### 多头注意力（Multi-Head Attention）

为了捕获不同类型的关系，Transformer 使用**多头注意力**（Multi-Head Attention）：

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h) W^O
$$

其中每个头是独立的自注意力：

$$
\text{head}_i = \text{Attention}(X W_i^Q, X W_i^K, X W_i^V)
$$

$W^O \in \mathbb{R}^{h \times d_v \times d_{\text{model}}}$ 是输出投影矩阵，$d_{\text{model}}$ 是模型维度。

---

## 七、残差连接：深层网络的关键（2015）

**时间：2015 年 - 何恺明等**

### 深度网络的训练困境

随着网络深度增加，我们遇到了两个问题：
1. **梯度消失**：深层网络的梯度难以传播到早期层
2. **退化问题**：网络深度增加后，训练误差反而增加（即使没有过拟合）

2015 年，何恺明等人提出了**残差连接**（Residual Connection），解决了这个问题。

### 数学形式

#### 残差块（Residual Block）

普通层的映射是 $F(x, \{W_i\})$，残差块学习的是**残差映射**：

$$
R(x, \{W_i\}) = F(x, \{W_i\}) - x
$$

其中 $F(x, \{W_i\})$ 是残差函数（通常由 2-3 个卷积层组成），$\{W_i\}$ 是可学习的权重。

残差块的输出是：

$$
y = F(x, {W_i}) + x
$$

这被称为**跳跃连接**（skip connection）或**快捷路径**（shortcut path）。

#### 为什么有效？

考虑 L 层残差网络。前向传播可以写成：

$$
x_{l+1} = x_l + F(x_l, W_l)
$$

因此：

$$
x_L = x_0 + \sum_{l=0}^{L-1} F(x_l, W_l)
$$

这意味着梯度可以直接从第 L 层传播到第 0 层：

$$
\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_L} · \prod_{l=0}^{L-1} \left(1 + \frac{\partial F(x_l, W_l)}{\partial x_l}\right)
$$

残差连接中的恒等映射（identity mapping）确保梯度至少为 1，解决了梯度消失问题。

---

## 八、现代架构：大模型的前奏（2018）

**时间：2018 年 - BERT；2020 年 - GPT-3；2022 年 - ChatGPT**

### 从 NLP 到通用智能

2018 年，Google 提出了**BERT**（Bidirectional Encoder Representations from Transformers），将预训练-微调（pre-training and fine-tuning）范式推向主流。

BERT 的创新点：
1. **掩码语言模型**（Masked Language Model, MLM）：随机掩盖输入 tokens 的 15%，让模型预测
2. **下一句预测**（Next Sentence Prediction, NSP）：预测两个句子是否相邻
3. **双向编码**：使用 Transformer 的编码器，同时看到左右上下文

2020 年，OpenAI 发布了**GPT-3**（Generative Pre-trained Transformer 3），展示了超大规模模型的涌现能力。

GPT-3 的关键：
1. **参数规模**：1750 亿参数
2. **少样本学习**（Few-shot Learning）：只需几个例子就能学会新任务
3. **零样本学习**（Zero-shot Learning）：无需任何例子

### Transformer 的编码器-解码器架构

#### 编码器（Encoder）

编码器处理输入序列，输出固定维度的表示：

$$
Z = \text{Encoder}(X)
$$

其中 $Z \in \mathbb{R}^{T \times d_{\text{model}}}$ 是编码后的表示。

#### 解码器（Decoder）

解码器生成输出序列：

$$
\hat{y}_t = \text{softmax}(z_t W_{vocab})
$$

其中 $W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times |\text{Vocab}|}$ 是词表矩阵，$z_t$ 是解码器在时间 $t$ 的隐藏状态。

#### 编码器-解码器注意力

解码器通过**交叉注意力**（Cross-Attention）关注编码器的输出：

$$
z_t = \text{Attention}(Q_t, K, V)
$$

其中 $Q_t = z_{t-1} W^Q$ 是解码器的查询，$K = Z W^K$ 和 $V = Z W^V$ 是编码器的键和值。

---

## 结语：从单神经元到通用智能

1957 年的感知机只是一个线性分类器。但七十年后的今天，我们有：
- 数千亿参数的模型
- 能理解复杂语言
- 能生成艺术作品
- 能辅助科学发现

这七十年的征程，本质上是数学和思想的演进：
1. **感知机**：理解单个神经元
2. **反向传播**：理解如何学习多层网络
3. **卷积网络**：理解空间结构
4. **循环网络**：理解时间依赖
5. **LSTM**：理解长期记忆
6. **注意力**：理解全局关系
7. **残差连接**：理解深层网络
8. **预训练模型**：理解大规模学习

每一个突破都建立在前人的基础上，用数学公式表达了对智能的新理解。

今天的深度学习仍有很多未解之谜：如何实现真正的推理？如何获得常识？如何解释模型的决策？

但回望过去，我们有理由相信：只要坚持用数学和实验探索未知，终有一天，我们会解开这些谜题，创造出真正的通用智能。

---

**参考文献**：

1. Rosenblatt, F. (1958). *The perceptron: A probabilistic model for information storage and organization in the brain*.
2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors". *Nature*.
3. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition". *Proceedings of the IEEE*.
4. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet classification with deep convolutional neural networks". *NIPS*.
5. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory". *Neural Computation*.
6. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). "Attention is all you need". *NIPS*.
7. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition". *CVPR*.
8. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". *NAACL-HLT*.
