---
title: "傅里叶变换：从正弦波到频谱的秘密"
date: 2026-01-12T22:30:00+08:00
draft: false
description: "从零开始详细推导傅里叶级数和傅里叶变换，包括三角函数正交性、复数形式、频谱分析，以及信号处理、图像处理等实际案例"
categories: ["数学", "信号处理"]
tags: []
cover:
    image: "images/covers/1550684848-fac1c5b4e853.jpg"
    alt: "傅里叶变换与信号处理"
    caption: "傅里叶变换：频谱分析的核心"
---

## 引言：分解的艺术

### 一个古老的问题

1822年，法国数学家约瑟夫·傅里叶（Joseph Fourier）在研究热传导问题时，提出了一个革命性的观点：**任何复杂的周期函数都可以分解为简单正弦波的叠加**。

这个想法在当时引起了巨大的争议。拉格朗日等数学家认为这是不可能的——毕竟，三角函数和任意的周期函数看起来如此不同。

然而，傅里叶是正确的。这个看似简单的主张，打开了信号处理、分析数学乃至整个现代工程学的大门。

### 傅里叶变换的力量

今天，傅里叶变换无处不在：

- **音乐**：你的Spotify音乐被压缩时，背后是傅里叶变换在工作
- **图像**：手机摄像头的图像处理、JPEG压缩，都依赖傅里叶方法
- **医学**：CT扫描和MRI使用傅里叶重建技术生成人体内部图像
- **通信**：WiFi、5G、蓝牙——所有无线通信都使用傅里叶变换来传输数据
- **金融**：分析师用傅里叶方法来发现数据中的周期性模式

**核心思想**：在"时间域"或"空间域"中复杂的信号，在"频率域"中可能变得极其简单。

### 这篇文章的目标

在接下来的篇幅中，我们将从最基本的三角函数开始，一步一步地推导出傅里叶级数和傅里叶变换。我们会看到：

1. 为什么正弦波是"最基本"的函数
2. 如何将任意函数分解为正弦波的叠加
3. 傅里叶变换的数学本质是什么
4. 傅里叶变换在实际问题中的强大应用

让我们开始这段数学之旅。

---

## 第一章：三角函数的正交性

### 1.1 什么是正交？

在向量空间中，**正交**（orthogonal）意味着两个向量垂直，它们的点积为零。

$$\mathbf{u} \cdot \mathbf{v} = 0 \quad \Rightarrow \quad \mathbf{u} \perp \mathbf{v}$$

这个概念可以推广到函数。一个函数集合如果满足某种"点积"为零的条件，我们就说它们是**正交**的。

### 1.2 函数的"点积"

对于定义在区间 $[a, b]$ 上的两个函数 $f(x)$ 和 $g(x)$，我们定义它们的"点积"为：

$$\langle f, g \rangle = \int_a^b f(x) g(x) \, dx$$

如果 $\langle f, g \rangle = 0$，我们就说 $f$ 和 $g$ 在区间 $[a, b]$ 上**正交**。

### 1.3 三角函数的正交性

考虑三角函数系：

$$\{1, \cos(x), \sin(x), \cos(2x), \sin(2x), \cos(3x), \sin(3x), \dots\}$$

这些函数在任意长度为 $2\pi$ 的区间上都是正交的。

**证明1：常数与正弦/余弦正交**

$$\int_{-\pi}^{\pi} 1 \cdot \cos(nx) \, dx = \left[\frac{\sin(nx)}{n}\right]_{-\pi}^{\pi} = 0$$

$$\int_{-\pi}^{\pi} 1 \cdot \sin(nx) \, dx = \left[-\frac{\cos(nx)}{n}\right]_{-\pi}^{\pi} = 0$$

**证明2：不同频率的正弦函数正交**

$$\int_{-\pi}^{\pi} \sin(mx) \sin(nx) \, dx = \begin{cases} 0 & m \neq n \\ \pi & m = n \end{cases}$$

使用积化和差公式：$\sin A \sin B = \frac{1}{2}[\cos(A-B) - \cos(A+B)]$

$$\int_{-\pi}^{\pi} \sin(mx) \sin(nx) \, dx = \frac{1}{2} \int_{-\pi}^{\pi} [\cos((m-n)x) - \cos((m+n)x)] \, dx$$

当 $m \neq n$ 时，两个积分都为零。

当 $m = n$ 时：

$$\int_{-\pi}^{\pi} \sin^2(nx) \, dx = \frac{1}{2} \int_{-\pi}^{\pi} [1 - \cos(2nx)] \, dx = \frac{1}{2} \cdot 2\pi = \pi$$

**证明3：正弦与余弦正交**

$$\int_{-\pi}^{\pi} \cos(mx) \sin(nx) \, dx = 0$$

无论 $m$ 和 $n$ 是否相等，结果都是零。

### 1.4 正交归一化

为了方便计算，我们可以将函数"归一化"，使它们的范数（"长度"）为1：

$$\|f\|^2 = \langle f, f \rangle = \int_a^b f^2(x) \, dx$$

归一化后的函数：

- $\phi_0(x) = \frac{1}{\sqrt{2\pi}}$
- $\phi_{2n-1}(x) = \frac{\cos(nx)}{\sqrt{\pi}}$
- $\phi_{2n}(x) = \frac{\sin(nx)}{\sqrt{\pi}}$

归一化后，任意两个不同函数的点积为0，相同函数的点积为1。

---

## 第二章：傅里叶级数

### 2.1 问题的提出

假设我们有一个周期为 $2\pi$ 的函数 $f(x)$，我们想把它写成三角函数的线性组合：

$$f(x) = a_0 + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right]$$

**问题**：如何确定系数 $a_0, a_1, a_2, \dots$ 和 $b_1, b_2, \dots$？

### 2.2 利用正交性求系数

**第一步：求 $a_0$**

对等式两边从 $-\pi$ 到 $\pi$ 积分：

$$\int_{-\pi}^{\pi} f(x) \, dx = \int_{-\pi}^{\pi} \left[ a_0 + \sum_{n=1}^{\infty} (a_n \cos(nx) + b_n \sin(nx)) \right] \, dx$$

由于三角函数的正交性，除 $a_0$ 外的所有项积分都为零：

$$\int_{-\pi}^{\pi} f(x) \, dx = \int_{-\pi}^{\pi} a_0 \, dx = a_0 \cdot 2\pi$$

因此：

$$a_0 = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x) \, dx = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x) \, dx$$

**第二步：求 $a_n$（$n \geq 1$）**

两边乘以 $\cos(kx)$，然后积分：

$$\int_{-\pi}^{\pi} f(x) \cos(kx) \, dx = \int_{-\pi}^{\pi} a_k \cos^2(kx) \, dx = a_k \cdot \pi$$

因此：

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx$$

**第三步：求 $b_n$（$n \geq 1$）**

两边乘以 $\sin(kx)$，然后积分：

$$\int_{-\pi}^{\pi} f(x) \sin(kx) \, dx = \int_{-\pi}^{\pi} b_k \sin^2(kx) \, dx = b_k \cdot \pi$$

因此：

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx$$

### 2.3 傅里叶级数的完整形式

**周期为 $2\pi$ 的函数** $f(x)$ 的傅里叶级数为：

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right]$$

其中：

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx$$
$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx$$

**注意**：$a_0$ 的系数是 $\frac{1}{2}$，这是为了与 $a_n$ 的公式保持一致（当 $n=0$ 时，$a_0$ 的公式给出 $\frac{2}{\pi}$ 倍的积分，所以要除以2）。

### 2.4 周期为 $T$ 的函数

如果函数 $f(t)$ 的周期是 $T$（不是 $2\pi$），我们可以进行变量替换：

令 $\omega_0 = \frac{2\pi}{T}$（基波频率），$x = \omega_0 t$，则：

$$f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(n\omega_0 t) + b_n \sin(n\omega_0 t) \right]$$

其中：

$$a_n = \frac{2}{T} \int_{t_0}^{t_0+T} f(t) \cos(n\omega_0 t) \, dt$$
$$b_n = \frac{2}{T} \int_{t_0}^{t_0+T} f(t) \sin(n\omega_0 t) \, dt$$

### 2.5 傅里叶级数的收敛性

**狄利克雷条件**（函数可以展开为傅里叶级数的充分条件）：

1. 函数在周期内连续，或只有有限个第一类间断点
2. 函数只有有限个极大值和极小值
3. 函数绝对可积：$\int_{-\pi}^{\pi} |f(x)| \, dx < \infty$

**吉布斯现象**：在函数的间断点附近，傅里叶级数会出现约9%的过冲，这个过冲不随项数增加而消失。

### 2.6 傅里叶级数的例子

**例子1：方波**

考虑周期为 $2\pi$ 的方波：

$$f(x) = \begin{cases} 1 & 0 < x < \pi \\ -1 & -\pi < x < 0 \end{cases}$$

这是一个奇函数（$f(-x) = -f(x)$），所以所有余弦系数 $a_n = 0$。

计算正弦系数：

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx = \frac{2}{\pi} \int_{0}^{\pi} \sin(nx) \, dx = \frac{2}{\pi} \left[ -\frac{\cos(nx)}{n} \right]_0^{\pi} = \frac{2(1 - (-1)^n)}{n\pi}$$

因此：

$$b_n = \begin{cases} \frac{4}{n\pi} & n \text{ 为奇数} \\ 0 & n \text{ 为偶数} \end{cases}$$

傅里叶级数：

$$f(x) = \frac{4}{\pi} \left( \sin(x) + \frac{1}{3}\sin(3x) + \frac{1}{5}\sin(5x) + \dots \right)$$

这就是著名的**方波的正弦分解**。

**例子2：锯齿波**

考虑周期为 $2\pi$ 的锯齿波：$f(x) = x$（在 $[-\pi, \pi]$ 上）

由于是奇函数，$a_n = 0$：

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} x \sin(nx) \, dx = \frac{2}{\pi} \int_{0}^{\pi} x \sin(nx) \, dx = \frac{2(-1)^{n+1}}{n}$$

傅里叶级数：

$$f(x) = 2 \sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} \sin(nx)$$

---

## 第三章：复数形式的傅里叶级数

### 3.1 欧拉公式

**欧拉公式**是复数分析中最优美的公式之一：

$$e^{i\theta} = \cos\theta + i\sin\theta$$

从中可以推导出：

$$\cos\theta = \frac{e^{i\theta} + e^{-i\theta}}{2}, \quad \sin\theta = \frac{e^{i\theta} - e^{-i\theta}}{2i} = -i\frac{e^{i\theta} - e^{-i\theta}}{2}$$

### 3.2 复数形式的傅里叶级数

使用欧拉公式，我们可以将傅里叶级数写成更紧凑的形式：

$$f(x) = \sum_{n=-\infty}^{\infty} c_n e^{inx}$$

其中系数为：

$$c_n = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x) e^{-inx} \, dx$$

**推导**：将正弦和余弦用复指数表示，合并正负频率项。

$$a_n \cos(nx) + b_n \sin(nx) = a_n \frac{e^{inx} + e^{-inx}}{2} + b_n \frac{e^{inx} - e^{-inx}}{2i}$$

$$= \frac{a_n - ib_n}{2} e^{inx} + \frac{a_n + ib_n}{2} e^{-inx}$$

定义 $c_n = \frac{a_n - ib_n}{2}$，则 $c_{-n} = \frac{a_n + ib_n}{2}$。

### 3.3 复数形式的优点

1. **公式更简洁**：只需一个求和，一个系数公式
2. **对称性好**：正负频率自然出现
3. **易于推广到傅里叶变换**
4. **便于计算**：复数乘法比三角函数乘法更简单

---

## 第四章：从傅里叶级数到傅里叶变换

### 4.1 非周期函数的挑战

傅里叶级数只能处理**周期函数**。对于非周期函数，我们需要傅里叶变换。

**关键思想**：非周期函数可以看作周期趋向无穷大的周期函数。

### 4.2 傅里叶变换的推导

考虑周期为 $T$ 的函数 $f_T(t)$，其傅里叶级数为：

$$f_T(t) = \sum_{n=-\infty}^{\infty} c_n e^{in\omega_0 t}$$

其中 $\omega_0 = \frac{2\pi}{T}$，系数：

$$c_n = \frac{1}{T} \int_{-T/2}^{T/2} f_T(t) e^{-in\omega_0 t} \, dt$$

当 $T \to \infty$ 时，$f_T(t) \to f(t)$（非周期函数），$\omega_0 \to d\omega$（连续频率）。

令 $F(\omega) = T c_n = \int_{-T/2}^{T/2} f_T(t) e^{-i\omega t} \, dt$

当 $T \to \infty$ 时：

$$F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} \, dt$$

这称为 $f(t)$ 的**傅里叶变换**。

### 4.3 傅里叶变换的定义

**傅里叶变换**（从时域到频域）：

$$\mathcal{F}\{f(t)\} = F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} \, dt$$

**傅里叶逆变换**（从频域到时域）：

$$\mathcal{F}^{-1}\{F(\omega)\} = f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i\omega t} \, d\omega$$

**频率与角频率**：

有时使用频率 $f$（单位：Hz）而非角频率 $\omega$（单位：rad/s）：

$$F(f) = \int_{-\infty}^{\infty} f(t) e^{-i2\pi ft} \, dt$$

$$f(t) = \int_{-\infty}^{\infty} F(f) e^{i2\pi ft} \, df$$

### 4.4 傅里叶变换的例子

**例子1：矩形脉冲**

考虑矩形脉冲：

$$f(t) = \begin{cases} 1 & |t| < \tau/2 \\ 0 & |t| > \tau/2 \end{cases}$$

傅里叶变换：

$$F(\omega) = \int_{-\tau/2}^{\tau/2} e^{-i\omega t} \, dt = \left[ \frac{e^{-i\omega t}}{-i\omega} \right]_{-\tau/2}^{\tau/2} = \frac{2\sin(\omega\tau/2)}{\omega} = \tau \cdot \text{sinc}\left(\frac{\omega\tau}{2}\right)$$

其中 $\text{sinc}(x) = \frac{\sin x}{x}$。

**重要结论**：时域的矩形脉冲对应频域的 sinc 函数。

**例子2：指数衰减**

考虑 $f(t) = e^{-at} u(t)$（$a > 0$，$u(t)$ 是单位阶跃函数）

$$F(\omega) = \int_{0}^{\infty} e^{-at} e^{-i\omega t} \, dt = \int_{0}^{\infty} e^{-(a+i\omega)t} \, dt = \frac{1}{a+i\omega}$$

**例子3：高斯函数**

$f(t) = e^{-\pi t^2}$ 是一个特殊的例子：

$$F(\omega) = e^{-\pi \omega^2}$$

高斯函数的傅里叶变换还是高斯函数！

---

## 第五章：傅里叶变换的性质

### 5.1 线性性

$$\mathcal{F}\{af(t) + bg(t)\} = aF(\omega) + bG(\omega)$$

**证明**：

$$\int_{-\infty}^{\infty} [af(t) + bg(t)] e^{-i\omega t} \, dt = a\int f(t)e^{-i\omega t}dt + b\int g(t)e^{-i\omega t}dt$$

### 5.2 时移性质

$$\mathcal{F}\{f(t - t_0)\} = e^{-i\omega t_0} F(\omega)$$

**证明**：

$$\int f(t - t_0) e^{-i\omega t} dt = \int f(u) e^{-i\omega(u+t_0)} du = e^{-i\omega t_0} \int f(u) e^{-i\omega u} du$$

**物理解释**：时域的延迟对应频域的相位偏移。

### 5.3 频移性质

$$\mathcal{F}\{f(t) e^{i\omega_0 t}\} = F(\omega - \omega_0)$$

**物理解释**：时域的调制对应频域的搬移。这是AM广播和无线电通信的基础。

### 5.4 导数性质

$$\mathcal{F}\{f'(t)\} = i\omega F(\omega)$$

$$\mathcal{F}\{f^{(n)}(t)\} = (i\omega)^n F(\omega)$$

**证明**：

$$\int f'(t) e^{-i\omega t} dt = [f(t)e^{-i\omega t}]_{-\infty}^{\infty} + i\omega\int f(t)e^{-i\omega t}dt$$

假设 $f(t) \to 0$ 当 $|t| \to \infty$，边界项为零。

**物理解释**：微分在时域对应乘以 $i\omega$ 在频域。

### 5.5 卷积定理

**卷积定义**：

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) \, d\tau$$

**时域卷积 ↔ 频域乘积**：

$$\mathcal{F}\{f * g\} = F(\omega) \cdot G(\omega)$$

**频域卷积 ↔ 时域乘积**：

$$\mathcal{F}\{f \cdot g\} = \frac{1}{2\pi} F * G(\omega)$$

**重要性**：卷积定理是傅里叶变换最重要的性质之一，它将复杂的卷积运算转化为简单的乘法运算。

### 5.6 帕塞瓦尔定理

$$\int_{-\infty}^{\infty} |f(t)|^2 \, dt = \frac{1}{2\pi} \int_{-\infty}^{\infty} |F(\omega)|^2 \, d\omega$$

**物理解释**：时域的能量等于频域的能量（能量守恒）。

---

## 第六章：离散傅里叶变换（DFT）

### 6.1 离散化的必要性

在实际应用中，我们无法处理连续信号。计算机只能处理离散数据。

**采样**：将连续信号 $f(t)$ 在时刻 $t_n = n\Delta t$ 采样，得到离散序列 $f[n] = f(n\Delta t)$。

**奈奎斯特-香农采样定理**：如果信号带宽有限（最高频率 $f_{\text{max}}$），当采样频率 $f_s > 2f_{\text{max}}$ 时，可以无失真地重建原信号。

### 6.2 DFT的定义

对于长度为 $N$ 的离散序列 $x[0], x[1], \dots, x[N-1]$，其**离散傅里叶变换**为：

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-i\frac{2\pi}{N}nk}, \quad k = 0, 1, \dots, N-1$$

**逆变换**：

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{i\frac{2\pi}{N}nk}$$

### 6.3 DFT的频率分辨率

DFT的频率分辨率为：

$$\Delta f = \frac{f_s}{N} = \frac{1}{N\Delta t}$$

$N$ 越大，频率分辨率越高。

### 6.4 快速傅里叶变换（FFT）

1965年，Cooley和Tukey发表了FFT算法，将DFT的计算复杂度从 $O(N^2)$ 降低到 $O(N\log N)$。

**FFT的意义**：

- $N = 1024$：$N^2 = 1,048,576$，$N\log_2 N = 10,240$——加速100倍！
- 实时信号处理成为可能
- 推动了数字信号处理的发展

**FFT的原理**：利用DFT的周期性和对称性，递归地将长序列分解为短序列。

---

## 第七章：实际应用案例

### 7.1 案例一：音频频谱分析

**问题**：分析一段音频的频率成分

**步骤**：

1. **采样**：以 $f_s = 44.1$ kHz采样音频信号
2. **加窗**：取一小段（如1024个样本），防止频谱泄漏
3. **FFT**：计算该段的频谱
4. **可视化**：绘制频谱图

**代码思路**：

```python
import numpy as np
from scipy.fft import fft

# 采样
fs = 44100  # 采样率
t = np.arange(0, 1, 1/fs)  # 1秒
signal = np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*880*t)

# FFT
N = len(signal)
yf = fft(signal)
xf = np.linspace(0, fs/2, N//2)

# 绘制频谱
import matplotlib.pyplot as plt
plt.plot(xf, 2/N * np.abs(yf[:N//2]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Audio Spectrum')
plt.show()
```

**结果**：在440 Hz和880 Hz处出现峰值，分别对应音叉的基频和泛音。

**应用**：
- 音乐均衡器
- 语音识别
- 乐器调音

### 7.2 案例二：图像边缘检测

**问题**：检测图像中的边缘

**原理**：边缘是图像亮度变化剧烈的地方，对应高频成分。

**步骤**：

1. **2D FFT**：对图像进行二维傅里叶变换
2. **高通滤波**：保留高频成分，滤除低频
3. **逆FFT**：得到边缘增强后的图像

**代码思路**：

```python
import numpy as np
import cv2

# 读取图像
img = cv2.imread('image.jpg', 0)

# 2D FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 高通滤波
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols), np.uint8)
r = 30  # 滤波半径
cv2.circle(mask, (ccol, crow), r, 0, -1)

# 应用滤波
fshift_filtered = fshift * mask
fimg = np.fft.ifftshift(fshift_filtered)
img_filtered = np.abs(np.fft.ifft2(fimg))

# 显示结果
cv2.imshow('Edge Detection', img_filtered)
cv2.waitKey(0)
```

**结果**：图像中的边缘被突出显示。

**应用**：
- 机器视觉
- 医学图像分析
- 自动驾驶

### 7.3 案例三：音频降噪

**问题**：消除音频中的背景噪声

**原理**：噪声通常是宽带的，而语音信号集中在特定频段。

**步骤**：

1. **FFT**：将音频转换到频域
2. **谱减**：减去噪声的频谱估计
3. **逆FFT**：重建降噪后的音频

**代码思路**：

```python
import numpy as np
from scipy.fft import fft, ifft

# 带噪声的信号
noisy = original + noise

# FFT
Y = fft(noisy)
N = len(Y)

# 估计噪声（取信号开始的一段静音）
noise_fft = fft(noise_segment)
noise_power = np.abs(noise_fft)**2

# 谱减
Y_clean = np.zeros(N, dtype=complex)
for k in range(N):
    magnitude = np.abs(Y[k])
    if magnitude > np.sqrt(noise_power[k]):
        Y_clean[k] = Y[k] * (1 - np.sqrt(noise_power[k])/magnitude)
    else:
        Y_clean[k] = 0

# 逆FFT
cleaned = np.real(ifft(Y_clean))
```

**应用**：
- 语音通话降噪
- 录音后期处理
- 助听器

### 7.4 案例四：数据压缩

**问题**：压缩图像或音频数据

**原理**：大部分信息集中在低频，高频成分可以量化或丢弃。

**JPEG压缩步骤**：

1. **分块**：将图像分成8×8的块
2. **DCT**：对每块进行二维离散余弦变换（DFT的变体）
3. **量化**：对DCT系数进行量化（丢弃高频信息）
4. **编码**：对量化后的系数进行熵编码

```python
import cv2
import numpy as np

img = cv2.imread('photo.jpg', 0)
rows, cols = img.shape

# 分块并DCT
compressed = np.zeros_like(img, dtype=np.float32)
for i in range(0, rows, 8):
    for j in range(0, cols, 8):
        block = img[i:i+8, j:j+8].astype(np.float32)
        dct_block = cv2.dct(block)
        # 量化（简化版）
        dct_block[4:8, 4:8] = 0  # 丢弃高频
        compressed[i:i+8, j:j+8] = cv2.idct(dct_block)

# 转换为uint8
compressed = np.clip(compressed, 0, 255).astype(np.uint8)
```

**应用**：
- JPEG图像压缩
- MP3音频压缩
- 视频压缩

### 7.5 案例五：通信系统中的调制

**问题**：如何在有限的频谱内传输更多信息？

**正交频分复用（OFDM）**是现代通信的核心技术。

**原理**：

1. 将高速数据流分成多个低速子流
2. 每个子流调制到不同的正交子载波上
3. 子载波之间正交（频谱重叠但不干扰）

**数学表示**：

$$s(t) = \sum_{k=0}^{N-1} c_k e^{i2\pi k \Delta f t}$$

其中 $c_k$ 是复数调制符号（QAM或PSK），$\Delta f$ 是子载波间隔。

**应用**：
- WiFi（802.11a/g/n/ac/ax）
- 4G/5G移动通信
- 数字电视

### 7.6 案例六：金融数据分析

**问题**：发现股票价格中的周期性模式

**步骤**：

1. 收集股票价格时间序列
2. 去趋势（去除长期趋势）
3. FFT变换
4. 分析频谱中的峰值

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 股票价格数据
df = pd.read_csv('stock.csv')
price = df['Close'].values

# 去趋势
from scipy import signal
detrended = signal.detrend(price)

# FFT
N = len(detrended)
yf = np.fft.fft(detrended)
xf = np.fft.fftfreq(N, 1/252)  # 假设每天采样，252个交易日/年

# 只看正频率
pos_mask = xf > 0
plt.plot(xf[pos_mask], np.abs(yf[pos_mask]))
plt.xlabel('Frequency (cycles/year)')
plt.ylabel('Amplitude')
plt.title('Stock Price Cycle Analysis')
plt.show()
```

**结果**：频谱中的峰值对应股票价格的周期性波动。

---

## 第八章：相关变换

### 8.1 拉普拉斯变换

傅里叶变换要求函数绝对可积。拉普拉斯变换放宽了这个条件。

**定义**：

$$F(s) = \int_{0}^{\infty} f(t) e^{-st} \, dt$$

其中 $s = \sigma + i\omega$ 是复数频率。

**应用**：
- 控制系统分析
- 微分方程求解
- 电路分析

### 8.2 Z变换

Z变换是离散信号的拉普拉斯变换。

**定义**：

$$X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}$$

**应用**：
- 数字滤波器设计
- 离散系统分析

### 8.3 离散余弦变换（DCT）

DCT是实信号的优化版本，只使用余弦函数。

**定义**：

$$X[k] = \sum_{n=0}^{N-1} x[n] \cos\left[\frac{\pi}{N}(n + \frac{1}{2})k\right]$$

**应用**：
- JPEG压缩
- MP3压缩
- 视频压缩

---

## 结语：数学的力量

### 傅里叶变换的意义

回顾我们走过的旅程，从三角函数的正交性到傅里叶级数，从傅里叶变换到实际应用，我们见证了数学的力量。

傅里叶变换不仅仅是一个数学工具，更是一种**世界观**：

1. **分解的思维**：复杂问题可以分解为简单问题的叠加
2. **频率的视角**：从频率角度看问题，往往能发现隐藏的规律
3. **变换的思想**：通过变换，将困难的问题变得简单

### 从傅里叶到小波

傅里叶变换的局限是：它只能告诉我们"有哪些频率"，不能告诉我们"这些频率在什么时候出现"。

**小波变换**（Wavelet Transform）解决了这个问题，它同时具有时间和频率的分辨率。

### 给读者的话

如果你读到这里，恭喜你！你已经掌握了傅里叶分析的基本概念。

傅里叶变换是现代工程和科学的基石。从音乐到医学，从通信到金融，它无处不在。

当你下次听音乐、拍照、使用手机时，请记住：这些技术的背后，都有傅里叶变换在默默地工作。

---

## 附录：重要公式汇总

### 傅里叶级数

**周期为 $2\pi$ 的函数**：

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} [a_n \cos(nx) + b_n \sin(nx)]$$

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx$$
$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx$$

**复数形式**：

$$f(x) = \sum_{n=-\infty}^{\infty} c_n e^{inx}$$

$$c_n = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x) e^{-inx} \, dx$$

### 傅里叶变换

**连续时间傅里叶变换（CTFT）**：

$$F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} \, dt$$

$$f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i\omega t} \, d\omega$$

### 离散傅里叶变换（DFT）

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-i\frac{2\pi}{N}nk}$$

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{i\frac{2\pi}{N}nk}$$

### 傅里叶变换性质

| 性质 | 时域 | 频域 |
|------|------|------|
| 线性 | $af(t) + bg(t)$ | $aF(\omega) + bG(\omega)$ |
| 时移 | $f(t - t_0)$ | $e^{-i\omega t_0}F(\omega)$ |
| 频移 | $f(t)e^{i\omega_0 t}$ | $F(\omega - \omega_0)$ |
| 微分 | $f'(t)$ | $i\omega F(\omega)$ |
| 卷积 | $f * g(t)$ | $F(\omega)G(\omega)$ |
| 乘积 | $f(t)g(t)$ | $\frac{1}{2\pi}F * G(\omega)$ |

### 常见傅里叶变换对

| 信号 | 傅里叶变换 |
|------|-----------|
| $\delta(t)$ | $1$ |
| $1$ | $2\pi\delta(\omega)$ |
| $e^{i\omega_0 t}$ | $2\pi\delta(\omega - \omega_0)$ |
| $\cos(\omega_0 t)$ | $\pi[\delta(\omega - \omega_0) + \delta(\omega + \omega_0)]$ |
| $\text{rect}(t/\tau)$ | $\tau\text{sinc}(\omega\tau/2)$ |
| $e^{-at}u(t)$ $(a>0)$ | $\frac{1}{a+i\omega}$ |
| $e^{-\pi t^2}$ | $e^{-\pi \omega^2}$ |

---

*本文旨在为有一定数学基础的读者提供傅里叶分析的入门导引。更深入的学习建议参考专业教材，如 Oppenheim 的《Signals and Systems》、Bracewell 的《The Fourier Transform and Its Applications》等。*
