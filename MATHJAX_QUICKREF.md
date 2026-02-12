# MathJax 快速参考卡

## 启用 MathJax

在文章 front matter 中添加:
```yaml
mathjax: true
```

## 基础语法

### 行内公式
```markdown
$E = mc^2$
```

### 块级公式
```markdown
$$
f(x) = \int_{-\infty}^{\infty} \hat{f}(\xi)\,e^{2\pi i \xi x} \,d\xi
$$
```

## 常用符号

### 上下标
```latex
x_2      下标
x^2      上标
x_{2n}   复杂下标
x^{2n+1} 复杂上标
```

### 分数
```latex
\frac{a}{b}           简单分数
\frac{x^2+1}{x-1}     复杂分数
```

### 根号
```latex
\sqrt{x}      平方根
\sqrt[n]{x}   n次根
```

### 求和与积分
```latex
\sum_{i=1}^{n} x_i          求和
\int_{a}^{b} f(x) dx         积分
\lim_{x \to \infty} f(x)     极限
\prod_{i=1}^{n} x_i          乘积
```

### 希腊字母
```latex
\alpha \beta \gamma \delta \epsilon \zeta \eta \theta
\iota \kappa \lambda \mu \nu \xi \pi \rho \sigma \tau
\upsilon \phi \chi \psi \omega

\Gamma \Delta \Theta \Lambda \Sigma \Phi \Psi \Omega
```

### 关系符号
```latex
\leq  小于等于
\geq  大于等于
\neq  不等于
\approx 约等于
\equiv 恒等于
\sim  相似
```

### 集合符号
```latex
\in    属于
\notin 不属于
\subset 子集
\subseteq 子集或等于
\cup   并集
\cap   交集
\emptyset 空集
```

### 逻辑符号
```latex
\forall  对于所有
\exists  存在
\Rightarrow 蕴含
\Leftrightarrow 当且仅当
\neg     非
\vee     或
\wedge   且
```

### 箭头
```latex
\to        →
\rightarrow →
\Rightarrow ⇒
\leftarrow ←
\leftrightarrow ↔
```

### 空格
```latex
\,         小空格
\;         中等空格
\quad      大空格
\qquad     超大空格
```

## 高级功能

### 矩阵
```latex
$$
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$
```

### 方程组
```latex
$$
\begin{cases}
3x + 5y + z = 1 \\
7x - 2y + 4z = 2
\end{cases}
$$
```

### 对齐
```latex
$$
\begin{align}
f(x) &= x^2 + 2x + 1 \\
     &= (x + 1)^2
\end{align}
$$
```

### 括号
```latex
( )    小括号
[ ]    中括号
\{ \}  大括号
\langle \rangle  尖括号
| |    绝对值
\| \|  范数
```

自动缩放的括号:
```latex
\left( \frac{1}{2} \right)
\left[ \sum_{i=1}^{n} x_i \right]
```

### 导数和偏导数
```latex
f'           一阶导数
f''          二阶导数
\frac{df}{dx} 导数
\frac{\partial f}{\partial x} 偏导数
\nabla        梯度
\partial      偏导数符号
```

### 统计符号
```latex
\bar{x}       均值
\hat{\theta}  估计值
\tilde{x}     波浪号
\dot{x}       导数点
\ddot{x}      二阶导数点
```

## 自定义宏

已在配置中预定义:
```latex
\R, \N, \Z, \Q, \C    数集
\vec{v}               向量
\norm{x}              范数
\abs{x}               绝对值
\bm{x}                粗体
```

## 常见公式示例

### 二次公式
```latex
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
```

### 欧拉公式
```latex
e^{i\pi} + 1 = 0
```

### 正态分布
```latex
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
```

### 傅里叶级数
```latex
f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left( a_n \cos(nx) + b_n \sin(nx) \right)
```

### 泰勒展开
```latex
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x-a)^n
```

### 柯西-施瓦茨不等式
```latex
\left| \sum_{i=1}^{n} a_i b_i \right|^2 \leq \left( \sum_{i=1}^{n} |a_i|^2 \right) \left( \sum_{i=1}^{n} |b_i|^2 \right)
```

### 贝叶斯定理
```latex
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
```

### 信息熵
```latex
H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)
```

## 提示

1. **空格**: LaTeX 会自动处理大多数空格,必要时使用 `\quad` 或 `\qquad`
2. **换行**: 在块级公式中使用 `\\` 换行
3. **大括号**: 在 LaTeX 中 `\{` 和 `\}` 表示大括号字符
4. **注释**: 使用 `%` 在 LaTeX 代码中添加注释
5. **转义**: 在某些情况下需要使用 `\\` 表示单个反斜杠

## 在线工具

- [Overleaf](https://www.overleaf.com/) - 在线 LaTeX 编辑器
- [MathJax Demo](https://demo.mathjax.org/) - MathJax 在线演示
- [Detexify](http://detexify.kirelabs.org/) - 手写符号识别
- [Codecogs Equation Editor](https://www.codecogs.com/latex/eqneditor.php) - 可视化公式编辑器

## 故障排除

### 公式不显示
1. 检查是否添加了 `mathjax: true`
2. 检查浏览器控制台错误
3. 验证 LaTeX 语法
4. 清除浏览器缓存

### 性能问题
1. 减少公式数量
2. 使用更简单的公式
3. 考虑延迟加载

---

更多信息请参阅 [MATHJAX_GUIDE.md](./MATHJAX_GUIDE.md)
