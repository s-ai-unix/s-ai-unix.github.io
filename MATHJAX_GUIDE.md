# MathJax 使用指南

## 概述

本博客已成功集成 MathJax,支持在文章中渲染 LaTeX 数学公式。

## 配置文件

### 1. MathJax 配置
位置: `/layouts/partials/mathjax.html`

功能:
- 支持行内公式: `$...$` 或 `\(...\)...\(...\)`
- 支持块级公式: `$$...$$` 或 `\[...\]...\[...\]`
- 自动编号支持
- 自定义数学宏

### 2. 自动加载
位置: `/layouts/partials/extend_footer.html`

通过这个文件,MathJax 会自动在所有页面的底部加载。

## 全局配置

在 `hugo.yaml` 中添加了全局配置:
```yaml
params:
  mathjax: true
```

## 在文章中使用 MathJax

### 方法一:启用全局配置

如果你希望所有文章都支持 MathJax,在 `hugo.yaml` 中设置:
```yaml
params:
  mathjax: true
```

### 方法二:按文章启用

在文章的 front matter 中添加:
```yaml
---
title: "文章标题"
mathjax: true
---
```

## LaTeX 语法示例

### 行内公式

```
这是行内公式 $E = mc^2$ 的例子。
```

渲染效果: 这是行内公式 $E = mc^2$ 的例子。

### 块级公式

```
这是块级公式:
$$f(x) = \int_{-\infty}^{\infty} \hat{f}(\xi)\,e^{2\pi i \xi x} \,d\xi$$
```

渲染效果:
$$f(x) = \int_{-\infty}^{\infty} \hat{f}(\xi)\,e^{2\pi i \xi x} \,d\xi$$

## 常用语法

### 下标和上标
```
下标: x_1, x_2, ..., x_n
上标: x^2, x^3, e^{i\pi}
组合: a_{ij}, b^{n+1}
```

### 分数
```
简单分数: \frac{a}{b}
复杂分数: \frac{x^2 + 1}{x - 1}
```

### 根号
```
平方根: \sqrt{x}
n次根: \sqrt[n]{x}
```

### 求和和积分
```
求和: \sum_{i=1}^{n} x_i
积分: \int_{a}^{b} f(x) dx
极限: \lim_{x \to \infty} f(x)
```

### 希腊字母
```
小写: \alpha, \beta, \gamma, \delta, \theta, \lambda, \mu, \sigma, \phi, \omega
大写: \Gamma, \Delta, \Theta, \Lambda, \Sigma, \Phi, \Omega
```

### 矩阵
```
$$
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$
```

### 方程组
```
$$
\begin{cases}
3x + 5y + z = 1 \\
7x - 2y + 4z = 2 \\
-6x + 3y + 2z = 3
\end{cases}
$$
```

## 自定义宏

已在配置中预定义的快捷命令:
- `\R`: 实数集 $\mathbb{R}$
- `\N`: 自然数集 $\mathbb{N}$
- `\Z`: 整数集 $\mathbb{Z}$
- `\Q`: 有理数集 $\mathbb{Q}$
- `\C`: 复数集 $\mathbb{C}$
- `\vec{v}`: 向量 $\mathbf{v}$
- `\norm{x}`: 范数 $\left\lVert x \right\rVert$
- `\abs{x}`: 绝对值 $|x|$
- `\bm{x}`: 粗体 $\boldsymbol{x}$

## 注意事项

1. **转义字符**: 在 Markdown 中,某些字符需要转义
   - 使用 `\\` 表示单个反斜杠
   - 在代码块中不需要转义

2. **性能考虑**:
   - MathJax 会自动检测页面上的数学公式并渲染
   - 公式过多可能影响页面加载速度
   - 建议只在需要的文章中启用

3. **兼容性**:
   - 支持所有现代浏览器
   - 移动设备也能正常显示
   - 支持缩放和高分辨率屏幕

## 测试

访问 `/posts/test-mathjax/` 查看完整的 MathJax 测试页面,包含各种类型的数学公式示例。

## 故障排除

如果公式无法渲染:

1. 检查 front matter 中是否添加了 `mathjax: true`
2. 检查浏览器控制台是否有错误信息
3. 确保 LaTeX 语法正确
4. 尝试清除浏览器缓存

## 参考资源

- [MathJax 官方文档](https://docs.mathjax.org/)
- [LaTeX 数学公式大全](https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols)
- [在线 LaTeX 编辑器](https://www.overleaf.com/)

## 已转换的文章

以下文章已成功转换 Unicode 下标为 LaTeX 格式:
1. `2019-08-27-rnn-lstm-gru-complete-guide.md`
2. `2019-08-14-gradient-boosting-algorithm-guide.md`
3. `2020-02-16-mathematical-statistics-theory.md`

这些文章现在都包含正确的 LaTeX 数学公式,可以在浏览器中完美渲染。
