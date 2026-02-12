# LaTeX 数学公式规范

## 基本原则

**最重要的规则**：所有数学符号和变量**必须**包裹在数学模式中（`$...$` 或 `$$...$$`）

禁止在文本中直接使用数学符号（如 `x`, `y`, `w`, `b`, `∈`, `×`, `→` 等）

确保公式在 Markdown 编辑器和 MathJax 中都能正确渲染

## 数学模式使用

### 行内公式
使用：`$公式$`（在段落中使用）

示例：
```
变量 $x$ 的值是 5
函数 $f(x) = x^2$
```

### 独立公式
使用：`$$公式$$`（单独一行，居中显示）

示例：
```
$$
f(x) = \int_0^1 x^2 dx
$$
```

### 所有数学变量必须包裹
✅ 正确：`$x$`, `$y$`, `$w_i$`, `$b$`
❌ 错误：`x`, `y`, `w_i`, `b`（未包裹）

## 必须遵守的 LaTeX 规范

### 1. 变量和标量
- ✅ 正确：`$x$`, `$y$`, `$w_i$`, `$b$`
- ❌ 错误：`x`, `y`, `w_i`, `b`（未包裹）

### 2. 向量
- ✅ 正确：`$\mathbf{x}$`, `$\mathbf{w}$`, `$\vec{v}$`
- ✅ 希腊字母向量：`$\mathbf{\lambda}$`, `$\mathbf{\theta}$`（使用 `\mathbf` 而非 `\boldsymbol`）
- ❌ 错误：`x`, `w`, `v`（未包裹，未标记为向量）

**注意**：`\boldsymbol` 命令在某些 MathJax 配置中需要额外的 `boldsymbol` 扩展支持，可能显示为源码。建议使用 `\mathbf` 命令，兼容性更好。

### 3. 上标和下标
- ✅ 正确：`$x^{(1)}$`, `$w_{ij}$`, `$\sigma'$`, `$f_t$`
- ⚠️ 可以：`$x^2$`, `$W^T$`（单个字符可以不用花括号）
- ❌ 错误：`$x^(1)$`, `$w_ij$`, `f_t`（括号格式错误或未包裹）

**重要**：上标必须使用 `^{...}` 而非 `^(...)`，下标必须使用 `_{...}` 而非 `_(...)`

### 4. 希腊字母
- ✅ 正确：`$\alpha$`, `$\beta$`, `$\sigma$`, `$\delta$`, `$\theta$`
- ❌ 错误：`α`, `β`, `σ`, `δ`, `θ`（直接使用 Unicode）

**常见希腊字母**：
```
\alpha, \beta, \gamma, \delta, \epsilon, \theta, \lambda, \mu, \pi, \sigma, \tau, \phi, \omega
\Delta, \Sigma, \Phi, \Psi, \Omega
```

### 5. 特殊符号
- ✅ 正确：`$\times$`（乘法）, `$\to$`（箭头）, `$\in$`（属于）
- ✅ 正确：`$\mathbb{R}$`（实数集）, `$\subset$`（子集）, `$\neq$`（不等）
- ❌ 错误：`×`, `→`, `∈`, `R`（直接使用 Unicode 或未包裹）

**常见特殊符号**：
```
\times, \to, \in, \subset, \approx, \equiv, \neq, \leq, \geq
\infty, \partial, \nabla, \cdot
\mathbb{R}, \mathbb{N}, \mathbb{Z}, \mathbb{C}
```

### 6. 分数和导数
- ✅ 正确：`$\frac{\partial f}{\partial x}$`, `$\frac{a}{b}$`
- ✅ 正确：`$f'(x)$`, `$\frac{dy}{dx}$`
- ❌ 错误：`∂f/∂x`, `dy/dx`（未包裹）

### 7. 集合和区间
- ✅ 正确：`$\{1, 2, 3\}$`, `$[0, 1]$`, `$\mathbb{R}^d$`
- ❌ 错误：`{1, 2, 3}`, `[0, 1]`, `R^d`（未包裹或花括号未转义）

**注意**：在 Markdown 中，花括号需要转义为 `\{` 和 `\}`

### 8. 矩阵
- ✅ 正确：`$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$`
- ✅ 正确：`$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$`

### 9. 积分和求和
- ✅ 正确：`$\int_a^b f(x) dx$`, `$\sum_{i=1}^{n} x_i$`
- ❌ 错误：`∫`, `Σ`（直接使用 Unicode）

### 10. 表格中的数学
- ✅ 正确：`| $x_1$ | $x_2$ | $x_1 + x_2$ |`
- ❌ 错误：`| x_1 | x_2 | x_1 + x_2 |`（变量未包裹）

## 复杂公式的特殊处理（重要）

### 问题：复杂公式中的下划线被解析为 Markdown 斜体

当公式包含多个下划线（如多重下标、张量指标等）时，Markdown 解析器可能将 `_` 误认为斜体标记。

**症状**：公式中的 `_` 变成 `<em>` 标签，公式显示为源码或渲染错误。

**示例问题公式**：
```latex
$$\mathbf{X}_{i_1 i_2 \cdots i_N} = \sum_{\alpha_1, \ldots, \alpha_{N-1}} A^{(1)}_{i_1 \alpha_1} A^{(2)}_{\alpha_1 i_2 \alpha_2} \cdots$$
```

### 解决方案

#### 方案1：使用 HTML div 包裹（推荐）

对于包含多个下划线的复杂独立公式，使用 `<div class="math">` 包裹：

```markdown
<div class="math">
$$\mathbf{X}_{i_1 i_2 \cdots i_N} = \sum_{\alpha_1, \ldots, \alpha_{N-1}} A^{(1)}_{i_1 \alpha_1} A^{(2)}_{\alpha_1 i_2 \alpha_2} \cdots A^{(N)}_{\alpha_{N-1} i_N}$$
</div>
```

**优点**：
- Markdown 解析器跳过 `<div>` 标签内部内容
- MathJax 正常接收完整的 LaTeX 代码
- 兼容性最好

#### 方案2：使用 LaTeX 原生语法

使用 `\[` 和 `\]` 替代 `$$`：

```markdown
\[\mathbf{X}_{i_1 i_2 \cdots i_N} = \sum_{\alpha_1, \ldots, \alpha_{N-1}} \cdots\]
```

**注意**：需要 Hugo 配置支持 `passthrough` 扩展。

#### 方案3：简化公式（备选）

将复杂公式拆分为多个简单公式，减少单个公式中的下划线数量。

### 需要特殊处理的场景

以下情况建议使用 `<div class="math">` 包裹：

1. **张量指标**：多个带下标的变量相乘
   ```latex
   $$A^{(1)}_{i_1 \alpha_1} A^{(2)}_{\alpha_1 i_2 \alpha_2} \cdots A^{(N)}_{\alpha_{N-1} i_N}$$
   ```

2. **多重求和/积分**：嵌套的求和或积分
   ```latex
   $$\sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p a_{ijk}$$
   ```

3. **矩阵元序列**：长串的矩阵元素定义
   ```latex
   $$H_{ij} = \langle i | H | j \rangle = \int \psi_i^*(x) H \psi_j(x) dx$$
   ```

4. **张量网络**：矩阵乘积态、张量分解等
   ```latex
   $$\mathbf{X}_{i_1 i_2 \cdots i_N} = \sum_{\alpha_1, \ldots} A^{(1)}_{i_1 \alpha_1} \cdots$$
   ```

## 常见错误示例及修复

| 错误写法 | 正确写法 | 原因 |
|---------|---------|------|
| `x = (x_1, x_2, ..., x_d)^T` | `$\mathbf{x} = (x_1, x_2, \ldots, x_d)^T$` | 未包裹，省略号格式错误 |
| `权重 w_i` | `权重 $w_i$` | 变量未包裹 |
| `训练样本 (x_i, y_i)` | `训练样本 $(\mathbf{x}_i, y_i)$` | 未包裹 |
| `y ∈ {−1, +1}` | `$y \in \{-1, +1\}$` | 未包裹，花括号未转义 |
| `输入 → 隐藏层 → 输出层` | `输入 $\to$ 隐藏层 $\to$ 输出层` | 箭头未包裹 |
| `W ∈ R^(m×n)` | `$W \in \mathbb{R}^{m \times n}$` | 未包裹，符号格式错误 |
| `32×32 图像` | `$32 \times 32$ 图像` | 乘号未包裹 |
| `α_{ij} = 0.5` | `$\alpha_{ij} = 0.5$` | 希腊字母未包裹 |
| `f_t 接近 1` | `$f_t$ 接近 1` | 下标变量未包裹 |
| `1-NN 算法` | `1{-}NN 算法` 或 `1NN 算法` | 连字符在下标中导致歧义 |
| `\\[8pt]` | `\\` | cases 环境中方括号被误解析，导致换行失败 |
| `\boldsymbol{\lambda}` | `\mathbf{\lambda}` | `\boldsymbol` 需要额外宏包支持，可能显示为源码 |
| `\mathbf{v}'_{w}^T` | `(\mathbf{v}'_{w})^T` | 双重上标必须用括号明确，否则报错 "Double exponent" |
| `50,000` | `50{,}000` | 数字中的逗号会被误解析为下标分隔符 |
| `\llbracket` `\rrbracket` | `[\![` `]\!]` | MathJax 不支持，需要 `stmaryrd` 宏包 |
| `\dfrac{a}{b}` | `\frac{a}{b}` | MathJax 有限支持，建议用标准 `\frac` |

## 最佳实践

### 1. 编写公式时的检查清单
- [ ] 所有数学变量是否都用 `$...$` 或 `$$...$$` 包裹？
- [ ] 上标是否用 `^{...}` 而非 `^(...)`？
- [ ] 下标是否用 `_{...}` 而非 `_(...)`？
- [ ] 希腊字母是否用 LaTeX 命令（如 `\alpha`）而非 Unicode？
- [ ] 特殊符号（×, →, ∈ 等）是否用 LaTeX 命令？
- [ ] 花括号在文本中是否转义为 `\{` 和 `\}`？
- [ ] 向量是否用 `\mathbf{}` 或 `\vec{}` 标记？
- [ ] 复杂公式（多下划线）是否用 `<div class="math">` 包裹？
- [ ] 是否有双重上标（如 `v'^T`）？必须用括号：`(v')^T`
- [ ] 是否使用了 MathJax 不支持的命令（`\llbracket`, `\dfrac`, `\boldsymbol`）？

### 2. 常用模板
```
函数定义：$f: \mathbb{R}^n \to \mathbb{R}^m$
向量：$\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$
矩阵维度：$W \in \mathbb{R}^{m \times n}$
求和：$\sum_{i=1}^{n} x_i$
梯度：$\nabla f(\mathbf{x})$
偏导数：$\frac{\partial f}{\partial x_i}$
激活函数：$\sigma(z) = \frac{1}{1 + e^{-z}}$
Softmax：$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{c} e^{z_j}}$
```

### 3. 复杂公式模板
```markdown
<div class="math">
$$\mathbf{X}_{i_1 i_2 \cdots i_N} = \sum_{\alpha_1, \ldots, \alpha_{N-1}} A^{(1)}_{i_1 \alpha_1} A^{(2)}_{\alpha_1 i_2 \alpha_2} \cdots A^{(N)}_{\alpha_{N-1} i_N}$$
</div>
```

### 4. 测试验证
- 在支持 MathJax 的 Markdown 编辑器中预览
- 检查浏览器控制台是否有 MathJax 错误
- 确保所有公式正确渲染，没有显示为源代码
- 复杂公式务必检查下划线是否被正确解析

## MathJax 解析失败的常见原因

### 症状：浏览器控制台有 MathJax 错误

**常见原因**：
1. 未闭合的 `$` 或 `$$`
2. 在数学模式中使用了未转义的特殊字符（如 `#`, `%`, `&`）
3. 上标/下标格式错误（`^(...)` 应为 `^{...}`）
4. 连字符在下标中导致歧义（`1-NN` 应为 `1{-}NN` 或 `1NN`）
5. Markdown 解析器将复杂公式中的 `_` 解析为 `<em>`（使用 `<div class="math">` 包裹）

**解决**：使用 `hugo --quiet` 测试，检查构建输出

## 特殊场景

### 连字符在下标中
当在下标中使用连字符时，MathJax 可能会误解析。

- ❌ 错误：`$x_{1-NN}$`
- ✅ 正确：`$x_{1{-}NN}$` 或 `$x_{1NN}$`

### 星号在上标中
星号 `*` 在上标位置可能被 MathJax 误解析为 Markdown 强调符号或特殊标记，导致 "Missing open brace for superscript" 错误。

- ❌ 错误：`$x^*$`（在某些 MathJax 配置中会报错）
- ✅ 正确：`$x^{\ast}$` 或 `$x^{\ast}$`

**示例**：
```latex
% ❌ 可能导致错误
$\mathbf{x}^* = \arg\min f(x)$

% ✅ 推荐写法
$\mathbf{x}^{\ast} = \arg\min f(x)$
```

**注意**：这个问题在 Hugo + MathJax 环境中较为常见，建议在技术博客写作中统一使用 `^{\ast}` 格式。

### 表达式中的文本
在数学模式中包含文本时，使用 `\text{}` 命令：

- ✅ 正确：`$\text{对于所有 } x \in \mathbb{R}$`
- ❌ 错误：`对于所有 $x \in \mathbb{R}$`（中英文混排）

### 特殊字符 `#` 在数学模式中

`#` 在 LaTeX 中是特殊字符（用于宏参数），即使在数学模式中也不能直接使用 `\#`。

**错误示例**（表示"数据量"或"特征数"）：
- ❌ 错误：`$O(\#data \times \#features)$`
- ❌ 错误：`$O(\#samples)$`

这些写法会导致 MathJax 报错："You can't use 'macro parameter character #' in math mode"

**正确替代方案**：
- ✅ 使用下标变量：`$O(n_{samples} \times n_{features})$`
- ✅ 使用普通文本：`$\text{样本数} \times \text{特征数}$`
- ✅ 使用其他符号：`$O(N \times D)$`

**常见场景**：
- 算法复杂度：`$O(n^2)$` 而非 `$O(\#data^2)$`
- 张量维度：`$n_{channels}$` 而非 `$\#channels$`
- 统计量：`$n_{samples}$` 而非 `$\#samples$`

### 多行公式推导
使用 `align` 环境对齐多行公式：

```latex
$$
\begin{align}
E &= mc^2 \\
  &= \sqrt{p^2c^2 + m^2c^4}
\end{align}
$$
```

### cases 环境中的换行间距

在 `cases` 环境中使用 `\\[间距]` 添加垂直间距时，方括号 `[8pt]` 会被 MathJax 误解析为数学符号，导致换行失败或渲染异常。

- ❌ 错误：
```latex
$$
\begin{cases}
\frac{\partial F}{\partial x} = 0 \\[8pt]
\frac{\partial G}{\partial x} = 0
\end{cases}
$$
```

- ✅ 正确：使用简单的 `\\` 或增加行内空间
```latex
$$
\begin{cases}
\frac{\partial F}{\partial x} = 0 \\
\frac{\partial G}{\partial x} = 0
\end{cases}
$$
```

## 调试技巧

### 1. 逐步检查
从简单的公式开始，逐步增加复杂度：

```
测试1：$x$ （单个变量）
测试2：$x^2$ （上标）
测试3：$\alpha$ （希腊字母）
测试4：$\frac{a}{b}$ （分数）
测试5：复杂公式（多下划线，用 <div class="math"> 包裹）
```

### 2. 浏览器控制台
打开浏览器开发者工具，查看 Console 中的 MathJax 错误信息

### 3. 隔离问题
将有问题的公式单独放到一个测试文件中，验证是否能正确渲染

### 4. 检查 HTML 源码
在浏览器中查看页面源码，搜索 `<em>` 标签，确认是否被错误插入到公式中

## 图注/图片标题中的公式特殊处理

### 问题：Markdown 与 MathJax 的冲突

图注通常使用 `*图：描述*` 或 `_图：描述_` 格式，这会与公式中的 `*` 和 `_` 冲突：

```markdown
❌ *图：词向量关系 $\mathbf{v}_{king} - \mathbf{v}_{man}$*  
   ↑ 这里的 _ 会被 Markdown 解析为斜体结束符

❌ <em>图：词向量关系 $\mathbf{v}_{king} - \mathbf{v}_{man}$</em>  
   ↑ 即使使用 HTML 标签，内部的 _ 仍可能被解析
```

### 解决方案

**方案1：图注不用 LaTeX，用纯文本描述（推荐）**
```markdown
![图片描述](path.png)
<p class="caption">图：词向量关系，king - man + woman ≈ queen</p>
```

**方案2：使用 HTML 实体转义**
```markdown
<p class="caption">图：词向量关系 $\mathbf{v}\_{\text{king}}$</p>
   ↑ 使用 \_ 转义下划线
```

**方案3：避免在图注中使用复杂公式**
将复杂公式放在图注上方的正文中，图注只保留简单文字说明。

### 其他 Markdown 冲突场景

| 场景 | 问题 | 解决方案 |
|------|------|----------|
| 列表项中的公式 | `- $x_i$` 的 `_` 可能被解析 | 使用 `* $x_i$` 或加空格 `-  $x_i$` |
| 加粗文本中的公式 | `**$x_i$**` 的 `_` 冲突 | 改用 HTML `<strong>$x_i$</strong>` |
| 斜体文本中的公式 | `*$x_i$*` 的 `_` 冲突 | 改用 HTML `<em>$x_i$</em>` 或避免 |
