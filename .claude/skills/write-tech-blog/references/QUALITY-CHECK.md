# 文章质量检查清单

## 完成文章后的必做检查

在完成文章撰写后，必须执行以下检查确保文章质量。

### 1. 编码和乱码检查

**最重要**：检查是否有 Unicode 替换字符或乱码。

#### 检查方法

使用 Python 脚本检查：

```bash
python3 << 'EOF'
def check_encoding(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查替换字符
    replacement_count = content.count('\ufffd')
    if replacement_count > 0:
        print(f"❌ 发现 {replacement_count} 个替换字符")
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if '\ufffd' in line:
                print(f"  行 {i}: {line.strip()[:100]}")
        return False

    # 检查常见的乱码模式
    issues = []
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        # 检查不完整的中文字符
        if '薛定' in line and '薛定谔' not in line:
            issues.append((i, "不完整的'薛定谔'", line.strip()[:80]))
        elif '定谔' in line and '薛定谔' not in line:
            issues.append((i, "缺少'薛'字", line.strip()[:80]))

    if issues:
        print(f"⚠️  发现 {len(issues)} 处编码问题")
        for line_num, issue_type, text in issues[:10]:
            print(f"  行 {line_num}: {issue_type}")
        return False

    print("✅ 编码检查通过")
    return True

check_encoding('content/posts/[文章文件].md')
EOF
```

#### 常见乱码模式

- `薛定谔��程` → `薛定谔方程`（缺少"方"）
- `定谔方程` → `薛定谔方程`（缺少"薛"）
- `德布罗意关系出发` → `从德布罗意关系出发`（缺少"从"）
- 任何包含 `` 的行都是乱码

#### 修复方法

1. 在编辑器中查看文章，搜索常见的不完整字符
2. 使用 Read 工具检查可疑行
3. 使用 Edit 工具修复乱码
4. 重新运行检查脚本验证

### 2. 数学公式检查

使用 [LATEX-MATH.md](LATEX-MATH.md) 中的检查清单：

#### 基础检查
- [ ] 所有数学变量都用 `$...$` 或 `$$...$$` 包裹
- [ ] 上标使用 `^{...}` 而非 `^(...)`
- [ ] 下标使用 `_{...}` 而非 `_(...)`
- [ ] 希腊字母使用 LaTeX 命令（`\alpha`）而非 Unicode
- [ ] 特殊符号（×, →, ∈）在数学模式中
- [ ] 向量用 `\mathbf{}` 或 `\vec{}` 标记

#### Hugo + MathJax 特殊检查 ⚠️

**重要**：这些检查项针对 Hugo 博客的特殊要求。

- [ ] **无 `^*` 复共轭**：检查是否有 `\psi^*`，必须改为 `\psi^{\ast}`
- [ ] **无 `|` 绝对值**：检查是否有 `$|x|$`，应改为 `$\vert x \vert$`
- [ ] **无 `\|` 范数**：检查是否有 `$\|x\|$`，应改为 `$\lVert x\rVert$`
- [ ] **无 `\bm{}` ���令**：检查是否有 `\bm{}`，改为 `$\alpha$` 或 `$\mathbf{p}$`
- [ ] **无 `\boldsymbol{}` 命令**：检查是否有 `\boldsymbol{}`，改为标准格式
- [ ] **数学模式无中文标点**：检查 `$...$` 中是否有中文逗号、句号
- [ ] **无 `\\\\` 换行符**：矩阵中使用 `\\` 而非 `\\\\`

**检查命令**：
```bash
# 检查 ^* 复共轭
grep -n '\\\^\*' 文章.md

# 检查 | 绝对值
grep -n '\$.*\|.*\|.*\$' 文章.md

# 检查 \bm 命令
grep -n '\\bm{' 文章.md

# 检查 \boldsymbol 命令
grep -n '\\boldsymbol{' 文章.md
```

**6. 花括号配对检查** ⚠️

**重要**：花括号必须正确配对，特别是在下标和上标嵌套时。

**常见错误模式**：
```latex
❌ u_{i,j-1}}^{(k)}     # 多余的 }
❌ u^{(k)}_{new}        # 顺序错误
```

**检查命令**：
```bash
# 检查 }{ 模式（多余的花括号）
grep -n '\}{' 文章.md

# 检查 }} 模式（连续的右花括号）
grep -n '\}\}' 文章.md
```

**预防措施**：
- 写完公式立即用编辑器的括号匹配功能检查
- 复杂公式在单独文件中测试
- 发布前运行 Hugo 构建验证

#### 快速验证命令

```bash
# 检查是否有未包裹的 Unicode 数学符号
grep -n 'α\|β\|γ\|×\|→\|∈' content/posts/[文章文件].md
```

### 3. 图表检查

#### 3.1 Plotly 数理图形检查

数学和物理图形需满足：
- [ ] 使用 Plotly 生成（非 Mermaid）
- [ ] 配色符合苹果风格（主色 #007AFF）
- [ ] 使用 `plotly_white` 模板
- [ ] 图形保存为 HTML 文件在 `static/images/plots/`
- [ ] 文件大小 > 5KB（包含 JavaScript）
- [ ] 在文章中正确嵌入：
  ```html
  <div class="plot-container">
    <iframe src="/images/plots/图形文件.html" width="100%" height="500" frameborder="0"></iframe>
  </div>
  ```

#### 3.2 对比图可视化检查（多子图时）

当使用多子图展示对比概念时，必须检查：

**视觉区分度检查**：
- [ ] **每个子图有独特视觉特征**（不仅是标题不同）
- [ ] **关键概念有对应视觉元素**：
  - 动态过程 → 箭头
  - 区间/范围 → 背景色块或填充
  - 边界线 → 虚线或点线
  - 关键点 → 特殊标记（菱形、星形等）
- [ ] **标注清晰可见**：
  - 不与曲线重叠
  - 字体大小适中（≥10pt）
  - 颜色与背景有足够对比度

**快速验证方法**：
```bash
# 生成图片后，目视检查
# 1. 遮挡标题，能否区分三图？
# 2. 每个子图是否有独特的颜色/线型/标记？
# 3. 关键概念（如 dx, ε, δ）是否有对应的视觉元素？
```

**常见错误**：
```python
# ❌ 错误：三图几乎相同
fig.add_trace(go.Scatter(x=x, y=y), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=y), row=1, col=2)  # 完全一样的曲线
fig.add_trace(go.Scatter(x=x, y=y), row=1, col=3)  # 完全一样的曲线

# ✅ 正确：每图有独特元素
# 左图：箭头 + 特殊标记
fig.add_annotation(x=-0.8, y=0.85, text='趋近', arrowcolor='#FF3B30', ...)
fig.add_trace(go.Scatter(x=[0], y=[1], marker=dict(symbol='diamond', color='#34C759')), ...)

# 中图：背景区间
fig.add_vrect(x0=-dx, x1=dx, fillcolor='rgba(175, 82, 222, 0.15)', ...)

# 右图：填充区域 + 虚线
fig.add_trace(go.Scatter(fill='toself', fillcolor='rgba(255, 149, 0, 0.25)', ...))
fig.add_hline(y=1+epsilon, line=dict(dash='dot', color='#FF9500'), ...)
```

#### 3.2 Mermaid 流程图检查

仅用于非数理的流程图、概念图：
- [ ] 所有节点包含 `color:#ffffff`
- [ ] 使用苹果风格配色
- [ ] 边框宽度正确（核心 3px，重要 2px，次要 1px）
- [ ] 使用 subgraph 分组相关元素

### 4. 图片检查

```bash
# 检查图片文件大小（必须 > 10KB）
ls -lh static/images/covers/[图片文件].jpg

# 统计信息
file content/posts/[文章文件].md
```

验证：
- ✅ 图片大小 > 10KB
- ✅ 文件编码为 UTF-8
- ✅ 图片路径正确

#### 3.3 Plotly 图形检查

验证 Plotly 生成的数理图形：
- ✅ HTML 文件大小 > 5KB
- ✅ 文件位于 `static/images/plots/` 目录
- ✅ 使用苹果风格配色
- ✅ 图表标题和标签清晰
- ✅ 在文章中正确嵌入 iframe

### 5. Front Matter 检查

验证 YAML front matter 格式：

```yaml
---
title: "文章标题"
date: YYYY-MM-DDTHH:mm:ss+08:00
draft: false
description: "文章简介"
categories: ["分类1", "分类2"]
tags: ["标签1", "标签2"]
cover:
    image: "images/covers/[图片文件].jpg"
    alt: "图片描述"
    caption: "图片标题"
math: true
---
```

检查要点：
- ✅ 三个短横线 `---` 正确
- ✅ date 格式包含时区
- ✅ categories 和 tags 是数组
- ✅ cover 字段缩进正确（2 空格）

### 6. 内容完整性检查

快速扫描文章结构：

```bash
# 显示文章标题
grep "^## " content/posts/[文章文件].md | head -20
```

验证：
- ✅ 有引言部分
- ✅ 主要章节完整
- ✅ 有结语部分
- ✅ 参考文献已列出（如需要）

### 7. 链接检查

如果文章包含外部链接：

```bash
# 提取所有链接
grep -o 'https://[^)]*' content/posts/[文章文件].md | sort -u
```

验证：
- ✅ 所有链接格式正确
- ✅ 引用链接有效

## 自动化检查脚本

创建一个完整的检查脚本：

```bash
#!/bin/bash
# check_article.sh

ARTICLE="$1"

echo "=== 检查文章: $ARTICLE ==="

# 1. 编码检查
python3 << EOF
import sys
with open('$ARTICLE', 'r', encoding='utf-8') as f:
    content = f.read()

if '\ufffd' in content:
    print("❌ 发现替换字符")
    sys.exit(1)

print("✅ 编码检查通过")
EOF

if [ $? -ne 0 ]; then
    echo "编码检查失败"
    exit 1
fi

# 2. 数学公式检查
echo "检查数学符号..."
if grep -E 'α|β|γ|δ|θ|λ|μ|π|σ|φ|ω|Δ|Σ|Φ|Ψ' "$ARTICLE" | grep -v '\\\\(alpha\|beta\|gamma\|delta\|theta\|lambda\|mu\|pi\|sigma\|phi\|omega\|Delta\|Sigma\|Phi\|Psi' | grep -v '\$'; then
    echo "⚠️  发现未包裹的希腊字母"
else
    echo "✅ 数学符号检查通过"
fi

# 3. 图片检查
IMAGE=$(grep 'image:' "$ARTICLE" | head -1 | sed 's/.*image: "\(.*\)".*/\1/')
if [ -n "$IMAGE" ]; then
    SIZE=$(stat -f%z "static/images/covers/$IMAGE" 2>/dev/null || stat -c%s "static/images/covers/$IMAGE" 2>/dev/null)
    if [ "$SIZE" -lt 10240 ]; then
        echo "❌ 图片太小: $SIZE bytes"
    else
        echo "✅ 图片检查通过 ($SIZE bytes)"
    fi
fi

echo "=== 检查完成 ==="
```

使用方法：
```bash
chmod +x check_article.sh
./check_article.sh content/posts/[文章文件].md
```

## 修复流程

发现问题后的修复流程：

1. **定位问题行**：使用检查脚本或 Read 工具
2. **理解问题**：确定是编码问题、数学公式还是其他
3. **修复问题**：使用 Edit 工具修复
4. **重新验证**：再次运行检查脚本
5. **确认修复**：所有检查都通过

## 质量标准

文章完成并通过所有检查后，应满足：

- ✅ **无乱码**：无替换字符，无编码错误
- ✅ **数学规范**：所有公式符合 LaTeX 规范
- ✅ **Plotly 图形专业**：数理图形美观，配色协调，交互性好
- ✅ **图片有效**：封面图 > 10KB
- ✅ **格式正确**：Front Matter 格式正确
- ✅ **内容完整**：章节完整，逻辑清晰

只有通过所有检查，文章才能被认为是"完成"的。
