# paper-master 论文大师

一篇论文，两个版本：技术版（X光机解构）+ 解读版（乔木风格）。

## 目录结构

```
paper-master/
├── SKILL.md              # 主技能文件
├── README.md             # 本说明
├── templates/           # 模板目录
│   ├── technical_template.md    # 技术版 Markdown 模板
│   ├── interpretive_template.md # 解读版 Markdown 模板
│   ├── technical_template.html  # 技术版 HTML 模板
│   ├── interpretive_template.html# 解读版 HTML 模板
│   └── diagram_flowchart.mmd    # Mermaid 流程图模板
└── examples/            # 示例目录（待添加）
    └── sample_paper/
```

## 快速开始

```
/paper-master
请分析这篇论文：{PDF 路径或 URL}
```

## 输出位置

所有文件保存到：
```
~/Gitlab/Personal/Experimental/Paper-Reading/paper_html/{paper_id}/
```

## 输出示例

```
{paper_id}/
├── {paper_id}_技术版.md
├── {paper_id}_技术版.html
├── {paper_id}_解读版.md
├── {paper_id}_解读版.html
├── diagram_xxx.mmd         # Mermaid 流程图（可选）
└── diagram_xxx.png         # 渲染后的图（可选）
```

## 核心特点

| 特性 | 技术版 | 解读版 |
|------|--------|--------|
| 定位 | X光机解构 | 乔木对话 |
| 读者 | 研究者 | 大众 |
| LaTeX 公式 | ✅ 严格 | ✅ 可选 |
| 术语解释 | ❌ | ✅ ≥15处 |
| 生活类比 | ❌ | ✅ ≥3处 |
| 批判分析 | ✅ | ❌ |
| Mermaid 图 | ✅ | 可选 |

## 模板使用

### 技术版 Markdown
```bash
cat templates/technical_template.md
# 复制并替换 {} 中的内容
```

### 解读版 Markdown
```bash
cat templates/interpretive_template.md
# 复制并替换 {} 中的内容
```

### Mermaid 流程图
```bash
cat templates/diagram_flowchart.mmd
# 复制需要的图表模板
```

## 配图生成

### Mermaid → PNG
```bash
mmdc -i diagram.mmd -o diagram.png -H 600
```

### Mermaid → SVG
```bash
mmdc -i diagram.mmd -o diagram.svg -t neutral
```

## LaTeX 速查

| 符号 | LaTeX | 符号 | LaTeX |
|------|-------|------|-------|
| ω | `$\omega$` | α | `$\alpha$` |
| β | `$\beta$` | μ | `$\mu$` |
| ∈ | `$\in$` | ∀ | `$\forall$` |
| → | `$\rightarrow$` | ⊙ | `$\odot$` |
| ⊗ | `$\otimes$` | ⊕ | `$\oplus$` |
| 分数 | `$\frac{a}{b}$` | 向量 | `$\mathbf{x}$` |

## 质量检查

### 技术版
- [ ] ≥3 个核心痛点/解题机制条目
- [ ] ≥1 个 LaTeX 公式
- [ ] ≥1 个隐形假设
- [ ] ≥1 个未解之谜

### 解读版
- [ ] 术语解释 ≥15 处
- [ ] 生活化类比 ≥3 处
- [ ] 破折号 = 0 个
- [ ] 中文标点 100%
- [ ] 重要观点加粗
- [ ] 结尾有升华

## 相关技能

- **paper-analysis**：技术版核心引擎
- **qiaomu-paper-interpreter**：解读版核心引擎
