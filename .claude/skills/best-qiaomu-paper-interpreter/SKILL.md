---
name: best-qiaomu-paper-interpreter
description: Transform academic papers into conversational Chinese articles in Qiaomu's style. Use when user provides PDF URL with keywords "解读论文", "理解paper", or "乔木风格". Runs fully automatically.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch, TodoWrite
---

# 乔木论文解读

## 概述

将学术论文自动转化为乔木风格的通俗易懂解读文章。**全自动执行**，无需用户中途确认。

**核心特点**：
- 对话式语言，像和朋友聊天
- 每个术语都有引用块解释
- 生活化类比帮助理解
- 自动提取论文图表
- 生成《纽约客》风格配图

**输出**：
- 8000-10000字深度解读文章
- 完整论文档案（原始论文 PDF + 图表 + 元数据）
- 发布就绪的 Markdown 文件
- **精美 HTML 单文件**（内嵌所有图片，base64编码）


---

## 📁 文件管理规范

**核心原则**：
- 工作目录保留完整档案（原始论文 PDF + 图表 + 配图 + 所有中间文件）
- 最终输出文件复制到用户指定的阅读目录

### 默认工作目录

**主工作目录**（所有论文资源）：
```
~/Gitlab/Personal/Experimental/Paper-Reading/01.inbox/papers/{paper_id}/
```

```
~/Gitlab/Personal/Experimental/Paper-Reading/01.inbox/papers/SOTIF_Fuzzy_Cause_Trees_2025/
├── {paper_id}.pdf          # 原始论文
├── metadata.json           # 元数据
├── extracted_text.md       # 提取的文本
├── {标题}.md               # Markdown 源文件（工作副本）
├── images/                 # 原文图表（PDF 提取）
├── illustrations/          # AI生成配图（纽约客风格）
└── visual_config.json      # 配图配置（临时）
```

**临时工作目录**（仅用于单篇论文处理，避免污染主目录）：
```
01.inbox/papers/{paper_id}/  # 作为 ~/Gitlab/.../01.inbox/papers/ 的替身
```

### 最终输出目录（默认）

**HTML 输出目录**（默认，可被用户指定覆盖）：
```
~/Gitlab/Personal/Experimental/Paper-Reading/paper_html/{paper_id}/
```

**文件命名格式**：
文件命名格式：
```
{paper_id}_{skill名称}_{配图模型}_{时间戳}.html
```

示例：
```
SOTIF_Fuzzy_Cause_Trees_2025_best-qiaomu_jimeng-4.5_20260210_180000.html
```

**重要**：
- 如果用户**没有指定路径**，默认使用 `~/Gitlab/Personal/Experimental/Paper-Reading/01.inbox/`
- 如果用户**指定了路径**（如 `--output /custom/path`），则使用用户指定的路径
- **单篇论文处理**可以使用临时目录 `01.inbox/papers/`，处理完成后手动移动到主目录

**优势**：
- ✅ 工作目录保留完整档案（所有资源）
- ✅ 输出目录只有最终的 HTML（便于分享）
- ✅ 支持用户自定义输出路径（优先级高于默认）

详见：`FILE_MANAGEMENT.md` 和 `FILE_LOCATIONS.md`

---

## 自动化工作流程

**执行原则**：
1. ✅ 全程自动（**配图模型选择除外**）
2. ✅ 使用TodoWrite显示进度
3. ✅ 静默修复质量问题
4. ✅ 生成完整最终版本

**配图模型选择（强制）**：
- 每次触发本 skill 且涉及配图时，先询问用户选择 `jimeng-4.1`（免费）或 `jimeng-4.5`（付费）。
- 若用户未明确指定，默认使用 `jimeng-4.5`。
 - **自动降级**：若 `jimeng-4.5` 在网关侧出现超时/`524`（常见于生成耗时过长），会自动降级到 `jimeng-4.1` 继续完成配图，避免整篇文章因配图失败而中断。

**推荐顺序**：步骤0 → 1 → 5 → 2 → 3-7

---

## 图表质量闸门（强制）

**重要**：一个“逻辑/语义错误”的流程图比一张“不好看”的图危害更大。因此本 skill 对图表采取 **Fail-Closed** 策略：

- **Mermaid 图**：
  - 生成 HTML 时会做 **严格语法解析**；解析失败则 **阻止渲染**，并在页面中显示错误信息（避免读者被“看起来像对的”图误导）。
  - 对过大/过密/容易溢出的图（超长、节点过多、并行边过多、疑似 Markdown 列表标签等）会 **直接挡住**，要求先简化再呈现。
  - 生成器会对过长的节点标签做保守自动换行（插入 `<br/>`），优先保证“框能包住字”。

- **Canvas / Excalidraw**：
  - 最终文章中 **禁止直接嵌入 `<canvas>`** 或仅引用 `.excalidraw.md`。
  - 必须导出为 **静态 PNG** 再插入（可读、可审查、可归档）。

**离线质量检查脚本**（可在生成 HTML 前先跑一遍）：

```bash
python3 /Users/sun1/.agents/skills/best-qiaomu-paper-interpreter/scripts/quality_gate_markdown.py <article.md>
```

---

### Python 运行环境（强制）

统一使用 conda Python（避免系统 Python 缺包）：

```bash
PYTHON_CMD="/Users/sun1/miniconda3/envs/py3.13env/bin/python3"
```

说明：
- 本 skill 默认按上面的 conda Python 执行所有脚本。
- `generate_illustrations_v2.py` 已内置运行时兜底：如果当前 Python 缺少 `requests`，会自动切换到可用的 conda Python 重启执行。

---

### 初始化：创建进度追踪

**第一步**：创建todo list，让用户看到进度

```python
TodoWrite([
    {"content": "下载PDF并创建论文目录", "status": "in_progress", "activeForm": "下载PDF并创建论文目录"},
    {"content": "提取PDF文本内容", "status": "pending", "activeForm": "提取PDF文本内容"},
    {"content": "生成乔木风格解读文章", "status": "pending", "activeForm": "生成乔木风格解读文章"},
    {"content": "提取论文图表", "status": "pending", "activeForm": "提取论文图表"},
    {"content": "生成纽约客风格配图", "status": "pending", "activeForm": "生成纽约客风格配图"},
    {"content": "保存最终文件", "status": "pending", "activeForm": "保存最终文件"},
    {"content": "生成HTML", "status": "pending", "activeForm": "生成HTML"}
])
```

**每完成一步**，立即更新状态为`completed`，下一步为`in_progress`

---

### 步骤0：智能PDF管理

**目标**：下载PDF，创建规范化目录结构

**执行**：
```bash
# 使用脚本自动处理
$PYTHON_CMD /Users/sun1/.agents/skills/best-qiaomu-paper-interpreter/scripts/extract_pdf_metadata.py \
  <temp_pdf> papers <url>

# 参数说明：
# 1. <temp_pdf>: 临时下载的PDF路径（如 /tmp/paper.pdf）
# 2. papers: 输出基础目录
# 3. <url>: 原始PDF的URL（可选，用于提取年份等信息）
```

**输出**：
- `01.inbox/papers/{paper_id}/` 目录（如 `papers/SAM_2025/`）
- `{paper_id}.pdf` 重命名的PDF
- `metadata.json` 元数据文件

**paper_id生成规则**：
- 提取论文标题中的缩写词（如 SAM, BERT, GPT）
- 如无缩写，提取前2-3个关键词
- 加上年份，生成简洁标识（如 `SAM_2025`, `BERT_2018`）

**元数据包含**：
- paper_id（如 "SAM_2025"）
- title（完整标题）
- year（发表年份）
- authors（作者列表）
- source_url（来源URL）

**完成后**：更新todo状态

---

### 步骤1：提取PDF文本

**目标**：提取完整文本内容，保留格式和结构

**执行**：使用 Markitdown 将 PDF 转换为 Markdown

```bash
markitdown {paper_dir}/{paper_id}.pdf -o {paper_dir}/extracted_text.md
```

**输出**：`{paper_dir}/extracted_text.md`

**架构优势**（相比 pdfplumber）：
- ✅ 保留完整格式（空格、标点、换行）
- ✅ 数学公式结构完整
- ✅ 表格自动转换为 Markdown 表格
- ✅ 图表描述清晰可读
- ✅ 文档结构语义化（标题、引用、列表）

**重点关注**：
- 摘要和结论
- 方法描述
- 实验结果
- 图表标题（Figure X, Table X）

**完成后**：更新todo状态

---

### 步骤5：提前提取图表

**目标**：提取所有Figure和Table，供写作时引用

**执行**：
```bash
cd {paper_dir}
$PYTHON_CMD /Users/sun1/.agents/skills/best-qiaomu-paper-interpreter/scripts/extract_all_figures.py \
  {paper_id}.pdf images {paper_id}
```

**输出**：
- `images/{paper_id}_figure1.png`
- `images/{paper_id}_table1.png`
- `images/figure_list.md`（引用清单）

**特性**：
- 全自动识别Figure/Table标记
- 智能定位边界
- 2x高清分辨率

**完成后**：更新todo状态

---

### 步骤2：生成完整解读

**目标**：一次性生成最终完整版本（不要分初稿和完善版）

#### 内容结构（必须包含）

1. **引入**：用故事/场景引入，不直接讲技术
2. **核心概念**：术语解释（引用块） + 生活化类比
3. **技术细节**：是什么 → 为什么 → 怎么做
4. **实验数据**：表格展示，加粗重点
5. **深度洞察**：方法论启发、历史意义
6. **结尾升华**：延伸到认知层面

#### 术语解释格式（强制）

```markdown
> **Transformer**：一种神经网络架构，核心是"自注意力机制"。可以想象成你在读一句话时，会自动关注句子中最重要的几个词，而不是平均分配注意力。
```

#### 图表引用策略

- 查看`images/figure_list.md`，了解可用图表
- **自然引用**，不刻意堆砌
- 引用格式：`![描述](01.inbox/papers/{paper_id}/images/{paper_id}_figure1.png)`
- 建议：核心架构图、关键实验数据、可视化分析

#### 风格要求（严格遵守）

详见`references/style-guide.md`，核心：
- ✅ 短段落，多留白
- ✅ "就像""比如""试想一下"
- ✅ 中文标点（，。：！？）
- ✅ 重要观点加粗
- ❌ 绝对不用破折号
- ❌ 不用"首先""其次""值得注意的是"

#### 内部质量检查（静默执行）

生成后自检：
- 核心贡献覆盖？
- 术语解释完整？（至少15处）
- 生活化类比？（至少3处）
- 数据表格？（至少1个）
- 图表引用？（至少2处）
- 破折号？（必须0个）
- 中文标点？（100%）

发现问题→静默修复→继续

**完成后**：
- 保存到`{paper_dir}/{中文标题}_解读.md`
- 更新todo状态

---

### 步骤5.5：生成《纽约客》配图

**目标**：为每个H2标题生成配图（**并发加速，3倍性能提升**）

**工作流**：
1. 创建配置模板：
```bash
$PYTHON_CMD /Users/sun1/.agents/skills/best-qiaomu-paper-interpreter/scripts/generate_illustrations_v2.py \
  --create-template {filename}
```

2. Claude填写`visual_config.json`：
   - 读取文章，理解每个H2核心观点
   - 设计50-80字的具体视觉场景
   - **用具象物体/场景隐喻抽象概念**
   - 必须用英文描述，不包含风格指令和文字指令（详见`visual_description_guide.md`）

3. 批量并发生成：
```bash
# 默认付费版（jimeng-4.5）
$PYTHON_CMD /Users/sun1/.agents/skills/best-qiaomu-paper-interpreter/scripts/generate_illustrations_v2.py \
  {filename} --workers 3

# 显式指定免费版（jimeng-4.1）
$PYTHON_CMD /Users/sun1/.agents/skills/best-qiaomu-paper-interpreter/scripts/generate_illustrations_v2.py \
  {filename} --free --workers 3
```

**性能优化**：
- ⚡ 并发生成：3线程并发，12张图约2分钟（vs 串行6分钟）
- 🔒 线程安全：文件读写加锁，防止竞态条件
- 🛡️ 防重复：自动检测并跳过已插入的配图
- 🎯 智能重试：2次重试，友好错误提示

**配图要求**：
- 钢笔墨水速写，16:9比例
- 黑白线条 + 朱红色点缀
- 简洁留白，松弛线条
- 底部中文标题

**配图模型选项**（v2.1新增）：
- **jimeng-4.5**（默认）：付费模型，图片质量更高
- **jimeng-4.1**：免费模型，质量略低但完全免费
- 每次配图前先询问用户模型选择；未指定时按默认 `jimeng-4.5` 执行
- 用户明确要求免费模型时，使用 `--free` 参数

**API配置**：
- 默认使用即梦API（单张约29秒）
- Gemini当前不可用（503错误）
- 可通过`--provider jimeng|gemini|auto`切换

**完成后**：更新todo状态

---

### 步骤6：保存最终文件

**目标**：用H1标题作为文件名，保存到根目录

**执行**：
```bash
$PYTHON_CMD /Users/sun1/.agents/skills/best-qiaomu-paper-interpreter/scripts/finalize_markdown.py \
  {paper_dir}/{filename} .
```

**处理**：
- 提取H1标题
- 删除文章中的H1行
- 用H1标题命名保存到根目录

**效果**：
- 根目录：`AI的突破.md`（最终版，无H1）
- 工作目录：`papers/xxx/xxx_解读.md`（保留H1）

**完成后**：更新todo状态为completed

---

### 步骤6.5：生成 HTML

**目标**：生成精美的独立 HTML 单文件，适合分享与离线阅读

**执行**：
```bash
# 1. 生成 HTML（在工作目录）
$PYTHON_CMD /Users/sun1/.agents/skills/best-qiaomu-paper-interpreter/scripts/generate_html.py \
  {paper_dir}/{标题}.md \
  {paper_dir}/{标题}.html

# 2. 质量检查：验证图表提取质量
$PYTHON_CMD /Users/sun1/.agents/skills/best-qiaomu-paper-interpreter/scripts/validate_figure_extraction.py \
  {paper_dir}/images \
  {paper_dir}/{paper_id}.pdf

# 3. 发布到 Paper-Reading 目录（带时间戳）
$PYTHON_CMD /Users/sun1/.agents/skills/best-qiaomu-paper-interpreter/scripts/publish_to_paper_reading.py \
  {paper_dir}/{标题}.html \
  --model {使用的模型，如 jimeng-4.5}
```

**输出**：
- **工作目录**：
  - `{标题}.html`：单文件 HTML，所有图片内嵌为 base64（约 8-10MB）

- **Paper-Reading 目录**（默认）：
  ```
  /Users/sun1/Gitlab/Personal/Experimental/Paper-Reading/paper_html/{paper_id}/
  └── {paper_id}_best-qiaomu_{model}_{timestamp}.html
  ```

**特性**：
- ✅ 响应式设计（手机/平板/电脑自适应）
- ✅ MathJax 3 数学公式渲染
- ✅ 优雅的阅读体验（模仿 Medium/纽约客风格）
- ✅ 单文件便携（HTML 包含所有资源，离线可读）
- ✅ 自动时间戳命名（避免覆盖旧版本）
- ✅ 图表质量验证（防止提取不完整）

**依赖**：
- Pillow（图表质量检查）
- 自动内嵌图片为 base64（无需外部图片文件）

**自定义输出路径**：
如果用户指定了输出路径（如 `--output /custom/path`），则使用用户指定的路径而非默认路径。

**完成后**：更新todo状态为completed

---

### 步骤7：完成报告

简短告知用户：

```
✅ 论文解读完成！

📄 最终文件：
   - Markdown: {标题}.md
   - HTML: {paper_id}_best-qiaomu_{model}_{timestamp}.html

📊 文档统计：
   - 字数：约X字
   - 术语解释：X处
   - 生活化类比：X处
   - 图表引用：X张（Figure Y张，Table Z张）
   - 纽约客配图：X张

📁 工作档案：01.inbox/papers/{paper_id}/
   ├── {paper_id}.pdf（原始论文）
   ├── metadata.json
   ├── extracted_text.md
   ├── {标题}.md（工作副本）
   ├── {标题}.html（独立HTML）
   ├── images/（论文原始图表，已通过质量检查）
   └── illustrations/（AI生成配图）

📂 发布位置：
   - HTML 已复制到:
   /Users/sun1/Gitlab/Personal/Experimental/Paper-Reading/paper_html/{paper_id}/

🌐 在线阅读：
   - HTML 版本支持响应式设计，推荐用浏览器打开

✅ 图表质量检查：已通过
```

---

## 质量检查清单

生成的文档必须通过：

- [x] 所有术语有引用块解释（≥15处）
- [x] 生活化类比（≥3处）
- [x] 语言口语化
- [x] 破折号（=0个）
- [x] 中文标点（100%）
- [x] 重要观点加粗
- [x] 图表引用（≥2处）
- [x] 数据表格（≥1个）
- [x] 结尾有升华

---

## 参考文档

- **风格指南**：`references/style-guide.md`
- **使用示例**：`examples.md`
- **故障排查**：`TROUBLESHOOTING.md`
- **配图设计**：`visual_description_guide.md`
- **版本历史**：`CHANGELOG.md`
