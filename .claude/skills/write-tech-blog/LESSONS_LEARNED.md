# Skill 经验教训记录

本文档记录 write-tech-blog skill 使用过程中发现的问题及修复，用于驱动 Skill 的自升级。

---

## [2026-02-01] 图片路径格式错误导致网页无法显示

**问题类型**: 规范遗漏

**问题描述**: 
epsilon-delta 数学分析文章的 Plotly 图片在网页上无法正常显示。文章中使用的是相对路径 `images/plots/xxx.png`，但在 Hugo 生成的网站中，文章页面的 URL 可能是嵌套路径（如 `/posts/2026-02-01-epsilon-delta数学分析的严格化革命/`），导致相对路径无法正确解析到图片位置。

**影响范围**: 
- 所有使用 Plotly/PNG 图片的文章
- 所有插入图片的场景

**根本原因**: 
SKILL.md 中图片插入示例使用相对路径 `images/plots/xxx.png`，未明确说明 Hugo 静态站点环境下应使用以 `/` 开头的绝对路径。

**修复方案**: 
将文章中的 5 处图片路径从相对路径改为绝对路径：
- `images/plots/limit_concept_evolution.png` → `/images/plots/limit_concept_evolution.png`
- `images/plots/weierstrass_function.png` → `/images/plots/weierstrass_function.png`
- `images/plots/epsilon_delta_illustration.png` → `/images/plots/epsilon_delta_illustration.png`
- `images/plots/continuity_types.png` → `/images/plots/continuity_types.png`
- `images/plots/uniform_continuity.png` → `/images/plots/uniform_continuity.png`

**预防机制**: 
在 SKILL.md 中添加专门的"图片路径规范"章节，明确：
1. 必须使用以 `/` 开头的绝对路径
2. 说明原因（Hugo 文章可能是嵌套路径）
3. 提供正例和反例对比

**Skill 更新**:
- [x] 已更新 SKILL.md - 添加"图片路径规范"小节
- [ ] 已更新 QUALITY-CHECK.md - 添加图片路径检查项
- [ ] 已更新其他文档

**相关文件**:
- 文章文件: `content/posts/2026-02-01-epsilon-delta数学分析的严格化革命.md`
- Skill 文件: `.claude/skills/write-tech-blog/SKILL.md`

---

## [2026-02-01] 对比图可视化效果不佳，三个子图无明显区别

**问题类型**: 示例不足 / 规范遗漏

**问题描述**: 
极限概念演变图 (`limit_concept_evolution.png`) 中，三个子图（直观理解、无穷小方法、Epsilon-Delta）视觉效果几乎相同，都是简单的函数曲线。用户反馈"没看到有啥区别"，无法直观理解三种方法的差异。

具体缺陷：
1. 左图"趋近于"箭头未正确显示
2. 中图无穷小标记不明显，缺乏 dx 区间可视化
3. 右图 epsilon-delta 的误差带和邻域概念不够突出
4. 三图缺乏各自的视觉特征和区分度

**影响范围**: 
- 所有需要对比展示的 Plotly 图形
- 多子图布局的可视化效果

**根本原因**: 
1. SKILL.md 中 Plotly 作图原则缺少"对比图设计规范"
2. 没有强调多子图时每个子图应有独特的视觉元素
3. 标注位置和样式不够醒目

**修复方案**: 
1. 左图：添加红色"趋近"箭头 + 绿色"极限=1"菱形标记
2. 中图：添加紫色 dx 区间背景 + 公式标注 + dx 标记点
3. 右图：明确区分 epsilon 水平虚线（橙色）和 delta 垂直虚线（绿色）+ 误差带填充
4. 统一添加坐标轴标签 $x$ 和 $y$

代码改进点：
- 使用 `add_vrect` 添加区间背景色
- 使用不同颜色和线型区分元素
- 调整标注位置和字体大小
- 增加子图间距 `horizontal_spacing`

**预防机制**: 
在 SKILL.md 中添加"对比图设计规范"：
1. 每个子图必须有独特的视觉特征
2. 使用颜色、线型、标记、背景区等多种手段区分
3. 关键概念必须有对应的视觉元素（如 dx 对应紫色区间）
4. 标注必须清晰可见，避免与曲线重叠

**Skill 更新**:
- [x] 已更新 SKILL.md - 添加"对比图设计规范"小节（第7条原则）
- [x] 已更新 generate_epsilon_delta_plots.py - 修复代码
- [x] 已更新 QUALITY-CHECK.md - 添加"3.2 对比图可视化检查"

**相关文件**:
- 文章文件: `content/posts/2026-02-01-epsilon-delta数学分析的严格化革命.md`
- 图形脚本: `generate_epsilon_delta_plots.py`

---

## 模板条目

**问题类型**: Bug / 优化 / 规范补充 / 流程改进

**问题描述**: 
发生了什么，现象是什么

**根本原因**: 
为什么会发生这个问题

**修复方案**: 
如何解决的具体步骤

**预防机制**: 
如何避免再次发生

**Skill 更新**:
- [ ] 已更新 SKILL.md
- [ ] 已更新 QUALITY-CHECK.md
- [ ] 已更新其他文档

**相关文件**:
- 文章文件: `content/posts/xxx.md`
- Skill 文件: `.claude/skills/write-tech-blog/xxx.md`
