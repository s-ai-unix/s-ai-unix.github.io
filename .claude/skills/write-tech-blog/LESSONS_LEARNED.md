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
