# Write Tech Blog 技能更新日志

## 更新日期
2026-02-01

## 主要更新内容

### 1. 新增自升级机制（Self-Improvement Protocol）

#### 新增文件
1. **`SELF_IMPROVEMENT.md`** - Skill 自升级协议
   - 定义自升级触发条件（Bug 修复后、优化后、模式重复等）
   - 五阶段升级流程：问题记录 → 反思分析 → 更新决策 → 执行更新 → 验证
   - 四种更新类型：紧急补丁/规范补充/重构优化/暂不更新
   - 更新原则：DRY、正例+反例、可检查性、最小侵入

2. **`LESSONS_LEARNED.md`** - 经验教训记录
   - 记录使用中发现的问题和修复
   - 驱动 Skill 的持续改进
   - 定期回顾识别模式

#### 首次应用案例：图片路径规范
- **问题**：epsilon-delta 文章使用相对路径导致图片无法显示
- **分析**：规范遗漏 - SKILL.md 未明确 Hugo 环境下应使用绝对路径
- **更新类型**：B - 规范补充
- **Skill 更新**：在 SKILL.md 中添加"图片路径规范"小节

### 2. 图片路径规范

#### SKILL.md 新增章节
- 明确要求使用以 `/` 开头的绝对路径
- 提供正例和反例对比
- 解释 Hugo 嵌套路径的工作原理
- 添加检查清单

### 3. 对比图设计规范

#### 第二次自升级案例
- **问题**：极限概念演变图三子图无明显区别，用户反馈"没看到区别"
- **分析**：示例不足 - 缺少对比图设计的具体规范
- **更新类型**：B - 规范补充

#### SKILL.md 新增第7条原则
- **对比图设计规范**：多子图时每个子图必须有独特视觉特征
- 使用表格列出区分手段：背景色块、箭头方向、标记形状、线型差异、填充区域
- 提供正例（极限概念三图）和反例（代码对比）

#### QUALITY-CHECK.md 新增检查项
- **3.2 对比图可视化检查**：专门的多子图检查清单
- 视觉区分度检查
- 关键概念视觉元素检查
- 标注清晰度检查
- 提供快速验证方法和常见错误示例

#### LESSONS_LEARNED.md 新增记录
- 记录对比图可视化问题及修复
- 更新 Skill 更新状态

---

## 历史更新

## 更新日期
2024-12-24

## 主要更新内容

### 1. 支持 Plotly 数理图形

#### 改进点
- **传统方式**：仅支持 Mermaid 图表，适合流程图
- **新方式**：
  - 数理图形（函数图像、几何演化、物理可视化）使用 Plotly
  - 流程图、概念关系图继续使用 Mermaid
  - 明确区分使用场景

#### 新增文件
1. **`generate_plots.py`** - 图形生成脚本
   - 预定义 Ricci Flow 相关图形
   - 测地线和曲率可视化
   - 爱因斯坦场方程图示
   - 热方程对比图

2. **`PLOTLY-GUIDE.md`** - Plotly 使用指南
   - 嵌入方法
   - 样式要求
   - 自定义图形模板
   - 最佳实践

### 2. 更新质量检查

#### 质量检查文档 (`QUALITY-CHECK.md`)
- 新增 **Plotly 图形检查**章节
- 更新 **图表检查**为两部分：
  1. Plotly 数理图形检查
  2. Mermaid 流程图检查
- 调整质量标准

### 3. 技能描述更新

#### SKILL.md
- 更新描述：加入 "Plotly 数理图形"
- 新增 **图表生成策略**章节
- 明确区分不同类型图表的使用场景

### 4. 配色规范

#### 苹果风格配色
- 主色：`#007AFF`（蓝色）
- 辅助色：`#34C759`（绿色）
- 强调色：`#FF9500`（橙色）
- 背景：`plotly_white` 模板

## 使用指南

### 生成图形
```bash
cd /Users/sun1/.claude/skills/write-tech-blog
python3 generate_plots.py
```

### 在文章中使用
```html
<div class="plot-container">
  <iframe src="/images/plots/图形文件.html" width="100%" height="500" frameborder="0"></iframe>
</div>
```

### 图形类型选择

| 场景 | 推荐工具 | 示例 |
|------|---------|------|
| 函数图像 | Plotly | y=sin(x), 曲面图 |
| 几何演化 | Plotly | Ricci Flow 演化 |
| 物理可视化 | Plotly | 时空弯曲 |
| 流程图 | Mermaid | 概念关系 |
| 结构图 | Mermaid | 算法流程 |

## 预生成图形

### Ricci Flow 系列
1. `ricci-flow-evolution.html` - 不同维度球面演化
2. `ricci-curvature-initial.html` - 初始曲率分布

### 几何系列
3. `geodesics-sphere.html` - 球面测地线
4. `gaussian-curvature.html` - 双曲面曲率

### 物理系列
5. `einstein-field-equations.html` - 爱因斯坦方程
6. `heat-equation-comparison.html` - 热方程对比

## 注意事项

1. **文件位置**：Plotly 图形保存在 `static/images/plots/`
2. **服务器要求**：需支持 HTML 文件和 iframe
3. **首次加载**：可能需要等待 JavaScript 加载
4. **响应式设计**：使用 width="100%" 自适应

## 向后兼容性

- 现有 Mermaid 图表继续有效
- 只需在新文章中采用新的图形策略
- 不影响已发布文章

## 示例文章

Ricci Flow 文章已更新：
- 6个 Plotly 交互图形
- 3个 Mermaid 流程图
- 混合使用最佳实践