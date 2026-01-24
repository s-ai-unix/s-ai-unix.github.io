# Write Tech Blog 技能更新日志

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