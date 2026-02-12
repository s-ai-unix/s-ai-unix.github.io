#!/usr/bin/env python3
"""
生成 Scala 核心特性雷达图和应用对比图
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess

# 创建子图布局
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'polar'}, {'type': 'xy'}]],
    subplot_titles=('Scala 核心能力评估', 'Scala vs Java vs Kotlin 特性对比')
)

# ========== 雷达图：Scala 核心能力 ==========
categories = ['面向对象', '函数式编程', '类型系统', '并发编程', 
              'JVM 互操作', '代码简洁性', '性能', '工具生态']

# Scala 在各维度的评分 (1-10)
scala_scores = [10, 10, 10, 9, 10, 9, 9, 8]

fig.add_trace(go.Scatterpolar(
    r=scala_scores + [scala_scores[0]],  # 闭合
    theta=categories + [categories[0]],
    fill='toself',
    fillcolor='rgba(0, 122, 255, 0.25)',
    line=dict(color='#007AFF', width=2),
    name='Scala',
    hovertemplate='%{theta}: %{r}/10<extra></extra>'
), row=1, col=1)

# 更新极坐标轴
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10],
            tickfont=dict(size=10)
        ),
        angularaxis=dict(
            tickfont=dict(size=11)
        )
    )
)

# ========== 分组柱状图：语言对比 ==========
categories_comp = ['类型推断', '函数式特性', '代码简洁性', '空安全', '协程支持', '学习曲线']
scala_vals = [9, 10, 9, 7, 8, 5]
java_vals = [6, 5, 5, 8, 7, 7]
kotlin_vals = [8, 7, 8, 10, 10, 8]

x = list(range(len(categories_comp)))
width = 0.25

fig.add_trace(go.Bar(
    x=[i - width for i in x],
    y=scala_vals,
    width=width,
    name='Scala',
    marker=dict(color='#007AFF', line=dict(width=1, color='#333')),
    text=scala_vals,
    textposition='outside',
    textfont=dict(size=9),
    showlegend=True,
    hovertemplate='Scala - %{x}: %{y}/10<extra></extra>'
), row=1, col=2)

fig.add_trace(go.Bar(
    x=x,
    y=java_vals,
    width=width,
    name='Java',
    marker=dict(color='#FF9500', line=dict(width=1, color='#333')),
    text=java_vals,
    textposition='outside',
    textfont=dict(size=9),
    showlegend=True,
    hovertemplate='Java - %{x}: %{y}/10<extra></extra>'
), row=1, col=2)

fig.add_trace(go.Bar(
    x=[i + width for i in x],
    y=kotlin_vals,
    width=width,
    name='Kotlin',
    marker=dict(color='#34C759', line=dict(width=1, color='#333')),
    text=kotlin_vals,
    textposition='outside',
    textfont=dict(size=9),
    showlegend=True,
    hovertemplate='Kotlin - %{x}: %{y}/10<extra></extra>'
), row=1, col=2)

fig.update_xaxes(
    tickvals=x,
    ticktext=categories_comp,
    tickangle=15,
    row=1, col=2
)
fig.update_yaxes(title_text='评分 (1-10)', range=[0, 12], row=1, col=2)

# 整体布局
fig.update_layout(
    title=dict(
        text='Scala 语言特性与生态位分析',
        font=dict(size=16, color='#333'),
        x=0.5
    ),
    barmode='group',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial, sans-serif', size=12),
    margin=dict(l=80, r=80, t=100, b=100),
    height=450,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.2,
        xanchor='center',
        x=0.5
    )
)

# 保存
output_path = 'static/images/plots/scala-features.png'
fig.write_image(output_path, scale=2)

# 压缩
subprocess.run([
    'pngquant', '--quality=70-85', '--force',
    '--output', output_path, output_path
], check=False)

print(f"✅ 已生成并压缩: {output_path}")
