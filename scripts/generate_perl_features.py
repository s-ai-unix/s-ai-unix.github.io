#!/usr/bin/env python3
"""
生成 Perl 核心特性雷达图
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess

# 创建子图布局
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'polar'}, {'type': 'xy'}]],
    subplot_titles=('Perl 核心能力评估', 'Perl 版本代码量变化趋势')
)

# ========== 雷达图：Perl 核心能力 ==========
categories = ['文本处理', '正则表达式', '系统管理', 'CPAN生态', 
              '跨平台性', '开发效率', '运行性能', '向后兼容']

# Perl 在各维度的评分 (1-10)
perl_scores = [10, 10, 9, 9, 8, 9, 6, 10]

fig.add_trace(go.Scatterpolar(
    r=perl_scores + [perl_scores[0]],  # 闭合
    theta=categories + [categories[0]],
    fill='toself',
    fillcolor='rgba(0, 122, 255, 0.25)',
    line=dict(color='#007AFF', width=2),
    name='Perl',
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

# ========== 柱状图：版本代码量趋势 ==========
versions = ['5.0', '5.6', '5.8', '5.10', '5.20', '5.30', '5.40']
core_lines = [50000, 75000, 85000, 95000, 110000, 125000, 140000]

fig.add_trace(go.Bar(
    x=versions,
    y=core_lines,
    marker=dict(
        color=['#34C759', '#34C759', '#34C759', '#007AFF', 
               '#007AFF', '#007AFF', '#FF9500'],
        line=dict(width=1, color='#333')
    ),
    text=core_lines,
    textposition='outside',
    textfont=dict(size=10),
    name='核心代码行数',
    showlegend=False,
    hovertemplate='Perl %{x}: %{y:,} 行<extra></extra>'
), row=1, col=2)

fig.update_xaxes(title_text='Perl 版本', row=1, col=2)
fig.update_yaxes(title_text='核心代码行数', row=1, col=2)

# 整体布局
fig.update_layout(
    title=dict(
        text='Perl 语言特性与演进分析',
        font=dict(size=16, color='#333'),
        x=0.5
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial, sans-serif', size=12),
    margin=dict(l=80, r=80, t=100, b=60),
    height=450,
    showlegend=False
)

# 保存
output_path = 'static/images/plots/perl-features.png'
fig.write_image(output_path, scale=2)

# 压缩
subprocess.run([
    'pngquant', '--quality=70-85', '--force',
    '--output', output_path, output_path
], check=False)

print(f"✅ 已生成并压缩: {output_path}")
