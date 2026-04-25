#!/usr/bin/env python3
"""生成 rsync 语义对比图"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2,
    horizontal_spacing=0.25,
    subplot_titles=("镜像同步（rsync --delete）", "合并添加（cp 或 git merge）")
)

# 配色
color_src = '#007AFF'
color_dst = '#34C759'
color_delete = '#FF3B30'
color_arrow = '#86868B'

# 左图：镜像同步
fig.add_trace(go.Scatter(
    x=[0.5], y=[1.0],
    mode='markers+text',
    marker=dict(size=180, color=color_src, opacity=0.7),
    text=['SOURCE'],
    textposition='middle center',
    textfont=dict(size=14, color='white'),
    name='SOURCE'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[0.5], y=[0.0],
    mode='markers+text',
    marker=dict(size=180, color=color_dst, opacity=0.7),
    text=['DEST'],
    textposition='middle center',
    textfont=dict(size=14, color='white'),
    name='DEST'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[0.3, 0.5, 0.7],
    y=[0.5, 0.65, 0.5],
    mode='text',
    text=['删除', '同步', '添加'],
    textposition='middle center',
    textfont=dict(size=11, color=[color_delete, color_arrow, color_dst]),
    showlegend=False
), row=1, col=1)

# 右图：合并添加
fig.add_trace(go.Scatter(
    x=[0.5], y=[1.0],
    mode='markers+text',
    marker=dict(size=180, color=color_src, opacity=0.7),
    text=['SOURCE'],
    textposition='middle center',
    textfont=dict(size=14, color='white'),
    name='SOURCE'
), row=1, col=2)

fig.add_trace(go.Scatter(
    x=[0.5], y=[0.0],
    mode='markers+text',
    marker=dict(size=180, color=color_dst, opacity=0.7),
    text=['DEST'],
    textposition='middle center',
    textfont=dict(size=14, color='white'),
    name='DEST'
), row=1, col=2)

fig.add_trace(go.Scatter(
    x=[0.5],
    y=[0.5],
    mode='text',
    text=['只添加，不删除'],
    textposition='middle center',
    textfont=dict(size=13, color=color_dst),
    showlegend=False
), row=1, col=2)

fig.add_annotation(x=0.5, y=-0.35, text='结果：DEST = 原有文件 + SOURCE 的新文件',
                  row=1, col=2, showarrow=False, font=dict(size=12, color='#86868B'))

fig.update_layout(
    width=900, height=450,
    plot_bgcolor='white',
    showlegend=False,
    margin=dict(t=60, b=80)
)

fig.write_image('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/rsync-delete-semantics.png', scale=2)
print('done')
