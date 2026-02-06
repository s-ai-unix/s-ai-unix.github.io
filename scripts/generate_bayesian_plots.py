#!/usr/bin/env python3
"""
生成贝叶斯网络相关的 Plotly 图形
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import subprocess
import os

# 苹果风格配色
APPLE_BLUE = '#007AFF'
APPLE_GREEN = '#34C759'
APPLE_ORANGE = '#FF9500'
APPLE_RED = '#FF3B30'
APPLE_PURPLE = '#AF52DE'
APPLE_GRAY = '#8E8E93'

def save_and_compress(fig, filepath, width=900, height=600):
    """保存并压缩图片"""
    # 先保存为 PNG
    fig.write_image(filepath, width=width, height=height, scale=2)
    
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")


def draw_arrow(fig, x0, y0, x1, y1, color, node_radius=0.05):
    """在图中绘制带箭头的线段"""
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    
    if length < 0.001:
        return
    
    # 调整起点和终点，使其不进入节点圆圈内
    x0_adj = x0 + dx * (node_radius / length)
    y0_adj = y0 + dy * (node_radius / length)
    x1_adj = x0 + dx * (1 - node_radius / length)
    y1_adj = y0 + dy * (1 - node_radius / length)
    
    # 绘制线段
    fig.add_trace(go.Scatter(
        x=[x0_adj, x1_adj],
        y=[y0_adj, y1_adj],
        mode='lines',
        line=dict(color=color, width=2.5),
        showlegend=False
    ))


def plot_simple_bayesian_network():
    """
    简单的贝叶斯网络示例 - 洒水器-草地湿润问题
    """
    fig = go.Figure()
    
    # 节点位置
    nodes = {
        'Cloudy': (0.5, 0.8),
        'Sprinkler': (0.2, 0.4),
        'Rain': (0.8, 0.4),
        'WetGrass': (0.5, 0.0)
    }
    
    # 绘制边
    draw_arrow(fig, 0.5, 0.8, 0.2, 0.4, APPLE_BLUE, 0.08)
    draw_arrow(fig, 0.5, 0.8, 0.8, 0.4, APPLE_BLUE, 0.08)
    draw_arrow(fig, 0.2, 0.4, 0.5, 0.0, APPLE_GREEN, 0.08)
    draw_arrow(fig, 0.8, 0.4, 0.5, 0.0, APPLE_GREEN, 0.08)
    
    # 绘制节点
    node_colors = {
        'Cloudy': APPLE_BLUE,
        'Sprinkler': APPLE_PURPLE,
        'Rain': APPLE_PURPLE,
        'WetGrass': APPLE_ORANGE
    }
    
    for node, (x, y) in nodes.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=55, color=node_colors[node]),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[node],
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 添加节点标签
    labels = {'Cloudy': '多云', 'Sprinkler': '洒水器', 'Rain': '下雨', 'WetGrass': '草地湿'}
    for node, (x, y) in nodes.items():
        fig.add_annotation(
            x=x, y=y-0.12,
            text=labels[node],
            showarrow=False,
            font=dict(size=12, color='#333333', family='Arial')
        )
    
    fig.update_layout(
        title=dict(text='贝叶斯网络示例：洒水器-草地湿润问题', font=dict(size=16, family='Arial')),
        xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-0.2, 1], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False,
        width=700,
        height=500
    )
    
    return fig


def plot_conditional_probability():
    """
    条件概率的直观解释 - 韦恩图风格
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('全集 Ω', '已知 B 发生时'),
        horizontal_spacing=0.15
    )
    
    # 左图：全集
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 集合 B
    bx, by = 0.5 + 0.35*np.cos(theta), 0.5 + 0.35*np.sin(theta)
    fig.add_trace(go.Scatter(
        x=bx, y=by,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        line=dict(color=APPLE_BLUE, width=2),
        name='B',
        showlegend=True
    ), row=1, col=1)
    
    # 集合 A
    ax, ay = 0.35 + 0.3*np.cos(theta), 0.5 + 0.3*np.sin(theta)
    fig.add_trace(go.Scatter(
        x=ax, y=ay,
        fill='toself',
        fillcolor='rgba(52, 199, 89, 0.2)',
        line=dict(color=APPLE_GREEN, width=2),
        name='A',
        showlegend=True
    ), row=1, col=1)
    
    fig.add_annotation(x=0.35, y=0.5, text='A', showarrow=False, 
                      font=dict(size=16, color=APPLE_GREEN), row=1, col=1)
    fig.add_annotation(x=0.55, y=0.5, text='B', showarrow=False,
                      font=dict(size=16, color=APPLE_BLUE), row=1, col=1)
    fig.add_annotation(x=0.45, y=0.5, text='A∩B', showarrow=False,
                      font=dict(size=12, color='#333333'), row=1, col=1)
    
    # 右图：条件概率
    fig.add_trace(go.Scatter(
        x=bx, y=by,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.4)',
        line=dict(color=APPLE_BLUE, width=3),
        name='B (条件空间)',
        showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=ax, y=ay,
        fill='toself',
        fillcolor='rgba(52, 199, 89, 0.4)',
        line=dict(color=APPLE_GREEN, width=2),
        name='A',
        showlegend=True
    ), row=1, col=2)
    
    fig.add_annotation(x=0.5, y=0.5, text='P(A|B)', showarrow=False,
                      font=dict(size=18, color='#333333', family='Arial'),
                      row=1, col=2)
    
    fig.update_xaxes(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False, scaleanchor='x')
    
    fig.update_layout(
        title=dict(text='条件概率的直观理解：P(A|B) = P(A∩B) / P(B)', font=dict(size=16, family='Arial')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(x=0.5, y=-0.1, orientation='h', xanchor='center'),
        width=900,
        height=450
    )
    
    return fig


def plot_chain_structure():
    """
    链式结构：A -> B -> C
    """
    fig = go.Figure()
    
    # 节点位置
    nodes = {'A': (0.15, 0.5), 'B': (0.5, 0.5), 'C': (0.85, 0.5)}
    
    # 绘制边
    draw_arrow(fig, 0.15, 0.5, 0.5, 0.5, APPLE_BLUE, 0.06)
    draw_arrow(fig, 0.5, 0.5, 0.85, 0.5, APPLE_BLUE, 0.06)
    
    # 绘制节点
    node_labels = {'A': '感冒', 'B': '发烧', 'C': '咳嗽'}
    node_colors = {'A': APPLE_BLUE, 'B': APPLE_PURPLE, 'C': APPLE_ORANGE}
    
    for node, (x, y) in nodes.items():
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                marker=dict(size=60, color=node_colors[node]), showlegend=False))
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='text',
                                text=[node], textposition='middle center',
                                textfont=dict(size=16, color='white', family='Arial'),
                                showlegend=False))
        fig.add_annotation(x=x, y=y-0.12, text=node_labels[node],
                          showarrow=False, font=dict(size=12, color='#333333', family='Arial'))
    
    fig.add_annotation(x=0.5, y=0.85, text='链式结构：给定 B 时，A 与 C 条件独立',
                      showarrow=False, font=dict(size=14, color='#333333'))
    fig.add_annotation(x=0.5, y=0.78, text='A ⊥ C | B',
                      showarrow=False, font=dict(size=16, color=APPLE_RED, family='Arial'))
    
    fig.update_layout(
        title=dict(text='链式结构中的条件独立性', font=dict(size=16, family='Arial')),
        xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        width=700,
        height=400
    )
    
    return fig


def plot_v_structure():
    """
    V 结构：A -> C <- B
    """
    fig = go.Figure()
    
    # 节点位置
    nodes = {'A': (0.2, 0.8), 'B': (0.8, 0.8), 'C': (0.5, 0.2)}
    
    # 绘制边
    draw_arrow(fig, 0.2, 0.8, 0.5, 0.2, APPLE_BLUE, 0.06)
    draw_arrow(fig, 0.8, 0.8, 0.5, 0.2, APPLE_BLUE, 0.06)
    
    # 绘制节点
    node_labels = {'A': '认真学习', 'B': '聪明', 'C': '考高分'}
    node_colors = {'A': APPLE_GREEN, 'B': APPLE_GREEN, 'C': APPLE_ORANGE}
    
    for node, (x, y) in nodes.items():
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                marker=dict(size=60, color=node_colors[node]), showlegend=False))
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='text',
                                text=[node], textposition='middle center',
                                textfont=dict(size=16, color='white', family='Arial'),
                                showlegend=False))
        y_offset = -0.12 if node != 'C' else 0.12
        fig.add_annotation(x=x, y=y+y_offset, text=node_labels[node],
                          showarrow=False, font=dict(size=11, color='#333333', family='Arial'))
    
    fig.add_annotation(x=0.5, y=0.95, text='V 结构：给定 C 时，A 与 B 变得相关',
                      showarrow=False, font=dict(size=14, color='#333333'))
    fig.add_annotation(x=0.5, y=0.88, text='解释消除效应',
                      showarrow=False, font=dict(size=13, color=APPLE_RED, family='Arial'))
    
    fig.add_annotation(x=0.5, y=0.52, text='知道学生考高分(C)后，',
                      showarrow=False, font=dict(size=11, color='#666666'))
    fig.add_annotation(x=0.5, y=0.46, text='"认真学习"(A)和"聪明"(B) 互相影响',
                      showarrow=False, font=dict(size=11, color='#666666'))
    
    fig.update_layout(
        title=dict(text='V 结构与解释消除效应', font=dict(size=16, family='Arial')),
        xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        width=700,
        height=450
    )
    
    return fig


def plot_d_separation():
    """
    D-分离的三种基本结构对比
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('链式', '分叉', 'V 结构'),
        horizontal_spacing=0.1
    )
    
    # ===== 链式 =====
    draw_arrow(fig, 0.2, 0.5, 0.5, 0.5, APPLE_BLUE, 0.05)
    draw_arrow(fig, 0.5, 0.5, 0.8, 0.5, APPLE_BLUE, 0.05)
    
    for node, x, y in [('A', 0.2, 0.5), ('B', 0.5, 0.5), ('C', 0.8, 0.5)]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                marker=dict(size=45, color=APPLE_BLUE), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='text',
                                text=[node], textposition='middle center',
                                textfont=dict(size=14, color='white'), showlegend=False), row=1, col=1)
    
    fig.add_annotation(x=0.5, y=0.15, text='A ⊥ C | B ✓', showarrow=False,
                      font=dict(size=12, color=APPLE_GREEN), row=1, col=1)
    
    # ===== 分叉 =====
    draw_arrow(fig, 0.5, 0.7, 0.2, 0.3, APPLE_PURPLE, 0.05)
    draw_arrow(fig, 0.5, 0.7, 0.8, 0.3, APPLE_PURPLE, 0.05)
    
    for node, x, y, color in [('A', 0.2, 0.3, APPLE_ORANGE), ('B', 0.5, 0.7, APPLE_PURPLE), ('C', 0.8, 0.3, APPLE_ORANGE)]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                marker=dict(size=45, color=color), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='text',
                                text=[node], textposition='middle center',
                                textfont=dict(size=14, color='white'), showlegend=False), row=1, col=2)
    
    fig.add_annotation(x=0.5, y=0.05, text='A ⊥ C | B ✓', showarrow=False,
                      font=dict(size=12, color=APPLE_GREEN), row=1, col=2)
    
    # ===== V 结构 =====
    draw_arrow(fig, 0.2, 0.7, 0.5, 0.3, APPLE_RED, 0.05)
    draw_arrow(fig, 0.8, 0.7, 0.5, 0.3, APPLE_RED, 0.05)
    
    for node, x, y, color in [('A', 0.2, 0.7, APPLE_GREEN), ('B', 0.8, 0.7, APPLE_GREEN), ('C', 0.5, 0.3, APPLE_RED)]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                marker=dict(size=45, color=color), showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='text',
                                text=[node], textposition='middle center',
                                textfont=dict(size=14, color='white'), showlegend=False), row=1, col=3)
    
    fig.add_annotation(x=0.5, y=0.05, text='A ⊥̸ C | C ✗', showarrow=False,
                      font=dict(size=12, color=APPLE_RED), row=1, col=3)
    
    fig.update_xaxes(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False, scaleanchor='x')
    
    fig.update_layout(
        title=dict(text='D-分离的三种基本结构', font=dict(size=16, family='Arial')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        width=950,
        height=400
    )
    
    return fig


def plot_inference_types():
    """
    贝叶斯网络中的三种推理类型
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('因果推理', '诊断推理', '解释消除'),
        horizontal_spacing=0.12
    )
    
    # ===== 因果推理 =====
    draw_arrow(fig, 0.2, 0.5, 0.5, 0.5, APPLE_BLUE, 0.05)
    draw_arrow(fig, 0.5, 0.5, 0.8, 0.5, APPLE_BLUE, 0.05)
    
    for node, x, y, color in [('A', 0.2, 0.5, APPLE_GREEN), ('B', 0.5, 0.5, APPLE_BLUE), ('C', 0.8, 0.5, APPLE_ORANGE)]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                marker=dict(size=45, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='text',
                                text=[node], textposition='middle center',
                                textfont=dict(size=14, color='white'), showlegend=False), row=1, col=1)
    
    fig.add_annotation(x=0.5, y=0.15, text='已知: A', showarrow=False,
                      font=dict(size=11, color=APPLE_GREEN), row=1, col=1)
    fig.add_annotation(x=0.5, y=0.08, text='推断: P(C|A)', showarrow=False,
                      font=dict(size=11, color=APPLE_ORANGE), row=1, col=1)
    
    # ===== 诊断推理 =====
    draw_arrow(fig, 0.2, 0.5, 0.5, 0.5, APPLE_BLUE, 0.05)
    draw_arrow(fig, 0.5, 0.5, 0.8, 0.5, APPLE_BLUE, 0.05)
    
    for node, x, y, color in [('A', 0.2, 0.5, APPLE_GREEN), ('B', 0.5, 0.5, APPLE_BLUE), ('C', 0.8, 0.5, APPLE_ORANGE)]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                marker=dict(size=45, color=color), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='text',
                                text=[node], textposition='middle center',
                                textfont=dict(size=14, color='white'), showlegend=False), row=1, col=2)
    
    fig.add_annotation(x=0.5, y=0.15, text='已知: C', showarrow=False,
                      font=dict(size=11, color=APPLE_ORANGE), row=1, col=2)
    fig.add_annotation(x=0.5, y=0.08, text='推断: P(A|C)', showarrow=False,
                      font=dict(size=11, color=APPLE_GREEN), row=1, col=2)
    
    # ===== 解释消除 =====
    draw_arrow(fig, 0.25, 0.7, 0.5, 0.3, APPLE_RED, 0.05)
    draw_arrow(fig, 0.75, 0.7, 0.5, 0.3, APPLE_RED, 0.05)
    
    for node, x, y, color in [('A', 0.25, 0.7, APPLE_BLUE), ('B', 0.75, 0.7, APPLE_BLUE), ('C', 0.5, 0.3, APPLE_RED)]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                marker=dict(size=45, color=color), showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='text',
                                text=[node], textposition='middle center',
                                textfont=dict(size=14, color='white'), showlegend=False), row=1, col=3)
    
    fig.add_annotation(x=0.5, y=0.95, text='已知: B, C', showarrow=False,
                      font=dict(size=11, color='#333333'), row=1, col=3)
    fig.add_annotation(x=0.5, y=0.88, text='推断: P(A|B,C) < P(A|C)', showarrow=False,
                      font=dict(size=10, color=APPLE_RED), row=1, col=3)
    
    fig.update_xaxes(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False, scaleanchor='x')
    
    fig.update_layout(
        title=dict(text='贝叶斯网络中的三种推理类型', font=dict(size=16, family='Arial')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        width=1000,
        height=400
    )
    
    return fig


def plot_bayes_timeline():
    """
    贝叶斯方法发展历史时间线
    """
    fig = go.Figure()
    
    events = [
        (1763, '贝叶斯定理', '贝叶斯去世后\n发表其论文'),
        (1812, '拉普拉斯', '系统阐述\n逆概率'),
        (1950, '图灵测试', '贝叶斯方法\n应用于AI'),
        (1985, '置信度网络', '提出贝叶斯网络'),
        (1988, 'Judea Pearl', '系统建立\n理论框架'),
        (2011, 'Watson', 'IBM Watson\n使用贝叶斯方法'),
        (2020, '现代应用', '广泛应用于\n各个领域'),
    ]
    
    years = [e[0] for e in events]
    y_positions = [3.5, -2.5, 3.2, -2.8, 2.8, -3.2, 3.0]
    
    # 主时间线
    fig.add_trace(go.Scatter(
        x=[min(years)-10, max(years)+10],
        y=[0, 0],
        mode='lines',
        line=dict(color=APPLE_GRAY, width=3),
        showlegend=False
    ))
    
    colors = [APPLE_BLUE, APPLE_PURPLE, APPLE_GREEN, APPLE_ORANGE, APPLE_RED, '#5856D6', '#FF2D55']
    
    for i, (year, title, desc) in enumerate(events):
        y = y_positions[i]
        color = colors[i % len(colors)]
        
        # 垂直连接线
        fig.add_trace(go.Scatter(
            x=[year, year],
            y=[0, y * 0.6],
            mode='lines',
            line=dict(color=color, width=2, dash='dot'),
            showlegend=False
        ))
        
        # 事件点
        fig.add_trace(go.Scatter(
            x=[year],
            y=[0],
            mode='markers',
            marker=dict(size=12, color=color, line=dict(color='white', width=2)),
            showlegend=False
        ))
        
        # 事件卡片
        fig.add_annotation(
            x=year, y=y,
            text=f'<b>{year}</b><br>{title}',
            showarrow=False,
            font=dict(size=11, color=color),
            bgcolor='white',
            bordercolor=color,
            borderwidth=2,
            borderpad=4,
            align='center'
        )
        
        # 描述文字
        fig.add_annotation(
            x=year, y=y - (0.8 if y > 0 else -0.8),
            text=desc,
            showarrow=False,
            font=dict(size=9, color='#666666'),
            align='center'
        )
    
    # 时期标注
    fig.add_annotation(x=1780, y=6, text='经典时期', showarrow=False,
                      font=dict(size=13, color=APPLE_BLUE, family='Arial'))
    fig.add_annotation(x=1980, y=-5, text='现代贝叶斯时期', showarrow=False,
                      font=dict(size=13, color=APPLE_ORANGE, family='Arial'))
    fig.add_annotation(x=2015, y=5.5, text='AI 应用时期', showarrow=False,
                      font=dict(size=13, color=APPLE_GREEN, family='Arial'))
    
    fig.update_layout(
        title=dict(text='贝叶斯方法与贝叶斯网络发展简史', font=dict(size=16, family='Arial')),
        xaxis=dict(range=[1750, 2030], showgrid=False, zeroline=False, tickfont=dict(size=10), tickvals=years),
        yaxis=dict(range=[-7, 7.5], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=60, t=80, b=60),
        width=1100,
        height=550,
        showlegend=False
    )
    
    return fig


def plot_naive_bayes():
    """
    朴素贝叶斯分类器示意图
    """
    fig = go.Figure()
    
    # 类别节点
    fig.add_trace(go.Scatter(x=[0.5], y=[0.85], mode='markers',
                            marker=dict(size=70, color=APPLE_RED), showlegend=False))
    fig.add_trace(go.Scatter(x=[0.5], y=[0.85], mode='text',
                            text=['Class'], textposition='middle center',
                            textfont=dict(size=14, color='white', family='Arial')))
    
    # 特征节点
    features = ['F₁', 'F₂', 'F₃', 'F₄', 'F₅']
    y_pos = 0.25
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for i, (feat, x) in enumerate(zip(features, x_positions)):
        color = [APPLE_BLUE, APPLE_GREEN, APPLE_PURPLE, APPLE_ORANGE, '#5856D6'][i]
        
        # 边
        draw_arrow(fig, 0.5, 0.85, x, y_pos, color, 0.07)
        
        # 节点
        fig.add_trace(go.Scatter(x=[x], y=[y_pos], mode='markers',
                                marker=dict(size=50, color=color), showlegend=False))
        fig.add_trace(go.Scatter(x=[x], y=[y_pos], mode='text',
                                text=[feat], textposition='middle center',
                                textfont=dict(size=14, color='white', family='Arial')))
    
    # 条件独立性标注
    fig.add_annotation(x=0.5, y=0.55, text='条件独立', showarrow=False,
                      font=dict(size=11, color='#666666'))
    fig.add_annotation(x=0.5, y=0.50, text='P(Fᵢ, Fⱼ|C) = P(Fᵢ|C) · P(Fⱼ|C)', showarrow=False,
                      font=dict(size=10, color=APPLE_GRAY, family='Arial'))
    
    fig.update_layout(
        title=dict(text='朴素贝叶斯分类器结构', font=dict(size=16, family='Arial')),
        xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        width=700,
        height=500
    )
    
    return fig


def main():
    """生成所有图形"""
    os.makedirs('static/images/plots', exist_ok=True)
    
    print("开始生成贝叶斯网络相关图形...")
    
    fig1 = plot_simple_bayesian_network()
    save_and_compress(fig1, 'static/images/plots/bayesian-network-example.png')
    
    fig2 = plot_conditional_probability()
    save_and_compress(fig2, 'static/images/plots/conditional-probability.png', width=1000, height=500)
    
    fig3 = plot_chain_structure()
    save_and_compress(fig3, 'static/images/plots/chain-structure.png', width=800, height=450)
    
    fig4 = plot_v_structure()
    save_and_compress(fig4, 'static/images/plots/v-structure.png', width=800, height=500)
    
    fig5 = plot_d_separation()
    save_and_compress(fig5, 'static/images/plots/d-separation.png', width=1000, height=450)
    
    fig6 = plot_inference_types()
    save_and_compress(fig6, 'static/images/plots/inference-types.png', width=1050, height=450)
    
    fig7 = plot_bayes_timeline()
    save_and_compress(fig7, 'static/images/plots/bayesian-timeline.png', width=1200, height=600)
    
    fig8 = plot_naive_bayes()
    save_and_compress(fig8, 'static/images/plots/naive-bayes.png', width=800, height=550)
    
    print("\n✅ 所有图形生成完成！")


if __name__ == '__main__':
    main()
