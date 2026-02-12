#!/usr/bin/env python3
"""
VAE 网络架��可视化
使用 Plotly 创建交互式网络图
"""

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np

# 设置全局样式
pio.templates.default = "plotly_white"

def create_vae_network_diagram():
    """创建 VAE 网络架构的交互式图表"""

    # 创建主图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('编码器', '重参数化采样', '解码器', '损失计算'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # 定义节点位置
    nodes = {
        # 编码器部分
        'x': (0, 0),  # 输入
        'h1_enc': (1, 1),  # 隐藏层1
        'h2_enc': (2, 1),  # 隐藏层2
        'mu': (3, 0.5),   # 均值
        'log_sigma2': (3, 1.5),  # 对数方差

        # 重参数化采样
        'epsilon': (4, 1),  # 噪声
        'z': (5, 1),       # 潜在变量

        # 解码器部分
        'h1_dec': (6, 1),  # 隐藏层1
        'h2_dec': (7, 1),  # 隐藏层2
        'x_hat': (8, 1),   # 重建输出

        # 损失计算
        'reconstruction_loss': (6, 2.5),  # 重建误差
        'kl_loss': (3, 2.5),              # KL 散度
        'total_loss': (5, 3.5)           # 总损失
    }

    # 定义边
    edges = [
        # 编码器
        ('x', 'h1_enc'),
        ('h1_enc', 'h2_enc'),
        ('h2_enc', 'mu'),
        ('h2_enc', 'log_sigma2'),

        # 重参数化采样
        ('epsilon', 'z'),
        ('mu', 'z'),
        ('log_sigma2', 'z'),

        # 解码器
        ('z', 'h1_dec'),
        ('h1_dec', 'h2_dec'),
        ('h2_dec', 'x_hat'),

        # 损失计算
        ('x_hat', 'reconstruction_loss'),
        ('mu', 'kl_loss'),
        ('log_sigma2', 'kl_loss'),
        ('reconstruction_loss', 'total_loss'),
        ('kl_loss', 'total_loss')
    ]

    # 节点文本
    node_text = {
        'x': '输入 x',
        'h1_enc': '隐藏层1<br/>h₁ = ReLU W₁ x + b₁',
        'h2_enc': '隐藏层2<br/>h₂ = ReLU W₂ h₁ + b₂',
        'mu': '均值 μφ(x)',
        'log_sigma2': '对数方差<br/>log σ²φ(x)',
        'epsilon': '噪声 ε ∼ N(0,I)',
        'z': 'z = μφ(x) + σφ(x) ⊙ ε',
        'h1_dec': '隐藏层1<br/>h\'₁ = ReLU W₃ z + b₃',
        'h2_dec': '隐藏层2<br/>h\'₂ = ReLU W₄ h\'₁ + b₄',
        'x_hat': '重建输出 x̂ = μθ(z)',
        'reconstruction_loss': '-log pθ(x|z)',
        'kl_loss': 'D<sub>KL</sub>(qφ(z|x)||p(z))',
        'total_loss': '总损失 ℒ = K + L'
    }

    # 节点颜色
    node_colors = {
        'x': '#007AFF',
        'h1_enc': '#007AFF',
        'h2_enc': '#007AFF',
        'mu': '#007AFF',
        'log_sigma2': '#007AFF',
        'epsilon': '#FF9500',
        'z': '#FF9500',
        'h1_dec': '#34C759',
        'h2_dec': '#34C759',
        'x_hat': '#34C759',
        'reconstruction_loss': '#FF3B30',
        'kl_loss': '#FF3B30',
        'total_loss': '#FF3B30'
    }

    # 添加边
    for edge in edges:
        x_coords = [nodes[edge[0]][0], nodes[edge[1]][0]]
        y_coords = [nodes[edge[0]][1], nodes[edge[1]][1]]
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='none'
            ),
            row=1, col=1
        )

    # 添加节点
    for node, pos in nodes.items():
        color = node_colors[node]
        fig.add_trace(
            go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode='markers+text',
                text=[node_text[node]],
                textposition='middle center',
                marker=dict(
                    size=20 if node in ['x', 'z', 'x_hat', 'total_loss'] else 15,
                    color=color,
                    symbol='circle' if node not in ['epsilon', 'z'] else 'diamond',
                    line=dict(width=3 if node in ['x', 'z', 'x_hat', 'total_loss'] else 2)
                ),
                showlegend=False,
                hovertemplate=f"{node_text[node]}<extra></extra>"
            ),
            row=1, col=1
        )

    # 更新布局
    fig.update_layout(
        title='VAE 网络架构',
        height=800,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # 更新子图布局
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.5, 8.5],
                row=i, col=j
            )
            fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.5, 4],
                row=i, col=j
            )

    return fig

def create_vae_flow_diagram():
    """创建 VAE 流程图（改进版）"""

    fig = go.Figure()

    # 定义流程步骤
    steps = [
        {'name': '输入数据', 'x': 0, 'y': 3, 'color': '#007AFF'},
        {'name': '编码器', 'x': 1, 'y': 3, 'color': '#007AFF'},
        {'name': '获取参数', 'x': 2, 'y': 3, 'color': '#007AFF'},
        {'name': '生成噪声', 'x': 3, 'y': 3, 'color': '#FF9500'},
        {'name': '重参数化', 'x': 4, 'y': 3, 'color': '#FF9500'},
        {'name': '解码器', 'x': 5, 'y': 3, 'color': '#34C759'},
        {'name': '重建数据', 'x': 6, 'y': 3, 'color': '#34C759'},
        {'name': '计算损失', 'x': 7, 'y': 3, 'color': '#FF3B30'}
    ]

    # 添加连接线
    for i in range(len(steps)-1):
        fig.add_trace(go.Scatter(
            x=[steps[i]['x'], steps[i+1]['x']],
            y=[steps[i]['y'], steps[i+1]['y']],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))

    # 添加节点
    for step in steps:
        fig.add_trace(go.Scatter(
            x=[step['x']],
            y=[step['y']],
            mode='markers+text',
            text=[step['name']],
            textposition='bottom center',
            marker=dict(
                size=20,
                color=step['color'],
                line=dict(width=3)
            ),
            showlegend=False,
            hovertemplate=f"{step['name']}<extra></extra>"
        ))

    # 添加详细说明
    details = [
        {'x': 1, 'y': 2, 'text': 'x → μ, log σ²'},
        {'x': 3, 'y': 2, 'text': 'ε ∼ N(0,I)'},
        {'x': 4, 'y': 2, 'text': 'z = μ + σ⊙ε'},
        {'x': 5, 'y': 2, 'text': 'z → x̂'},
        {'x': 7, 'y': 2, 'text': 'ℒ = -log p(x|z) + D<sub>KL</sub>'}
    ]

    for detail in details:
        fig.add_annotation(
            x=detail['x'],
            y=detail['y'],
            text=detail['text'],
            showarrow=False,
            font=dict(size=10, color='gray'),
            xref='x',
            yref='y'
        )

    # 更新布局
    fig.update_layout(
        title='VAE 数据流',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def create_vae_equation_visualization():
    """创建 VAE 公式可视化"""

    fig = go.Figure()

    # 公式
    formulas = [
        "VAE 损失函数：",
        "ℒ(θ, φ; x) = -ℰ[log p_θ(x|z)] + D<sub>KL</sub>(q_φ(z|x)||p(z))",
        "",
        "其中：",
        "• q_φ(z|x) = N(μ_φ(x), diag(σ_φ²(x)))",
        "• p(z) = N(0, I)",
        "• p_θ(x|z) = N(μ_θ(z), σ_θ²(z)I)",
        "",
        "重参数化：",
        "z = μ_φ(x) + σ_φ(x) ⊙ ε, ε ∼ N(0,I)",
        "",
        "KL 散度（对角协方差）：",
        "D<sub>KL</sub> = ½∑[σ² + μ² - 1 - log σ²]"
    ]

    y_positions = list(range(len(formulas)-1, -1, -1))

    for i, (formula, y_pos) in enumerate(zip(formulas, y_positions)):
        fig.add_annotation(
            x=0.5,
            y=y_pos,
            text=formula,
            showarrow=False,
            font=dict(
                size=16 if i == 0 else 14,
                color="black" if i == 0 else "gray"
            ),
            xref='paper',
            yref='paper'
        )

    # 添加视觉分隔
    fig.add_shape(
        type="line",
        x0=0, y0=2.5, x1=1, y1=2.5,
        line=dict(color="lightgray", width=1),
        xref='paper',
        yref='paper'
    )

    fig.add_shape(
        type="line",
        x0=0, y0=7.5, x1=1, y1=7.5,
        line=dict(color="lightgray", width=1),
        xref='paper',
        yref='paper'
    )

    fig.add_shape(
        type="line",
        x0=0, y0=12.5, x1=1, y1=12.5,
        line=dict(color="lightgray", width=1),
        xref='paper',
        yref='paper'
    )

    fig.update_layout(
        title="VAE 核心公式",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

if __name__ == "__main__":
    # 创建三个图形
    fig1 = create_vae_network_diagram()
    fig2 = create_vae_flow_diagram()
    fig3 = create_vae_equation_visualization()

    # 保存图形
    fig1.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/vae_network.html")
    fig2.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/vae_flow.html")
    fig3.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/vae_equations.html")

    print("VAE 网络图形已生成并保存到 static/images/plots/ 目录")
    print("1. vae_network.html - 交互式网络架构图")
    print("2. vae_flow.html - 数据流程图")
    print("3. vae_equations.html - 核心公式可视化")