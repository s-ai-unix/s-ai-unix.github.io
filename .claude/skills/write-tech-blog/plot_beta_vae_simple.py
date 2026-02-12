#!/usr/bin/env python3
"""
β-VAE 损失权衡可视化（简化版）
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# 设置全局样式
pio.templates.default = "plotly_white"

def create_beta_vae_tradeoff():
    """创建 β-VAE 损失权衡图"""

    # 创建数据
    beta_values = np.linspace(0.1, 5, 100)

    # 假设的权衡关系
    kl_divergence = 0.5 * np.log(beta_values) + 0.1 * beta_values
    reconstruction_error = 0.5 + 0.1 * (beta_values - 1)**2

    # 创建图形
    fig = go.Figure()

    # 添加 KL 散度曲线
    fig.add_trace(go.Scatter(
        x=beta_values,
        y=kl_divergence,
        mode='lines',
        name='KL 散度',
        line=dict(color='red', width=3),
        hovertemplate='β: %{x:.2f}<br>KL 散度: %{y:.3f}<extra></extra>'
    ))

    # 添加重建误差曲线
    fig.add_trace(go.Scatter(
        x=beta_values,
        y=reconstruction_error,
        mode='lines',
        name='重建误差',
        line=dict(color='blue', width=3),
        hovertemplate='β: %{x:.2f}<br>重建误差: %{y:.3f}<extra></extra>'
    ))

    # 标记标准 VAE (β=1)
    fig.add_trace(go.Scatter(
        x=[1],
        y=[0.5],
        mode='markers',
        marker=dict(
            size=12,
            color='green',
            symbol='diamond',
            line=dict(width=2)
        ),
        name='标准 VAE (β=1)'
    ))

    # 更新布局
    fig.update_layout(
        title='β-VAE 损失权衡',
        xaxis_title='β 参数',
        yaxis_title='损失值',
        width=800,
        height=600,
        font=dict(family="Arial, sans-serif", size=14),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0.02
        )
    )

    return fig

def create_beta_distributions():
    """创建不同 β 值的分布对比"""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('β = 0.5 (重建优先)', 'β = 1.0 (标准)', 'β = 2.0 (平衡)', 'β = 5.0 (解耦优先)'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    # 生成数据
    z = np.linspace(-3, 3, 100)

    beta_values = [0.5, 1.0, 2.0, 5.0]
    colors = ['blue', 'green', 'orange', 'red']

    for i, (beta, color) in enumerate(zip(beta_values, colors)):
        row = i // 2 + 1
        col = i % 2 + 1

        # 计算分布
        if beta == 0.5:
            sigma = 1.5
            mu = 0
        elif beta == 1.0:
            sigma = 1.0
            mu = 0
        elif beta == 2.0:
            sigma = 0.8
            mu = 0.2
        else:  # beta = 5.0
            sigma = 0.5
            mu = 1.0

        # 计算高斯分布
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(z - mu)**2 / (2 * sigma**2))

        # 添加到子图
        fig.add_trace(go.Scatter(
            x=z,
            y=pdf,
            mode='lines',
            name=f'β = {beta}',
            line=dict(color=color, width=3),
            showlegend=False
        ), row=row, col=col)

    # 更新布局
    fig.update_layout(
        title='不同 β 值对潜在分布的影响',
        height=600,
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=True
    )

    return fig

if __name__ == "__main__":
    # 创建图形
    fig1 = create_beta_vae_tradeoff()
    fig2 = create_beta_distributions()

    # 保存图形
    fig1.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/beta_vae_tradeoff.html")
    fig2.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/beta_vae_distributions.html")

    print("β-VAE 可视化已生成并保存到 static/images/plots/ 目录")
    print("1. beta_vae_tradeoff.html - 损失权衡曲线")
    print("2. beta_vae_distributions.html - 不同 β 值分布对比")