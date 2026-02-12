#!/usr/bin/env python3
"""
潜在空间插值可视化（简化版）
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import multivariate_normal

# 设置全局样式
pio.templates.default = "plotly_white"

def create_latent_interpolation():
    """创建潜在空间插值可视化"""

    # 创建 2D 潜在空间
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # 先验分布 p(z) = N(0, I)
    z_grid = np.dstack([X, Y])
    prior_pdf = multivariate_normal.pdf(z_grid, mean=[0, 0], cov=[[1, 0], [0, 1]])

    # 插值路径
    z1 = np.array([-1.5, -1.5])  # 起点
    z2 = np.array([1.5, 1.5])   # 终点

    # 创建插值点
    t_values = np.linspace(0, 1, 20)
    interpolation_points = []
    labels = []

    for i, t in enumerate(t_values):
        z_interp = z1 * (1 - t) + z2 * t
        interpolation_points.append(z_interp)
        labels.append(f'z<sub>{i}</sub>')

    interpolation_points = np.array(interpolation_points)

    # 创建图形
    fig = go.Figure()

    # 添加先验分布的等高线
    fig.add_trace(go.Contour(
        x=x, y=y, z=prior_pdf,
        contours=dict(start=0.01, end=0.4, size=0.05),
        colorscale='Greens',
        name='先验分布 p(z)',
        line=dict(width=1, color='green', dash='solid')
    ))

    # 添加插值路径
    fig.add_trace(go.Scatter(
        x=interpolation_points[:, 0],
        y=interpolation_points[:, 1],
        mode='lines+markers',
        name='插值路径',
        line=dict(color='blue', width=3),
        marker=dict(
            size=8,
            color='blue',
            symbol='circle'
        )
    ))

    # 添加起点和终点
    fig.add_trace(go.Scatter(
        x=[z1[0], z2[0]],
        y=[z1[1], z2[1]],
        mode='markers',
        marker=dict(
            size=12,
            color=['red', 'purple'],
            symbol=['diamond', 'diamond'],
            line=dict(width=2)
        ),
        name='端点',
        text=['z₁', 'z₂'],
        textposition='top center'
    ))

    # 添加插值点标签（每隔几个点显示一个）
    for i in range(0, len(t_values), 5):
        fig.add_annotation(
            x=interpolation_points[i, 0],
            y=interpolation_points[i, 1],
            text=f'Step {i}',
            showarrow=True,
            arrowhead=1,
            arrowsize=0.5,
            arrowwidth=1,
            ax=10,
            ay=-20
        )

    # 添加解释文本
    fig.add_annotation(
        x=-2.5, y=2.8,
        text="平滑插值是 VAE 生成质量的重要指标",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12)
    )

    # 更新布局
    fig.update_layout(
        title='潜在空间插值',
        xaxis_title='z₁',
        yaxis_title='z₂',
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

def create_latent_space_demonstration():
    """创建潜在空间演示图"""

    # 创建图形
    fig = go.Figure()

    # 先验分布
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    z_grid = np.dstack([X, Y])
    prior_pdf = multivariate_normal.pdf(z_grid, mean=[0, 0], cov=[[1, 0], [0, 1]])

    # 添加先验分布
    fig.add_trace(go.Contour(
        x=x, y=y, z=prior_pdf,
        contours=dict(start=0.01, end=0.4, size=0.05),
        colorscale='Greens',
        name='先验分布 p(z)',
        showscale=False
    ))

    # 添加一些样本点
    np.random.seed(42)
    sample_points = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1, 0.5], [0.5, 1]],
        size=50
    )

    fig.add_trace(go.Scatter(
        x=sample_points[:, 0],
        y=sample_points[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.7,
            symbol='circle'
        ),
        name='样本点'
    ))

    # 添加一个插值路径示例
    z_start = np.array([-1.5, 1.5])
    z_end = np.array([1.5, -1.5])

    t_values = np.linspace(0, 1, 10)
    interp_path = np.array([z_start * (1-t) + z_end * t for t in t_values])

    fig.add_trace(go.Scatter(
        x=interp_path[:, 0],
        y=interp_path[:, 1],
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=6, color='red'),
        name='插值路径'
    ))

    # 更新布局
    fig.update_layout(
        title='潜在空间可视化',
        xaxis_title='z₁',
        yaxis_title='z₂',
        width=800,
        height=600,
        font=dict(family="Arial, sans-serif", size=14),
        showlegend=True
    )

    return fig

if __name__ == "__main__":
    # 创建图形
    fig1 = create_latent_interpolation()
    fig2 = create_latent_space_demonstration()

    # 保存图形
    fig1.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/latent_interpolation.html")
    fig2.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/latent_space_demo.html")

    print("潜在空间可视化已生成并保存到 static/images/plots/ 目录")
    print("1. latent_interpolation.html - 插值路径可视化")
    print("2. latent_space_demo.html - 潜在空间演示")