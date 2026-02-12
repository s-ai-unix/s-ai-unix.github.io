#!/usr/bin/env python3
"""
潜在空间插值可视化
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
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
        name='点',
        text=['z₁', 'z₂'],
        textposition='top center'
    ))

    # 添加插值点标签
    for i, point in enumerate(interpolation_points[::3]):  # 每 3 个点显示一个标签
        fig.add_annotation(
            x=point[0],
            y=point[1],
            text=f'z<sub>{i*3}</sub>',
            showarrow=True,
            arrowhead=1,
            arrowsize=0.5,
            arrowwidth=1,
            ax=0,
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

def create_latent_grid():
    """创建潜在空间网格采样"""

    # 创建网格点
    n_points = 7
    z_values = np.linspace(-2, 2, n_points)

    # 创建图形
    fig = make_subplots(
        rows=n_points, cols=n_points,
        subplot_titles=[[f'z₁={z:.1f}, z₂={y:.1f}' for y in z_values[::-1]] for z in z_values],
        specs=[[{"type": "scatter"}]*n_points for _ in range(n_points)],
        vertical_spacing=0.01,
        horizontal_spacing=0.01
    )

    # 为每个网格点添加分布
    for i, z1 in enumerate(z_values):
        for j, z2 in enumerate(z_values[::-1]):
            # 计算以 (z1, z2) 为中心的分布
            x = np.linspace(-1, 1, 50)
            y = np.linspace(-1, 1, 50)
            X, Y = np.meshgrid(x, y)

            # 移动到中心点
            X_centered = X + z1
            Y_centered = Y + z2

            # 创建高斯分布
            z_grid = np.dstack([X_centered, Y_centered])
            pdf = multivariate_normal.pdf(z_grid, mean=[z1, z2], cov=[[0.3, 0], [0, 0.3]])

            # 添加等高线
            fig.add_trace(go.Contour(
                x=X_centered[0, :],
                y=Y_centered[:, 0],
                z=pdf,
                contours=dict(start=0.1, end=1.0, size=0.1),
                showscale=False,
                line=dict(width=0.5, color='blue')
            ), row=i+1, col=j+1)

    # 更新布局
    fig.update_layout(
        title='潜在空间网格采样',
        height=600,
        width=600,
        font=dict(family="Arial, sans-serif", size=10),
        plot_bgcolor='white'
    )

    # 隐藏坐标轴
    for i in range(1, n_points+1):
        for j in range(1, n_points+1):
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)

    return fig

def create_manifold_walk():
    """创建流形漫步可视化"""

    # 创建螺旋路径
    t = np.linspace(0, 4*np.pi, 100)
    r = t / (4*np.pi)  # 半径随角度增加

    z1 = r * np.cos(t)
    z2 = r * np.sin(t)

    # 创建图形
    fig = go.Figure()

    # 添加先验分布背景
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    z_grid = np.dstack([X, Y])
    prior_pdf = multivariate_normal.pdf(z_grid, mean=[0, 0], cov=[[1, 0], [0, 1]])

    fig.add_trace(go.Contour(
        x=x, y=y, z=prior_pdf,
        contours=dict(start=0.01, end=0.4, size=0.05),
        colorscale='Greens',
        showscale=False,
        opacity=0.3,
        name='先验分布'
    ))

    # 添加螺旋路径
    fig.add_trace(go.Scatter(
        x=z1,
        y=z2,
        mode='lines+markers',
        name='流形漫步路径',
        line=dict(color='purple', width=3),
        marker=dict(
            size=6,
            color='purple',
            symbol='circle'
        )
    ))

    # 添加路径点
    for i in range(0, len(t), 10):
        fig.add_annotation(
            x=z1[i],
            y=z2[i],
            text=f'Step {i}',
            showarrow=True,
            arrowhead=1,
            arrowsize=0.5,
            arrowwidth=1,
            ax=10,
            ay=0
        )

    # 更新布局
    fig.update_layout(
        title='流形漫步：探索潜在空间',
        xaxis_title='z₁',
        yaxis_title='z₂',
        width=800,
        height=600,
        font=dict(family="Arial, sans-serif", size=14),
        showlegend=True
    )

    return fig

if __name__ == "__main__":
    # 创建三个图形
    fig1 = create_latent_interpolation()
    fig2 = create_latent_grid()
    fig3 = create_manifold_walk()

    # 保存图形
    fig1.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/latent_interpolation.html")
    fig2.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/latent_grid.html")
    fig3.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/manifold_walk.html")

    print("潜在空间可视化已生成并保存到 static/images/plots/ 目录")
    print("1. latent_interpolation.html - 线性插值")
    print("2. latent_grid.html - 网格采样")
    print("3. manifold_walk.html - 流形漫步")