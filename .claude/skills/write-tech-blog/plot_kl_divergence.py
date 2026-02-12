#!/usr/bin/env python3
"""
KL 散度可视化：对角协方差矩阵的简化
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import multivariate_normal
import plotly.io as pio

# 设置全局样式
pio.templates.default = "plotly_white"

def create_kl_2d_surface():
    """创建 2D KL 散度表面图"""

    # 定义网格
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # 先验分布：标准正态分布 N(0, I)
    prior_mean = np.array([0, 0])
    prior_cov = np.array([[1, 0], [0, 1]])

    # 近似后验分布：N(μ, diag(σ²))
    mu_x = 1.5  # 偏移
    mu_y = 0.5  # 偏移
    sigma_x = 0.8  # 标准差
    sigma_y = 1.2  # 标准差

    posterior_mean = np.array([mu_x, mu_y])
    posterior_cov = np.array([[sigma_x**2, 0], [0, sigma_y**2]])

    # 计算两个分布的 PDF
    prior_pdf = multivariate_normal.pdf(np.dstack([X, Y]), mean=prior_mean, cov=prior_cov)
    posterior_pdf = multivariate_normal.pdf(np.dstack([X, Y]), mean=posterior_mean, cov=posterior_cov)

    # 计算 KL 散度（使用解析解）
    kl_divergence = 0.5 * (sigma_x**2 + sigma_y**2 + mu_x**2 + mu_y**2 - 2 - np.log(sigma_x**2) - np.log(sigma_y**2))

    # 创建图形
    fig = go.Figure()

    # 添加先验分布的等高线
    fig.add_trace(go.Contour(
        x=x, y=y, z=prior_pdf,
        contours=dict(start=0.01, end=0.4, size=0.05),
        colorscale='Blues',
        name='先验分布 p(z)',
        line=dict(width=1, color='blue')
    ))

    # 添加近似后验分布的等高线
    fig.add_trace(go.Contour(
        x=x, y=y, z=posterior_pdf,
        contours=dict(start=0.01, end=0.4, size=0.05),
        colorscale='Reds',
        name='近似后验 q(z|x)',
        line=dict(width=1, color='red', dash='dash')
    ))

    # 添加均值点
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=10, color='blue', symbol='x'),
        name='先验均值 (0,0)'
    ))

    fig.add_trace(go.Scatter(
        x=[mu_x], y=[mu_y],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='近似后验均值'
    ))

    # 添加 KL 散度文本
    fig.add_annotation(
        x=2, y=2.5,
        text=f"KL 散度: {kl_divergence:.3f}",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    # 更新布局
    fig.update_layout(
        title='2D 高斯分布的 KL 散度可视化',
        xaxis_title='z₁',
        yaxis_title='z₂',
        width=800,
        height=600,
        font=dict(family="Arial, sans-serif", size=14),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    return fig

def create_kl_parameter_analysis():
    """创建 KL 散度参数分析图"""

    # 创建参数空间
    mu_values = np.linspace(-2, 2, 50)
    sigma_values = np.linspace(0.1, 2, 50)
    MU, SIGMA = np.meshgrid(mu_values, sigma_values)

    # 计算 KL 散度
    # D_KL = 0.5 * (σ² + μ² - 1 - log(σ²))
    kl_values = 0.5 * (SIGMA**2 + MU**2 - 1 - np.log(SIGMA**2))

    # 确保 KL 值为正
    kl_values = np.maximum(kl_values, 0)

    # 创建图形
    fig = go.Figure(data=[
        go.Surface(
            x=MU,
            y=SIGMA,
            z=kl_values,
            colorscale='Viridis',
            contours=dict(
                x=dict(show=True, usecolormap=True, project_z=True),
                y=dict(show=True, usecolormap=True, project_z=True),
                z=dict(show=True, usecolormap=True, project_z=True)
            ),
            opacity=0.8
        )
    ])

    # 添加等高线
    fig.add_trace(go.Contour(
        x=mu_values,
        y=sigma_values,
        z=kl_values,
        contours=dict(start=0, end=2, size=0.2),
        showscale=False,
        line=dict(width=1, color='black')
    ))

    # 标记标准点 (μ=0, σ=1)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[1], z=[0],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='标准点 (0,1)'
    ))

    # 更新布局
    fig.update_layout(
        title='KL 散度：均值和标准差的影响',
        scene=dict(
            xaxis_title='μ',
            yaxis_title='σ',
            zaxis_title='KL 散度',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600,
        font=dict(family="Arial, sans-serif", size=14)
    )

    return fig

def create_kl_formula_visualization():
    """创建 KL 公式分解可视化"""

    # 创建图形
    fig = go.Figure()

    # 公式部分
    formulas = [
        "D<sub>KL</sub>(q||p) = ½[σ² + μ² - 1 - ln(σ²)]",
        "其中：",
        "• q ~ N(μ, σ²I)",
        "• p ~ N(0, I)",
        "• σ²: 对角协方差矩阵的对角元素",
        "• μ: 潜在空间的均值"
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
            xref="paper",
            yref="paper"
        )

    # 添加可视化示例
    # 简单的 1D 例子
    x = np.linspace(-4, 4, 200)

    # 先验分布
    prior = (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)

    # 近似后验分布
    mu = 1.0
    sigma = 0.8
    posterior = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x-mu)**2/(2*sigma**2))

    # 计算 KL 值
    kl_1d = 0.5 * (sigma**2 + mu**2 - 1 - np.log(sigma**2))

    # 添加 1D 分布
    fig.add_trace(go.Scatter(
        x=x, y=prior,
        mode='lines',
        name='先验分布 p(z)',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=x, y=posterior,
        mode='lines',
        name='近似后验 q(z|x)',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.add_annotation(
        x=3, y=0.3,
        text=f"1D KL 值: {kl_1d:.3f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black"
    )

    # 更新布局
    fig.update_layout(
        title="KL 散度公式可视化",
        xaxis_title="z",
        yaxis_title="概率密度",
        height=600,
        showlegend=True,
        font=dict(family="Arial, sans-serif", size=14),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )

    return fig

if __name__ == "__main__":
    # 创建三个图形
    fig1 = create_kl_2d_surface()
    fig2 = create_kl_parameter_analysis()
    fig3 = create_kl_formula_visualization()

    # 保存图形
    fig1.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/kl_divergence_2d.html")
    fig2.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/kl_divergence_3d.html")
    fig3.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/kl_divergence_formula.html")

    print("KL 散度图形已生成并保存到 static/images/plots/ 目录")
    print("1. kl_divergence_2d.html - 2D 高斯分布的 KL 散度")
    print("2. kl_divergence_3d.html - 参数空间分析")
    print("3. kl_divergence_formula.html - 公式可视化")