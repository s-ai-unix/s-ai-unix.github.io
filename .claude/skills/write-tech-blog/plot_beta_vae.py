#!/usr/bin/env python3
"""
β-VAE 损失权衡可视化
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# 设置全局样式
pio.templates.default = "plotly_white"

def create_beta_vae_tradeoff():
    """创建 β-VAE 损失权衡图"""

    # 创建数据
    beta_values = np.linspace(0.1, 5, 100)

    # 假设的权衡关系（简化模型）
    # 随着 β 增加，KL 散度增加，重建误差增加
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

    # 添加权衡包络线
    tradeoff_curve = 0.5 * (kl_divergence + reconstruction_error)
    fig.add_trace(go.Scatter(
        x=beta_values,
        y=tradeoff_curve,
        mode='lines',
        name='总损失',
        line=dict(color='purple', width=2, dash='dash'),
        hovertemplate='β: %{x:.2f}<br>总损失: %{y:.3f}<extra></extra>'
    ))

    # 标记标准 VAE (β=1)
    fig.add_trace(go.Scatter(
        x=[1],
        y=[0.5 + 0.1 * (1 - 1)**2],
        mode='markers',
        marker=dict(
            size=12,
            color='green',
            symbol='diamond',
            line=dict(width=2)
        ),
        name='标准 VAE (β=1)',
        hovertemplate='标准 VAE<br>β=1<br><extra></extra>'
    ))

    # 添加区域标注
    fig.add_annotation(
        x=0.5,
        y=0.6,
        text="低 β：更好的重建",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="blue",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="blue",
        borderwidth=1,
        font=dict(size=12, color="blue")
    )

    fig.add_annotation(
        x=3,
        y=1.2,
        text="高 β：更好的解耦",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=1,
        font=dict(size=12, color="red")
    )

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

def create_beta_vae_3d_surface():
    """创建 3D β-VAE 损失表面图"""

    # 创建参数网格
    beta_values = np.linspace(0.1, 3, 50)
    dimensionality = np.linspace(2, 10, 50)
    BETA, DIM = np.meshgrid(beta_values, dimensionality)

    # 计算损失表面
    # 简化的损失模型
    reconstruction_loss = 1.0 + 0.1 * (BETA - 1)**2
    kl_loss = 0.5 * BETA * np.log(DIM)
    total_loss = reconstruction_loss + kl_loss

    # 创建图形
    fig = go.Figure(data=[
        go.Surface(
            x=BETA,
            y=DIM,
            z=total_loss,
            colorscale='Viridis',
            opacity=0.8,
            contours=dict(
                x=dict(show=True, usecolormap=True, project_z=True),
                y=dict(show=True, usecolormap=True, project_z=True),
                z=dict(show=True, usecolormap=True, project_z=True)
            )
        )
    ])

    # 添加等高线
    fig.add_trace(go.Contour(
        x=beta_values,
        y=dimensionality,
        z=total_loss,
        contours=dict(start=0.5, end=3, size=0.2),
        showscale=False,
        line=dict(width=1, color='white'),
        opacity=0.5
    ))

    # 标记最佳点示例
    optimal_beta = 1.0
    optimal_dim = 5.0
    optimal_loss = 1.0 + 0.1 * (optimal_beta - 1)**2 + 0.5 * optimal_beta * np.log(optimal_dim)

    fig.add_trace(go.Scatter3d(
        x=[optimal_beta],
        y=[optimal_dim],
        z=[optimal_loss],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='diamond'
        ),
        name='最佳点示例'
    ))

    # 更新布局
    fig.update_layout(
        title='β-VAE 3D 损失表面',
        scene=dict(
            xaxis_title='β',
            yaxis_title='维度数',
            zaxis_title='总损失',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600,
        font=dict(family="Arial, sans-serif", size=14)
    )

    return fig

def create_beta_effect_comparison():
    """创建 β 参数影响的对比图"""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('β = 0.5 (重建优先)', 'β = 1.0 (标准 VAE)', 'β = 2.0 (平衡)', 'β = 5.0 (解耦优先)'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # 生成数据
    z = np.linspace(-3, 3, 100)

    beta_values = [0.5, 1.0, 2.0, 5.0]
    colors = ['blue', 'green', 'orange', 'red']

    for i, (beta, color) in enumerate(zip(beta_values, colors)):
        row = i // 2 + 1
        col = i % 2 + 1

        # 计算分布（简化示例）
        if beta == 0.5:
            # 重建优先，分布较宽
            sigma = 1.5
            mu = 0
        elif beta == 1.0:
            # 标准 VAE
            sigma = 1.0
            mu = 0
        elif beta == 2.0:
            # 平衡
            sigma = 0.8
            mu = 0.2
        else:  # beta = 5.0
            # 解耦优先，分布较窄
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

        # 添加 KL 散度标注
        kl_value = 0.5 * (sigma**2 + mu**2 - 1 - np.log(sigma**2))
        fig.add_annotation(
            x=2.5,
            y=max(pdf) * 0.8,
            text=f'KL = {kl_value:.2f}',
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="rgba(255,255,255,0.8)"
        ), row=row, col=col

    # 更新布局
    fig.update_layout(
        title='不同 β 值对潜在分布的影响',
        height=600,
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=True
    )

    # 隐藏坐标轴标签
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="z", row=i, col=j)
            fig.update_yaxes(title_text="p(z)", row=i, col=j)

    return fig

if __name__ == "__main__":
    # 导入必要的库
    from plotly.subplots import make_subplots

    # 创建三个图形

    # 创建三个图形
    fig1 = create_beta_vae_tradeoff()
    fig2 = create_beta_vae_3d_surface()
    fig3 = create_beta_effect_comparison()

    # 保存图形
    fig1.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/beta_vae_tradeoff.html")
    fig2.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/beta_vae_3d.html")
    fig3.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/beta_vae_comparison.html")

    print("β-VAE 可视化已生成并保存到 static/images/plots/ 目录")
    print("1. beta_vae_tradeoff.html - 损失权衡曲线")
    print("2. beta_vae_3d.html - 3D 损失表面")
    print("3. beta_vae_comparison.html - 不同 β 值对比")