#!/usr/bin/env python3
"""
VAE 训练过程曲线可视化
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.ndimage import gaussian_filter1d

# 设置全局样式
pio.templates.default = "plotly_white"

def create_training_curves():
    """创建 VAE 训练曲线"""

    # 创建训练轮次数据
    epochs = np.arange(1, 101)

    # 模拟训练数据（添加噪声和趋势）
    np.random.seed(42)

    # 重建误差：逐渐下降，然后趋于稳定
    base_reconstruction = 2.5 * np.exp(-epochs / 30) + 0.3
    reconstruction_noise = np.random.normal(0, 0.05, len(epochs))
    reconstruction_loss = base_reconstruction + reconstruction_noise
    reconstruction_loss = gaussian_filter1d(reconstruction_loss, sigma=2)

    # KL 散度：逐渐上升，然后趋于稳定
    base_kl = 2 * (1 - np.exp(-epochs / 50))
    kl_noise = np.random.normal(0, 0.02, len(epochs))
    kl_loss = base_kl + kl_noise
    kl_loss = gaussian_filter1d(kl_loss, sigma=2)

    # 总损失 = 重建误差 + KL 散度
    total_loss = reconstruction_loss + kl_loss

    # 创建图形
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('总损失', '重建误差', 'KL 散度'),
        vertical_spacing=0.05,
        shared_xaxes=True
    )

    # 总损失
    fig.add_trace(go.Scatter(
        x=epochs,
        y=total_loss,
        mode='lines',
        name='总损失',
        line=dict(color='purple', width=2),
        showlegend=False
    ), row=1, col=1)

    # 重建误差
    fig.add_trace(go.Scatter(
        x=epochs,
        y=reconstruction_loss,
        mode='lines',
        name='重建误差',
        line=dict(color='blue', width=2),
        showlegend=False
    ), row=2, col=1)

    # KL 散度
    fig.add_trace(go.Scatter(
        x=epochs,
        y=kl_loss,
        mode='lines',
        name='KL 散度',
        line=dict(color='red', width=2),
        showlegend=False
    ), row=3, col=1)

    # 添加平衡点标记
    balance_epoch = 50
    balance_total = total_loss[balance_epoch - 1]
    balance_reconstruction = reconstruction_loss[balance_epoch - 1]
    balance_kl = kl_loss[balance_epoch - 1]

    fig.add_vline(x=balance_epoch, line_dash="dash", line_color="gray", row="all", col=1)
    fig.add_annotation(
        x=balance_epoch,
        y=balance_total,
        text="平衡点",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        row=1, col=1
    )

    # 更新布局
    fig.update_layout(
        title='VAE 训练过程曲线',
        height=600,
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    # 更新 x 轴
    fig.update_xaxes(title_text="训练轮次", row=3, col=1)

    # 更新 y 轴
    fig.update_yaxes(title_text="损失值", row=1, col=1)
    fig.update_yaxes(title_text="重建误差", row=2, col=1)
    fig.update_yaxes(title_text="KL 散度", row=3, col=1)

    return fig

def create_convergence_analysis():
    """创建收敛分析图"""

    # 创建参数空间
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    latent_dims = [2, 5, 10, 20]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('学习率 = 0.001', '学习率 = 0.005', '学习率 = 0.01', '学习率 = 0.05'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    # 为每个学习率创建收敛曲线
    colors = ['blue', 'green', 'orange', 'red']

    for i, (lr, color) in enumerate(zip(learning_rates, colors)):
        row = i // 2 + 1
        col = i % 2 + 1

        # 模拟不同潜在维度的收敛情况
        epochs = np.arange(1, 101)

        for dim in latent_dims:
            # 学习率越高，收敛越快但可能不稳定
            if lr == 0.001:
                base_loss = 1.5 / np.sqrt(dim) * np.exp(-epochs / 40)
            elif lr == 0.005:
                base_loss = 1.2 / np.sqrt(dim) * np.exp(-epochs / 30)
            elif lr == 0.01:
                base_loss = 1.0 / np.sqrt(dim) * np.exp(-epochs / 25) + 0.1 * np.sin(epochs / 10)
            else:  # lr = 0.05
                base_loss = 0.8 / np.sqrt(dim) * np.exp(-epochs / 20) + 0.2 * np.sin(epochs / 5)

            loss = base_loss + np.random.normal(0, 0.02, len(epochs))
            loss = gaussian_filter1d(loss, sigma=1)

            fig.add_trace(go.Scatter(
                x=epochs,
                y=loss,
                mode='lines',
                name=f'd={dim}',
                line=dict(color=color, width=1),
                showlegend=False,
                opacity=0.7
            ), row=row, col=col)

        # 添加收敛点
        convergence_epoch = int(40 / lr)
        fig.add_vline(
            x=convergence_epoch,
            line_dash="dash",
            line_color=color,
            line_width=1,
            row=row,
            col=col
        )

    # 更新布局
    fig.update_layout(
        title='不同学习率和潜在维度的收敛分析',
        height=600,
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=False
    )

    return fig

def create_loss_landscape():
    """创建损失景观图"""

    # 创建参数网格
    lr_values = np.logspace(-4, -1, 50)
    beta_values = np.linspace(0.1, 5, 50)
    LR, BETA = np.meshgrid(lr_values, beta_values)

    # 计算损失景观（简化模型）
    # 学习率太小：收敛慢
    # 学习率太大：不稳定
    # β 太小：重建质量好但解耦差
    # β 太大：解耦好但重建质量差
    loss = 2 * np.log10(LR) * (BETA / 5) + 1 / np.sqrt(LR) * (5 / BETA)
    loss = loss + np.random.normal(0, 0.1, loss.shape)

    # 创建图形
    fig = go.Figure(data=[
        go.Contour(
            x=lr_values,
            y=beta_values,
            z=loss,
            colorscale='Viridis',
            contours=dict(
                start=0,
                end=10,
                size=0.5
            ),
            colorbar=dict(title="损失值")
        )
    ])

    # 添加等高线
    fig.add_trace(go.Contour(
        x=lr_values,
        y=beta_values,
        z=loss,
        contours=dict(
            start=1,
            end=9,
            size=1
        ),
        showscale=False,
        line=dict(width=1, color='white'),
        opacity=0.5
    ))

    # 标记最佳区域
    fig.add_shape(
        type="rect",
        x0=0.001, y0=0.5,
        x1=0.01, y1=2.0,
        fillcolor="green",
        opacity=0.2,
        line_width=0
    )

    fig.add_annotation(
        x=0.005,
        y=1.25,
        text="最佳区域",
        showarrow=False,
        font=dict(size=12, color="green")
    )

    # 更新布局
    fig.update_layout(
        title='VAE 参数损失景观',
        xaxis_title='学习率',
        yaxis_title='β 参数',
        width=800,
        height=600,
        font=dict(family="Arial, sans-serif", size=14)
    )

    return fig

if __name__ == "__main__":
    # 导入必要的库
    from plotly.subplots import make_subplots

    # 创建三个图形
    fig1 = create_training_curves()
    fig2 = create_convergence_analysis()
    fig3 = create_loss_landscape()

    # 保存图形
    fig1.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/vae_training_curves.html")
    fig2.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/vae_convergence.html")
    fig3.write_html("/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/vae_loss_landscape.html")

    print("VAE 训练可视化已生成并保存到 static/images/plots/ 目录")
    print("1. vae_training_curves.html - 训练过程曲线")
    print("2. vae_convergence.html - 收敛分析")
    print("3. vae_loss_landscape.html - 参数损失景观")