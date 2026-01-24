#!/usr/bin/env python3
"""
VAE 数学图形可视化
使用 Plotly 创建交互式数学图表
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 创建输出目录
os.makedirs('static/images/vae', exist_ok=True)

# ============================================================================
# 图1：ELBO 分解示意图
# ============================================================================
def plot_elbo_decomposition():
    """
    可视化 ELBO = log p(x) - KL(q||p)
    展示证据下界的组成部分
    """
    fig = go.Figure()
    
    # 模拟 log p(x) 随潜在维度变化的曲线
    d_values = np.linspace(1, 50, 100)
    log_p_x = 2.5 * np.log(d_values + 10) - 3
    kl_divergence = 0.8 * np.exp(-0.05 * d_values) * d_values**0.5
    elbo = log_p_x - kl_divergence
    
    # 绘制 log p(x)
    fig.add_trace(go.Scatter(
        x=d_values, y=log_p_x,
        mode='lines',
        name='log p(x) (证据)',
        line=dict(color='#007AFF', width=3)
    ))
    
    # 绘制 KL 散度（向下）
    fig.add_trace(go.Scatter(
        x=d_values, y=-kl_divergence,
        mode='lines',
        name='-KL(q||p)',
        line=dict(color='#FF3B30', width=3, dash='dash')
    ))
    
    # 绘制 ELBO
    fig.add_trace(go.Scatter(
        x=d_values, y=elbo,
        mode='lines',
        name='ELBO (证据下界)',
        line=dict(color='#34C759', width=4)
    ))
    
    fig.update_layout(
        title='ELBO 分解：ELBO = log p(x) - KL(q||p)',
        xaxis_title='潜在空间维度 (d)',
        yaxis_title='值',
        template='plotly_white',
        height=500,
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.write_html('static/images/vae/elbo_decomposition.html')
    fig.write_image('static/images/vae/elbo_decomposition.png', width=800, height=500, scale=2)
    print("✓ ELBO 分解图已保存")

# ============================================================================
# 图2：重参数化技巧可视化
# ============================================================================
def plot_reparameterization():
    """
    可视化重参数化技巧：z = μ + σ * ε
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '标准正态噪声 ε ~ N(0,1)',
            '变换 z = μ + σ·ε',
            '不同均值 μ 的效果',
            '不同方差 σ² 的效果'
        ],
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'histogram'}, {'type': 'histogram'}]]
    )
    
    # 生成标准正态分布
    epsilon = np.random.randn(10000)
    
    # 子图1：标准正态噪声
    fig.add_trace(go.Histogram(
        x=epsilon,
        nbinsx=50,
        name='ε ~ N(0,1)',
        marker_color='#007AFF',
        opacity=0.7
    ), row=1, col=1)
    
    # 子图2：变换后的分布 (μ=2, σ=1.5)
    z1 = 2 + 1.5 * epsilon
    fig.add_trace(go.Histogram(
        x=z1,
        nbinsx=50,
        name='z = 2 + 1.5·ε',
        marker_color='#34C759',
        opacity=0.7
    ), row=1, col=2)
    
    # 子图3：不同均值
    for mu, color in zip([-2, 0, 2], ['#FF3B30', '#007AFF', '#34C759']):
        z = mu + 1 * epsilon
        fig.add_trace(go.Histogram(
            x=z,
            nbinsx=50,
            name=f'μ={mu}, σ=1',
            marker_color=color,
            opacity=0.5
        ), row=2, col=1)
    
    # 子图4：不同方差
    for sigma, color in zip([0.5, 1, 2], ['#FF3B30', '#007AFF', '#34C759']):
        z = 0 + sigma * epsilon
        fig.add_trace(go.Histogram(
            x=z,
            nbinsx=50,
            name=f'μ=0, σ={sigma}',
            marker_color=color,
            opacity=0.5
        ), row=2, col=2)
    
    fig.update_layout(
        title='重参数化技巧：z = μ + σ·ε',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="值")
    fig.update_yaxes(title_text="频数")
    
    fig.write_html('static/images/vae/reparameterization.html')
    fig.write_image('static/images/vae/reparameterization.png', width=1000, height=600, scale=2)
    print("✓ 重参数化技巧图已保存")

# ============================================================================
# 图3：KL 散度计算公式可视化
# ============================================================================
def plot_kl_divergence_2d():
    """
    可视化 2D 高斯分布之间的 KL 散度
    """
    fig = go.Figure()
    
    # 创建网格
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    
    # 标准正态分布 p(z) = N(0, I)
    Z_p = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    
    # 近似后验 q(z) = N([1.5, 1], diag([1.5², 0.8²]))
    mu_q = np.array([1.5, 1.0])
    sigma_q = np.array([1.5, 0.8])
    Z_q = np.exp(-((X-mu_q[0])**2/(2*sigma_q[0]**2) + (Y-mu_q[1])**2/(2*sigma_q[1]**2))) / (2*np.pi*sigma_q[0]*sigma_q[1])
    
    # 绘制 p(z)
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z_p,
        colorscale='Blues',
        name='p(z) ~ N(0, I)',
        contours=dict(
            start=0.01,
            end=0.16,
            size=0.01
        ),
        showscale=False,
        line=dict(width=2)
    ))
    
    # 绘制 q(z)
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z_q,
        colorscale='Reds',
        name='q(z|x)',
        contours=dict(
            start=0.01,
            end=0.16,
            size=0.01
        ),
        showscale=False,
        line=dict(width=2, dash='dash')
    ))
    
    # 添加均值点（分别添加两个点）
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=15, color='#007AFF'),
        name='μ_p'
    ))
    
    fig.add_trace(go.Scatter(
        x=[1.5], y=[1.0],
        mode='markers',
        marker=dict(size=15, color='#FF3B30'),
        name='μ_q'
    ))
    
    # 计算 KL 散度
    kl = 0.5 * (np.sum(sigma_q**2) + np.sum(mu_q**2) - 2 - np.sum(np.log(sigma_q**2)))
    
    fig.update_layout(
        title=f'KL 散度可视化：D_KL(q||p) = {kl:.3f}',
        xaxis_title='z₁',
        yaxis_title='z₂',
        template='plotly_white',
        height=500,
        width=500
    )
    
    fig.write_html('static/images/vae/kl_divergence_2d.html')
    fig.write_image('static/images/vae/kl_divergence_2d.png', width=500, height=500, scale=2)
    print("✓ KL 散度可视化图已保存")

# ============================================================================
# 图4：潜在空间插值
# ============================================================================
def plot_latent_interpolation():
    """
    可视化潜在空间的插值过程
    """
    # 模拟两个潜在向量
    z1 = np.array([-2, -2])
    z2 = np.array([2, 2])
    
    # 生成插值路径
    n_steps = 10
    alphas = np.linspace(0, 1, n_steps)
    interpolation = np.outer(1-alphas, z1) + np.outer(alphas, z2)
    
    fig = go.Figure()
    
    # 绘制先验分布等高线
    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Greens',
        opacity=0.3,
        showscale=False,
        contours=dict(
            start=0.01,
            end=0.16,
            size=0.01
        )
    ))
    
    # 绘制插值路径
    fig.add_trace(go.Scatter(
        x=interpolation[:, 0],
        y=interpolation[:, 1],
        mode='lines+markers',
        line=dict(color='#007AFF', width=3),
        marker=dict(size=10, color=interpolation[:, 0], 
                    colorscale='Bluered', showscale=False),
        name='插值路径'
    ))
    
    # 标记起点和终点
    fig.add_trace(go.Scatter(
        x=[z1[0], z2[0]],
        y=[z1[1], z2[1]],
        mode='markers+text',
        marker=dict(size=20, color=['#34C759', '#FF3B30']),
        text=['z₁', 'z₂'],
        textposition='top center',
        textfont=dict(size=16, color='black'),
        name='端点'
    ))
    
    fig.update_layout(
        title='潜在空间插值：z(α) = (1-α)z₁ + αz₂',
        xaxis_title='z₁',
        yaxis_title='z₂',
        template='plotly_white',
        height=500,
        width=500
    )
    
    fig.write_html('static/images/vae/latent_interpolation.html')
    fig.write_image('static/images/vae/latent_interpolation.png', width=500, height=500, scale=2)
    print("✓ 潜在空间插值图已保存")

# ============================================================================
# 图5：β-VAE 损失权衡
# ============================================================================
def plot_beta_vae_tradeoff():
    """
    可视化 β-VAE 中重建误差与 KL 散度的权衡
    """
    fig = go.Figure()
    
    beta_values = np.logspace(-1, 1, 50)
    
    # 模拟不同 β 值下的重建误差和 KL 散度
    reconstruction_error = 10 / (beta_values + 0.5)
    kl_divergence = beta_values * 0.8
    
    # 绘制重建误差
    fig.add_trace(go.Scatter(
        x=beta_values, y=reconstruction_error,
        mode='lines',
        name='重建误差',
        line=dict(color='#007AFF', width=3)
    ))
    
    # 绘制 KL 散度
    fig.add_trace(go.Scatter(
        x=beta_values, y=kl_divergence,
        mode='lines',
        name='KL 散度',
        line=dict(color='#FF3B30', width=3)
    ))
    
    # 标注标准 VAE (β=1)
    fig.add_trace(go.Scatter(
        x=[1], y=[10/1.5],
        mode='markers',
        marker=dict(size=15, color='#34C759', symbol='diamond'),
        name='标准 VAE (β=1)',
        text=['β=1'],
        textposition='top center'
    ))
    
    fig.update_layout(
        title='β-VAE：重建误差 vs KL 散度',
        xaxis_title='β (KL 散度权重)',
        yaxis_title='损失值',
        xaxis_type='log',
        template='plotly_white',
        height=500,
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.write_html('static/images/vae/beta_vae_tradeoff.html')
    fig.write_image('static/images/vae/beta_vae_tradeoff.png', width=800, height=500, scale=2)
    print("✓ β-VAE 权衡图已保存")

# ============================================================================
# 图6：VAE 训练曲线模拟
# ============================================================================
def plot_vae_training_curves():
    """
    模拟 VAE 训练过程中的损失变化
    """
    epochs = np.arange(1, 201)
    
    # 模拟训练损失（随时间下降）
    reconstruction_train = 100 * np.exp(-0.02 * epochs) + 10
    kl_train = 15 * (1 - np.exp(-0.03 * epochs)) + 5
    
    # 模拟验证损失
    reconstruction_val = 100 * np.exp(-0.018 * epochs) + 12
    kl_val = 15 * (1 - np.exp(-0.025 * epochs)) + 6
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['重建误差', 'KL 散度'],
        shared_xaxes=True
    )
    
    # 重建误差
    fig.add_trace(go.Scatter(
        x=epochs, y=reconstruction_train,
        mode='lines',
        name='训练集',
        line=dict(color='#007AFF', width=2),
        legendgroup='train'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=epochs, y=reconstruction_val,
        mode='lines',
        name='验证集',
        line=dict(color='#FF9500', width=2),
        legendgroup='val'
    ), row=1, col=1)
    
    # KL 散度
    fig.add_trace(go.Scatter(
        x=epochs, y=kl_train,
        mode='lines',
        name='训练集',
        line=dict(color='#007AFF', width=2),
        showlegend=False,
        legendgroup='train'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=epochs, y=kl_val,
        mode='lines',
        name='验证集',
        line=dict(color='#FF9500', width=2),
        showlegend=False,
        legendgroup='val'
    ), row=2, col=1)
    
    fig.update_layout(
        title='VAE 训练过程',
        template='plotly_white',
        height=600,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text='Epoch')
    fig.update_yaxes(title_text='重建误差', row=1, col=1)
    fig.update_yaxes(title_text='KL 散度', row=2, col=1)
    
    fig.write_html('static/images/vae/training_curves.html')
    fig.write_image('static/images/vae/training_curves.png', width=800, height=600, scale=2)
    print("✓ VAE 训练曲线图已保存")

# ============================================================================
# 主函数
# ============================================================================
if __name__ == '__main__':
    print("开始生成 VAE 数学可视化图表...\n")
    
    plot_elbo_decomposition()
    plot_reparameterization()
    plot_kl_divergence_2d()
    plot_latent_interpolation()
    plot_beta_vae_tradeoff()
    plot_vae_training_curves()
    
    print("\n✅ 所有图表生成完成！")
    print("📁 图表保存位置: static/images/vae/")
