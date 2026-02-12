#!/usr/bin/env python3
"""
为 GAN 论文解读文章生成 Plotly 配图
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    fig.write_image(filepath, width=width, height=height, scale=2)
    
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")


def plot_gan_architecture():
    """绘制 GAN 架构示意图 - 生成器与判别器的对抗关系"""
    fig = go.Figure()
    
    # 节点位置
    # 左侧：噪声 z
    # 中间：生成器 G -> 生成样本
    # 右侧：判别器 D -> 真假判断
    # 下方：真实数据
    
    # 噪声 z (圆形)
    fig.add_trace(go.Scatter(
        x=[1], y=[5],
        mode='markers',
        marker=dict(size=60, color=APPLE_GRAY, line=dict(width=2, color='white')),
        name='潜在空间'
    ))
    fig.add_trace(go.Scatter(
        x=[1], y=[5],
        mode='text',
        text=['$z$'],
        textposition='middle center',
        textfont=dict(size=16, color='white', family='Arial')
    ))
    
    # 生成器 G (方形)
    fig.add_trace(go.Scatter(
        x=[3], y=[5],
        mode='markers',
        marker=dict(size=55, color=APPLE_BLUE, symbol='square', line=dict(width=2, color='white')),
        name='生成器'
    ))
    fig.add_trace(go.Scatter(
        x=[3], y=[5],
        mode='text',
        text=['G'],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial')
    ))
    
    # 生成样本 G(z) (圆形)
    fig.add_trace(go.Scatter(
        x=[5], y=[5],
        mode='markers',
        marker=dict(size=50, color=APPLE_GREEN, line=dict(width=2, color='white')),
        name='生成样本'
    ))
    fig.add_trace(go.Scatter(
        x=[5], y=[5],
        mode='text',
        text=['$G(z)$'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial')
    ))
    
    # 真实数据 x (圆形)
    fig.add_trace(go.Scatter(
        x=[5], y=[2.5],
        mode='markers',
        marker=dict(size=50, color=APPLE_ORANGE, line=dict(width=2, color='white')),
        name='真实数据'
    ))
    fig.add_trace(go.Scatter(
        x=[5], y=[2.5],
        mode='text',
        text=['$x$'],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial')
    ))
    
    # 判别器 D (菱形)
    fig.add_trace(go.Scatter(
        x=[7], y=[3.75],
        mode='markers',
        marker=dict(size=55, color=APPLE_PURPLE, symbol='diamond', line=dict(width=2, color='white')),
        name='判别器'
    ))
    fig.add_trace(go.Scatter(
        x=[7], y=[3.75],
        mode='text',
        text=['D'],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial')
    ))
    
    # 输出 (真假判断)
    fig.add_trace(go.Scatter(
        x=[9], y=[3.75],
        mode='markers',
        marker=dict(size=45, color=APPLE_RED, line=dict(width=2, color='white')),
        name='输出'
    ))
    fig.add_trace(go.Scatter(
        x=[9], y=[3.75],
        mode='text',
        text=['0/1'],
        textposition='middle center',
        textfont=dict(size=11, color='white', family='Arial')
    ))
    
    # 箭头 - 从 z 到 G
    fig.add_annotation(
        x=2, y=5,
        ax=1.5, ay=5,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_GRAY
    )
    
    # 箭头 - 从 G 到 G(z)
    fig.add_annotation(
        x=4, y=5,
        ax=3.5, ay=5,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_BLUE
    )
    
    # 箭头 - 从 G(z) 到 D
    fig.add_annotation(
        x=6, y=4.5,
        ax=5.5, ay=4.9,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_GREEN
    )
    
    # 箭头 - 从 x 到 D
    fig.add_annotation(
        x=6, y=3,
        ax=5.5, ay=2.6,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_ORANGE
    )
    
    # 箭头 - 从 D 到 输出
    fig.add_annotation(
        x=8, y=3.75,
        ax=7.5, ay=3.75,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_PURPLE
    )
    
    # 添加标签
    fig.add_annotation(x=3, y=6.2, text='生成器网络', showarrow=False, 
                      font=dict(size=12, color=APPLE_BLUE))
    fig.add_annotation(x=7, y=5, text='判别器网络', showarrow=False, 
                      font=dict(size=12, color=APPLE_PURPLE))
    fig.add_annotation(x=1, y=6.2, text='潜在噪声', showarrow=False, 
                      font=dict(size=11, color=APPLE_GRAY))
    fig.add_annotation(x=5, y=1.5, text='真实数据分布', showarrow=False, 
                      font=dict(size=11, color=APPLE_ORANGE))
    
    # 对抗箭头 (从输出回到 G)
    fig.add_annotation(
        x=3, y=4.2,
        ax=8.5, ay=3.3,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_RED
    )
    fig.add_annotation(x=5.5, y=3.2, text='反向传播', showarrow=False, 
                      font=dict(size=10, color=APPLE_RED))
    
    fig.update_layout(
        title=dict(text='GAN 对抗架构示意图', x=0.5, font=dict(size=16)),
        xaxis=dict(visible=False, range=[0, 10]),
        yaxis=dict(visible=False, range=[0.5, 7]),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20),
        height=450
    )
    
    return fig


def plot_training_dynamics():
    """绘制 GAN 训练动态 - 损失曲线和分布演化"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('训练损失曲线', '分布距离演化'),
        horizontal_spacing=0.12
    )
    
    # 模拟训练过程
    iterations = np.arange(0, 1000, 10)
    
    # 判别器损失：先快速下降，然后震荡
    d_loss = 2.0 * np.exp(-iterations / 200) + 0.3 + 0.1 * np.sin(iterations / 50)
    
    # 生成器损失：先上升（判别器变强），然后逐渐下降
    g_loss = np.where(iterations < 200, 
                      1.0 + iterations / 200,
                      1.5 * np.exp(-(iterations - 200) / 300) + 0.2)
    
    # 左图：损失曲线
    fig.add_trace(
        go.Scatter(x=iterations, y=d_loss, mode='lines', name='判别器损失',
                  line=dict(color=APPLE_PURPLE, width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=g_loss, mode='lines', name='生成器损失',
                  line=dict(color=APPLE_GREEN, width=2)),
        row=1, col=1
    )
    
    # 右图：JS 散度演化
    js_div = 0.7 * np.exp(-iterations / 250)
    fig.add_trace(
        go.Scatter(x=iterations, y=js_div, mode='lines', name='JS 散度',
                  line=dict(color=APPLE_BLUE, width=2), fill='tozeroy',
                  fillcolor='rgba(0, 122, 255, 0.1)'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='迭代次数', row=1, col=1)
    fig.update_xaxes(title_text='迭代次数', row=1, col=2)
    fig.update_yaxes(title_text='损失值', row=1, col=1)
    fig.update_yaxes(title_text='JS 散度', row=1, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.35, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=40, t=60, b=50),
        height=400
    )
    
    return fig


def plot_distribution_evolution():
    """绘制真实分布与生成分布的演化过程"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('初始阶段', '训练中期', '训练后期', '收敛状态'),
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )
    
    x = np.linspace(-4, 4, 200)
    
    # 真实分布：双峰高斯
    def real_dist(x):
        return 0.5 * (1/np.sqrt(2*np.pi*0.5)) * np.exp(-(x-1.5)**2/0.5) + \
               0.5 * (1/np.sqrt(2*np.pi*0.5)) * np.exp(-(x+1.5)**2/0.5)
    
    real = real_dist(x)
    
    # 四个阶段的生成分布
    stages = [
        # 初始：单峰高斯（完全不对）
        (1/np.sqrt(2*np.pi*2)) * np.exp(-x**2/2),
        # 中期：开始分裂
        0.6 * (1/np.sqrt(2*np.pi*1.2)) * np.exp(-(x-0.5)**2/1.2) + \
        0.4 * (1/np.sqrt(2*np.pi*1.2)) * np.exp(-(x+0.5)**2/1.2),
        # 后期：接近真实
        0.5 * (1/np.sqrt(2*np.pi*0.7)) * np.exp(-(x-1.2)**2/0.7) + \
        0.5 * (1/np.sqrt(2*np.pi*0.7)) * np.exp(-(x+1.2)**2/0.7),
        # 收敛：几乎一致
        0.5 * (1/np.sqrt(2*np.pi*0.5)) * np.exp(-(x-1.5)**2/0.5) + \
        0.5 * (1/np.sqrt(2*np.pi*0.5)) * np.exp(-(x+1.5)**2/0.5) + 0.01
    ]
    
    colors_gen = [APPLE_GRAY, APPLE_ORANGE, APPLE_BLUE, APPLE_GREEN]
    
    for i, (stage_dist, color) in enumerate(zip(stages, colors_gen)):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # 真实分布
        fig.add_trace(
            go.Scatter(x=x, y=real, mode='lines', name='真实分布' if i == 0 else None,
                      line=dict(color=APPLE_RED, width=2, dash='dash'),
                      showlegend=(i==0)),
            row=row, col=col
        )
        
        # 生成分布
        fig.add_trace(
            go.Scatter(x=x, y=stage_dist, mode='lines', name='生成分布' if i == 0 else None,
                      line=dict(color=color, width=2),
                      showlegend=(i==0)),
            row=row, col=col
        )
        
        fig.update_xaxes(range=[-4, 4], row=row, col=col)
        fig.update_yaxes(range=[0, 0.6], row=row, col=col)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=11),
        legend=dict(x=0.75, y=0.35, bgcolor='rgba(255,255,255,0.9)'),
        margin=dict(l=50, r=30, t=60, b=40),
        height=500
    )
    
    return fig


def plot_optimal_discriminator():
    """绘制最优判别器的几何解释"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('概率密度函数', '最优判别器输出'),
        horizontal_spacing=0.12
    )
    
    x = np.linspace(-4, 4, 500)
    
    # 真实分布 p_data
    p_data = (1/np.sqrt(2*np.pi*0.8)) * np.exp(-(x-1)**2/0.8)
    
    # 生成分布 p_g (初始时不同)
    p_g = (1/np.sqrt(2*np.pi*0.8)) * np.exp(-(x+0.5)**2/0.8)
    
    # 左图：概率密度
    fig.add_trace(
        go.Scatter(x=x, y=p_data, mode='lines', name='$p_{data}(x)$',
                  line=dict(color=APPLE_RED, width=2.5), fill='tozeroy',
                  fillcolor='rgba(255, 59, 48, 0.1)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=p_g, mode='lines', name='$p_g(x)$',
                  line=dict(color=APPLE_BLUE, width=2.5), fill='tozeroy',
                  fillcolor='rgba(0, 122, 255, 0.1)'),
        row=1, col=1
    )
    
    # 右图：最优判别器 D*(x) = p_data / (p_data + p_g)
    D_optimal = p_data / (p_data + p_g + 1e-8)
    
    fig.add_trace(
        go.Scatter(x=x, y=D_optimal, mode='lines', name='$D^{\ast}(x)$',
                  line=dict(color=APPLE_PURPLE, width=2.5)),
        row=1, col=2
    )
    
    # 添加参考线
    fig.add_hline(y=0.5, line_dash="dash", line_color=APPLE_GRAY, 
                  annotation_text="不确定边界", row=1, col=2)
    
    fig.update_xaxes(title_text='$x$', row=1, col=1)
    fig.update_xaxes(title_text='$x$', row=1, col=2)
    fig.update_yaxes(title_text='概率密度', row=1, col=1)
    fig.update_yaxes(title_text='$D^{\ast}(x)$', range=[-0.05, 1.05], row=1, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.65, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=40, t=60, b=50),
        height=400
    )
    
    return fig


def plot_value_function_landscape():
    """绘制价值函数的等高线图 - 展示纳什均衡点"""
    fig = go.Figure()
    
    # 创建网格
    theta_g = np.linspace(0, 4, 100)  # 生成器参数
    theta_d = np.linspace(0, 4, 100)  # 判别器参数
    Theta_G, Theta_D = np.meshgrid(theta_g, theta_d)
    
    # 简化的价值函数 V(D, G) = log(D) + log(1-D) * (1 - exp(-(G-2)^2))
    # 纳什均衡点在 (2, 0.5)
    V = np.log(Theta_D + 0.1) + np.log(1.1 - Theta_D) * (1 - np.exp(-(Theta_G - 2)**2/2))
    
    # 等高线
    contour = go.Contour(
        x=theta_g, y=theta_d, z=V,
        colorscale='RdYlBu',
        contours=dict(start=-5, end=0, size=0.5),
        colorbar=dict(title='V(D,G)', titleside='right'),
        line=dict(width=0.5)
    )
    fig.add_trace(contour)
    
    # 标记纳什均衡点
    fig.add_trace(go.Scatter(
        x=[2], y=[0.5],
        mode='markers',
        marker=dict(size=15, color=APPLE_RED, symbol='star', 
                   line=dict(width=2, color='white')),
        name='纳什均衡'
    ))
    
    fig.add_annotation(x=2, y=0.5, text='纳什均衡<br>(G最优, D无法区分)', 
                      showarrow=True, ax=40, ay=-40,
                      font=dict(size=11, color=APPLE_RED))
    
    fig.update_layout(
        title=dict(text='GAN 价值函数 landscape', x=0.5, font=dict(size=14)),
        xaxis_title='$\\theta_G$ (生成器参数)',
        yaxis_title='$\\theta_D$ (判别器参数)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=60, r=80, t=60, b=50),
        height=450
    )
    
    return fig


def main():
    """生成所有配图"""
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始生成 GAN 文章配图...\n")
    
    # 1. GAN 架构图
    fig1 = plot_gan_architecture()
    save_and_compress(fig1, f'{output_dir}/gan-architecture.png', width=800, height=450)
    
    # 2. 训练动态
    fig2 = plot_training_dynamics()
    save_and_compress(fig2, f'{output_dir}/gan-training-dynamics.png', width=900, height=400)
    
    # 3. 分布演化
    fig3 = plot_distribution_evolution()
    save_and_compress(fig3, f'{output_dir}/gan-distribution-evolution.png', width=900, height=500)
    
    # 4. 最优判别器
    fig4 = plot_optimal_discriminator()
    save_and_compress(fig4, f'{output_dir}/gan-optimal-discriminator.png', width=900, height=400)
    
    # 5. 价值函数 landscape
    fig5 = plot_value_function_landscape()
    save_and_compress(fig5, f'{output_dir}/gan-value-landscape.png', width=800, height=450)
    
    print("\n✅ 所有配图生成完成!")
    
    # 检查文件大小
    print("\n文件大小统计:")
    for fname in ['gan-architecture.png', 'gan-training-dynamics.png', 
                  'gan-distribution-evolution.png', 'gan-optimal-discriminator.png',
                  'gan-value-landscape.png']:
        fpath = f'{output_dir}/{fname}'
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  {fname}: {size/1024:.1f} KB")


if __name__ == '__main__':
    main()
