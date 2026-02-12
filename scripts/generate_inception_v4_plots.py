#!/usr/bin/env python3
"""
生成 Inception-v4 论文解读相关的 Plotly 图形
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import subprocess
import os

# 苹果风格配色
APPLE_BLUE = '#007AFF'
APPLE_GREEN = '#34C759'
APPLE_ORANGE = '#FF9500'
APPLE_RED = '#FF3B30'
APPLE_PURPLE = '#AF52DE'
APPLE_CYAN = '#5AC8FA'
APPLE_GRAY = '#8E8E93'

def save_and_compress(fig, filepath, width=900, height=600):
    """保存并压缩图片"""
    fig.write_image(filepath, width=width, height=height, scale=2)
    
    # 压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force',
            '--output', filepath, filepath
        ], check=False, capture_output=True)
    
    print(f"✅ 已保存: {filepath}")

def plot_inception_evolution():
    """Inception 系列演进时间线"""
    fig = go.Figure()
    
    # 时间节点 - 稍微错开X轴位置避免重叠
    years = [2014, 2015, 2015.15, 2016, 2016.15, 2016.3]
    versions = ['Inception-v1\n(GoogLeNet)', 'Inception-v2', 'Inception-v3', 
                'Inception-v4', 'Inception-ResNet-v1', 'Inception-ResNet-v2']
    top5_errors = [6.67, 6.2, 5.6, 3.08, 4.3, 3.08]  # top-5 error
    colors = [APPLE_BLUE, APPLE_GREEN, APPLE_GREEN, APPLE_RED, APPLE_ORANGE, APPLE_ORANGE]
    
    # 添加节点
    for i, (year, version, error, color) in enumerate(zip(years, versions, top5_errors, colors)):
        fig.add_trace(go.Scatter(
            x=[year],
            y=[error],
            mode='markers+text',
            marker=dict(size=40, color=color, line=dict(width=2, color='white')),
            text=[version],
            textposition='top center',
            textfont=dict(size=9, color='#333'),
            showlegend=False,
            hovertemplate=f'{version}<br>Top-5 Error: {error}%<extra></extra>'
        ))
    
    # 连接线 - 使用原始年份位置
    line_years = [2014, 2015, 2015.15, 2016, 2016.15, 2016.3]
    fig.add_trace(go.Scatter(
        x=line_years,
        y=top5_errors,
        mode='lines',
        line=dict(color=APPLE_GRAY, width=2, dash='dot'),
        showlegend=False
    ))
    
    # 添加残差革命标注 - 调整位置避免重叠
    fig.add_annotation(
        x=2015.5, y=6.0,
        text="残差革命<br>ResNet-2015",
        showarrow=True,
        arrowhead=2,
        arrowcolor=APPLE_PURPLE,
        ax=0, ay=-50,
        font=dict(size=11, color=APPLE_PURPLE),
        bgcolor='white',
        bordercolor=APPLE_PURPLE,
        borderwidth=1
    )
    
    fig.update_layout(
        title='Inception 系列演进与 ImageNet Top-5 错误率',
        xaxis_title='年份',
        yaxis_title='Top-5 错误率 (%)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        xaxis=dict(range=[2013.5, 2016.8], dtick=1),
        yaxis=dict(range=[0, 8]),
        height=500,
        width=900
    )
    
    return fig

def plot_inception_v4_architecture():
    """Inception-v4 整体架构示意图"""
    fig = go.Figure()
    
    # 模块定义
    modules = [
        ('Stem', 0, APPLE_BLUE),
        ('Inception-A\n×4', 1, APPLE_GREEN),
        ('Reduction-A', 2, APPLE_ORANGE),
        ('Inception-B\n×7', 3, APPLE_GREEN),
        ('Reduction-B', 4, APPLE_ORANGE),
        ('Inception-C\n×3', 5, APPLE_GREEN),
        ('Average\nPool', 6, APPLE_PURPLE),
        ('Dropout', 7, APPLE_GRAY),
        ('FC +\nSoftmax', 8, APPLE_RED),
    ]
    
    # 绘制模块
    for name, x, color in modules:
        # 模块框
        fig.add_shape(
            type="rect",
            x0=x-0.35, y0=0.3, x1=x+0.35, y1=0.7,
            fillcolor=color,
            line=dict(color=color, width=2),
            opacity=0.8
        )
        # 模块名称
        fig.add_trace(go.Scatter(
            x=[x], y=[0.5],
            mode='text',
            text=[name],
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial'),
            showlegend=False
        ))
        
        # 添加尺寸标注
        sizes = ['299×299×3', '35×35×384', '17×17×1024', '17×17×1024', '8×8×1536', '8×8×1536', '1×1×1536', '1×1×1536', '1000']
        fig.add_trace(go.Scatter(
            x=[x], y=[0.15],
            mode='text',
            text=[sizes[x]],
            textposition='middle center',
            textfont=dict(size=9, color='#666'),
            showlegend=False
        ))
    
    # 连接线
    for i in range(len(modules)-1):
        fig.add_annotation(
            x=i+0.35, y=0.5,
            ax=i+0.65, ay=0.5,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#999'
        )
    
    fig.update_layout(
        title='Inception-v4 整体架构流程',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        xaxis=dict(range=[-0.5, 8.5], visible=False),
        yaxis=dict(range=[0, 0.85], visible=False),
        height=400,
        width=1100,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def plot_inception_modules_comparison():
    """Inception 模块结构对比"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Inception-A', 'Inception-B', 'Inception-C'),
        horizontal_spacing=0.1
    )
    
    # Inception-A: 35x35 grid
    # 可视化分支结构
    branches_a = [
        ('1×1 conv', 0.5, APPLE_BLUE),
        ('1×1→3×3', 0.3, APPLE_GREEN),
        ('1×1→3×3→3×3', 0.1, APPLE_ORANGE),
        ('3×3 pool→1×1', -0.1, APPLE_PURPLE),
    ]
    
    for name, y, color in branches_a:
        fig.add_trace(go.Scatter(
            x=[0, 0.5, 1], y=[y, y, 0.5],
            mode='lines+markers',
            line=dict(color=color, width=3),
            marker=dict(size=15, color=color),
            showlegend=False,
            hoverinfo='text',
            hovertext=name
        ), row=1, col=1)
    
    # Inception-B: 17x17 grid
    branches_b = [
        ('1×1 conv', 0.5, APPLE_BLUE),
        ('1×1→1×7→7×1', 0.3, APPLE_GREEN),
        ('1×1→7×1→1×7→7×1→1×7', 0.1, APPLE_ORANGE),
        ('3×3 pool→1×1', -0.1, APPLE_PURPLE),
    ]
    
    for name, y, color in branches_b:
        fig.add_trace(go.Scatter(
            x=[0, 0.5, 1], y=[y, y, 0.5],
            mode='lines+markers',
            line=dict(color=color, width=3),
            marker=dict(size=15, color=color),
            showlegend=False,
            hoverinfo='text',
            hovertext=name
        ), row=1, col=2)
    
    # Inception-C: 8x8 grid
    branches_c = [
        ('1×1 conv', 0.5, APPLE_BLUE),
        ('1×1→1×3→3×1', 0.3, APPLE_GREEN),
        ('1×1→3×1→1×3', 0.1, APPLE_ORANGE),
        ('3×3 pool→1×1', -0.1, APPLE_PURPLE),
    ]
    
    for name, y, color in branches_c:
        fig.add_trace(go.Scatter(
            x=[0, 0.5, 1], y=[y, y, 0.5],
            mode='lines+markers',
            line=dict(color=color, width=3),
            marker=dict(size=15, color=color),
            showlegend=False,
            hoverinfo='text',
            hovertext=name
        ), row=1, col=3)
    
    fig.update_layout(
        title='Inception 模块多分支结构对比',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        height=450,
        width=1000,
        showlegend=False
    )
    
    for i in range(1, 4):
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, range=[-0.3, 0.8], row=1, col=i)
    
    return fig

def plot_training_comparison():
    """训练过程对比"""
    fig = go.Figure()
    
    # 模拟训练曲线 (epochs vs top-5 error)
    epochs = np.arange(0, 160)
    
    # Inception-v3 (无残差)
    v3_error = 15 * np.exp(-epochs/50) + 5.6 + 0.5 * np.random.randn(len(epochs)) * 0.1
    
    # Inception-v4 (无残差，更深)
    v4_error = 16 * np.exp(-epochs/45) + 3.08 + 0.5 * np.random.randn(len(epochs)) * 0.1
    
    # Inception-ResNet-v1 (有残差)
    res_v1_error = 14 * np.exp(-epochs/35) + 4.3 + 0.5 * np.random.randn(len(epochs)) * 0.1
    
    # Inception-ResNet-v2 (有残差，更深)
    res_v2_error = 15 * np.exp(-epochs/32) + 3.08 + 0.5 * np.random.randn(len(epochs)) * 0.1
    
    fig.add_trace(go.Scatter(
        x=epochs, y=v3_error,
        mode='lines',
        name='Inception-v3',
        line=dict(color=APPLE_BLUE, width=2),
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=v4_error,
        mode='lines',
        name='Inception-v4',
        line=dict(color=APPLE_GREEN, width=2),
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=res_v1_error,
        mode='lines',
        name='Inception-ResNet-v1',
        line=dict(color=APPLE_ORANGE, width=2),
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=res_v2_error,
        mode='lines',
        name='Inception-ResNet-v2',
        line=dict(color=APPLE_RED, width=2),
        opacity=0.8
    ))
    
    # 添加收敛区域标注
    fig.add_vrect(
        x0=100, x1=160,
        fillcolor=APPLE_GRAY, opacity=0.1,
        layer="below", line_width=0,
    )
    
    fig.add_annotation(
        x=130, y=12,
        text="收敛区域",
        showarrow=False,
        font=dict(size=11, color=APPLE_GRAY)
    )
    
    fig.update_layout(
        title='训练过程对比：Top-5 错误率随 Epoch 变化',
        xaxis_title='Epoch',
        yaxis_title='Top-5 错误率 (%)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(
            x=0.99, y=0.99,
            xanchor='right', yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'
        ),
        height=500,
        width=900
    )
    
    return fig

def plot_residual_scaling():
    """残差缩放效果示意图"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('无残差缩放（不稳定）', '有残差缩放（稳定）'),
        horizontal_spacing=0.15
    )
    
    iterations = np.arange(0, 50000, 100)
    
    # 无残差缩放：后期震荡
    unstable = 100 * np.exp(-iterations/10000) * (1 + 0.1 * np.sin(iterations/500))
    unstable[300:] = unstable[300:] + 20 * np.random.randn(len(unstable[300:])) * (iterations[300:] / 50000)
    
    # 有残差缩放：平稳收敛
    stable = 100 * np.exp(-iterations/10000) * (1 + 0.05 * np.sin(iterations/500))
    
    fig.add_trace(go.Scatter(
        x=iterations, y=unstable,
        mode='lines',
        line=dict(color=APPLE_RED, width=2),
        name='训练损失',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=iterations, y=stable,
        mode='lines',
        line=dict(color=APPLE_GREEN, width=2),
        name='训练损失',
        showlegend=False
    ), row=1, col=2)
    
    # 添加不稳定区域标注
    fig.add_annotation(
        x=40000, y=40,
        text="震荡发散",
        showarrow=True,
        arrowhead=2,
        arrowcolor=APPLE_RED,
        ax=-50, ay=-30,
        font=dict(size=11, color=APPLE_RED),
        row=1, col=1
    )
    
    fig.add_annotation(
        x=40000, y=10,
        text="稳定收敛",
        showarrow=True,
        arrowhead=2,
        arrowcolor=APPLE_GREEN,
        ax=-50, ay=-30,
        font=dict(size=11, color=APPLE_GREEN),
        row=1, col=2
    )
    
    fig.update_layout(
        title='残差缩放对训练稳定性的影响',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        height=400,
        width=900
    )
    
    fig.update_xaxes(title_text='迭代次数', row=1, col=1)
    fig.update_xaxes(title_text='迭代次数', row=1, col=2)
    fig.update_yaxes(title_text='损失值', row=1, col=1)
    fig.update_yaxes(title_text='损失值', row=1, col=2)
    
    return fig

def plot_filter_factorization():
    """卷积分解示意图"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('传统大卷积核', '非对称分解'),
        horizontal_spacing=0.1
    )
    
    # 传统 5x5 卷积
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=5, y1=5,
        fillcolor=APPLE_BLUE,
        line=dict(color=APPLE_BLUE, width=2),
        opacity=0.6,
        row=1, col=1
    )
    
    fig.add_annotation(
        x=2.5, y=2.5,
        text="5×5 conv<br>25 params",
        showarrow=False,
        font=dict(size=12, color='white'),
        row=1, col=1
    )
    
    # 非对称分解：5x1 + 1x5
    fig.add_shape(
        type="rect",
        x0=0, y0=2, x1=5, y1=3,
        fillcolor=APPLE_GREEN,
        line=dict(color=APPLE_GREEN, width=2),
        opacity=0.8,
        row=1, col=2
    )
    
    fig.add_shape(
        type="rect",
        x0=2, y0=0, x1=3, y1=5,
        fillcolor=APPLE_ORANGE,
        line=dict(color=APPLE_ORANGE, width=2),
        opacity=0.8,
        row=1, col=2
    )
    
    fig.add_annotation(
        x=2.5, y=4.2,
        text="5×1 conv",
        showarrow=False,
        font=dict(size=10, color='white'),
        row=1, col=2
    )
    
    fig.add_annotation(
        x=4.2, y=2.5,
        text="1×5 conv",
        showarrow=False,
        font=dict(size=10, color='white'),
        row=1, col=2
    )
    
    fig.add_annotation(
        x=2.5, y=-0.8,
        text="共 10 个参数",
        showarrow=False,
        font=dict(size=11, color='#333'),
        row=1, col=2
    )
    
    # 添加效率对比
    fig.add_annotation(
        x=0.5, y=-0.5,
        xref='paper', yref='paper',
        text='计算量: 25次乘法',
        showarrow=False,
        font=dict(size=11, color=APPLE_BLUE)
    )
    
    fig.add_annotation(
        x=0.75, y=-0.5,
        xref='paper', yref='paper',
        text='计算量: 10次乘法 (节省60%)',
        showarrow=False,
        font=dict(size=11, color=APPLE_GREEN)
    )
    
    fig.update_layout(
        title='卷积核非对称分解：5×5 → 5×1 + 1×5',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        height=450,
        width=800,
        margin=dict(l=50, r=50, t=80, b=80)
    )
    
    for i in range(1, 3):
        fig.update_xaxes(range=[-0.5, 5.5], visible=False, row=1, col=i)
        fig.update_yaxes(range=[-1.5, 5.5], visible=False, row=1, col=i)
    
    return fig

def main():
    output_dir = '/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成各个图形
    print("生成图形...")
    
    # 1. Inception 演进时间线
    fig1 = plot_inception_evolution()
    save_and_compress(fig1, f'{output_dir}/inception-evolution.png')
    
    # 2. Inception-v4 整体架构
    fig2 = plot_inception_v4_architecture()
    save_and_compress(fig2, f'{output_dir}/inception-v4-architecture.png', width=1100, height=400)
    
    # 3. Inception 模块对比
    fig3 = plot_inception_modules_comparison()
    save_and_compress(fig3, f'{output_dir}/inception-modules.png', width=1000, height=450)
    
    # 4. 训练对比
    fig4 = plot_training_comparison()
    save_and_compress(fig4, f'{output_dir}/inception-training-comparison.png', width=900, height=500)
    
    # 5. 残差缩放
    fig5 = plot_residual_scaling()
    save_and_compress(fig5, f'{output_dir}/residual-scaling.png', width=900, height=400)
    
    # 6. 卷积分解
    fig6 = plot_filter_factorization()
    save_and_compress(fig6, f'{output_dir}/filter-factorization.png', width=800, height=450)
    
    print("\n所有图形生成完成！")

if __name__ == '__main__':
    main()
