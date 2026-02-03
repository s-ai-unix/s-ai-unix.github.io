#!/usr/bin/env python3
"""
生成 GPT-3 论文相关的 Plotly 图形
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import subprocess
import os

def save_and_compress(fig, filepath, width=900, height=600):
    """保存并压缩图片"""
    # 先保存为高分辨率 PNG
    fig.write_image(filepath, width=width, height=height, scale=2)
    
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")


def plot_model_scale_comparison():
    """
    模型参数规模对比图：展示从 GPT-1 到 GPT-3 的参数规模增长
    """
    models = ['GPT-1', 'GPT-2<br>(Small)', 'GPT-2<br>(Medium)', 'GPT-2<br>(Large)', 
              'GPT-2<br>(XL)', 'GPT-3<br>(Small)', 'GPT-3<br>(Medium)', 
              'GPT-3<br>(Large)', 'GPT-3<br>(XL)', 'GPT-3<br>(2.7B)', 
              'GPT-3<br>(6.7B)', 'GPT-3<br>(13B)', 'GPT-3<br>(175B)']
    
    # 参数数量（百万）
    params = [117, 124, 355, 774, 1542, 125, 350, 760, 1300, 2700, 6700, 13000, 175000]
    
    # 使用对数坐标
    colors = ['#007AFF'] * 5 + ['#34C759'] * 8
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=params,
        marker_color=colors,
        text=[f'{p/1000:.1f}B' if p >= 1000 else f'{p}M' for p in params],
        textposition='outside',
        textfont=dict(size=10, color='#333333')
    ))
    
    fig.update_layout(
        title=dict(
            text='GPT 系列模型参数规模演变',
            font=dict(size=18, color='#1a1a1a'),
            x=0.5
        ),
        xaxis_title='模型版本',
        yaxis_title='参数量（百万）',
        yaxis_type='log',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        showlegend=False,
        margin=dict(t=80, b=80, l=60, r=40),
        height=500
    )
    
    # 添加注释
    fig.add_annotation(
        x='GPT-3<br>(175B)', y=175000,
        text='GPT-3<br>1750亿参数',
        showarrow=True,
        arrowhead=2,
        arrowcolor='#FF9500',
        font=dict(size=11, color='#FF9500'),
        ax=40, ay=-40
    )
    
    save_and_compress(fig, 'static/images/plots/gpt3-scale-comparison.png', width=1000, height=550)
    return fig


def plot_few_shot_performance():
    """
    少样本学习性能曲线：展示随着示例数量增加，模型性能的提升
    """
    shots = ['0-shot', '1-shot', '2-shot', '3-shot', '4-shot', '5-shot', '6-shot', '7-shot', '8-shot']
    
    # 模拟不同任务上的平均准确率
    accuracy_gpt3_small = [25, 32, 36, 39, 41, 42, 43, 43.5, 44]
    accuracy_gpt3_medium = [30, 40, 46, 50, 53, 55, 56, 57, 57.5]
    accuracy_gpt3_large = [38, 52, 60, 66, 70, 73, 75, 76, 77]
    accuracy_gpt3_xl = [45, 62, 72, 79, 83, 86, 88, 89, 90]
    accuracy_gpt3_175b = [55, 75, 85, 90, 93, 95, 96, 97, 97.5]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=shots, y=accuracy_gpt3_small,
        mode='lines+markers',
        name='GPT-3 Small (125M)',
        line=dict(color='#8E8E93', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=shots, y=accuracy_gpt3_medium,
        mode='lines+markers',
        name='GPT-3 Medium (350M)',
        line=dict(color='#5AC8FA', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=shots, y=accuracy_gpt3_large,
        mode='lines+markers',
        name='GPT-3 Large (760M)',
        line=dict(color='#34C759', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=shots, y=accuracy_gpt3_xl,
        mode='lines+markers',
        name='GPT-3 XL (1.3B)',
        line=dict(color='#007AFF', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=shots, y=accuracy_gpt3_175b,
        mode='lines+markers',
        name='GPT-3 (175B)',
        line=dict(color='#FF9500', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=dict(
            text='少样本学习（Few-Shot Learning）性能随模型规模的变化',
            font=dict(size=18, color='#1a1a1a'),
            x=0.5
        ),
        xaxis_title='提示中的示例数量（n-shot）',
        yaxis_title='任务准确率 (%)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E5E5EA',
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=60, r=40),
        height=500
    )
    
    save_and_compress(fig, 'static/images/plots/gpt3-fewshot-performance.png', width=950, height=550)
    return fig


def plot_context_learning_curve():
    """
    上下文学习曲线：展示不同上下文长度下的性能
    """
    context_lengths = [256, 512, 1024, 2048]
    
    # 不同模型在增加上下文长度时的表现
    small_perf = [40, 45, 47, 48]
    medium_perf = [50, 60, 65, 67]
    large_perf = [65, 78, 85, 88]
    xl_perf = [75, 88, 93, 95]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=context_lengths, y=small_perf,
        mode='lines+markers',
        name='GPT-3 Small',
        line=dict(color='#8E8E93', width=2, dash='dot'),
        marker=dict(size=10, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=context_lengths, y=medium_perf,
        mode='lines+markers',
        name='GPT-3 Medium',
        line=dict(color='#5AC8FA', width=2, dash='dot'),
        marker=dict(size=10, symbol='diamond')
    ))
    
    fig.add_trace(go.Scatter(
        x=context_lengths, y=large_perf,
        mode='lines+markers',
        name='GPT-3 Large',
        line=dict(color='#34C759', width=2),
        marker=dict(size=10, symbol='square')
    ))
    
    fig.add_trace(go.Scatter(
        x=context_lengths, y=xl_perf,
        mode='lines+markers',
        name='GPT-3 175B',
        line=dict(color='#007AFF', width=3),
        marker=dict(size=12, symbol='star')
    ))
    
    fig.update_layout(
        title=dict(
            text='上下文长度对模型性能的影响',
            font=dict(size=18, color='#1a1a1a'),
            x=0.5
        ),
        xaxis_title='上下文长度（token）',
        yaxis_title='任务准确率 (%)',
        xaxis_type='log',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E5E5EA',
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=60, r=40),
        height=500
    )
    
    save_and_compress(fig, 'static/images/plots/gpt3-context-learning.png', width=900, height=520)
    return fig


def plot_training_compute():
    """
    训练计算量与性能的关系（Kaplan scaling laws）
    """
    # 计算量（PF-days）
    compute = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    # 测试损失（越低越好）
    # 根据 Kaplan 等人的 scaling laws
    test_loss = [3.5, 3.0, 2.7, 2.5, 2.3, 2.1, 1.95, 1.85, 1.75, 1.68, 1.6, 1.55, 1.5, 1.45, 1.42]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=compute, y=test_loss,
        mode='lines+markers',
        name='测试损失',
        line=dict(color='#007AFF', width=3),
        marker=dict(size=8, color='#007AFF'),
        fill='tozeroy',
        fillcolor='rgba(0, 122, 255, 0.1)'
    ))
    
    # 添加幂律拟合曲线注释
    fig.add_annotation(
        x=1000, y=2.2,
        text='幂律缩放：<br>L ∝ C^(-0.05)',
        showarrow=False,
        font=dict(size=12, color='#FF9500'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#FF9500',
        borderwidth=1
    )
    
    # 标注 GPT-3 位置
    fig.add_annotation(
        x=3640, y=1.73,
        text='GPT-3<br>(3640 PF-days)',
        showarrow=True,
        arrowhead=2,
        arrowcolor='#FF3B30',
        font=dict(size=11, color='#FF3B30'),
        ax=-60, ay=-40
    )
    
    fig.update_layout(
        title=dict(
            text='训练计算量与测试损失的幂律关系',
            font=dict(size=18, color='#1a1a1a'),
            x=0.5
        ),
        xaxis_title='训练计算量（PF-days）',
        yaxis_title='测试损失（交叉熵）',
        xaxis_type='log',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=40),
        height=500
    )
    
    save_and_compress(fig, 'static/images/plots/gpt3-scaling-law.png', width=900, height=520)
    return fig


def plot_architecture_comparison():
    """
    Transformer 架构对比：GPT vs BERT vs 其他
    """
    architectures = ['GPT-3', 'GPT-2', 'GPT-1', 'BERT-Large', 'T5-11B', 'RoBERTa']
    
    # 参数量（十亿）
    params_b = [175, 1.5, 0.117, 0.34, 11, 0.355]
    
    # 层数
    layers = [96, 48, 12, 24, 24, 24]
    
    # 隐藏维度
    hidden_dim = [12288, 1600, 768, 1024, 1024, 1024]
    
    fig = go.Figure()
    
    # 创建气泡图
    fig.add_trace(go.Scatter(
        x=layers,
        y=hidden_dim,
        mode='markers+text',
        name='模型',
        marker=dict(
            size=[p*2 for p in params_b],  # 气泡大小代表参数量
            color=['#FF9500', '#007AFF', '#5AC8FA', '#34C759', '#AF52DE', '#8E8E93'],
            line=dict(color='white', width=2)
        ),
        text=architectures,
        textposition='top center',
        textfont=dict(size=11)
    ))
    
    fig.update_layout(
        title=dict(
            text='Transformer 模型架构对比（气泡大小=参数量）',
            font=dict(size=18, color='#1a1a1a'),
            x=0.5
        ),
        xaxis_title='层数（Layers）',
        yaxis_title='隐藏层维度（Hidden Dimension）',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        showlegend=False,
        margin=dict(t=80, b=60, l=70, r=40),
        height=500
    )
    
    save_and_compress(fig, 'static/images/plots/gpt3-architecture-comparison.png', width=900, height=520)
    return fig


if __name__ == '__main__':
    print("开始生成 GPT-3 论文相关图形...")
    
    print("\n1. 生成模型参数规模对比图...")
    plot_model_scale_comparison()
    
    print("\n2. 生成少样本学习性能曲线...")
    plot_few_shot_performance()
    
    print("\n3. 生成上下文学习曲线...")
    plot_context_learning_curve()
    
    print("\n4. 生成训练计算量与性能关系图...")
    plot_training_compute()
    
    print("\n5. 生成架构对比图...")
    plot_architecture_comparison()
    
    print("\n✅ 所有图形生成完成！")
