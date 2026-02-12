#!/usr/bin/env python3
"""
生成 Llama 3 论文解读文章的可视化图形
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import subprocess
import os

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

# 苹果风格配色
APPLE_BLUE = '#007AFF'
APPLE_GREEN = '#34C759'
APPLE_ORANGE = '#FF9500'
APPLE_RED = '#FF3B30'
APPLE_PURPLE = '#AF52DE'
APPLE_CYAN = '#5AC8FA'

def plot_scaling_laws():
    """绘制 Scaling Laws 曲线 - 计算最优与过训练"""
    
    # 训练数据量 (tokens)
    tokens = np.array([1e12, 5e12, 1e13, 5e13, 1e14, 5e14, 1e15, 1.56e13, 1.56e13])
    
    # 模型参数量 (parameters)
    params_chinchilla = np.array([1e9, 5e9, 1e10, 5e10, 1e11, 5e11, 1e12, 4.05e11, 4.05e11])
    
    # Chinchilla 最优曲线 (L(N,D) = A/N^alpha + B/D^beta)
    # 使用简化的 scaling law: loss ∝ N^(-0.5) + D^(-0.5)
    loss_chinchilla = 2.0 * (params_chinchilla / 1e9) ** (-0.5) + 0.5 * (tokens / 1e12) ** (-0.5)
    
    # Llama 3 405B 的位置标记
    llama3_tokens = 15.6e12  # 15.6T tokens
    llama3_params = 405e9    # 405B params
    
    fig = go.Figure()
    
    # 绘制 Chinchilla 最优线
    fig.add_trace(go.Scatter(
        x=params_chinchilla[:-2] / 1e9,
        y=loss_chinchilla[:-2],
        mode='lines',
        name='Chinchilla 最优曲线',
        line=dict(color=APPLE_BLUE, width=3, dash='dash'),
        hovertemplate='参数量: %{x:.1f}B<br>损失: %{y:.4f}<extra></extra>'
    ))
    
    # 标记 Llama 3 各模型
    models = {
        'Llama 3 8B': (8, 15.6, APPLE_GREEN),
        'Llama 3 70B': (70, 15.6, APPLE_ORANGE),
        'Llama 3 405B': (405, 15.6, APPLE_RED),
    }
    
    for name, (p, t, color) in models.items():
        # 计算该模型的理论损失 (简化公式)
        loss = 2.0 * (p) ** (-0.5) + 0.5 * (t) ** (-0.5)
        fig.add_trace(go.Scatter(
            x=[p],
            y=[loss],
            mode='markers+text',
            name=name,
            marker=dict(size=25, color=color, symbol='circle'),
            text=[name],
            textposition='top center',
            textfont=dict(size=12, color='#333'),
            hovertemplate=f'{name}<br>参数量: {p}B<br>训练数据: {t}T tokens<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Scaling Laws: 计算最优 vs Llama 3 过训练策略',
            font=dict(size=18, family='Arial, sans-serif'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='模型参数量 (B)', font=dict(size=14)),
            type='log',
            tickfont=dict(size=12),
            gridcolor='#E5E5E5'
        ),
        yaxis=dict(
            title=dict(text='验证损失', font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='#E5E5E5'
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        width=900,
        height=600,
        margin=dict(t=100, b=80, l=80, r=50)
    )
    
    # 添加注释说明过训练
    fig.add_annotation(
        x=405, y=0.35,
        text='Llama 3 405B<br>训练数据量 15.6T<br>远超计算最优配置',
        showarrow=True,
        arrowhead=2,
        arrowcolor=APPLE_RED,
        ax=80,
        ay=-60,
        font=dict(size=11, color=APPLE_RED),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=APPLE_RED,
        borderwidth=1
    )
    
    save_and_compress(fig, 'static/images/plots/llama3-scaling-laws.png')
    return fig

def plot_context_length_evolution():
    """绘制上下文长度演化和损失曲线"""
    
    # 训练阶段
    stages = ['预训练\n(8K)', '继续预训练\n(128K)']
    context_lengths = [8192, 131072]
    
    # 各阶段损失 (示意数据)
    train_loss = [2.1, 2.15]
    val_loss_short = [2.15, 2.18]  # 短上下文验证
    val_loss_long = [None, 2.25]   # 长上下文验证
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('上下文长度扩展', '长上下文性能'),
        horizontal_spacing=0.15
    )
    
    # 左图：上下文长度对比
    fig.add_trace(
        go.Bar(
            x=stages,
            y=context_lengths,
            marker_color=[APPLE_BLUE, APPLE_GREEN],
            text=[f'{c/1024:.0f}K' for c in context_lengths],
            textposition='outside',
            textfont=dict(size=14),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 右图：不同长度下的性能
    lengths = np.array([4096, 8192, 16384, 32768, 65536, 131072])
    # 模拟 needle-in-haystack 准确率
    accuracy = np.array([99.5, 99.2, 98.8, 98.0, 97.5, 96.8])
    
    fig.add_trace(
        go.Scatter(
            x=lengths / 1024,
            y=accuracy,
            mode='lines+markers',
            name='检索准确率',
            line=dict(color=APPLE_BLUE, width=3),
            marker=dict(size=10, color=APPLE_BLUE),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(
            text='Llama 3 长上下文能力演化',
            font=dict(size=18, family='Arial, sans-serif'),
            x=0.5
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=1000,
        height=500,
        margin=dict(t=100, b=80, l=80, r=50)
    )
    
    # 更新坐标轴
    fig.update_yaxes(title_text='上下文长度 (tokens)', type='log', row=1, col=1)
    fig.update_yaxes(title_text='Needle-in-Haystack 准确率 (%)', row=1, col=2)
    fig.update_xaxes(title_text='训练阶段', row=1, col=1)
    fig.update_xaxes(title_text='上下文长度 (K tokens)', row=1, col=2)
    
    save_and_compress(fig, 'static/images/plots/llama3-context-evolution.png')
    return fig

def plot_model_comparison():
    """绘制模型性能对比图"""
    
    benchmarks = ['MMLU', 'HumanEval', 'GSM8K', 'MATH', 'GPQA']
    
    # 各模型得分 (百分比)
    llama3_8b = [68.4, 72.6, 84.5, 30.4, 32.8]
    llama3_70b = [79.5, 80.5, 95.1, 52.8, 46.7]
    llama3_405b = [85.1, 89.0, 96.8, 73.8, 51.1]
    gpt4 = [86.4, 87.6, 92.0, 52.9, 47.1]  # GPT-4 参考
    
    fig = go.Figure()
    
    x = np.arange(len(benchmarks))
    width = 0.18
    
    # 添加各模型条形图
    fig.add_trace(go.Bar(
        name='Llama 3 8B',
        x=benchmarks,
        y=llama3_8b,
        marker_color=APPLE_CYAN,
        text=[f'{v:.1f}' for v in llama3_8b],
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    fig.add_trace(go.Bar(
        name='Llama 3 70B',
        x=benchmarks,
        y=llama3_70b,
        marker_color=APPLE_GREEN,
        text=[f'{v:.1f}' for v in llama3_70b],
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    fig.add_trace(go.Bar(
        name='Llama 3 405B',
        x=benchmarks,
        y=llama3_405b,
        marker_color=APPLE_RED,
        text=[f'{v:.1f}' for v in llama3_405b],
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    fig.add_trace(go.Bar(
        name='GPT-4 (参考)',
        x=benchmarks,
        y=gpt4,
        marker_color=APPLE_PURPLE,
        marker_pattern_shape='/',
        text=[f'{v:.1f}' for v in gpt4],
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    fig.update_layout(
        title=dict(
            text='Llama 3 系列与 GPT-4 性能对比',
            font=dict(size=18, family='Arial, sans-serif'),
            x=0.5
        ),
        xaxis=dict(
            title='基准测试',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='准确率 (%)',
            range=[0, 105],
            gridcolor='#E5E5E5'
        ),
        barmode='group',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        width=1000,
        height=600,
        margin=dict(t=120, b=80, l=80, r=50)
    )
    
    save_and_compress(fig, 'static/images/plots/llama3-benchmark-comparison.png')
    return fig

def plot_training_compute():
    """绘制训练计算量分布"""
    
    # 训练阶段
    stages = ['预训练\n(8K)', '继续预训练\n(128K)', '退火\n(高质量)', 'SFT', 'DPO']
    
    # 各阶段消耗的 FLOPs (估算)
    flops = [3.5e25, 2.5e24, 5e23, 2e23, 1e23]
    
    # 转换为百分比
    total = sum(flops)
    percentages = [f/sum(flops) * 100 for f in flops]
    
    colors = [APPLE_BLUE, APPLE_GREEN, APPLE_CYAN, APPLE_ORANGE, APPLE_PURPLE]
    
    fig = go.Figure(data=[go.Pie(
        labels=stages,
        values=percentages,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12),
        hovertemplate='%{label}<br>%{percent}<br>FLOPs: %{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text='Llama 3 405B 训练计算量分布',
            font=dict(size=18, family='Arial, sans-serif'),
            x=0.5
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.05
        ),
        width=800,
        height=600,
        margin=dict(t=80, b=50, l=50, r=200),
        annotations=[dict(
            text=f'总计<br>3.8×10²⁵<br>FLOPs',
            x=0.5, y=0.5,
            font=dict(size=14, color='#333'),
            showarrow=False
        )]
    )
    
    save_and_compress(fig, 'static/images/plots/llama3-training-compute.png')
    return fig

if __name__ == '__main__':
    print("正在生成 Llama 3 论文可视化图形...")
    
    print("\n1. 生成 Scaling Laws 曲线...")
    plot_scaling_laws()
    
    print("\n2. 生成上下文长度演化图...")
    plot_context_length_evolution()
    
    print("\n3. 生成模型性能对比图...")
    plot_model_comparison()
    
    print("\n4. 生成训练计算量分布图...")
    plot_training_compute()
    
    print("\n✅ 所有图形生成完成！")
    
    # 列出生成的文件
    print("\n生成的文件:")
    for f in os.listdir('static/images/plots'):
        if f.startswith('llama3'):
            size = os.path.getsize(f'static/images/plots/{f}')
            print(f"  - {f}: {size/1024:.1f} KB")
