#!/usr/bin/env python3
"""
生成 Seq2Seq 论文解读的可视化图形
使用 Plotly 生成专业的数理图形
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os

def save_and_compress(fig, filepath, width=900, height=600):
    """保存并压缩图片"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存图片
    fig.write_image(filepath, width=width, height=height, scale=2)
    
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False, capture_output=True)
    
    print(f"✅ 已保存并压缩: {filepath}")


def plot_encoder_decoder_architecture():
    """
    绘制 Seq2Seq 编码器-解码器架构示意图
    """
    fig = go.Figure()
    
    # 编码器部分
    encoder_x = [1, 2, 3, 4]
    encoder_y = [3, 3, 3, 3]
    encoder_labels = ['$x_1$', '$x_2$', '$x_3$', '$\\langle\\text{EOS}\\rangle$']
    
    for i, (x, y, label) in enumerate(zip(encoder_x, encoder_y, encoder_labels)):
        # 输入节点
        fig.add_trace(go.Scatter(
            x=[x], y=[y+0.8],
            mode='markers',
            marker=dict(size=45, color='#007AFF', line=dict(width=2, color='#0051D5')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y+0.8],
            mode='text',
            text=[label],
            textposition='middle center',
            textfont=dict(size=14, color='white', family='Arial'),
            showlegend=False
        ))
        
        # LSTM 单元
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=55, color='#34C759', line=dict(width=2, color='#248A3D')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=['LSTM'],
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial'),
            showlegend=False
        ))
        
        # 连接线
        if i < len(encoder_x) - 1:
            fig.add_trace(go.Scatter(
                x=[x+0.3, x+0.7], y=[y, y],
                mode='lines',
                line=dict(color='#8E8E93', width=2),
                showlegend=False
            ))
    
    # 上下文向量
    fig.add_trace(go.Scatter(
        x=[5.5], y=[3],
        mode='markers',
        marker=dict(size=70, color='#FF9500', line=dict(width=3, color='#B86E00')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[5.5], y=[3],
        mode='text',
        text=['上下文<br>向量 $c$'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 连接编码器到上下文
    fig.add_trace(go.Scatter(
        x=[4.3, 5.1], y=[3, 3],
        mode='lines',
        line=dict(color='#007AFF', width=3),
        showlegend=False
    ))
    
    # 解码器部分
    decoder_x = [7, 8, 9, 10]
    decoder_y = [3, 3, 3, 3]
    decoder_labels = ['$y_1$', '$y_2$', '$y_3$', '$\\langle\\text{EOS}\\rangle$']
    
    for i, (x, y, label) in enumerate(zip(decoder_x, decoder_y, decoder_labels)):
        # LSTM 单元
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=55, color='#34C759', line=dict(width=2, color='#248A3D')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=['LSTM'],
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial'),
            showlegend=False
        ))
        
        # 输出节点
        fig.add_trace(go.Scatter(
            x=[x], y=[y-0.8],
            mode='markers',
            marker=dict(size=45, color='#AF52DE', line=dict(width=2, color='#7B3491')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y-0.8],
            mode='text',
            text=[label],
            textposition='middle center',
            textfont=dict(size=14, color='white', family='Arial'),
            showlegend=False
        ))
        
        # 连接线
        if i < len(decoder_x) - 1:
            fig.add_trace(go.Scatter(
                x=[x+0.3, x+0.7], y=[y, y],
                mode='lines',
                line=dict(color='#8E8E93', width=2),
                showlegend=False
            ))
    
    # 连接上下文到解码器
    fig.add_trace(go.Scatter(
        x=[5.9, 6.7], y=[3, 3],
        mode='lines',
        line=dict(color='#007AFF', width=3),
        showlegend=False
    ))
    
    # 添加标签
    fig.add_trace(go.Scatter(
        x=[2.5], y=[4.5],
        mode='text',
        text=['编码器 (Encoder)'],
        textfont=dict(size=16, color='#007AFF', family='Arial', weight='bold'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[8.5], y=[4.5],
        mode='text',
        text=['解码器 (Decoder)'],
        textfont=dict(size=16, color='#007AFF', family='Arial', weight='bold'),
        showlegend=False
    ))
    
    # 添加输入输出标注
    fig.add_trace(go.Scatter(
        x=[1], y=[5],
        mode='text',
        text=['输入序列: "Hello"'],
        textfont=dict(size=12, color='#333333', family='Arial'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[10], y=[1.5],
        mode='text',
        text=['输出序列: "Bonjour"'],
        textfont=dict(size=12, color='#333333', family='Arial'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text='Seq2Seq 编码器-解码器架构',
            font=dict(size=18, family='Arial', color='#1D1D1F')
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 11]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0.5, 5.5]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif'),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        width=900,
        height=500
    )
    
    save_and_compress(fig, 'static/images/plots/seq2seq-architecture.png', width=900, height=500)


def plot_lstm_cell():
    """
    绘制 LSTM 单元内部结构
    """
    fig = go.Figure()
    
    # 主单元框
    fig.add_shape(
        type="rect",
        x0=2, y0=1, x1=6, y1=5,
        fillcolor="#F5F5F7",
        line=dict(color="#007AFF", width=2),
        layer="below"
    )
    
    # 遗忘门
    fig.add_trace(go.Scatter(
        x=[3], y=[4.2],
        mode='markers',
        marker=dict(size=50, color='#FF3B30', line=dict(width=2, color='#C41E3A')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[3], y=[4.2],
        mode='text',
        text=['$\\sigma$'],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[3], y=[4.9],
        mode='text',
        text=['遗忘门'],
        textposition='top center',
        textfont=dict(size=10, color='#FF3B30', family='Arial'),
        showlegend=False
    ))
    
    # 输入门
    fig.add_trace(go.Scatter(
        x=[3], y=[3],
        mode='markers',
        marker=dict(size=50, color='#34C759', line=dict(width=2, color='#248A3D')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[3], y=[3],
        mode='text',
        text=['$\\sigma$'],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2.2], y=[3],
        mode='text',
        text=['输入门'],
        textposition='middle right',
        textfont=dict(size=10, color='#34C759', family='Arial'),
        showlegend=False
    ))
    
    # 候选状态 tanh
    fig.add_trace(go.Scatter(
        x=[4], y=[3],
        mode='markers',
        marker=dict(size=50, color='#AF52DE', line=dict(width=2, color='#7B3491')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[4], y=[3],
        mode='text',
        text=['tanh'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 输出门
    fig.add_trace(go.Scatter(
        x=[5], y=[2],
        mode='markers',
        marker=dict(size=50, color='#FF9500', line=dict(width=2, color='#B86E00')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[5], y=[2],
        mode='text',
        text=['$\\sigma$'],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[5.8], y=[2],
        mode='text',
        text=['输出门'],
        textposition='middle left',
        textfont=dict(size=10, color='#FF9500', family='Arial'),
        showlegend=False
    ))
    
    # 细胞状态线
    fig.add_trace(go.Scatter(
        x=[1.5, 3, 3, 5, 5, 6.5], y=[4.2, 4.2, 4.2, 4.2, 4.2, 4.2],
        mode='lines',
        line=dict(color='#007AFF', width=4),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[1.2], y=[4.2],
        mode='text',
        text=['$C_{t-1}$'],
        textfont=dict(size=12, color='#007AFF', family='Arial'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[6.8], y=[4.2],
        mode='text',
        text=['$C_t$'],
        textfont=dict(size=12, color='#007AFF', family='Arial'),
        showlegend=False
    ))
    
    # 隐藏状态线
    fig.add_trace(go.Scatter(
        x=[1.5, 2, 2, 5, 5, 5, 6.5], y=[1.5, 1.5, 1.5, 1.5, 1.5, 2, 2],
        mode='lines',
        line=dict(color='#34C759', width=3),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[1.2], y=[1.5],
        mode='text',
        text=['$h_{t-1}$'],
        textfont=dict(size=12, color='#34C759', family='Arial'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[6.8], y=[2],
        mode='text',
        text=['$h_t$'],
        textfont=dict(size=12, color='#34C759', family='Arial'),
        showlegend=False
    ))
    
    # 输入 x_t
    fig.add_trace(go.Scatter(
        x=[1.5, 2], y=[0.5, 0.5],
        mode='lines',
        line=dict(color='#8E8E93', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[1.2], y=[0.5],
        mode='text',
        text=['$x_t$'],
        textfont=dict(size=12, color='#8E8E93', family='Arial'),
        showlegend=False
    ))
    
    # 连接线
    # h_{t-1} 连接到各个门
    fig.add_trace(go.Scatter(
        x=[2, 2, 3], y=[1.5, 4.5, 4.5],
        mode='lines',
        line=dict(color='#8E8E93', width=1.5, dash='dot'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2, 3], y=[1.5, 3.4],
        mode='lines',
        line=dict(color='#8E8E93', width=1.5, dash='dot'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2, 2, 4], y=[1.5, 0.8, 0.8],
        mode='lines',
        line=dict(color='#8E8E93', width=1.5, dash='dot'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2, 2, 5, 5], y=[1.5, 0.3, 0.3, 1.6],
        mode='lines',
        line=dict(color='#8E8E93', width=1.5, dash='dot'),
        showlegend=False
    ))
    
    # x_t 连接
    fig.add_trace(go.Scatter(
        x=[2, 2, 3], y=[0.5, 0.5, 3.6],
        mode='lines',
        line=dict(color='#8E8E93', width=1.5, dash='dot'),
        showlegend=False
    ))
    
    # 输出 tanh
    fig.add_trace(go.Scatter(
        x=[5], y=[3],
        mode='markers',
        marker=dict(size=40, color='#5AC8FA', line=dict(width=2, color='#007AFF')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[5], y=[3],
        mode='text',
        text=['tanh'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 细胞状态到输出 tanh
    fig.add_trace(go.Scatter(
        x=[5, 5], y=[4.2, 3.4],
        mode='lines',
        line=dict(color='#007AFF', width=2),
        showlegend=False
    ))
    
    # 输出门到 h_t
    fig.add_trace(go.Scatter(
        x=[5, 5], y=[2.4, 2],
        mode='lines',
        line=dict(color='#FF9500', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text='LSTM 单元内部结构',
            font=dict(size=18, family='Arial', color='#1D1D1F')
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 8]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 6]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif'),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        width=800,
        height=500
    )
    
    save_and_compress(fig, 'static/images/plots/lstm-cell-structure.png', width=800, height=500)


def plot_attention_mechanism():
    """
    绘制注意力机制示意图
    """
    fig = go.Figure()
    
    # 编码器隐藏状态
    encoder_x = [1, 1, 1, 1]
    encoder_y = [1, 2.5, 4, 5.5]
    encoder_labels = ['$h_4$', '$h_3$', '$h_2$', '$h_1$']
    encoder_text = ['"jour"', '"bon"', '"Bon"', '"SOS"']
    
    for x, y, label, text in zip(encoder_x, encoder_y, encoder_labels, encoder_text):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=50, color='#007AFF', line=dict(width=2, color='#0051D5')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[label],
            textposition='middle center',
            textfont=dict(size=12, color='white', family='Arial'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x-0.5], y=[y],
            mode='text',
            text=[text],
            textposition='middle right',
            textfont=dict(size=10, color='#007AFF', family='Arial'),
            showlegend=False
        ))
    
    # 注意力权重
    attention_weights = [0.1, 0.2, 0.5, 0.2]
    for i, (y, weight) in enumerate(zip(encoder_y, attention_weights)):
        line_width = weight * 10
        alpha = 0.3 + weight * 0.7
        fig.add_trace(go.Scatter(
            x=[1.4, 3.6], y=[y, 3.5],
            mode='lines',
            line=dict(color=f'rgba(255, 149, 0, {alpha})', width=line_width),
            showlegend=False
        ))
        # 权重标注
        fig.add_trace(go.Scatter(
            x=[2.5], y=[y + (3.5 - y) * 0.3],
            mode='text',
            text=[f'{weight}'],
            textfont=dict(size=9, color='#FF9500', family='Arial'),
            showlegend=False
        ))
    
    # 上下文向量
    fig.add_trace(go.Scatter(
        x=[4], y=[3.5],
        mode='markers',
        marker=dict(size=60, color='#FF9500', line=dict(width=3, color='#B86E00')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[4], y=[3.5],
        mode='text',
        text=['$c_t$'],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 解码器隐藏状态
    fig.add_trace(go.Scatter(
        x=[6], y=[3.5],
        mode='markers',
        marker=dict(size=55, color='#34C759', line=dict(width=2, color='#248A3D')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[6], y=[3.5],
        mode='text',
        text=['$s_t$'],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 连接上下文到解码器
    fig.add_trace(go.Scatter(
        x=[4.4, 5.6], y=[3.5, 3.5],
        mode='lines',
        line=dict(color='#34C759', width=3),
        showlegend=False
    ))
    
    # 输出
    fig.add_trace(go.Scatter(
        x=[7.5], y=[3.5],
        mode='markers',
        marker=dict(size=50, color='#AF52DE', line=dict(width=2, color='#7B3491')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[7.5], y=[3.5],
        mode='text',
        text=['"jour"'],
        textposition='middle center',
        textfont=dict(size=11, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 连接解码器到输出
    fig.add_trace(go.Scatter(
        x=[6.4, 7.1], y=[3.5, 3.5],
        mode='lines',
        line=dict(color='#AF52DE', width=2),
        showlegend=False
    ))
    
    # 标题标签
    fig.add_trace(go.Scatter(
        x=[1], y=[6.5],
        mode='text',
        text=['编码器隐藏状态'],
        textfont=dict(size=12, color='#007AFF', family='Arial', weight='bold'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[4], y=[5],
        mode='text',
        text=['上下文向量<br>(加权和)'],
        textfont=dict(size=10, color='#FF9500', family='Arial'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[6], y=[5],
        mode='text',
        text=['解码器状态'],
        textfont=dict(size=12, color='#34C759', family='Arial', weight='bold'),
        showlegend=False
    ))
    
    # 公式说明
    fig.add_trace(go.Scatter(
        x=[4], y=[0.5],
        mode='text',
        text=['$c_t = \\sum_{i=1}^{n} \\alpha_{ti} h_i$'],
        textfont=dict(size=14, color='#333333', family='Arial'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text='注意力机制：编码器-解码器对齐',
            font=dict(size=18, family='Arial', color='#1D1D1F')
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 9]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 7]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif'),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        width=900,
        height=550
    )
    
    save_and_compress(fig, 'static/images/plots/attention-mechanism.png', width=900, height=550)


def plot_bleu_score_comparison():
    """
    绘制 BLEU 分数对比图（展示 Seq2Seq 相比传统方法的优势）
    """
    fig = go.Figure()
    
    methods = ['SMT<br>Baseline', 'Neural<br>LM', 'RNN<br>Encoder<br>Decoder', 'Seq2Seq<br>+ LSTM<br>+ Reverse']
    bleu_scores = [30.6, 31.5, 31.8, 34.8]
    colors = ['#8E8E93', '#5AC8FA', '#007AFF', '#34C759']
    
    fig.add_trace(go.Bar(
        x=methods,
        y=bleu_scores,
        marker=dict(color=colors, line=dict(width=2, color='#1D1D1F')),
        text=[f'{s:.1f}' for s in bleu_scores],
        textposition='outside',
        textfont=dict(size=14, color='#1D1D1F', family='Arial'),
        showlegend=False
    ))
    
    # 添加基准线
    fig.add_hline(y=33.3, line_dash="dash", line_color="#FF3B30", line_width=2,
                  annotation_text="Best WMT'14 Submission", 
                  annotation_position="right",
                  annotation_font=dict(color="#FF3B30", size=11))
    
    fig.update_layout(
        title=dict(
            text='WMT\'14 English to French BLEU Score Comparison',
            font=dict(size=16, family='Arial', color='#1D1D1F')
        ),
        xaxis=dict(
            title='Method',
            titlefont=dict(size=12, family='Arial'),
            tickfont=dict(size=10, family='Arial')
        ),
        yaxis=dict(
            title='BLEU Score',
            titlefont=dict(size=12, family='Arial'),
            tickfont=dict(size=11, family='Arial'),
            range=[28, 37],
            gridcolor='#E5E5EA'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif'),
        margin=dict(l=60, r=120, t=60, b=60),
        width=800,
        height=500,
        bargap=0.3
    )
    
    save_and_compress(fig, 'static/images/plots/bleu-score-comparison.png', width=800, height=500)


def plot_sequence_transduction():
    """
    绘制序列转导问题的可视化
    """
    fig = go.Figure()
    
    # 输入序列
    input_words = ['Hello', 'world', 'this', 'is', 'a', 'test']
    input_x = list(range(len(input_words)))
    input_y = [2] * len(input_words)
    
    for i, (x, word) in enumerate(zip(input_x, input_words)):
        fig.add_trace(go.Scatter(
            x=[x], y=[input_y[0]],
            mode='markers',
            marker=dict(size=55, color='#007AFF', line=dict(width=2, color='#0051D5')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[input_y[0]],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 输出序列
    output_words = ['Bonjour', 'monde', 'ceci', 'est', 'un', 'test']
    output_x = list(range(len(output_words)))
    output_y = [0] * len(output_words)
    
    for i, (x, word) in enumerate(zip(output_x, output_words)):
        fig.add_trace(go.Scatter(
            x=[x], y=[output_y[0]],
            mode='markers',
            marker=dict(size=55, color='#34C759', line=dict(width=2, color='#248A3D')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[output_y[0]],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 添加对齐连线（显示顺序可能不同）
    for i in range(len(input_words)):
        fig.add_trace(go.Scatter(
            x=[i, i], y=[1.7, 0.3],
            mode='lines',
            line=dict(color='#FF9500', width=2, dash='dot'),
            opacity=0.6,
            showlegend=False
        ))
    
    # 添加标签
    fig.add_trace(go.Scatter(
        x=[-1], y=[2],
        mode='text',
        text=['输入:'],
        textposition='middle right',
        textfont=dict(size=12, color='#007AFF', family='Arial', weight='bold'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[-1], y=[0],
        mode='text',
        text=['输出:'],
        textposition='middle right',
        textfont=dict(size=12, color='#34C759', family='Arial', weight='bold'),
        showlegend=False
    ))
    
    # 添加问题标注
    fig.add_trace(go.Scatter(
        x=[2.5], y=[3],
        mode='text',
        text=['问题：输入输出长度可能不同，词序可能变化'],
        textfont=dict(size=11, color='#FF3B30', family='Arial'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[2.5], y=[-1],
        mode='text',
        text=['$P(y_1, ..., y_{T^{\prime}} | x_1, ..., x_T) = \\prod_{t=1}^{T^{\prime}} P(y_t | c, y_1, ..., y_{t-1})$'],
        textfont=dict(size=11, color='#333333', family='Arial'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text='序列转导问题：从一种序列映射到另一种序列',
            font=dict(size=16, family='Arial', color='#1D1D1F')
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-2, 7]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-2, 4]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif'),
        showlegend=False,
        margin=dict(l=80, r=20, t=60, b=20),
        width=900,
        height=400
    )
    
    save_and_compress(fig, 'static/images/plots/sequence-transduction.png', width=900, height=400)


def plot_rnn_limitations():
    """
    绘制传统 RNN 的局限性（梯度消失）
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=['梯度消失问题', '长程依赖困难'])
    
    # 左图：梯度随时间的衰减
    timesteps = np.arange(1, 51)
    grad_vanish = np.exp(-0.1 * timesteps)
    grad_normal = np.ones_like(timesteps) * 0.5
    
    fig.add_trace(go.Scatter(
        x=timesteps, y=grad_vanish,
        mode='lines',
        name='梯度 (sigmoid)',
        line=dict(color='#FF3B30', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 59, 48, 0.2)'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=timesteps, y=grad_normal,
        mode='lines',
        name='理想梯度',
        line=dict(color='#34C759', width=2, dash='dash')
    ), row=1, col=1)
    
    # 右图：长距离依赖的信号衰减
    distances = np.arange(1, 31)
    info_retention = np.exp(-0.08 * distances)
    
    fig.add_trace(go.Scatter(
        x=distances, y=info_retention,
        mode='lines+markers',
        name='信息保留率',
        line=dict(color='#007AFF', width=3),
        marker=dict(size=6, color='#007AFF'),
        fill='tozeroy',
        fillcolor='rgba(0, 122, 255, 0.2)'
    ), row=1, col=2)
    
    # 添加临界线
    fig.add_hline(y=0.1, line_dash="dot", line_color="#FF9500", line_width=2,
                  annotation_text="有效阈值", row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text='RNN 训练中的梯度消失与长程依赖问题',
            font=dict(size=16, family='Arial', color='#1D1D1F')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif'),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=40, t=80, b=50),
        width=900,
        height=400
    )
    
    fig.update_xaxes(title_text='时间步', gridcolor='#E5E5EA', row=1, col=1)
    fig.update_xaxes(title_text='序列距离', gridcolor='#E5E5EA', row=1, col=2)
    fig.update_yaxes(title_text='梯度大小', gridcolor='#E5E5EA', row=1, col=1)
    fig.update_yaxes(title_text='信息保留比例', gridcolor='#E5E5EA', row=1, col=2)
    
    save_and_compress(fig, 'static/images/plots/rnn-limitations.png', width=900, height=400)


if __name__ == '__main__':
    print("开始生成 Seq2Seq 可视化图形...")
    
    plot_encoder_decoder_architecture()
    plot_lstm_cell()
    plot_attention_mechanism()
    plot_bleu_score_comparison()
    plot_sequence_transduction()
    plot_rnn_limitations()
    
    print("\n所有图形生成完成！")
