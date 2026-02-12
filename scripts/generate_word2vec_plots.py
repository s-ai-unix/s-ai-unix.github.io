#!/usr/bin/env python3
"""
生成 Word2Vec 相关的可视化图形
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os

def save_and_compress(fig, filepath):
    """保存并压缩图片"""
    fig.write_image(filepath, scale=2)
    
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")


def plot_one_hot_vs_dense():
    """对比独热编码和稠密向量表示"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('独热编码 (One-Hot)', '稠密向量 (Word2Vec)'),
        horizontal_spacing=0.15
    )
    
    # 独热编码示例
    words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    vocab_size = len(words)
    one_hot = np.eye(vocab_size)
    
    fig.add_trace(
        go.Heatmap(
            z=one_hot,
            x=[f'dim_{i+1}' for i in range(vocab_size)],
            y=words,
            colorscale=[[0, '#F2F2F7'], [1, '#007AFF']],
            showscale=False,
            text=one_hot.astype(int),
            texttemplate='%{text}',
            textfont={'size': 14}
        ),
        row=1, col=1
    )
    
    # 稠密向量示例（模拟词向量）
    np.random.seed(42)
    dense_vectors = np.array([
        [0.8, 0.6, -0.3],   # apple (水果，圆形，甜味)
        [0.7, 0.4, -0.2],   # banana (水果，长形，甜味)
        [0.9, 0.3, -0.4],   # cherry (水果，圆形，酸甜)
        [0.2, 0.5, 0.8],    # date (水果，但特征不同)
        [0.85, 0.4, -0.35], # elderberry (水果，圆形)
    ])
    
    fig.add_trace(
        go.Heatmap(
            z=dense_vectors,
            x=['语义维度1', '语义维度2', '语义维度3'],
            y=words,
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            colorbar=dict(title='值', x=0.97),
            text=np.round(dense_vectors, 2),
            texttemplate='%{text}',
            textfont={'size': 12}
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='词表示方式的演进：从稀疏到稠密',
        title_font_size=16,
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        height=400,
        width=900
    )
    
    fig.update_xaxes(side='top')
    
    return fig


def plot_cbow_architecture():
    """CBOW 模型架构图"""
    fig = go.Figure()
    
    # 颜色定义（苹果风格）
    colors = {
        'input': '#007AFF',      # 蓝色
        'projection': '#34C759',  # 绿色
        'output': '#FF9500',      # 橙色
        'text': '#FFFFFF'
    }
    
    # 输入层 - 上下文词
    input_words = ['the', 'cat', 'on', 'mat']
    input_x = [1, 1, 1, 1]
    input_y = [3, 2, 1, 0]
    
    for i, word in enumerate(input_words):
        # 圆圈
        fig.add_trace(go.Scatter(
            x=[input_x[i]],
            y=[input_y[i]],
            mode='markers',
            marker=dict(size=45, color=colors['input']),
            showlegend=False
        ))
        # 文字
        fig.add_trace(go.Scatter(
            x=[input_x[i]],
            y=[input_y[i]],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=11, color=colors['text']),
            showlegend=False
        ))
    
    # 投影层/隐藏层
    fig.add_trace(go.Scatter(
        x=[2.5], y=[1.5],
        mode='markers',
        marker=dict(size=60, color=colors['projection'], symbol='diamond'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2.5], y=[1.5],
        mode='text',
        text=['投影层'],
        textposition='middle center',
        textfont=dict(size=11, color=colors['text']),
        showlegend=False
    ))
    
    # 输出层
    fig.add_trace(go.Scatter(
        x=[4], y=[1.5],
        mode='markers',
        marker=dict(size=55, color=colors['output']),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[4], y=[1.5],
        mode='text',
        text=['sat'],
        textposition='middle center',
        textfont=dict(size=12, color=colors['text']),
        showlegend=False
    ))
    
    # 连接线
    for y in input_y:
        fig.add_trace(go.Scatter(
            x=[1.3, 2.2],
            y=[y, 1.5],
            mode='lines',
            line=dict(color='#C7C7CC', width=1.5),
            showlegend=False
        ))
    
    fig.add_trace(go.Scatter(
        x=[2.8, 3.7],
        y=[1.5, 1.5],
        mode='lines',
        line=dict(color='#C7C7CC', width=2),
        showlegend=False
    ))
    
    # 层标签
    fig.add_annotation(x=1, y=4.2, text='输入层', showarrow=False, font=dict(size=14))
    fig.add_annotation(x=2.5, y=3.0, text='投影层\n(平均)', showarrow=False, font=dict(size=12))
    fig.add_annotation(x=4, y=3.0, text='输出层', showarrow=False, font=dict(size=14))
    
    # 标题
    fig.add_annotation(x=2.5, y=-0.8, 
                       text='输入: ["the", "cat", "on", "mat"] → 预测: "sat"', 
                       showarrow=False, font=dict(size=13))
    
    fig.update_layout(
        title='CBOW (Continuous Bag of Words) 架构',
        title_font_size=16,
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        xaxis=dict(range=[0, 5], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-1.5, 5], showgrid=False, showticklabels=False, zeroline=False),
        height=450,
        width=700,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def plot_skipgram_architecture():
    """Skip-gram 模型架构图"""
    fig = go.Figure()
    
    colors = {
        'input': '#007AFF',
        'projection': '#34C759',
        'output': '#FF9500',
        'text': '#FFFFFF'
    }
    
    # 输入层 - 中心词
    fig.add_trace(go.Scatter(
        x=[1], y=[1.5],
        mode='markers',
        marker=dict(size=55, color=colors['input']),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[1], y=[1.5],
        mode='text',
        text=['sat'],
        textposition='middle center',
        textfont=dict(size=12, color=colors['text']),
        showlegend=False
    ))
    
    # 投影层
    fig.add_trace(go.Scatter(
        x=[2.5], y=[1.5],
        mode='markers',
        marker=dict(size=60, color=colors['projection'], symbol='diamond'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2.5], y=[1.5],
        mode='text',
        text=['投影层'],
        textposition='middle center',
        textfont=dict(size=11, color=colors['text']),
        showlegend=False
    ))
    
    # 输出层 - 多个上下文词
    output_words = ['the', 'cat', 'on', 'mat']
    output_x = [4, 4, 4, 4]
    output_y = [3, 2, 1, 0]
    
    for i, word in enumerate(output_words):
        fig.add_trace(go.Scatter(
            x=[output_x[i]],
            y=[output_y[i]],
            mode='markers',
            marker=dict(size=45, color=colors['output']),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[output_x[i]],
            y=[output_y[i]],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=11, color=colors['text']),
            showlegend=False
        ))
    
    # 连接线
    fig.add_trace(go.Scatter(
        x=[1.3, 2.2],
        y=[1.5, 1.5],
        mode='lines',
        line=dict(color='#C7C7CC', width=2),
        showlegend=False
    ))
    
    for y in output_y:
        fig.add_trace(go.Scatter(
            x=[2.8, 3.7],
            y=[1.5, y],
            mode='lines',
            line=dict(color='#C7C7CC', width=1.5),
            showlegend=False
        ))
    
    # 层标签
    fig.add_annotation(x=1, y=3.2, text='输入层', showarrow=False, font=dict(size=14))
    fig.add_annotation(x=2.5, y=3.0, text='投影层', showarrow=False, font=dict(size=12))
    fig.add_annotation(x=4, y=4.2, text='输出层', showarrow=False, font=dict(size=14))
    
    fig.add_annotation(x=2.5, y=-0.8, 
                       text='输入: "sat" → 预测: ["the", "cat", "on", "mat"]', 
                       showarrow=False, font=dict(size=13))
    
    fig.update_layout(
        title='Skip-gram 架构',
        title_font_size=16,
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        xaxis=dict(range=[0, 5], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-1.5, 5], showgrid=False, showticklabels=False, zeroline=False),
        height=450,
        width=700,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def plot_training_loss():
    """训练损失随时间变化"""
    np.random.seed(42)
    iterations = np.arange(0, 1001, 10)
    
    # 模拟训练损失
    base_loss = 8.0
    loss_cbow = base_loss * np.exp(-iterations / 300) + 0.5 + np.random.normal(0, 0.05, len(iterations))
    loss_skipgram = base_loss * np.exp(-iterations / 250) + 0.3 + np.random.normal(0, 0.08, len(iterations))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=loss_cbow,
        mode='lines',
        name='CBOW',
        line=dict(color='#007AFF', width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=loss_skipgram,
        mode='lines',
        name='Skip-gram',
        line=dict(color='#34C759', width=2.5)
    ))
    
    fig.update_layout(
        title='训练损失随迭代次数变化',
        xaxis_title='迭代次数',
        yaxis_title='负对数似然损失',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.75, y=0.95),
        height=450,
        width=750
    )
    
    return fig


def plot_word_analogy():
    """词类比关系的向量可视化"""
    # 模拟词向量（2D投影）
    words = {
        'king': np.array([2.0, 1.5]),
        'queen': np.array([2.2, -0.5]),
        'man': np.array([0.5, 1.5]),
        'woman': np.array([0.7, -0.5]),
    }
    
    fig = go.Figure()
    
    # 绘制点
    for word, vec in words.items():
        color = '#007AFF' if word in ['king', 'queen'] else '#34C759'
        fig.add_trace(go.Scatter(
            x=[vec[0]],
            y=[vec[1]],
            mode='markers+text',
            marker=dict(size=20, color=color),
            text=[word],
            textposition='top center',
            textfont=dict(size=14),
            showlegend=False
        ))
    
    # 绘制向量箭头
    # king - man + woman = queen
    fig.add_annotation(
        x=words['man'][0], y=words['man'][1],
        ax=words['king'][0], ay=words['king'][1],
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='#FF9500'
    )
    
    fig.add_annotation(
        x=words['woman'][0], y=words['woman'][1],
        ax=words['man'][0], ay=words['man'][1],
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='#FF9500'
    )
    
    # 结果向量（虚线）
    fig.add_trace(go.Scatter(
        x=[words['king'][0], words['king'][0] - words['man'][0] + words['woman'][0]],
        y=[words['king'][1], words['king'][1] - words['man'][1] + words['woman'][1]],
        mode='lines',
        line=dict(color='#FF3B30', width=2, dash='dash'),
        showlegend=False
    ))
    
    # 标注关系
    fig.add_annotation(x=1.3, y=2.5, text='king - man ≈ queen - woman', 
                       showarrow=False, font=dict(size=14, color='#FF3B30'))
    fig.add_annotation(x=1.3, y=2.2, text='king - man + woman ≈ queen', 
                       showarrow=False, font=dict(size=13, color='#FF3B30'))
    
    fig.update_layout(
        title='词向量类比：king - man + woman ≈ queen',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        xaxis=dict(range=[-0.5, 3], showgrid=True),
        yaxis=dict(range=[-1.5, 3], showgrid=True),
        height=500,
        width=700
    )
    
    return fig


def plot_hierarchical_softmax():
    """分层 Softmax 结构"""
    fig = go.Figure()
    
    # 二叉树节点位置
    nodes = {
        'root': (0, 0),
        'L': (-2, -1),
        'R': (2, -1),
        'LL': (-3, -2),
        'LR': (-1, -2),
        'RL': (1, -2),
        'RR': (3, -2),
    }
    
    colors = {
        'internal': '#007AFF',
        'leaf': '#34C759',
        'text': '#FFFFFF'
    }
    
    # 绘制边
    edges = [
        ('root', 'L'), ('root', 'R'),
        ('L', 'LL'), ('L', 'LR'),
        ('R', 'RL'), ('R', 'RR'),
    ]
    
    for parent, child in edges:
        x0, y0 = nodes[parent]
        x1, y1 = nodes[child]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color='#C7C7CC', width=2),
            showlegend=False
        ))
    
    # 绘制节点
    for name, (x, y) in nodes.items():
        if name in ['LL', 'LR', 'RL', 'RR']:
            color = colors['leaf']
            size = 35
            label = f'w_{list(nodes.keys()).index(name)-2}'
        else:
            color = colors['internal']
            size = 40
            label = 'n' if name == 'root' else name
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=size, color=color),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[label],
            textposition='middle center',
            textfont=dict(size=12, color=colors['text']),
            showlegend=False
        ))
    
    # 添加标签
    fig.add_annotation(x=-1, y=-0.3, text='左(0)', showarrow=False, font=dict(size=11))
    fig.add_annotation(x=1, y=-0.3, text='右(1)', showarrow=False, font=dict(size=11))
    fig.add_annotation(x=-2.5, y=-1.3, text='0', showarrow=False, font=dict(size=11))
    fig.add_annotation(x=-1.5, y=-1.3, text='1', showarrow=False, font=dict(size=11))
    fig.add_annotation(x=1.5, y=-1.3, text='0', showarrow=False, font=dict(size=11))
    fig.add_annotation(x=2.5, y=-1.3, text='1', showarrow=False, font=dict(size=11))
    
    # 路径示例
    fig.add_annotation(x=-3, y=-2.7, text='编码: 00', showarrow=False, font=dict(size=10))
    fig.add_annotation(x=-1, y=-2.7, text='编码: 01', showarrow=False, font=dict(size=10))
    fig.add_annotation(x=1, y=-2.7, text='编码: 10', showarrow=False, font=dict(size=10))
    fig.add_annotation(x=3, y=-2.7, text='编码: 11', showarrow=False, font=dict(size=10))
    
    fig.update_layout(
        title='Hierarchical Softmax 二叉树结构',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        xaxis=dict(range=[-4, 4], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-3.5, 1], showgrid=False, showticklabels=False, zeroline=False),
        height=400,
        width=700,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def plot_negative_sampling():
    """负采样示意图"""
    fig = go.Figure()
    
    # 正样本和负样本
    context_words = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    target = 'sat'
    
    # 可视化
    y_pos = 0
    colors = {
        'positive': '#34C759',
        'negative': '#FF3B30',
        'neutral': '#8E8E93'
    }
    
    # 中心词
    fig.add_trace(go.Scatter(
        x=[0], y=[y_pos],
        mode='markers+text',
        marker=dict(size=50, color='#007AFF'),
        text=['"sat"'],
        textposition='top center',
        textfont=dict(size=14),
        showlegend=False
    ))
    
    # 正样本（上下文）
    for i, word in enumerate(['"the"', '"cat"', '"on"', '"mat"']):
        x_pos = 2 + i * 1.5
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[y_pos + 0.5],
            mode='markers+text',
            marker=dict(size=40, color=colors['positive']),
            text=[word],
            textposition='top center',
            textfont=dict(size=11),
            showlegend=False
        ))
        # 连线
        fig.add_trace(go.Scatter(
            x=[0.4, x_pos - 0.3],
            y=[y_pos + 0.1, y_pos + 0.4],
            mode='lines',
            line=dict(color=colors['positive'], width=2),
            showlegend=False
        ))
    
    # 负样本（噪声）
    negative_words = ['"car"', '"blue"', '"run"']
    for i, word in enumerate(negative_words):
        x_pos = 2 + i * 1.5
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[y_pos - 0.8],
            mode='markers+text',
            marker=dict(size=40, color=colors['negative']),
            text=[word],
            textposition='bottom center',
            textfont=dict(size=11),
            showlegend=False
        ))
        # 虚线连接
        fig.add_trace(go.Scatter(
            x=[0.4, x_pos - 0.3],
            y=[y_pos - 0.1, y_pos - 0.7],
            mode='lines',
            line=dict(color=colors['negative'], width=2, dash='dash'),
            showlegend=False
        ))
    
    # 图例
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color=colors['positive']),
        name='正样本 (上下文)'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color=colors['negative']),
        name='负样本 (噪声)'
    ))
    
    fig.update_layout(
        title='负采样：区分正样本与随机负样本',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        xaxis=dict(range=[-1, 8], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-2, 2], showgrid=False, showticklabels=False, zeroline=False),
        height=400,
        width=800,
        legend=dict(x=0.7, y=0.95),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def main():
    """生成所有图形"""
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 独热编码 vs 稠密向量
    fig1 = plot_one_hot_vs_dense()
    save_and_compress(fig1, f'{output_dir}/word2vec_onehot_vs_dense.png')
    
    # 2. CBOW 架构
    fig2 = plot_cbow_architecture()
    save_and_compress(fig2, f'{output_dir}/word2vec_cbow_arch.png')
    
    # 3. Skip-gram 架构
    fig3 = plot_skipgram_architecture()
    save_and_compress(fig3, f'{output_dir}/word2vec_skipgram_arch.png')
    
    # 4. 训练损失
    fig4 = plot_training_loss()
    save_and_compress(fig4, f'{output_dir}/word2vec_training_loss.png')
    
    # 5. 词类比
    fig5 = plot_word_analogy()
    save_and_compress(fig5, f'{output_dir}/word2vec_analogy.png')
    
    # 6. Hierarchical Softmax
    fig6 = plot_hierarchical_softmax()
    save_and_compress(fig6, f'{output_dir}/word2vec_hierarchical_softmax.png')
    
    # 7. 负采样
    fig7 = plot_negative_sampling()
    save_and_compress(fig7, f'{output_dir}/word2vec_negative_sampling.png')
    
    print("\n所有图形生成完成！")


if __name__ == '__main__':
    main()
