#!/usr/bin/env python3
"""
生成 Word2Vec 文章配图
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import subprocess
import os


def save_and_compress(fig, filepath, width=900, height=600):
    """保存并压缩图片"""
    fig.write_image(filepath, scale=2, width=width, height=height)
    
    # 压缩 PNG
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force',
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")


def plot_word_analogy():
    """
    词向量类比可视化：king - man + woman ≈ queen
    """
    # 简化的二维投影示例
    np.random.seed(42)
    
    # 定义词语位置（示意性的二维投影）
    words = {
        'king': (2.5, 2.0),
        'queen': (2.5, -2.0),
        'man': (-2.5, 2.0),
        'woman': (-2.5, -2.0),
    }
    
    fig = go.Figure()
    
    # 颜色配置（苹果风格）
    colors = {
        'king': '#007AFF',
        'queen': '#007AFF',
        'man': '#34C759',
        'woman': '#34C759'
    }
    
    # 绘制词语点
    for word, (x, y) in words.items():
        # 节点圆圈
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=60, color=colors[word], opacity=0.8),
            name=word,
            showlegend=False
        ))
        # 文字标签
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=14, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 绘制关系箭头
    # king -> queen (gender axis)
    fig.add_annotation(
        x=2.5, y=-1.5, ax=2.5, ay=1.5,
        xref='x', yref='y', axref='x', ayref='y',
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='#FF9500'
    )
    
    # man -> woman (gender axis)
    fig.add_annotation(
        x=-2.5, y=-1.5, ax=-2.5, ay=1.5,
        xref='x', yref='y', axref='x', ayref='y',
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='#FF9500'
    )
    
    # king -> man (royalty axis)
    fig.add_annotation(
        x=-2.0, y=2.0, ax=2.0, ay=2.0,
        xref='x', yref='y', axref='x', ayref='y',
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='#5856D6'
    )
    
    # queen -> woman (royalty axis)
    fig.add_annotation(
        x=-2.0, y=-2.0, ax=2.0, ay=-2.0,
        xref='x', yref='y', axref='x', ayref='y',
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='#5856D6'
    )
    
    # 添加虚线表示类比关系
    fig.add_shape(type="line", x0=2.5, y0=2.0, x1=-2.5, y1=-2.0,
                  line=dict(color='#FF3B30', width=2, dash='dash'))
    
    # 添加公式标注
    fig.add_annotation(
        x=0, y=3.5,
        text='v<sub>king</sub> - v<sub>man</sub> + v<sub>woman</sub> ≈ v<sub>queen</sub>',
        showarrow=False,
        font=dict(size=16, color='#333333', family='Arial')
    )
    
    fig.update_layout(
        title=dict(
            text='词向量空间中的语义关系',
            font=dict(size=18, color='#333333', family='Arial')
        ),
        xaxis=dict(range=[-4, 4], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-4, 4], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=700,
        height=600,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def plot_cbow_architecture():
    """
    CBOW 架构示意图
    """
    fig = go.Figure()
    
    # 输入层（上下文词）
    input_y = [2.5, 1.5, -1.5, -2.5]
    input_labels = ['w<sub>t-2</sub>', 'w<sub>t-1</sub>', 'w<sub>t+1</sub>', 'w<sub>t+2</sub>']
    
    for i, (y, label) in enumerate(zip(input_y, input_labels)):
        # 输入节点
        fig.add_trace(go.Scatter(
            x=[1], y=[y],
            mode='markers',
            marker=dict(size=50, color='#007AFF', opacity=0.9),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[1], y=[y],
            mode='text',
            text=[label],
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 投影层（平均）
    fig.add_trace(go.Scatter(
        x=[3], y=[0],
        mode='markers',
        marker=dict(size=70, color='#34C759', opacity=0.9),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[3], y=[0],
        mode='text',
        text=['平均'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 输出层（中心词）
    fig.add_trace(go.Scatter(
        x=[5], y=[0],
        mode='markers',
        marker=dict(size=60, color='#FF9500', opacity=0.9),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[5], y=[0],
        mode='text',
        text=['w<sub>t</sub>'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 连接线
    for y in input_y:
        fig.add_shape(type="line", x0=1.3, y0=y, x1=2.7, y1=0,
                      line=dict(color='#CCCCCC', width=1.5))
    
    fig.add_shape(type="line", x0=3.35, y0=0, x1=4.65, y1=0,
                  line=dict(color='#CCCCCC', width=2))
    
    # 层标签
    fig.add_annotation(x=1, y=3.5, text='输入层', showarrow=False,
                       font=dict(size=13, color='#666666'))
    fig.add_annotation(x=3, y=3.5, text='投影层', showarrow=False,
                       font=dict(size=13, color='#666666'))
    fig.add_annotation(x=5, y=3.5, text='输出层', showarrow=False,
                       font=dict(size=13, color='#666666'))
    
    fig.update_layout(
        title=dict(
            text='CBOW 架构：用上下文预测中心词',
            font=dict(size=16, color='#333333', family='Arial')
        ),
        xaxis=dict(range=[0, 6], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-4, 4], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=600,
        height=450,
        showlegend=False,
        margin=dict(l=30, r=30, t=60, b=30)
    )
    
    return fig


def plot_skipgram_architecture():
    """
    Skip-gram 架构示意图
    """
    fig = go.Figure()
    
    # 输入层（中心词）
    fig.add_trace(go.Scatter(
        x=[1], y=[0],
        mode='markers',
        marker=dict(size=60, color='#FF9500', opacity=0.9),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[1], y=[0],
        mode='text',
        text=['w<sub>t</sub>'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 投影层
    fig.add_trace(go.Scatter(
        x=[3], y=[0],
        mode='markers',
        marker=dict(size=70, color='#34C759', opacity=0.9),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[3], y=[0],
        mode='text',
        text=['投影'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 输出层（上下文词）
    output_y = [2.5, 1.5, -1.5, -2.5]
    output_labels = ['w<sub>t-2</sub>', 'w<sub>t-1</sub>', 'w<sub>t+1</sub>', 'w<sub>t+2</sub>']
    
    for i, (y, label) in enumerate(zip(output_y, output_labels)):
        fig.add_trace(go.Scatter(
            x=[5], y=[y],
            mode='markers',
            marker=dict(size=50, color='#007AFF', opacity=0.9),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[5], y=[y],
            mode='text',
            text=[label],
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 连接线
    fig.add_shape(type="line", x0=1.35, y0=0, x1=2.65, y1=0,
                  line=dict(color='#CCCCCC', width=2))
    
    for y in output_y:
        fig.add_shape(type="line", x0=3.35, y0=0, x1=4.7, y1=y,
                      line=dict(color='#CCCCCC', width=1.5))
    
    # 层标签
    fig.add_annotation(x=1, y=3.5, text='输入层', showarrow=False,
                       font=dict(size=13, color='#666666'))
    fig.add_annotation(x=3, y=3.5, text='投影层', showarrow=False,
                       font=dict(size=13, color='#666666'))
    fig.add_annotation(x=5, y=3.5, text='输出层', showarrow=False,
                       font=dict(size=13, color='#666666'))
    
    fig.update_layout(
        title=dict(
            text='Skip-gram 架构：用中心词预测上下文',
            font=dict(size=16, color='#333333', family='Arial')
        ),
        xaxis=dict(range=[0, 6], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-4, 4], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=600,
        height=450,
        showlegend=False,
        margin=dict(l=30, r=30, t=60, b=30)
    )
    
    return fig


def plot_training_loss():
    """
    训练过程中损失函数的变化
    """
    np.random.seed(42)
    
    # 模拟训练曲线
    epochs = np.linspace(0, 5, 100)
    loss_cbow = 5 * np.exp(-0.8 * epochs) + 0.5 + np.random.normal(0, 0.05, 100)
    loss_skipgram = 6 * np.exp(-0.6 * epochs) + 0.6 + np.random.normal(0, 0.05, 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs, y=loss_cbow,
        mode='lines',
        name='CBOW',
        line=dict(color='#007AFF', width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=loss_skipgram,
        mode='lines',
        name='Skip-gram',
        line=dict(color='#FF9500', width=2.5)
    ))
    
    fig.update_layout(
        title=dict(
            text='Word2Vec 训练过程',
            font=dict(size=16, color='#333333', family='Arial')
        ),
        xaxis=dict(
            title='训练轮数 (Epoch)',
            titlefont=dict(size=13),
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor='#F0F0F0'
        ),
        yaxis=dict(
            title='负对数似然损失',
            titlefont=dict(size=13),
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor='#F0F0F0'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=700,
        height=450,
        legend=dict(
            x=0.7, y=0.95,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#E0E0E0',
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=60, b=50)
    )
    
    return fig


def plot_similarity_heatmap():
    """
    词向量相似度热力图示例
    """
    words = ['国王', '女王', '男人', '女人', '王子', '公主']
    
    # 模拟的余弦相似度矩阵（基于语义关系）
    similarity = np.array([
        [1.00, 0.85, 0.75, 0.65, 0.90, 0.70],
        [0.85, 1.00, 0.65, 0.80, 0.70, 0.92],
        [0.75, 0.65, 1.00, 0.88, 0.60, 0.55],
        [0.65, 0.80, 0.88, 1.00, 0.55, 0.75],
        [0.90, 0.70, 0.60, 0.55, 1.00, 0.78],
        [0.70, 0.92, 0.55, 0.75, 0.78, 1.00]
    ])
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity,
        x=words,
        y=words,
        colorscale='Blues',
        zmin=0, zmax=1,
        text=np.round(similarity, 2),
        texttemplate='%{text:.2f}',
        textfont=dict(size=10, color='#333333'),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(
            text='词向量余弦相似度',
            font=dict(size=16, color='#333333', family='Arial')
        ),
        xaxis=dict(
            tickfont=dict(size=11),
            side='bottom'
        ),
        yaxis=dict(
            tickfont=dict(size=11),
            autorange='reversed'
        ),
        width=550,
        height=500,
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    return fig


if __name__ == '__main__':
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成 Word2Vec 配图...")
    
    # 1. 词向量类比可视化
    fig1 = plot_word_analogy()
    save_and_compress(fig1, f'{output_dir}/word2vec-analogy.png')
    
    # 2. CBOW 架构
    fig2 = plot_cbow_architecture()
    save_and_compress(fig2, f'{output_dir}/word2vec-cbow.png')
    
    # 3. Skip-gram 架构
    fig3 = plot_skipgram_architecture()
    save_and_compress(fig3, f'{output_dir}/word2vec-skipgram.png')
    
    # 4. 训练曲线
    fig4 = plot_training_loss()
    save_and_compress(fig4, f'{output_dir}/word2vec-training.png')
    
    # 5. 相似度热力图
    fig5 = plot_similarity_heatmap()
    save_and_compress(fig5, f'{output_dir}/word2vec-similarity.png')
    
    print("\n所有配图生成完成！")
