#!/usr/bin/env python3
"""
生成 BERT 论文解读所需的 Plotly 图表
输出为 PNG 格式
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
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
APPLE_YELLOW = '#FFCC00'

def save_and_compress(fig, filepath, width=900, height=600, scale=2):
    """保存并压缩图片"""
    fig.write_image(filepath, width=width, height=height, scale=scale)
    
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False, capture_output=True)
    
    print(f"✅ 已保存: {filepath}")

def plot_attention_mechanism():
    """绘制注意力机制示意图"""
    fig = go.Figure()
    
    # 输入词向量位置
    input_words = ['Query', 'Key', 'Value']
    input_x = [0, 0, 0]
    input_y = [2, 1, 0]
    input_colors = [APPLE_BLUE, APPLE_ORANGE, APPLE_GREEN]
    
    # 绘制输入节点
    for x, y, word, color in zip(input_x, input_y, input_words, input_colors):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=65, color=color, line=dict(width=2, color='white')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial'),
            showlegend=False
        ))
    
    # Attention 计算框
    fig.add_trace(go.Scatter(
        x=[2.5], y=[1],
        mode='markers',
        marker=dict(size=80, color=APPLE_PURPLE, symbol='square',
                   line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2.5], y=[1],
        mode='text',
        text=['Attention<br>计算'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 输出
    fig.add_trace(go.Scatter(
        x=[5], y=[1],
        mode='markers',
        marker=dict(size=70, color=APPLE_RED, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[5], y=[1],
        mode='text',
        text=['输出'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 连接线
    # Query 和 Key 到 Attention
    for y in [2, 1]:
        fig.add_trace(go.Scatter(
            x=[0.4, 2.1], y=[y, 1.2],
            mode='lines',
            line=dict(color=APPLE_GRAY, width=2),
            showlegend=False
        ))
    # Value 到 Attention
    fig.add_trace(go.Scatter(
        x=[0.4, 2.1], y=[0, 0.8],
        mode='lines',
        line=dict(color=APPLE_GRAY, width=2),
        showlegend=False
    ))
    # Attention 到输出
    fig.add_trace(go.Scatter(
        x=[2.9, 4.6], y=[1, 1],
        mode='lines',
        line=dict(color=APPLE_GRAY, width=2.5),
        showlegend=False
    ))
    
    # 公式标注
    fig.add_annotation(x=1.3, y=2.5, text='Q', showarrow=False, font=dict(size=14))
    fig.add_annotation(x=1.3, y=1.5, text='K', showarrow=False, font=dict(size=14))
    fig.add_annotation(x=3.8, y=0.3, text='Attention(Q,K,V)', showarrow=False, font=dict(size=12))
    
    fig.update_layout(
        title=dict(text='注意力机制计算流程', font=dict(size=16, family='Arial')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 6.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 3.5]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    return fig

def plot_bert_architecture():
    """绘制 BERT 架构对比图（Bi-directional vs Left-to-Right）"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('传统语言模型（单向）', 'BERT（双向）'),
        horizontal_spacing=0.1
    )
    
    # 左图：单向
    words = ['[MASK]', '爱', '北京', '天安门']
    word_x = [0, 1, 2, 3]
    word_y = [0, 0, 0, 0]
    
    # 绘制词节点
    for x, y, word in zip(word_x, word_y, words):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=55, color=APPLE_BLUE, line=dict(width=2, color='white')),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial'),
            showlegend=False
        ), row=1, col=1)
    
    # 单向箭头（只能看左边）
    arrows = [(1, 0), (2, 1), (3, 2)]
    for end, start in arrows:
        fig.add_annotation(
            x=word_x[end]-0.25, y=0.15,
            ax=word_x[start]+0.25, ay=0.15,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=2, arrowsize=1.5, arrowwidth=2,
            arrowcolor=APPLE_GRAY,
            row=1, col=1
        )
    
    # 右图：双向
    for x, y, word in zip(word_x, word_y, words):
        color = APPLE_RED if word == '[MASK]' else APPLE_GREEN
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=55, color=color, line=dict(width=2, color='white')),
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial'),
            showlegend=False
        ), row=1, col=2)
    
    # 双向箭头（[MASK] 可以看到所有词）
    mask_idx = 0
    for i in [1, 2, 3]:
        # 从其他词指向 MASK
        fig.add_annotation(
            x=word_x[mask_idx]+0.1, y=0.25,
            ax=word_x[i]-0.1, ay=0.25,
            xref='x2', yref='y2', axref='x2', ayref='y2',
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
            arrowcolor=APPLE_ORANGE,
            row=1, col=2
        )
        # 双向箭头返回
        fig.add_annotation(
            x=word_x[i]-0.1, y=-0.25,
            ax=word_x[mask_idx]+0.1, ay=-0.25,
            xref='x2', yref='y2', axref='x2', ayref='y2',
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
            arrowcolor=APPLE_ORANGE,
            row=1, col=2
        )
    
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 3.5])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 0.8])
    
    fig.update_layout(
        title=dict(text='语言模型架构对比：单向 vs 双向', font=dict(size=16, family='Arial')),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=40, r=40, t=80, b=40),
        height=350
    )
    
    return fig

def plot_pretraining_tasks():
    """绘制 BERT 预训练任务对比"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('MLM: Masked Language Model', 'NSP: Next Sentence Prediction'),
        horizontal_spacing=0.12
    )
    
    # MLM 任务
    sentence = ['我', '[MASK]', '北京', '天安门']
    colors_mlm = [APPLE_BLUE, APPLE_RED, APPLE_GREEN, APPLE_GREEN]
    x_pos = [0, 1, 2, 3]
    
    for x, word, color in zip(x_pos, sentence, colors_mlm):
        fig.add_trace(go.Scatter(
            x=[x], y=[0],
            mode='markers',
            marker=dict(size=60, color=color, line=dict(width=2, color='white')),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[x], y=[0],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial'),
            showlegend=False
        ), row=1, col=1)
    
    # MLM 预测箭头
    fig.add_annotation(
        x=1, y=0.6,
        text='预测: 爱',
        showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_ORANGE,
        font=dict(size=12),
        row=1, col=1
    )
    
    # NSP 任务
    sent_a = ['今天', '天气', '很好']
    sent_b = ['我们', '去', '公园']
    
    # 句子A
    for i, word in enumerate(sent_a):
        fig.add_trace(go.Scatter(
            x=[i], y=[0.5],
            mode='markers',
            marker=dict(size=50, color=APPLE_BLUE, line=dict(width=2, color='white')),
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[i], y=[0.5],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial'),
            showlegend=False
        ), row=1, col=2)
    
    # 句子B
    for i, word in enumerate(sent_b):
        fig.add_trace(go.Scatter(
            x=[i + 0.5], y=[-0.5],
            mode='markers',
            marker=dict(size=50, color=APPLE_GREEN, line=dict(width=2, color='white')),
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[i + 0.5], y=[-0.5],
            mode='text',
            text=[word],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial'),
            showlegend=False
        ), row=1, col=2)
    
    # [SEP] 标记
    fig.add_trace(go.Scatter(
        x=[2.5], y=[0],
        mode='markers',
        marker=dict(size=45, color=APPLE_PURPLE, line=dict(width=2, color='white')),
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[2.5], y=[0],
        mode='text',
        text=['SEP'],
        textposition='middle center',
        textfont=dict(size=9, color='white', family='Arial'),
        showlegend=False
    ), row=1, col=2)
    
    # 预测箭头
    fig.add_annotation(
        x=2.5, y=1.2,
        text='IsNext?',
        showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_ORANGE,
        font=dict(size=12),
        row=1, col=2
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 3.5])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.5], row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.8], row=1, col=2)
    
    fig.update_layout(
        title=dict(text='BERT 预训练任务', font=dict(size=16, family='Arial')),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=40, r=40, t=80, b=40),
        height=380
    )
    
    return fig

def plot_transformer_encoder():
    """绘制 Transformer Encoder 结构"""
    fig = go.Figure()
    
    # 输入嵌入
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=80, color=APPLE_BLUE, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='text',
        text=['Input<br>Embedding'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial'),
        showlegend=False
    ))
    
    # Multi-Head Attention
    fig.add_trace(go.Scatter(
        x=[2], y=[1.5],
        mode='markers',
        marker=dict(size=75, color=APPLE_PURPLE, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2], y=[1.5],
        mode='text',
        text=['Multi-Head<br>Attention'],
        textposition='middle center',
        textfont=dict(size=9, color='white', family='Arial'),
        showlegend=False
    ))
    
    # Add & Norm
    fig.add_trace(go.Scatter(
        x=[3.5], y=[1.5],
        mode='markers',
        marker=dict(size=55, color=APPLE_CYAN, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[3.5], y=[1.5],
        mode='text',
        text=['Add&Norm'],
        textposition='middle center',
        textfont=dict(size=9, color='white', family='Arial'),
        showlegend=False
    ))
    
    # Feed Forward
    fig.add_trace(go.Scatter(
        x=[5], y=[1.5],
        mode='markers',
        marker=dict(size=70, color=APPLE_ORANGE, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[5], y=[1.5],
        mode='text',
        text=['Feed<br>Forward'],
        textposition='middle center',
        textfont=dict(size=9, color='white', family='Arial'),
        showlegend=False
    ))
    
    # Add & Norm 2
    fig.add_trace(go.Scatter(
        x=[6.5], y=[1.5],
        mode='markers',
        marker=dict(size=55, color=APPLE_CYAN, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[6.5], y=[1.5],
        mode='text',
        text=['Add&Norm'],
        textposition='middle center',
        textfont=dict(size=9, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 输出
    fig.add_trace(go.Scatter(
        x=[8], y=[0],
        mode='markers',
        marker=dict(size=80, color=APPLE_GREEN, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[8], y=[0],
        mode='text',
        text=['Output'],
        textposition='middle center',
        textfont=dict(size=11, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 连接线
    connections = [
        ([0.5, 1.5], [0.2, 1.2]),  # Input -> MHA
        ([2.5, 3.2], [1.5, 1.5]),   # MHA -> Add&Norm
        ([3.8, 4.6], [1.5, 1.5]),   # Add&Norm -> FF
        ([5.4, 6.2], [1.5, 1.5]),   # FF -> Add&Norm2
        ([6.8, 7.5], [1.2, 0.2]),   # Add&Norm2 -> Output
    ]
    for x_vals, y_vals in connections:
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines',
            line=dict(color=APPLE_GRAY, width=2),
            showlegend=False
        ))
    
    # 残差连接
    fig.add_trace(go.Scatter(
        x=[2, 3.5], y=[2.2, 2.2],
        mode='lines',
        line=dict(color=APPLE_YELLOW, width=2, dash='dash'),
        showlegend=False
    ))
    fig.add_annotation(x=2.75, y=2.4, text='残差连接', showarrow=False, font=dict(size=9))
    
    fig.update_layout(
        title=dict(text='Transformer Encoder 结构', font=dict(size=16, family='Arial')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 3]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    return fig

def plot_glue_benchmark():
    """绘制 GLUE 基准测试结果对比"""
    tasks = ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA', 'STS-B', 'MRPC', 'RTE', 'WNLI']
    bert_base = [84.6, 88.5, 90.5, 93.5, 52.1, 85.8, 88.9, 66.4, 71.2]
    bert_large = [86.7, 89.3, 92.3, 94.9, 60.5, 87.1, 89.3, 70.1, 73.5]
    previous_best = [80.6, 84.3, 87.4, 91.3, 45.0, 82.0, 84.0, 61.0, 65.0]
    
    fig = go.Figure()
    
    x = np.arange(len(tasks))
    width = 0.25
    
    fig.add_trace(go.Bar(
        name='Previous Best',
        x=[t + ' ' for t in tasks],
        y=previous_best,
        marker=dict(color=APPLE_GRAY, line=dict(width=1, color='white')),
        text=[f'{v:.1f}' for v in previous_best],
        textposition='outside',
        textfont=dict(size=9)
    ))
    
    fig.add_trace(go.Bar(
        name='BERT-Base',
        x=[t + ' ' for t in tasks],
        y=bert_base,
        marker=dict(color=APPLE_BLUE, line=dict(width=1, color='white')),
        text=[f'{v:.1f}' for v in bert_base],
        textposition='outside',
        textfont=dict(size=9)
    ))
    
    fig.add_trace(go.Bar(
        name='BERT-Large',
        x=[t + ' ' for t in tasks],
        y=bert_large,
        marker=dict(color=APPLE_GREEN, line=dict(width=1, color='white')),
        text=[f'{v:.1f}' for v in bert_large],
        textposition='outside',
        textfont=dict(size=9)
    ))
    
    fig.update_layout(
        title=dict(text='GLUE 基准测试结果对比', font=dict(size=16, family='Arial')),
        xaxis=dict(title='任务'),
        yaxis=dict(title='准确率 (%)', range=[0, 105]),
        barmode='group',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=11),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=20, t=80, b=80),
        height=450
    )
    
    return fig

def plot_fine_tuning_process():
    """绘制 BERT 微调流程"""
    fig = go.Figure()
    
    # 预训练阶段
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=90, color=APPLE_BLUE, line=dict(width=3, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='text',
        text=['预训练<br>BERT'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 箭头
    fig.add_annotation(
        x=1.5, y=0,
        ax=0.6, ay=0,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=2, arrowsize=2, arrowwidth=2.5,
        arrowcolor=APPLE_GRAY
    )
    
    # 微调阶段 - 任务1
    fig.add_trace(go.Scatter(
        x=[2.5], y=[1.5],
        mode='markers',
        marker=dict(size=75, color=APPLE_GREEN, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2.5], y=[1.5],
        mode='text',
        text=['情感分析'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 微调阶段 - 任务2
    fig.add_trace(go.Scatter(
        x=[2.5], y=[0],
        mode='markers',
        marker=dict(size=75, color=APPLE_ORANGE, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2.5], y=[0],
        mode='text',
        text=['问答系统'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 微调阶段 - 任务3
    fig.add_trace(go.Scatter(
        x=[2.5], y=[-1.5],
        mode='markers',
        marker=dict(size=75, color=APPLE_PURPLE, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[2.5], y=[-1.5],
        mode='text',
        text=['命名实体'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 从预训练到微调的连接线
    for y_target in [1.5, 0, -1.5]:
        fig.add_trace(go.Scatter(
            x=[0.5, 2.0], y=[0.1 * y_target, 0.7 * y_target],
            mode='lines',
            line=dict(color=APPLE_GRAY, width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(text='BERT 预训练 + 微调流程', font=dict(size=16, family='Arial')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 4]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    return fig

def main():
    """主函数：生成所有图表"""
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    plots = [
        (plot_attention_mechanism, 'bert_attention.png'),
        (plot_bert_architecture, 'bert_architecture.png'),
        (plot_pretraining_tasks, 'bert_pretraining.png'),
        (plot_transformer_encoder, 'bert_transformer.png'),
        (plot_glue_benchmark, 'bert_glue.png'),
        (plot_fine_tuning_process, 'bert_finetuning.png'),
    ]
    
    for plot_func, filename in plots:
        print(f"生成: {filename}")
        fig = plot_func()
        filepath = os.path.join(output_dir, filename)
        save_and_compress(fig, filepath)
    
    print("\n✅ 所有 BERT 图表生成完成!")

if __name__ == '__main__':
    main()
