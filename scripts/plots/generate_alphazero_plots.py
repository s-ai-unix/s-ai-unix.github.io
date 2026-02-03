#!/usr/bin/env python3
"""
生成 AlphaZero 论文解读所需的 Plotly 图表
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

def plot_alphazero_vs_alphago():
    """绘制 AlphaZero vs AlphaGo 对比"""
    fig = go.Figure()
    
    # AlphaGo 流程
    alphago_steps = [
        ('人类棋谱', 0, 2, APPLE_BLUE),
        ('监督学习', 2, 2, APPLE_ORANGE),
        ('策略网络', 4, 2, APPLE_PURPLE),
        ('强化学习', 6, 2, APPLE_GREEN),
        ('AlphaGo', 8, 2, APPLE_RED),
    ]
    
    # AlphaZero 流程
    alphazero_steps = [
        ('随机初始化', 0, 0, APPLE_GRAY),
        ('自我对弈', 2, 0, APPLE_CYAN),
        ('神经网络', 4, 0, APPLE_PURPLE),
        ('MCTS', 6, 0, APPLE_ORANGE),
        ('AlphaZero', 8, 0, APPLE_GREEN),
    ]
    
    # 绘制 AlphaGo
    for name, x, y, color in alphago_steps:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=55, color=color, line=dict(width=2, color='white')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[name],
            textposition='middle center',
            textfont=dict(size=9, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 绘制 AlphaZero
    for name, x, y, color in alphazero_steps:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=55, color=color, line=dict(width=2, color='white')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[name],
            textposition='middle center',
            textfont=dict(size=9, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 连接线
    for i in range(len(alphago_steps) - 1):
        # AlphaGo 连线
        fig.add_trace(go.Scatter(
            x=[alphago_steps[i][1] + 0.35, alphago_steps[i+1][1] - 0.35],
            y=[alphago_steps[i][2], alphago_steps[i+1][2]],
            mode='lines',
            line=dict(color=APPLE_GRAY, width=2),
            showlegend=False
        ))
        # AlphaZero 连线
        fig.add_trace(go.Scatter(
            x=[alphazero_steps[i][1] + 0.35, alphazero_steps[i+1][1] - 0.35],
            y=[alphazero_steps[i][2], alphazero_steps[i+1][2]],
            mode='lines',
            line=dict(color=APPLE_GRAY, width=2),
            showlegend=False
        ))
    
    # 标签
    fig.add_annotation(x=4, y=2.8, text='AlphaGo', showarrow=False, 
                      font=dict(size=14, color=APPLE_RED))
    fig.add_annotation(x=4, y=-0.8, text='AlphaZero', showarrow=False,
                      font=dict(size=14, color=APPLE_GREEN))
    
    fig.update_layout(
        title=dict(text='AlphaGo vs AlphaZero 训练流程对比', font=dict(size=16, family='Arial')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 3.5]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    return fig

def plot_mcts_puct():
    """绘制 MCTS + PUCT 算法流程"""
    fig = go.Figure()
    
    # 四个阶段节点
    stages = [
        ('选择', 0, 0, APPLE_BLUE),
        ('扩展', 2.5, 0, APPLE_ORANGE),
        ('评估', 5, 0, APPLE_PURPLE),
        ('回溯', 7.5, 0, APPLE_GREEN),
    ]
    
    for name, x, y, color in stages:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=70, color=color, line=dict(width=2, color='white')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[name],
            textposition='middle center',
            textfont=dict(size=12, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 循环箭头
    for i in range(len(stages)):
        next_i = (i + 1) % len(stages)
        x_start = stages[i][1] + 0.45
        x_end = stages[next_i][1] - 0.45
        y_start = stages[i][2]
        y_end = stages[next_i][2]
        
        # 如果是从回溯到选择，画曲线
        if i == 3:
            fig.add_trace(go.Scatter(
                x=[7.5, 8.2, 8.2, 0, 0],
                y=[0.5, 0.5, 1.5, 1.5, 0.5],
                mode='lines',
                line=dict(color=APPLE_GRAY, width=2),
                showlegend=False
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[x_start, x_end],
                y=[y_start, y_end],
                mode='lines',
                line=dict(color=APPLE_GRAY, width=2.5),
                showlegend=False
            ))
            # 箭头
            fig.add_annotation(
                x=x_end, y=y_end,
                ax=x_start, ay=y_end,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True,
                arrowhead=2, arrowsize=1.5, arrowwidth=2,
                arrowcolor=APPLE_GRAY
            )
    
    # 详细说明
    details = [
        ('PUCT 选择动作', 0, -1, APPLE_BLUE),
        ('添加新节点', 2.5, -1, APPLE_ORANGE),
        ('神经网络评估', 5, -1, APPLE_PURPLE),
        ('更新统计信息', 7.5, -1, APPLE_GREEN),
    ]
    
    for text, x, y, color in details:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[text],
            textposition='middle center',
            textfont=dict(size=9, color=color, family='Arial'),
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(text='MCTS 四个阶段循环', font=dict(size=16, family='Arial')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.8, 2]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=20, r=20, t=60, b=40),
        height=380
    )
    
    return fig

def plot_neural_network_structure():
    """绘制 AlphaZero 神经网络结构"""
    fig = go.Figure()
    
    # 输入层 - 棋盘状态
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=85, color=APPLE_BLUE, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='text',
        text=['棋盘状态<br>8×8×k'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 残差块 xN
    for i in range(3):
        fig.add_trace(go.Scatter(
            x=[1.5 + i*0.8], y=[0],
            mode='markers',
            marker=dict(size=45, color=APPLE_PURPLE, line=dict(width=2, color='white')),
            showlegend=False
        ))
    
    # ResNet Block 标签
    fig.add_trace(go.Scatter(
        x=[2.3], y=[0.8],
        mode='text',
        text=['ResNet Blocks x19/39'],
        textposition='middle center',
        textfont=dict(size=9, color=APPLE_PURPLE, family='Arial'),
        showlegend=False
    ))
    
    # 省略号
    fig.add_trace(go.Scatter(
        x=[4], y=[0],
        mode='text',
        text=['...'],
        textposition='middle center',
        textfont=dict(size=14, color=APPLE_GRAY, family='Arial'),
        showlegend=False
    ))
    
    # 策略头
    fig.add_trace(go.Scatter(
        x=[5.5], y=[1.2],
        mode='markers',
        marker=dict(size=65, color=APPLE_GREEN, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[5.5], y=[1.2],
        mode='text',
        text=['策略头<br>p(a|s)'],
        textposition='middle center',
        textfont=dict(size=9, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 价值头
    fig.add_trace(go.Scatter(
        x=[5.5], y=[-1.2],
        mode='markers',
        marker=dict(size=65, color=APPLE_ORANGE, line=dict(width=2, color='white')),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[5.5], y=[-1.2],
        mode='text',
        text=['价值头<br>v(s)'],
        textposition='middle center',
        textfont=dict(size=9, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 连接线
    connections = [
        ([0.5, 1.2], [0, 0]),  # input to resnet
        ([3.8, 4.3], [0, 0]),  # resnet to ...
        ([4.3, 5.1], [0, 1.0]),  # ... to policy
        ([4.3, 5.1], [0, -1.0]),  # ... to value
    ]
    for x_vals, y_vals in connections:
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines',
            line=dict(color=APPLE_GRAY, width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(text='AlphaZero 双头 ResNet 架构', font=dict(size=16, family='Arial')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 7]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.2, 2.2]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    return fig

def plot_training_progress():
    """绘制训练过程中 Elo 评级变化"""
    fig = go.Figure()
    
    # 模拟训练过程中的 Elo 提升
    np.random.seed(42)
    hours = np.arange(0, 24, 0.5)
    
    # 国际象棋 - 快速提升
    chess_elo = 3000 + 800 * (1 - np.exp(-hours/4)) + np.random.normal(0, 30, len(hours))
    
    # 将棋 - 中等提升
    shogi_elo = 3000 + 600 * (1 - np.exp(-hours/5)) + np.random.normal(0, 25, len(hours))
    
    # 围棋 - 较慢提升（因为 AlphaZero 从零开始）
    go_elo = 2000 + 1200 * (1 - np.exp(-hours/6)) + np.random.normal(0, 40, len(hours))
    
    fig.add_trace(go.Scatter(
        x=hours, y=chess_elo,
        mode='lines',
        name='国际象棋',
        line=dict(color=APPLE_BLUE, width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=shogi_elo,
        mode='lines',
        name='将棋',
        line=dict(color=APPLE_GREEN, width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=go_elo,
        mode='lines',
        name='围棋',
        line=dict(color=APPLE_PURPLE, width=2.5)
    ))
    
    # 添加 Stockfish 参考线
    fig.add_hline(y=3590, line=dict(color=APPLE_RED, width=2, dash='dash'),
                  annotation_text='Stockfish 8', annotation_position='right')
    
    fig.update_layout(
        title=dict(text='AlphaZero 训练过程中 Elo 评级提升', font=dict(size=16, family='Arial')),
        xaxis=dict(title='训练时间（小时）'),
        yaxis=dict(title='Elo 评级'),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=120, t=80, b=50),
        height=450
    )
    
    return fig

def plot_games_comparison():
    """绘制三种棋类游戏的对比"""
    fig = go.Figure()
    
    games = ['国际象棋', '将棋', '围棋']
    branching = [35, 80, 250]  # 分支因子
    game_length = [80, 115, 150]  # 平均步数
    complexity = [123, 226, 360]  # log10(复杂度)
    
    # 归一化到 0-100 范围用于可视化
    branching_norm = np.array(branching) / max(branching) * 100
    length_norm = np.array(game_length) / max(game_length) * 100
    
    fig.add_trace(go.Bar(
        name='分支因子',
        x=games,
        y=branching,
        marker=dict(color=APPLE_BLUE, line=dict(width=1, color='white')),
        text=[f'{v}' for v in branching],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='平均步数',
        x=games,
        y=game_length,
        marker=dict(color=APPLE_ORANGE, line=dict(width=1, color='white')),
        text=[f'{v}' for v in game_length],
        textposition='outside'
    ))
    
    fig.add_trace(go.Scatter(
        name='复杂度 (10^x)',
        x=games,
        y=complexity,
        mode='lines+markers+text',
        line=dict(color=APPLE_RED, width=3),
        marker=dict(size=12),
        text=[f'10^{v}' for v in complexity],
        textposition='top center',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=dict(text='三种棋类游戏复杂度对比', font=dict(size=16, family='Arial')),
        xaxis=dict(title='棋类游戏'),
        yaxis=dict(title='数值', range=[0, 200]),
        yaxis2=dict(title='复杂度指数', overlaying='y', side='right', range=[0, 400]),
        barmode='group',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=80, t=80, b=50),
        height=450
    )
    
    return fig

def plot_self_play_loop():
    """绘制自我对弈循环"""
    fig = go.Figure()
    
    # 循环节点
    nodes = [
        ('神经网络', 0, 2, APPLE_PURPLE),
        ('MCTS 搜索', 3, 2, APPLE_ORANGE),
        ('选择动作', 6, 2, APPLE_BLUE),
        ('自我对弈', 4.5, 0, APPLE_GREEN),
        ('生成数据', 1.5, 0, APPLE_CYAN),
    ]
    
    for name, x, y, color in nodes:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=70, color=color, line=dict(width=2, color='white')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[name],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 循环箭头
    # 神经网络 -> MCTS
    fig.add_trace(go.Scatter(
        x=[0.6, 2.4], y=[2, 2],
        mode='lines',
        line=dict(color=APPLE_GRAY, width=2.5),
        showlegend=False
    ))
    fig.add_annotation(
        x=1.5, y=2.15,
        text='p, v',
        showarrow=False,
        font=dict(size=10, color=APPLE_GRAY)
    )
    
    # MCTS -> 选择动作
    fig.add_trace(go.Scatter(
        x=[3.6, 5.4], y=[2, 2],
        mode='lines',
        line=dict(color=APPLE_GRAY, width=2.5),
        showlegend=False
    ))
    
    # 选择动作 -> 自我对弈
    fig.add_trace(go.Scatter(
        x=[5.6, 5.2], y=[1.5, 0.5],
        mode='lines',
        line=dict(color=APPLE_GRAY, width=2.5),
        showlegend=False
    ))
    
    # 自我对弈 -> 生成数据
    fig.add_trace(go.Scatter(
        x=[3.2, 2.2], y=[0, 0],
        mode='lines',
        line=dict(color=APPLE_GRAY, width=2.5),
        showlegend=False
    ))
    
    # 生成数据 -> 神经网络（循环）
    fig.add_trace(go.Scatter(
        x=[0.8, 0, 0], y=[0, 0.8, 1.5],
        mode='lines',
        line=dict(color=APPLE_RED, width=3),
        showlegend=False
    ))
    fig.add_annotation(
        x=-0.4, y=0.8,
        text='训练',
        showarrow=False,
        font=dict(size=10, color=APPLE_RED)
    )
    
    # 添加数据说明
    fig.add_annotation(
        x=3, y=-0.6,
        text='(s, π, z) - 状态、策略、结果',
        showarrow=False,
        font=dict(size=10, color=APPLE_GRAY)
    )
    
    fig.update_layout(
        title=dict(text='AlphaZero 自我对弈循环', font=dict(size=16, family='Arial')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 7.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 3.2]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=20, r=20, t=60, b=50),
        height=420
    )
    
    return fig

def main():
    """主函数：生成所有图表"""
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    plots = [
        (plot_alphazero_vs_alphago, 'alphazero_vs_alphago.png'),
        (plot_mcts_puct, 'alphazero_mcts.png'),
        (plot_neural_network_structure, 'alphazero_network.png'),
        (plot_training_progress, 'alphazero_training.png'),
        (plot_games_comparison, 'alphazero_games.png'),
        (plot_self_play_loop, 'alphazero_selfplay.png'),
    ]
    
    for plot_func, filename in plots:
        print(f"生成: {filename}")
        fig = plot_func()
        filepath = os.path.join(output_dir, filename)
        save_and_compress(fig, filepath)
    
    print("\n✅ 所有 AlphaZero 图表生成完成!")

if __name__ == '__main__':
    main()
