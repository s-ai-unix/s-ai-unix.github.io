#!/usr/bin/env python3
"""
生成 AlphaGo 论文解读所需的 Plotly 图表
输出为 PNG 格式（非 HTML）
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

def save_and_compress(fig, filepath, width=900, height=600, scale=2):
    """保存并压缩图片"""
    fig.write_image(filepath, width=width, height=height, scale=scale)
    
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False, capture_output=True)
    
    print(f"✅ 已保存: {filepath}")

def plot_mcts_tree():
    """绘制 MCTS 树搜索示意图"""
    fig = go.Figure()
    
    # 定义节点位置 - 三层树结构
    # 根节点
    root_x, root_y = 0, 0
    
    # 第二层节点
    level2_x = [-3, -1, 1, 3]
    level2_y = [-2, -2, -2, -2]
    
    # 第三层节点（部分展开）
    level3_x = []
    level3_y = []
    level3_parents = [0, 0, 1, 1, 2, 3]  # 对应 level2 的父节点索引
    
    # 为每个第二层节点添加子节点
    for i, (px, py) in enumerate(zip(level2_x, level2_y)):
        if i < 2:  # 前两个节点展开
            level3_x.extend([px - 0.8, px + 0.8])
            level3_y.extend([py - 2, py - 2])
        elif i == 2:  # 第三个节点展开一个
            level3_x.append(px)
            level3_y.append(py - 2)
    
    # 绘制连接线
    # 根到第二层
    for x, y in zip(level2_x, level2_y):
        fig.add_trace(go.Scatter(
            x=[root_x, x], y=[root_y, y],
            mode='lines',
            line=dict(color=APPLE_GRAY, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 第二层到第三层
    child_idx = 0
    for i, (px, py) in enumerate(zip(level2_x, level2_y)):
        if i < 2:
            for offset in [-0.8, 0.8]:
                fig.add_trace(go.Scatter(
                    x=[px, px + offset], y=[py, py - 2],
                    mode='lines',
                    line=dict(color=APPLE_GRAY, width=1.5),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            child_idx += 2
        elif i == 2:
            fig.add_trace(go.Scatter(
                x=[px, px], y=[py, py - 2],
                mode='lines',
                line=dict(color=APPLE_GRAY, width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # 绘制节点
    # 根节点 - 蓝色核心
    fig.add_trace(go.Scatter(
        x=[root_x], y=[root_y],
        mode='markers',
        marker=dict(size=60, color=APPLE_BLUE, line=dict(width=3, color='white')),
        showlegend=False,
        hovertemplate='根节点<br>Q=%{customdata:.3f}<extra></extra>',
        customdata=[0.5]
    ))
    fig.add_trace(go.Scatter(
        x=[root_x], y=[root_y],
        mode='text',
        text=['s₀'],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 第二层节点
    n_values = [12, 8, 5, 3]  # 访问次数
    q_values = [0.45, 0.52, 0.38, 0.61]  # Q 值
    colors_l2 = [APPLE_GREEN if q > 0.5 else APPLE_ORANGE for q in q_values]
    
    for x, y, n, q, color in zip(level2_x, level2_y, n_values, q_values, colors_l2):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=45, color=color, line=dict(width=2, color='white')),
            showlegend=False,
            hovertemplate=f'N={n}<br>Q={q:.3f}<extra></extra>'
        ))
    
    # 第二层文字
    l2_labels = ['a₁', 'a₂', 'a₃', 'a₄']
    for x, y, label in zip(level2_x, level2_y, l2_labels):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[label],
            textposition='middle center',
            textfont=dict(size=12, color='white', family='Arial'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 第三层节点
    level3_colors = [APPLE_PURPLE, APPLE_CYAN, APPLE_PURPLE, APPLE_CYAN, APPLE_GREEN, APPLE_ORANGE]
    for x, y, color in zip(level3_x, level3_y, level3_colors):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=35, color=color, line=dict(width=1.5, color='white')),
            showlegend=False,
            hovertemplate='叶节点<extra></extra>'
        ))
    
    # 添加图例说明
    legend_items = [
        ('根节点', APPLE_BLUE),
        ('高 Q 值', APPLE_GREEN),
        ('低 Q 值', APPLE_ORANGE),
        ('叶节点', APPLE_PURPLE)
    ]
    for i, (name, color) in enumerate(legend_items):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=color),
            name=name,
            showlegend=True
        ))
    
    fig.update_layout(
        title=dict(text='MCTS 树搜索结构示意', font=dict(size=18, family='Arial')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5.5, 1]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=20, r=20, t=60, b=20),
        height=500
    )
    
    return fig

def plot_ucb_formula():
    """绘制 UCB 公式与探索利用权衡图"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('UCB 随访问次数变化', '探索与利用的权衡'),
        horizontal_spacing=0.12
    )
    
    # 左图：UCB 值随 N 的变化
    N_parent = 100
    n_range = np.arange(1, 50, 0.5)
    c_puct = 1.5
    Q = 0.5  # 假设 Q 值
    P = 0.3  # 先验概率
    
    u_values = Q + c_puct * P * np.sqrt(N_parent) / (1 + n_range)
    
    fig.add_trace(go.Scatter(
        x=n_range, y=u_values,
        mode='lines',
        name='UCB = Q + U',
        line=dict(color=APPLE_BLUE, width=3),
        showlegend=False
    ), row=1, col=1)
    
    # 添加 Q 值参考线
    fig.add_trace(go.Scatter(
        x=[0, 50], y=[Q, Q],
        mode='lines',
        name='Q 值',
        line=dict(color=APPLE_GREEN, width=2, dash='dash'),
        showlegend=False
    ), row=1, col=1)
    
    # 右图：探索项 U 的变化
    U_values = c_puct * P * np.sqrt(N_parent) / (1 + n_range)
    
    fig.add_trace(go.Scatter(
        x=n_range, y=U_values,
        mode='lines',
        name='探索项 U',
        line=dict(color=APPLE_ORANGE, width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 149, 0, 0.2)',
        showlegend=False
    ), row=1, col=2)
    
    fig.update_xaxes(title_text='访问次数 n(s,a)', row=1, col=1)
    fig.update_yaxes(title_text='UCB 值', row=1, col=1)
    fig.update_xaxes(title_text='访问次数 n(s,a)', row=1, col=2)
    fig.update_yaxes(title_text='探索项 U(s,a)', row=1, col=2)
    
    fig.update_layout(
        title=dict(text='UCB 算法探索-利用权衡', font=dict(size=16, family='Arial')),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=60, r=40, t=80, b=50),
        height=450
    )
    
    return fig

def plot_training_pipeline():
    """绘制 AlphaGo 训练流程"""
    fig = go.Figure()
    
    # 阶段位置
    stages = [
        ('人类棋谱', -6, 0, APPLE_BLUE),
        ('监督学习', -3, 0, APPLE_GREEN),
        ('策略网络 pσ', 0, 0, APPLE_PURPLE),
        ('强化学习', 3, 0, APPLE_ORANGE),
        ('策略网络 pρ', 6, 0, APPLE_RED),
    ]
    
    # 添加阶段节点
    for name, x, y, color in stages:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=55, color=color, line=dict(width=2, color='white')),
            showlegend=False,
            hovertemplate=name + '<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[name],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 添加箭头连接
    arrow_x = [-4.5, -1.5, 1.5, 4.5]
    for ax in arrow_x:
        fig.add_annotation(
            x=ax + 0.7, y=0,
            ax=ax - 0.7, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=2, arrowsize=1.5, arrowwidth=2,
            arrowcolor=APPLE_GRAY
        )
    
    # 添加下方的价值网络分支
    fig.add_trace(go.Scatter(
        x=[0], y=[-2.5],
        mode='markers',
        marker=dict(size=55, color=APPLE_CYAN, line=dict(width=2, color='white')),
        showlegend=False,
        hovertemplate='价值网络<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[-2.5],
        mode='text',
        text=['vθ'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 连接线
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-0.5, -2],
        mode='lines',
        line=dict(color=APPLE_GRAY, width=2, dash='dot'),
        showlegend=False
    ))
    
    fig.add_annotation(
        x=0.5, y=-1.25,
        text='回归训练',
        showarrow=False,
        font=dict(size=10, color=APPLE_GRAY)
    )
    
    # 添加标题
    fig.add_annotation(
        x=0, y=2.5,
        text='AlphaGo 训练流程：从人类知识到自我对弈',
        showarrow=False,
        font=dict(size=16, family='Arial')
    )
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-8, 8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4, 3.5]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=20, r=20, t=80, b=20),
        height=450
    )
    
    return fig

def plot_win_rate_comparison():
    """绘制不同版本 AlphaGo 胜率对比"""
    versions = ['策略网络\n(SL)', '策略网络\n(RL)', '价值网络\n+MCTS', 'AlphaGo\n(完整)', 'AlphaGo\n(分布式)']
    win_rates = [55, 65, 85, 95, 99.8]
    colors = [APPLE_GRAY, APPLE_CYAN, APPLE_ORANGE, APPLE_GREEN, APPLE_BLUE]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=versions,
        y=win_rates,
        marker=dict(color=colors, line=dict(width=1, color='white')),
        text=[f'{v}%' for v in win_rates],
        textposition='outside',
        textfont=dict(size=12, family='Arial'),
        showlegend=False
    ))
    
    # 添加参考线
    fig.add_hline(y=50, line=dict(color=APPLE_RED, width=2, dash='dash'),
                  annotation_text='随机水平', annotation_position='right')
    
    fig.update_layout(
        title=dict(text='AlphaGo 各组件胜率对比（vs 其他围棋程序）', font=dict(size=16, family='Arial')),
        xaxis=dict(title='版本'),
        yaxis=dict(title='胜率 (%)', range=[0, 105]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=60, r=100, t=80, b=60),
        height=450
    )
    
    return fig

def plot_search_iterations():
    """绘制 MCTS 迭代过程中评估值的变化"""
    iterations = np.arange(0, 101, 1)
    
    # 模拟 Q 值收敛过程
    np.random.seed(42)
    Q_a1 = 0.3 + 0.4 * (1 - np.exp(-iterations/20)) + np.random.normal(0, 0.02, len(iterations))
    Q_a2 = 0.5 + 0.1 * (1 - np.exp(-iterations/25)) + np.random.normal(0, 0.015, len(iterations))
    Q_a3 = 0.4 + 0.15 * (1 - np.exp(-iterations/18)) + np.random.normal(0, 0.018, len(iterations))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations, y=Q_a1,
        mode='lines',
        name='动作 a₁ (高先验)',
        line=dict(color=APPLE_BLUE, width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations, y=Q_a2,
        mode='lines',
        name='动作 a₂ (中先验)',
        line=dict(color=APPLE_GREEN, width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations, y=Q_a3,
        mode='lines',
        name='动作 a₃ (低先验)',
        line=dict(color=APPLE_ORANGE, width=2.5)
    ))
    
    # 添加最终选择标记
    final_iter = 100
    fig.add_trace(go.Scatter(
        x=[final_iter], y=[Q_a1[-1]],
        mode='markers',
        marker=dict(size=15, color=APPLE_BLUE, symbol='star'),
        showlegend=False,
        hovertemplate='最终选择<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='MCTS 模拟过程中 Q 值的收敛', font=dict(size=16, family='Arial')),
        xaxis=dict(title='模拟次数', range=[0, 105]),
        yaxis=dict(title='Q(s,a) - 动作价值估计', range=[0.2, 0.8]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=40, t=80, b=50),
        height=450
    )
    
    return fig

def plot_neural_network_structure():
    """绘制策略网络和价值网络结构示意"""
    fig = go.Figure()
    
    # 定义所有节点位置 - 更大的节点，线从边缘连接
    nodes = {
        'input': {'x': 0, 'y': 0, 'size': 90, 'color': APPLE_BLUE, 'text': '输入<br>19×19'},
        'conv1': {'x': 1.8, 'y': 0.6, 'size': 65, 'color': APPLE_PURPLE, 'text': 'Conv1'},
        'conv2': {'x': 3.4, 'y': 0.3, 'size': 60, 'color': APPLE_PURPLE, 'text': 'Conv2'},
        'conv3': {'x': 5, 'y': 0, 'size': 55, 'color': APPLE_PURPLE, 'text': 'Conv3'},
        'policy': {'x': 7, 'y': 1.3, 'size': 75, 'color': APPLE_GREEN, 'text': '策略<br>p(a|s)'},
        'value': {'x': 7, 'y': -1.3, 'size': 75, 'color': APPLE_ORANGE, 'text': '价值<br>v(s)'},
    }
    
    # 定义连接关系
    connections = [
        ('input', 'conv1'),
        ('conv1', 'conv2'),
        ('conv2', 'conv3'),
        ('conv3', 'policy'),
        ('conv3', 'value'),
    ]
    
    # 辅助函数：计算节点边缘点
    def get_edge_points(n1, n2):
        """计算从n1到n2方向上两节点的边缘点坐标"""
        x1, y1 = nodes[n1]['x'], nodes[n1]['y']
        x2, y2 = nodes[n2]['x'], nodes[n2]['y']
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.01:
            return x1, y1, x2, y2
        
        # 将 marker size (像素直径) 转换为数据坐标
        px_to_data = 0.004
        r1 = nodes[n1]['size'] * px_to_data
        r2 = nodes[n2]['size'] * px_to_data
        
        # n1 的出边点
        x1_out = x1 + (dx / dist) * r1
        y1_out = y1 + (dy / dist) * r1
        # n2 的入边点
        x2_in = x2 - (dx / dist) * r2
        y2_in = y2 - (dy / dist) * r2
        
        return x1_out, y1_out, x2_in, y2_in
    
    # 先绘制连接线（在节点下方）
    for n1_name, n2_name in connections:
        x1, y1, x2, y2 = get_edge_points(n1_name, n2_name)
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2],
            mode='lines',
            line=dict(color=APPLE_GRAY, width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 再绘制节点（覆盖在线的上方）
    for name, node in nodes.items():
        fig.add_trace(go.Scatter(
            x=[node['x']], y=[node['y']],
            mode='markers',
            marker=dict(size=node['size'], color=node['color'], 
                       line=dict(width=3, color='white')),
            showlegend=False
        ))
    
    # 最后绘制文字（在最上层）
    for name, node in nodes.items():
        fig.add_trace(go.Scatter(
            x=[node['x']], y=[node['y']],
            mode='text',
            text=[node['text']],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial'),
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(text='AlphaGo 双头神经网络结构', font=dict(size=16, family='Arial')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.8, 2.8]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=20, r=20, t=80, b=20),
        height=450
    )
    
    return fig

def main():
    """主函数：生成所有图表"""
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    plots = [
        (plot_mcts_tree, 'alphago_mcts_tree.png'),
        (plot_ucb_formula, 'alphago_ucb_formula.png'),
        (plot_training_pipeline, 'alphago_training_pipeline.png'),
        (plot_win_rate_comparison, 'alphago_win_rate.png'),
        (plot_search_iterations, 'alphago_search_iterations.png'),
        (plot_neural_network_structure, 'alphago_network_structure.png'),
    ]
    
    for plot_func, filename in plots:
        print(f"生成: {filename}")
        fig = plot_func()
        filepath = os.path.join(output_dir, filename)
        save_and_compress(fig, filepath)
    
    print("\n✅ 所有图表生成完成!")

if __name__ == '__main__':
    main()
