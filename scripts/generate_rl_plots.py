#!/usr/bin/env python3
"""
强化学习综述文章配套图表生成
"""

import subprocess
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 苹果风格配色
APPLE_COLORS = {
    'primary': '#007AFF',
    'success': '#34C759',
    'warning': '#FF9500',
    'danger': '#FF3B30',
    'purple': '#AF52DE',
    'secondary': '#5856D6'
}

def save_and_compress(fig, filepath):
    """保存并压缩图片"""
    fig.write_image(filepath, scale=2, width=900, height=500)
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force',
            '--output', filepath, filepath
        ], check=False, capture_output=True)
    print(f"✅ 已保存并压缩: {filepath}")

def plot_mdp_framework():
    """绘制MDP框架示意图"""
    fig = go.Figure()

    # 绘制节点
    nodes = [
        {'x': 1, 'y': 2, 'label': '环境', 'color': APPLE_COLORS['primary']},
        {'x': 4, 'y': 2, 'label': '智能体', 'color': APPLE_COLORS['success']},
        {'x': 2.5, 'y': 3.2, 'label': '状态S', 'color': APPLE_COLORS['purple']},
        {'x': 2.5, 'y': 0.8, 'label': '奖励R', 'color': APPLE_COLORS['warning']},
        {'x': 5.5, 'y': 2, 'label': '动作A', 'color': APPLE_COLORS['danger']},
    ]

    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node['x']], y=[node['y']],
            mode='markers+text',
            marker=dict(size=55, color=node['color']),
            text=[node['label']],
            textposition='middle center',
            textfont=dict(size=13, color='white', family='Arial Black')
        ))

    # 绘制连接箭头（使用注释）
    fig.add_annotation(
        x=2.5, y=2.3, axref='x', ayref='y',
        ax=4.3, ay=2.3,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='#86868B'
    )
    fig.add_annotation(
        x=4.3, y=1.7, axref='x', ayref='y',
        ax=2.5, ay=1.7,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='#86868B'
    )
    fig.add_annotation(
        x=1.7, y=2.6, axref='x', ayref='y',
        ax=2.5, ay=3,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='#86868B'
    )
    fig.add_annotation(
        x=1.7, y=1.4, axref='x', ayref='y',
        ax=2.5, ay=1,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor='#86868B'
    )

    fig.update_layout(
        title='马尔可夫决策过程（MDP）框架',
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    save_and_compress(fig, 'static/images/plots/rl-mdp-framework.png')

def plot_bellman_equation():
    """绘制贝尔曼方程示意图"""
    t = np.linspace(0, 10, 100)
    V_current = 100 * np.exp(-0.3 * t)
    V_future = 100 * np.exp(-0.3 * (t + 1))
    immediate_reward = 20 * np.exp(-0.5 * t)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=V_current,
        mode='lines',
        name='当前状态价值 V(s)',
        line=dict(color=APPLE_COLORS['primary'], width=3)
    ))

    fig.add_trace(go.Scatter(
        x=t, y=V_future,
        mode='lines',
        name='下一状态价值 γV(s\')',
        line=dict(color=APPLE_COLORS['success'], width=3, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=t, y=immediate_reward,
        mode='lines',
        name='即时奖励 R(s,a)',
        line=dict(color=APPLE_COLORS['warning'], width=3)
    ))

    fig.update_layout(
        title='贝尔曼方程：价值函数的递归结构',
        xaxis_title='时间步',
        yaxis_title='价值',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )

    save_and_compress(fig, 'static/images/plots/rl-bellman-equation.png')

def plot_q_learning_convergence():
    """绘制Q-learning收敛过程"""
    episodes = np.arange(0, 500)
    # 模拟Q值收敛过程
    Q_values = 50 + 30 * (1 - np.exp(-episodes / 100)) + np.random.normal(0, 2, len(episodes))
    # 添加一些噪声但逐渐收敛
    Q_values = np.convolve(Q_values, np.ones(10)/10, mode='same')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=episodes, y=Q_values,
        mode='lines',
        name='Q值估计',
        line=dict(color=APPLE_COLORS['primary'], width=2)
    ))

    # 添加真实Q值（假设）
    fig.add_hline(
        y=80, line_dash='dash', line_color=APPLE_COLORS['success'],
        annotation_text='最优Q值 Q*'
    )

    fig.update_layout(
        title='Q-learning收敛过程',
        xaxis_title='回合数',
        yaxis_title='Q值',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14)
    )

    save_and_compress(fig, 'static/images/plots/rl-qlearning-convergence.png')

def plot_policy_gradient():
    """绘制策略梯度优化过程"""
    theta = np.linspace(-3, 3, 100)
    # 模拟目标函数（非凸）
    J = -0.5 * theta**4 + 2 * theta**2 + 10 * np.sin(2 * theta) + 50

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=theta, y=J,
        mode='lines',
        name='目标函数 J(θ)',
        line=dict(color=APPLE_COLORS['primary'], width=3),
        fill='tozeroy', fillcolor='rgba(0,122,255,0.1)'
    ))

    # 标记梯度方向
    fig.add_annotation(
        x=-1.5, y=50, axref='x', ayref='y',
        ax=-0.5, ay=55,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_COLORS['danger'],
        text='∇J(θ) > 0'
    )

    fig.add_annotation(
        x=1.5, y=50, axref='x', ayref='y',
        ax=0.5, ay=55,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_COLORS['danger'],
        text='∇J(θ) < 0'
    )

    # 标记局部最大值
    fig.add_trace(go.Scatter(
        x=[0.2], y=[max(J)],
        mode='markers',
        marker=dict(size=15, color=APPLE_COLORS['success'], symbol='diamond'),
        name='局部最优'
    ))

    fig.update_layout(
        title='策略梯度优化：沿梯度方向提升目标函数',
        xaxis_title='策略参数 θ',
        yaxis_title='目标函数 J(θ)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14)
    )

    save_and_compress(fig, 'static/images/plots/rl-policy-gradient.png')

def plot_actor_critic():
    """绘制Actor-Critic架构"""
    fig = go.Figure()

    # Actor网络
    fig.add_trace(go.Scatter(
        x=[1, 2, 3], y=[2, 2, 2],
        mode='lines+markers',
        line=dict(color=APPLE_COLORS['primary'], width=4),
        marker=dict(size=[40, 40, 40], color=APPLE_COLORS['primary']),
        text=['状态s', 'Actor', '动作a'],
        textposition='middle center',
        textfont=dict(size=11, color='white', family='Arial Black'),
        name='Actor'
    ))

    # Critic网络
    fig.add_trace(go.Scatter(
        x=[1, 2, 3], y=[1, 1, 1],
        mode='lines+markers',
        line=dict(color=APPLE_COLORS['success'], width=4),
        marker=dict(size=[40, 40, 40], color=APPLE_COLORS['success']),
        text=['状态s', 'Critic', '价值V'],
        textposition='middle center',
        textfont=dict(size=11, color='white', family='Arial Black'),
        name='Critic'
    ))

    # TD误差反馈
    fig.add_annotation(
        x=3, y=1.2, axref='x', ayref='y',
        ax=2, ay=1.7,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=APPLE_COLORS['warning'],
        text='TD误差'
    )

    fig.update_layout(
        title='Actor-Critic架构',
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 4]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 3]),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    save_and_compress(fig, 'static/images/plots/rl-actor-critic.png')

def plot_exploration_vs_exploitation():
    """绘制探索-利用权衡"""
    episodes = np.arange(0, 200)

    # 模拟探索率衰减
    epsilon = 1.0 * np.exp(-episodes / 50)

    # 模拟累积奖励
    reward_high_exp = 20 * episodes / (1 + 0.1 * episodes)  # 高探索
    reward_low_exp = 5 * episodes / (1 + 0.05 * episodes)   # 低探索
    reward_balanced = 25 * episodes / (1 + 0.08 * episodes)  # 平衡

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.2,
        row_heights=[0.4, 0.6]
    )

    fig.add_trace(go.Scatter(
        x=episodes, y=epsilon,
        mode='lines',
        name='ε-贪婪探索率',
        line=dict(color=APPLE_COLORS['primary'], width=3)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=episodes, y=reward_high_exp,
        mode='lines',
        name='过度探索',
        line=dict(color=APPLE_COLORS['danger'], width=2, dash='dot')
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=episodes, y=reward_low_exp,
        mode='lines',
        name='过度利用',
        line=dict(color=APPLE_COLORS['warning'], width=2, dash='dot')
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=episodes, y=reward_balanced,
        mode='lines',
        name='平衡策略',
        line=dict(color=APPLE_COLORS['success'], width=3)
    ), row=2, col=1)

    fig.update_xaxes(title_text='回合数', row=1, col=1)
    fig.update_xaxes(title_text='回合数', row=2, col=1)
    fig.update_yaxes(title_text='探索率 ε', row=1, col=1)
    fig.update_yaxes(title_text='累积奖励', row=2, col=1)

    fig.update_layout(
        title='探索-利用权衡：探索率衰减与累积奖励对比',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        height=700,
        margin=dict(t=80)
    )

    save_and_compress(fig, 'static/images/plots/rl-exploration-exploitation.png')

def plot_discount_factor():
    """绘制折扣因子影响"""
    t = np.arange(0, 20)

    gammas = [0.5, 0.8, 0.9, 0.95, 0.99]
    colors = [APPLE_COLORS['danger'], APPLE_COLORS['warning'],
              APPLE_COLORS['purple'], APPLE_COLORS['success'],
              APPLE_COLORS['primary']]

    fig = go.Figure()

    for gamma, color in zip(gammas, colors):
        discounted = gamma ** t
        fig.add_trace(go.Scatter(
            x=t, y=discounted,
            mode='lines',
            name=f'γ = {gamma}',
            line=dict(color=color, width=2)
        ))

    fig.update_layout(
        title='折扣因子对远期奖励的影响',
        xaxis_title='时间步 t',
        yaxis_title='折扣权重 γ^t',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        hovermode='x unified',
        legend=dict(x=0.98, y=0.98, xanchor='right', bgcolor='rgba(255,255,255,0.8)')
    )

    save_and_compress(fig, 'static/images/plots/rl-discount-factor.png')

if __name__ == '__main__':
    print("开始生成强化学习综述图表...")

    plot_mdp_framework()
    plot_bellman_equation()
    plot_q_learning_convergence()
    plot_policy_gradient()
    plot_actor_critic()
    plot_exploration_vs_exploitation()
    plot_discount_factor()

    print("\n✅ 所有图表生成完成！")
