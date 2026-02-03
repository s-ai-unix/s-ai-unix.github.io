#!/usr/bin/env python3
"""
生成概率论与数理统计相关的 Plotly 图形
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm, binom, poisson
import os

# 创建输出目录
os.makedirs('../static/images/plots', exist_ok=True)

# ========== 1. 正态分布的 PDF ==========
def plot_normal_distribution():
    """绘制不同参数的正态分布"""
    x = np.linspace(-5, 5, 500)

    fig = go.Figure()

    # 标准正态分布
    fig.add_trace(go.Scatter(
        x=x, y=norm.pdf(x, 0, 1),
        mode='lines',
        name='N(0, 1)',
        line=dict(color='#007AFF', width=3)
    ))

    # 均值为0，方差为0.5
    fig.add_trace(go.Scatter(
        x=x, y=norm.pdf(x, 0, 0.5),
        mode='lines',
        name='N(0, 0.5)',
        line=dict(color='#34C759', width=3)
    ))

    # 均值为0，方差为2
    fig.add_trace(go.Scatter(
        x=x, y=norm.pdf(x, 0, 2),
        mode='lines',
        name='N(0, 2)',
        line=dict(color='#FF9500', width=3)
    ))

    fig.update_layout(
        title='正态分布的概率密度函数',
        xaxis_title='x',
        yaxis_title='概率密度 f(x)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.7, y=0.95),
        width=800,
        height=500
    )

    fig.write_html('../static/images/plots/normal-distribution.html')
    print("✅ 正态分布图形已生成")
    return fig

# ========== 2. 二项分布的 PMF ==========
def plot_binomial_distribution():
    """绘制不同参数的二项分布"""
    n = 20
    k = np.arange(0, n+1)

    fig = go.Figure()

    # p = 0.3
    fig.add_trace(go.Bar(
        x=k, y=binom.pmf(k, n, 0.3),
        name='p = 0.3',
        marker_color='#007AFF',
        opacity=0.7
    ))

    # p = 0.5
    fig.add_trace(go.Bar(
        x=k, y=binom.pmf(k, n, 0.5),
        name='p = 0.5',
        marker_color='#34C759',
        opacity=0.7
    ))

    # p = 0.7
    fig.add_trace(go.Bar(
        x=k, y=binom.pmf(k, n, 0.7),
        name='p = 0.7',
        marker_color='#FF9500',
        opacity=0.7
    ))

    fig.update_layout(
        title='二项分布的概率质量函数 (n=20)',
        xaxis_title='成功次数 k',
        yaxis_title='概率 P(X = k)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.6, y=0.95),
        barmode='overlay',
        width=800,
        height=500
    )

    fig.write_html('../static/images/plots/binomial-distribution.html')
    print("✅ 二项分布图形已生成")
    return fig

# ========== 3. 中心极限定理演示 ==========
def plot_central_limit_theorem():
    """演示中心极限定理：不同样本量下样本均值的分布"""
    n_samples = 10000

    # 原始分布：均匀分布 U(0, 1)
    # 均值 = 0.5, 方差 = 1/12

    fig = go.Figure()

    # n = 1
    samples = np.mean(np.random.uniform(0, 1, (n_samples, 1)), axis=1)
    fig.add_trace(go.Histogram(
        x=samples,
        name='n = 1',
        nbinsx=50,
        marker_color='#007AFF',
        opacity=0.7,
        histnorm='probability density'
    ))

    # n = 5
    samples = np.mean(np.random.uniform(0, 1, (n_samples, 5)), axis=1)
    fig.add_trace(go.Histogram(
        x=samples,
        name='n = 5',
        nbinsx=50,
        marker_color='#34C759',
        opacity=0.7,
        histnorm='probability density'
    ))

    # n = 30
    samples = np.mean(np.random.uniform(0, 1, (n_samples, 30)), axis=1)
    fig.add_trace(go.Histogram(
        x=samples,
        name='n = 30',
        nbinsx=50,
        marker_color='#FF9500',
        opacity=0.7,
        histnorm='probability density'
    ))

    # 添加理论正态分布
    x = np.linspace(0.3, 0.7, 500)
    fig.add_trace(go.Scatter(
        x=x, y=norm.pdf(x, 0.5, np.sqrt(1/12/30)),
        mode='lines',
        name='理论正态分布',
        line=dict(color='#FF3B30', width=3, dash='dash')
    ))

    fig.update_layout(
        title='中心极限定理：均匀分布的样本均值分布',
        xaxis_title='样本均值',
        yaxis_title='概率密度',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.65, y=0.95),
        barmode='overlay',
        width=800,
        height=500
    )

    fig.write_html('../static/images/plots/central-limit-theorem.html')
    print("✅ 中心极限定理图形已生成")
    return fig

# ========== 4. 泊松分布的 PMF ==========
def plot_poisson_distribution():
    """绘制不同参数的泊松分布"""
    k = np.arange(0, 20)

    fig = go.Figure()

    # lambda = 1
    fig.add_trace(go.Bar(
        x=k, y=poisson.pmf(k, 1),
        name='λ = 1',
        marker_color='#007AFF',
        opacity=0.7
    ))

    # lambda = 4
    fig.add_trace(go.Bar(
        x=k, y=poisson.pmf(k, 4),
        name='λ = 4',
        marker_color='#34C759',
        opacity=0.7
    ))

    # lambda = 10
    fig.add_trace(go.Bar(
        x=k, y=poisson.pmf(k, 10),
        name='λ = 10',
        marker_color='#FF9500',
        opacity=0.7
    ))

    fig.update_layout(
        title='泊松分布的概率质量函数',
        xaxis_title='事件次数 k',
        yaxis_title='概率 P(X = k)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.6, y=0.95),
        barmode='overlay',
        width=800,
        height=500
    )

    fig.write_html('../static/images/plots/poisson-distribution.html')
    print("✅ 泊松分布图形已生成")
    return fig

# ========== 5. 交叉熵损失可视化 ==========
def plot_cross_entropy_loss():
    """绘制交叉熵损失函数"""
    y_hat = np.linspace(0.01, 0.99, 500)

    fig = go.Figure()

    # y = 1
    fig.add_trace(go.Scatter(
        x=y_hat, y=-np.log(y_hat),
        mode='lines',
        name='y = 1 (正样本)',
        line=dict(color='#007AFF', width=3)
    ))

    # y = 0
    fig.add_trace(go.Scatter(
        x=y_hat, y=-np.log(1 - y_hat),
        mode='lines',
        name='y = 0 (负样本)',
        line=dict(color='#FF9500', width=3)
    ))

    fig.update_layout(
        title='交叉熵损失函数',
        xaxis_title='预测概率 ŷ',
        yaxis_title='损失 L(y, ŷ)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.05, y=0.95),
        width=800,
        height=500
    )

    fig.write_html('../static/images/plots/cross-entropy-loss.html')
    print("✅ 交叉熵损失图形已生成")
    return fig

# ========== 6. KL 散度可视化 ==========
def plot_kl_divergence():
    """可视化 KL 散度：两个伯努利分布之间的 KL 散度"""
    p = np.linspace(0.01, 0.99, 100)
    q_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    fig = go.Figure()

    for q in q_values:
        kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        # 处理数值问题
        kl = np.where(np.isfinite(kl), kl, 0)

        fig.add_trace(go.Scatter(
            x=p, y=kl,
            mode='lines',
            name=f'q = {q}',
            line=dict(width=2)
        ))

    fig.update_layout(
        title='KL 散度：D_KL(Bernoulli(p) || Bernoulli(q))',
        xaxis_title='p',
        yaxis_title='KL 散度',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.05, y=0.95),
        width=800,
        height=500
    )

    fig.write_html('../static/images/plots/kl-divergence.html')
    print("✅ KL 散度图形已生成")
    return fig

# ========== 7. 熵函数可视化 ==========
def plot_entropy():
    """绘制伯努利分布的熵函数"""
    p = np.linspace(0, 1, 500)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    # 处理边界情况
    entropy = np.where(np.isfinite(entropy), entropy, 0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=p, y=entropy,
        mode='lines',
        name='H(X)',
        line=dict(color='#007AFF', width=3),
        fill='tozeroy'
    ))

    # 标记最大值
    fig.add_trace(go.Scatter(
        x=[0.5], y=[1],
        mode='markers',
        name='最大值 (p = 0.5)',
        marker=dict(color='#FF3B30', size=10)
    ))

    fig.update_layout(
        title='伯努利分布的熵函数',
        xaxis_title='p = P(X = 1)',
        yaxis_title='熵 H(X) (bits)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.05, y=0.95),
        width=800,
        height=500
    )

    fig.write_html('../static/images/plots/entropy.html')
    print("✅ 熵函数图形已生成")
    return fig

# ========== 生成所有图形 ==========
if __name__ == '__main__':
    print("开始生成 Plotly 图形...")
    plot_normal_distribution()
    plot_binomial_distribution()
    plot_central_limit_theorem()
    plot_poisson_distribution()
    plot_cross_entropy_loss()
    plot_kl_divergence()
    plot_entropy()
    print("\n所有图形生成完成！")
