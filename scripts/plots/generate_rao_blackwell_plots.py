#!/usr/bin/env python3
"""
生成Rao-Blackwell定理的配图
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os

OUTPUT_DIR = "static/images/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_and_compress(fig, filepath, width=900, height=600):
    """保存并压缩图片"""
    fig.write_image(filepath, width=width, height=height, scale=2)
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    print(f"✅ 已保存并压缩: {filepath}")


def plot_conditional_expectation():
    """图1：条件期望的直观理解"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('原始数据与估计量', '充分统计量T',
                       '条件期望E[δ|T]', '方差对比'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    np.random.seed(42)
    n = 1000
    
    # 生成数据：X ~ N(μ, 1)，估计μ
    mu_true = 2
    X = np.random.normal(mu_true, 1, n)
    
    # 原始估计量：δ(X) = X1 (只用第一个观测)
    delta_raw = X[0]
    
    # 充分统计量：T = X̄
    T = np.mean(X)
    
    # Rao-Blackwell改进：E[X1 | X̄] = X̄
    delta_rb = T
    
    # 左上：数据分布
    fig.add_trace(go.Histogram(
        x=X,
        nbinsx=30,
        opacity=0.7,
        marker_color='#007AFF',
        name='样本分布',
        showlegend=False
    ), row=1, col=1)
    
    # 标记估计量
    fig.add_vline(x=delta_raw, line=dict(color='#FF3B30', width=3), row=1, col=1)
    fig.add_vline(x=delta_rb, line=dict(color='#34C759', width=3), row=1, col=1)
    fig.add_vline(x=mu_true, line=dict(color='#000000', width=2, dash='dash'), row=1, col=1)
    
    # 右上：模拟多次试验，展示充分统计量的分布
    n_trials = 500
    T_values = []
    X1_values = []
    
    for _ in range(n_trials):
        X_trial = np.random.normal(mu_true, 1, n)
        T_values.append(np.mean(X_trial))
        X1_values.append(X_trial[0])
    
    fig.add_trace(go.Histogram(
        x=X1_values,
        nbinsx=40,
        opacity=0.5,
        marker_color='#FF3B30',
        name='δ=X₁',
        histnorm='probability density',
        showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Histogram(
        x=T_values,
        nbinsx=40,
        opacity=0.5,
        marker_color='#34C759',
        name='T=X̄',
        histnorm='probability density',
        showlegend=True
    ), row=1, col=2)
    
    fig.add_vline(x=mu_true, line=dict(color='#000000', width=2, dash='dash'), row=1, col=2)
    
    # 左下：条件期望的直观展示
    # 将T分组，展示每组内X1的条件期望
    n_sim = 2000
    T_all = []
    X1_all = []
    
    for _ in range(n_sim):
        X_s = np.random.normal(mu_true, 1, 100)
        T_all.append(np.mean(X_s))
        X1_all.append(X_s[0])
    
    T_all = np.array(T_all)
    X1_all = np.array(X1_all)
    
    # 分组计算条件期望
    bins = np.linspace(mu_true - 0.5, mu_true + 0.5, 15)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    conditional_means = []
    
    for i in range(len(bins) - 1):
        mask = (T_all >= bins[i]) & (T_all < bins[i+1])
        if np.sum(mask) > 10:
            conditional_means.append(np.mean(X1_all[mask]))
        else:
            conditional_means.append(np.nan)
    
    # 散点图展示关系
    sample_idx = np.random.choice(len(T_all), 500, replace=False)
    fig.add_trace(go.Scatter(
        x=T_all[sample_idx],
        y=X1_all[sample_idx],
        mode='markers',
        marker=dict(size=5, color='#007AFF', opacity=0.5),
        name='(T, X₁)',
        showlegend=False
    ), row=2, col=1)
    
    # 条件期望线
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=conditional_means,
        mode='lines+markers',
        line=dict(color='#FF3B30', width=3),
        marker=dict(size=8),
        name='E[X₁|T]',
        showlegend=True
    ), row=2, col=1)
    
    # 对角线
    t_range = np.linspace(mu_true - 0.6, mu_true + 0.6, 100)
    fig.add_trace(go.Scatter(
        x=t_range,
        y=t_range,
        mode='lines',
        line=dict(color='#000000', width=2, dash='dash'),
        name='y=x',
        showlegend=True
    ), row=2, col=1)
    
    fig.update_xaxes(title_text='T = X̄', row=2, col=1)
    fig.update_yaxes(title_text='X₁', row=2, col=1)
    
    # 右下：方差对比
    estimators = ['δ=X₁', 'T=X̄', 'δ_RB=X̄']
    variances = [np.var(X1_values), np.var(T_values), np.var(T_values)]
    colors = ['#FF3B30', '#34C759', '#007AFF']
    
    fig.add_trace(go.Bar(
        x=estimators,
        y=variances,
        marker_color=colors,
        showlegend=False
    ), row=2, col=2)
    
    fig.update_yaxes(title_text='方差', row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='Rao-Blackwell定理：条件期望改善估计量',
            font=dict(size=18)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/conditional_expectation.png', width=1000, height=800)
    return fig


def plot_variance_reduction():
    """图2：方差缩减的量化分析"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('样本量与方差缩减', '不同估计量的MSE对比',
                       'Rao-Blackwell化过程', 'Rao-Blackwell不等式'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    np.random.seed(123)
    
    # 左上：样本量对方差缩减的影响
    sample_sizes = np.arange(5, 101, 5)
    n_trials = 1000
    mu_true = 2
    
    var_raw = []
    var_rb = []
    
    for n in sample_sizes:
        estimates_raw = []
        estimates_rb = []
        
        for _ in range(n_trials):
            X = np.random.normal(mu_true, 1, n)
            estimates_raw.append(X[0])  # δ = X₁
            estimates_rb.append(np.mean(X))  # δ_RB = X̄
        
        var_raw.append(np.var(estimates_raw))
        var_rb.append(np.var(estimates_rb))
    
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=var_raw,
        mode='lines',
        line=dict(color='#FF3B30', width=2.5),
        name='原始估计量 (δ=X₁)',
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=var_rb,
        mode='lines',
        line=dict(color='#34C759', width=2.5),
        name='RB估计量 (δ_RB=X̄)',
        showlegend=True
    ), row=1, col=1)
    
    fig.update_xaxes(title_text='样本量 n', row=1, col=1)
    fig.update_yaxes(title_text='方差', row=1, col=1)
    
    # 右上：MSE对比（包含偏差情况）
    # 有偏估计量：δ = 2*X₁
    n = 50
    n_trials = 2000
    
    mse_raw = []
    mse_biased = []
    mse_rb = []
    sample_sizes_mse = np.arange(10, 201, 10)
    
    for n_mse in sample_sizes_mse:
        estimates_raw = []
        estimates_biased = []
        estimates_rb = []
        
        for _ in range(n_trials):
            X = np.random.normal(mu_true, 1, n_mse)
            estimates_raw.append(X[0])
            estimates_biased.append(2 * X[0])  # 有偏
            estimates_rb.append(np.mean(X))
        
        mse_raw.append(np.mean((np.array(estimates_raw) - mu_true)**2))
        mse_biased.append(np.mean((np.array(estimates_biased) - mu_true)**2))
        mse_rb.append(np.mean((np.array(estimates_rb) - mu_true)**2))
    
    fig.add_trace(go.Scatter(
        x=sample_sizes_mse,
        y=mse_raw,
        mode='lines',
        line=dict(color='#FF3B30', width=2),
        name='δ=X₁',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=sample_sizes_mse,
        y=mse_biased,
        mode='lines',
        line=dict(color='#FF9500', width=2),
        name='δ=2X₁ (有偏)',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=sample_sizes_mse,
        y=mse_rb,
        mode='lines',
        line=dict(color='#34C759', width=2),
        name='δ_RB=X̄',
        showlegend=False
    ), row=1, col=2)
    
    fig.update_xaxes(title_text='样本量 n', row=1, col=2)
    fig.update_yaxes(title_text='MSE', row=1, col=2)
    
    # 左下：Rao-Blackwell化过程的直观展示
    # 二维正态的例子
    n_points = 500
    rho = 0.8
    Sigma = np.array([[1, rho], [rho, 1]])
    mu = np.array([0, 0])
    
    X, Y = np.random.multivariate_normal(mu, Sigma, n_points).T
    
    fig.add_trace(go.Scatter(
        x=X,
        y=Y,
        mode='markers',
        marker=dict(size=5, color='#007AFF', opacity=0.5),
        name='原始数据',
        showlegend=False
    ), row=2, col=1)
    
    # 条件期望 E[Y|X=x] = rho * x
    x_range = np.linspace(-3, 3, 100)
    y_cond = rho * x_range
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_cond,
        mode='lines',
        line=dict(color='#FF3B30', width=3),
        name='E[Y|X]',
        showlegend=True
    ), row=2, col=1)
    
    # 展示投影
    x_val = 1.5
    y_orig = Y[np.abs(X - x_val) < 0.2]
    if len(y_orig) > 0:
        y_mean = np.mean(y_orig)
        fig.add_trace(go.Scatter(
            x=[x_val, x_val],
            y=[y_orig[0], rho * x_val],
            mode='lines+markers',
            line=dict(color='#000000', width=1.5),
            marker=dict(size=6),
            showlegend=False
        ), row=2, col=1)
    
    fig.update_xaxes(title_text='X', row=2, col=1)
    fig.update_yaxes(title_text='Y', row=2, col=1)
    
    # 右下：Rao-Blackwell不等式
    # Var(δ) ≥ Var(E[δ|T])
    n_samples = np.arange(2, 101)
    
    # 对于 δ = X₁，Var(δ) = 1
    var_delta = np.ones_like(n_samples)
    
    # Var(E[X₁|X̄]) = Var(X̄) = 1/n
    var_conditional = 1 / n_samples
    
    fig.add_trace(go.Scatter(
        x=n_samples,
        y=var_delta,
        mode='lines',
        line=dict(color='#FF3B30', width=3),
        name='Var(δ)',
        showlegend=True
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=n_samples,
        y=var_conditional,
        mode='lines',
        line=dict(color='#34C759', width=3),
        name='Var(E[δ|T])',
        showlegend=True
    ), row=2, col=2)
    
    # 填充差距
    fig.add_trace(go.Scatter(
        x=np.concatenate([n_samples, n_samples[::-1]]),
        y=np.concatenate([var_conditional, var_delta[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        line=dict(width=0),
        name='方差缩减',
        showlegend=True
    ), row=2, col=2)
    
    fig.update_xaxes(title_text='样本量 n', row=2, col=2)
    fig.update_yaxes(title_text='方差', row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='Rao-Blackwell定理：方差缩减的量化分析',
            font=dict(size=18)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/variance_reduction.png', width=1000, height=800)
    return fig


def plot_sufficiency():
    """图3：充分统计量的概念"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('因子分解定理示意', '充分统计量的信息保持',
                       '数据压缩与信息损失', '充分完备统计量'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    np.random.seed(456)
    
    # 左上：因子分解示意
    # 原始数据 -> 充分统计量 -> 估计
    x = np.linspace(-4, 4, 100)
    
    # 原始分布
    fig.add_trace(go.Scatter(
        x=x,
        y=np.exp(-0.5 * x**2) / np.sqrt(2*np.pi),
        mode='lines',
        line=dict(color='#007AFF', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 122, 255, 0.2)',
        name='原始分布',
        showlegend=True
    ), row=1, col=1)
    
    # 充分统计量的分布（更集中）
    n = 10
    fig.add_trace(go.Scatter(
        x=x,
        y=np.sqrt(n) * np.exp(-0.5 * n * x**2) / np.sqrt(2*np.pi),
        mode='lines',
        line=dict(color='#34C759', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(52, 199, 89, 0.2)',
        name='充分统计量分布',
        showlegend=True
    ), row=1, col=1)
    
    # 右上：信息保持的可视化
    # 熵的概念示意
    n_values = np.arange(1, 21)
    
    # 原始数据的熵（n个独立观测）
    entropy_raw = n_values * 0.5 * np.log(2 * np.pi * np.e)
    
    # 充分统计量的熵（相同信息）
    entropy_sufficient = 0.5 * np.log(2 * np.pi * np.e / n_values)
    
    fig.add_trace(go.Scatter(
        x=n_values,
        y=entropy_raw,
        mode='lines',
        line=dict(color='#FF3B30', width=3),
        name='原始数据熵',
        showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=n_values,
        y=entropy_sufficient,
        mode='lines',
        line=dict(color='#34C759', width=3),
        name='充分统计量熵',
        showlegend=True
    ), row=1, col=2)
    
    fig.update_xaxes(title_text='样本量 n', row=1, col=2)
    fig.update_yaxes(title_text='熵 (nats)', row=1, col=2)
    
    # 左下：数据压缩
    n_trials = 1000
    n = 20
    
    # 原始数据（n维）
    # 充分统计量（1维）
    
    compression_ratios = []
    mse_losses = []
    
    for trial in range(n_trials):
        X = np.random.normal(2, 1, n)
        T = np.mean(X)
        
        # 用T估计原始数据
        X_reconstructed = np.full(n, T)
        
        compression_ratios.append(n)  # n:1压缩
        mse_losses.append(np.mean((X - X_reconstructed)**2))
    
    # 展示压缩效果
    bins = np.linspace(0, 5, 30)
    fig.add_trace(go.Histogram(
        x=mse_losses,
        nbinsx=30,
        opacity=0.7,
        marker_color='#007AFF',
        name='重构误差',
        histnorm='probability density',
        showlegend=True
    ), row=2, col=1)
    
    fig.update_xaxes(title_text='均方误差', row=2, col=1)
    
    # 右下：完备性示意
    # 不同充分统计量的比较
    n = 50
    n_trials = 2000
    
    # 充分统计量1：样本均值
    estimates_mean = []
    # 充分统计量2：样本中位数（对于正态也是充分的）
    estimates_median = []
    
    for _ in range(n_trials):
        X = np.random.normal(2, 1, n)
        estimates_mean.append(np.mean(X))
        estimates_median.append(np.median(X))
    
    fig.add_trace(go.Histogram(
        x=estimates_mean,
        nbinsx=40,
        opacity=0.5,
        marker_color='#007AFF',
        name='样本均值',
        histnorm='probability density',
        showlegend=True
    ), row=2, col=2)
    
    fig.add_trace(go.Histogram(
        x=estimates_median,
        nbinsx=40,
        opacity=0.5,
        marker_color='#FF9500',
        name='样本中位数',
        histnorm='probability density',
        showlegend=True
    ), row=2, col=2)
    
    fig.add_vline(x=2, line=dict(color='#000000', width=2, dash='dash'), row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='充分统计量与数据压缩',
            font=dict(size=18)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/sufficiency_concept.png', width=1000, height=800)
    return fig


def plot_umvue():
    """图4：UMVUE与Lehmann-Scheffe定理"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('无偏估计量的方差下界', '不同估计量的效率比较',
                       'Rao-Blackwell + Lehmann-Scheffe', 'Cramer-Rao下界'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    import scipy.stats as stats
    np.random.seed(789)
    
    # 左上：方差下界
    n_trials = 2000
    sample_sizes = [10, 20, 50, 100]
    
    for i, n in enumerate(sample_sizes):
        # 不同的无偏估计量
        estimates_mean = []
        estimates_midrange = []
        
        for _ in range(n_trials):
            X = np.random.uniform(0, 1, n)
            estimates_mean.append(np.mean(X))
            estimates_midrange.append((np.min(X) + np.max(X)) / 2)
        
        var_mean = np.var(estimates_mean)
        var_mid = np.var(estimates_midrange)
        
        # C-R下界：1/(n*I(θ))
        # 对于Uniform(0,1)，I(θ)与n相关
        cr_bound = 1 / (12 * n)
        
        fig.add_trace(go.Bar(
            x=[f'n={n}\n均值', f'n={n}\n中程'],
            y=[var_mean, var_mid],
            name=f'n={n}',
            showlegend=True if i == 0 else False,
            marker_color=['#007AFF', '#FF9500']
        ), row=1, col=1)
    
    fig.update_yaxes(title_text='方差', row=1, col=1)
    
    # 右上：效率比较
    n = 30
    n_trials = 3000
    
    distributions = [
        ('Normal', lambda: np.random.normal(0, 1, n), '均值'),
        ('Uniform', lambda: np.random.uniform(-1, 1, n), '中程'),
        ('Exp', lambda: np.random.exponential(1, n), '均值')
    ]
    
    efficiencies = []
    labels = []
    
    for dist_name, dist_func, optimal in distributions:
        if dist_name == 'Normal':
            estimates_mle = [np.mean(dist_func()) for _ in range(n_trials)]
            estimates_median = [np.median(dist_func()) for _ in range(n_trials)]
            var_mle = np.var(estimates_mle)
            var_median = np.var(estimates_median)
            eff = var_mle / var_median
        elif dist_name == 'Uniform':
            estimates_mean = [np.mean(dist_func()) for _ in range(n_trials)]
            estimates_mid = [(np.min(dist_func()) + np.max(dist_func())) / 2 for _ in range(n_trials)]
            var_mean = np.var(estimates_mean)
            var_mid = np.var(estimates_mid)
            eff = var_mid / var_mean
        else:  # Exp
            estimates_mle = [np.mean(dist_func()) for _ in range(n_trials)]
            var_mle = np.var(estimates_mle)
            eff = 1.0
        
        efficiencies.append(eff)
        labels.append(dist_name)
    
    fig.add_trace(go.Bar(
        x=labels,
        y=efficiencies,
        marker_color=['#007AFF', '#34C759', '#FF9500'],
        showlegend=False
    ), row=1, col=2)
    
    fig.add_hline(y=1.0, line=dict(color='#FF3B30', width=2, dash='dash'), row=1, col=2)
    fig.update_yaxes(title_text='相对效率', row=1, col=2)
    
    # 左下：RB + LS 流程
    n = 50
    n_trials = 2000
    mu_true = 2
    
    estimates_raw = []
    estimates_rb = []
    estimates_umvue = []
    
    for _ in range(n_trials):
        X = np.random.normal(mu_true, 1, n)
        # 原始估计量（可能不充分）
        delta = X[0]
        # RB化（条件于充分统计量）
        delta_rb = np.mean(X)
        # 如果完备，这就是UMVUE
        delta_umvue = delta_rb
        
        estimates_raw.append(delta)
        estimates_rb.append(delta_rb)
        estimates_umvue.append(delta_umvue)
    
    x_range = np.linspace(mu_true - 1, mu_true + 1, 100)
    
    # 绘制估计量的分布
    for data, color, name in [(estimates_raw, '#FF3B30', '原始δ'),
                               (estimates_rb, '#34C759', 'RB化δ*'),
                               (estimates_umvue, '#007AFF', 'UMVUE')]:
        hist, bins = np.histogram(data, bins=40, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=hist,
            mode='lines',
            fill='tozeroy',
            line=dict(color=color, width=2),
            name=name,
            showlegend=True
        ), row=2, col=1)
    
    fig.add_vline(x=mu_true, line=dict(color='#000000', width=2, dash='dash'), row=2, col=1)
    
    # 右下：Cramer-Rao下界
    sample_sizes = np.arange(5, 101)
    
    # 正态分布 N(μ, σ²) 的C-R下界
    sigma_sq = 1
    cr_bound = sigma_sq / sample_sizes
    
    # UMVUE的方差达到C-R下界
    umvue_var = cr_bound
    
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=cr_bound,
        mode='lines',
        line=dict(color='#FF3B30', width=3),
        name='C-R下界',
        showlegend=True
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=umvue_var,
        mode='lines',
        line=dict(color='#34C759', width=2, dash='dash'),
        name='UMVUE方差',
        showlegend=True
    ), row=2, col=2)
    
    fig.update_xaxes(title_text='样本量 n', row=2, col=2)
    fig.update_yaxes(title_text='方差', row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='UMVUE与Lehmann-Scheffe定理',
            font=dict(size=18)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/umvue_lehmann_scheffe.png', width=1000, height=800)
    return fig


def plot_history():
    """图5：历史发展 - 全屏水平时间线"""
    fig = go.Figure()
    
    events = [
        (1922, 'Fisher《论理论统计学的数学基础》', '充分性概念', '#007AFF'),
        (1945, 'Rao《信息线与估计的精确性》', 'Rao-Blackwell定理', '#FF3B30'),
        (1947, 'Blackwell《条件期望与充分统计量》', '严格证明', '#FF3B30'),
        (1950, 'Lehmann-Scheffe定理', '完备性与UMVUE', '#34C759'),
        (1946, 'Cramer-Rao不等式', '方差下界', '#FF9500'),
        (1949, 'Rao《高级统计方法》', '系统阐述', '#AF52DE'),
        (1953, 'Lehmann《检验统计假设》', '经典教材', '#007AFF'),
    ]
    
    events.sort(key=lambda x: x[0])
    years = [e[0] for e in events]
    
    # 时间线占满整个宽度
    x_min, x_max = 1920, 1960
    
    # 分配y位置避免重叠：远离开来点，交错分布
    y_positions = [1.2, -1.0, 1.0, -1.2, 0.7, -0.7, 1.4]  # 自定义分布
    
    for i, (year, event, desc, color) in enumerate(events):
        y_pos = y_positions[i]
        y_offset = 1 if y_pos > 0 else -1
        
        # 标记点（在时间线上）
        fig.add_trace(go.Scatter(
            x=[year],
            y=[0],
            mode='markers',
            marker=dict(size=16, color=color, line=dict(color='white', width=2)),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'{year}: {event}'
        ))
        
        # 年份标签
        fig.add_trace(go.Scatter(
            x=[year],
            y=[0.12 if y_offset > 0 else -0.12],
            mode='text',
            text=[str(year)],
            textposition='top center' if y_offset > 0 else 'bottom center',
            textfont=dict(size=11, color='#333', family='Arial Black'),
            showlegend=False
        ))
        
        # 连接线
        fig.add_trace(go.Scatter(
            x=[year, year],
            y=[0.05 if y_offset > 0 else -0.05, y_pos * 0.75],
            mode='lines',
            line=dict(color=color, width=1.5),
            showlegend=False
        ))
        
        # 简化事件名称
        short_names = {
            'Fisher《论理论统计学的数学基础》': 'Fisher《理论统计学基础》',
            'Rao《信息线与估计的精确性》': 'Rao《信息线与估计精确性》',
            'Blackwell《条件期望与充分统计量》': 'Blackwell《条件期望》',
            'Lehmann-Scheffe定理': 'Lehmann-Scheffe定理',
            'Cramer-Rao不等式': 'Cramer-Rao不等式',
            'Rao《高级统计方法》': 'Rao《高级统计方法》',
            'Lehmann《检验统计假设》': 'Lehmann《统计假设检验》'
        }
        short_event = short_names.get(event, event)
        
        # 事件名称 - 分行显示
        fig.add_trace(go.Scatter(
            x=[year],
            y=[y_pos],
            mode='text',
            text=[short_event],
            textposition='top center' if y_offset > 0 else 'bottom center',
            textfont=dict(size=10, color='#222', family='Arial'),
            showlegend=False
        ))
        
        # 描述
        fig.add_trace(go.Scatter(
            x=[year],
            y=[y_pos - 0.25 if y_offset > 0 else y_pos + 0.25],
            mode='text',
            text=[desc],
            textposition='top center' if y_offset > 0 else 'bottom center',
            textfont=dict(size=8, color='#666'),
            showlegend=False
        ))
    
    # 主时间线
    fig.add_trace(go.Scatter(
        x=[x_min - 1, x_max + 1],
        y=[0, 0],
        mode='lines',
        line=dict(color='#888', width=2.5),
        showlegend=False
    ))
    
    # 阶段背景色块
    fig.add_vrect(
        x0=1920, x1=1945,
        fillcolor='rgba(0, 122, 255, 0.06)',
        line_width=0,
        layer='below'
    )
    
    fig.add_vrect(
        x0=1945, x1=1955,
        fillcolor='rgba(255, 59, 48, 0.06)',
        line_width=0,
        layer='below'
    )
    
    # 阶段标签
    fig.add_annotation(
        x=1932.5, y=1.8,
        text='充分性概念发展',
        showarrow=False,
        font=dict(size=12, color='#007AFF'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#007AFF',
        borderwidth=1,
        borderpad=3
    )
    
    fig.add_annotation(
        x=1950, y=1.8,
        text='RB定理与完善',
        showarrow=False,
        font=dict(size=12, color='#FF3B30'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#FF3B30',
        borderwidth=1,
        borderpad=3
    )
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=10),
        width=1400,
        height=480,
        title=dict(
            text='Rao-Blackwell定理发展历程（1922-1953）',
            font=dict(size=20, family='Arial'),
            x=0.5
        ),
        xaxis=dict(
            title='年份',
            tickmode='linear',
            dtick=5,
            range=[x_min - 1.5, x_max + 1.5],
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            gridwidth=1,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-1.8, 2.2]
        ),
        margin=dict(l=40, r=40, t=80, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/rao_blackwell_history.png', width=1400, height=480)
    return fig


if __name__ == '__main__':
    print("开始生成Rao-Blackwell定理配图...")
    
    print("\n1. 生成条件期望图...")
    plot_conditional_expectation()
    
    print("\n2. 生成方差缩减图...")
    plot_variance_reduction()
    
    print("\n3. 生成分充分统计量图...")
    plot_sufficiency()
    
    print("\n4. 生成UMVUE图...")
    plot_umvue()
    
    print("\n5. 生成历史发展图...")
    plot_history()
    
    print("\n✅ 所有配图生成完成！")
