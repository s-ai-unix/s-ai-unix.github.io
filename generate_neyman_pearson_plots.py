#!/usr/bin/env python3
"""
生成Neyman-Pearson引理的配图
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


def plot_hypothesis_testing_concept():
    """图1：假设检验的基本概念"""
    import scipy.stats as stats
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('零假设与备择假设的分布', '两类错误的关系',
                       '检验的拒绝域', '功效函数示意'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    np.random.seed(42)
    
    # 左上：两个假设的分布
    x = np.linspace(-4, 6, 500)
    
    # H0: N(0, 1)
    h0_dist = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
    # H1: N(2, 1)
    h1_dist = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * (x-2)**2)
    
    fig.add_trace(go.Scatter(
        x=x, y=h0_dist,
        mode='lines',
        line=dict(color='#007AFF', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 122, 255, 0.2)',
        name='H₀: N(0,1)',
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=h1_dist,
        mode='lines',
        line=dict(color='#FF3B30', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(255, 59, 48, 0.2)',
        name='H₁: N(2,1)',
        showlegend=True
    ), row=1, col=1)
    
    # 标记拒绝域
    rejection_threshold = 1.645
    fig.add_vline(x=rejection_threshold, line=dict(color='#FF9500', width=2, dash='dash'), row=1, col=1)
    
    # 右上：两类错误的关系
    alpha_values = np.linspace(0.01, 0.5, 100)
    # 简化的beta-alpha关系（实际取决于具体分布）
    beta_values = 1 - (1 - alpha_values)**1.5  # 示意关系
    
    fig.add_trace(go.Scatter(
        x=alpha_values,
        y=beta_values,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='α-β关系',
        showlegend=False
    ), row=1, col=2)
    
    # 标记Neyman-Pearson最优点的概念
    fig.add_trace(go.Scatter(
        x=[0.05],
        y=[0.2],
        mode='markers',
        marker=dict(size=12, color='#FF3B30', symbol='star'),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_annotation(x=0.05, y=0.25, text='NP最优点', showarrow=False, row=1, col=2)
    
    fig.update_xaxes(title_text='第一类错误率 α', row=1, col=2)
    fig.update_yaxes(title_text='第二类错误率 β', row=1, col=2)
    
    # 左下：拒绝域的可视化
    x_reject = x[x >= rejection_threshold]
    h0_reject = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_reject**2)
    h1_reject = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * (x_reject-2)**2)
    
    fig.add_trace(go.Scatter(
        x=x, y=h0_dist,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='H₀',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=h1_dist,
        mode='lines',
        line=dict(color='#FF3B30', width=2),
        name='H₁',
        showlegend=False
    ), row=2, col=1)
    
    # 填充拒绝域（alpha）
    fig.add_trace(go.Scatter(
        x=np.concatenate([[rejection_threshold], x_reject, [6]]),
        y=np.concatenate([[0], h0_reject, [0]]),
        fill='toself',
        fillcolor='rgba(255, 149, 0, 0.4)',
        line=dict(width=0),
        name='α',
        showlegend=True
    ), row=2, col=1)
    
    # 填充接受域中的H1部分（beta）
    x_accept = x[x < rejection_threshold]
    h1_accept = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * (x_accept-2)**2)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([[-4], x_accept, [rejection_threshold]]),
        y=np.concatenate([[0], h1_accept, [0]]),
        fill='toself',
        fillcolor='rgba(52, 199, 89, 0.3)',
        line=dict(width=0),
        name='β',
        showlegend=True
    ), row=2, col=1)
    
    fig.add_vline(x=rejection_threshold, line=dict(color='#000000', width=2, dash='dash'), row=2, col=1)
    
    # 右下：功效函数
    theta_values = np.linspace(-1, 4, 200)
    # 检验 H0: theta=0 vs H1: theta>0，拒绝域 X > c
    c = 1.645  # alpha = 0.05
    power = 1 - stats.norm.cdf(c - theta_values)  # P(X > c | theta)
    
    import scipy.stats as stats
    
    fig.add_trace(go.Scatter(
        x=theta_values,
        y=power,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='功效函数',
        showlegend=False
    ), row=2, col=2)
    
    # 标记alpha点
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0.05],
        mode='markers',
        marker=dict(size=10, color='#FF3B30'),
        showlegend=False
    ), row=2, col=2)
    
    fig.add_hline(y=0.05, line=dict(color='#FF9500', width=1, dash='dot'), row=2, col=2)
    fig.add_vline(x=0, line=dict(color='#FF9500', width=1, dash='dot'), row=2, col=2)
    
    fig.update_xaxes(title_text='参数 θ', row=2, col=2)
    fig.update_yaxes(title_text='功效 π(θ)', row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='假设检验的基本概念与两类错误',
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
    
    save_and_compress(fig, f'{OUTPUT_DIR}/hypothesis_testing_concept.png', width=1000, height=800)
    return fig


def plot_likelihood_ratio():
    """图2：似然比与最优检验"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('似然函数对比', '似然比 Λ(x)',
                       'Neyman-Pearson检验的拒绝域', 'ROC曲线示意'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    np.random.seed(123)
    
    import scipy.stats as stats
    
    # 设置参数
    mu0, mu1 = 0, 2
    sigma = 1
    
    x = np.linspace(-4, 6, 500)
    
    # 左上：似然函数
    L0 = stats.norm.pdf(x, mu0, sigma)
    L1 = stats.norm.pdf(x, mu1, sigma)
    
    fig.add_trace(go.Scatter(
        x=x, y=L0,
        mode='lines',
        line=dict(color='#007AFF', width=2.5),
        name='L(θ₀|x)',
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=L1,
        mode='lines',
        line=dict(color='#FF3B30', width=2.5),
        name='L(θ₁|x)',
        showlegend=True
    ), row=1, col=1)
    
    # 右上：似然比
    Lambda = L1 / L0
    # 限制范围以便可视化
    Lambda = np.clip(Lambda, 0, 20)
    
    fig.add_trace(go.Scatter(
        x=x, y=Lambda,
        mode='lines',
        line=dict(color='#AF52DE', width=3),
        name='Λ(x) = L(θ₁|x)/L(θ₀|x)',
        showlegend=True
    ), row=1, col=2)
    
    # 标记临界值
    k = np.exp(mu1 - mu0)  # 对应某个alpha
    fig.add_hline(y=k, line=dict(color='#FF9500', width=2, dash='dash'), row=1, col=2)
    fig.add_annotation(x=-2, y=k+1, text='临界值 k', showarrow=False, row=1, col=2)
    
    fig.update_yaxes(title_text='似然比 Λ(x)', row=1, col=2)
    
    # 左下：拒绝域
    # 找出拒绝域（似然比大的区域）
    rejection_region = x[Lambda > k]
    
    fig.add_trace(go.Scatter(
        x=x, y=L0,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='H₀密度',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=L1,
        mode='lines',
        line=dict(color='#FF3B30', width=2),
        name='H₁密度',
        showlegend=False
    ), row=2, col=1)
    
    # 标记拒绝域
    if len(rejection_region) > 0:
        reject_start = rejection_region[0]
        x_reject = x[x >= reject_start]
        L0_reject = stats.norm.pdf(x_reject, mu0, sigma)
        L1_reject = stats.norm.pdf(x_reject, mu1, sigma)
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([[reject_start], x_reject, [6]]),
            y=np.concatenate([[0], L0_reject, [0]]),
            fill='toself',
            fillcolor='rgba(255, 149, 0, 0.3)',
            line=dict(width=0),
            name='拒绝域',
            showlegend=True
        ), row=2, col=1)
    
    # 右下：ROC曲线
    # 不同阈值下的 (alpha, power)
    thresholds = np.linspace(-3, 5, 100)
    alphas = 1 - stats.norm.cdf(thresholds, mu0, sigma)
    powers = 1 - stats.norm.cdf(thresholds, mu1, sigma)
    
    fig.add_trace(go.Scatter(
        x=alphas,
        y=powers,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='NP检验的ROC曲线',
        showlegend=True
    ), row=2, col=2)
    
    # 对角线（随机猜测）
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='#CCCCCC', width=2, dash='dash'),
        name='随机猜测',
        showlegend=True
    ), row=2, col=2)
    
    # 标记特定点
    alpha_star = 0.05
    power_star = 1 - stats.norm.cdf(stats.norm.ppf(1-alpha_star, mu0, sigma), mu1, sigma)
    
    fig.add_trace(go.Scatter(
        x=[alpha_star],
        y=[power_star],
        mode='markers',
        marker=dict(size=12, color='#FF3B30', symbol='star'),
        name='α=0.05',
        showlegend=True
    ), row=2, col=2)
    
    fig.update_xaxes(title_text='假阳性率 α', row=2, col=2)
    fig.update_yaxes(title_text='真阳性率 (1-β)', row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='似然比与Neyman-Pearson最优检验',
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
    
    save_and_compress(fig, f'{OUTPUT_DIR}/likelihood_ratio.png', width=1000, height=800)
    return fig


def plot_ump_example():
    """图3：UMP检验示例（单调似然比）"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('不同备择假设的检验', '功效函数对比',
                       'Karlin-Rubin定理示意', '一致最优性'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    import scipy.stats as stats
    
    # 设置：X ~ N(theta, 1)，检验 H0: theta=0 vs H1: theta>0
    theta0 = 0
    theta_values = np.array([0.5, 1.0, 2.0])
    
    # 左上：不同theta下的分布
    x = np.linspace(-3, 5, 300)
    
    fig.add_trace(go.Scatter(
        x=x,
        y=stats.norm.pdf(x, theta0, 1),
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name=f'θ={theta0} (H₀)',
        showlegend=True
    ), row=1, col=1)
    
    colors = ['#34C759', '#FF9500', '#FF3B30']
    for i, theta in enumerate(theta_values):
        fig.add_trace(go.Scatter(
            x=x,
            y=stats.norm.pdf(x, theta, 1),
            mode='lines',
            line=dict(color=colors[i], width=2),
            name=f'θ={theta}',
            showlegend=True
        ), row=1, col=1)
    
    # 拒绝域（统一临界值）
    c = 1.645
    fig.add_vline(x=c, line=dict(color='#000000', width=2, dash='dash'), row=1, col=1)
    
    # 右上：功效函数
    theta_range = np.linspace(-0.5, 3, 200)
    power_function = 1 - stats.norm.cdf(c - theta_range)
    
    fig.add_trace(go.Scatter(
        x=theta_range,
        y=power_function,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='UMP检验功效',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_hline(y=0.05, line=dict(color='#FF9500', width=1, dash='dot'), row=1, col=2)
    fig.add_vline(x=0, line=dict(color='#FF9500', width=1, dash='dot'), row=1, col=2)
    
    # 标记不同theta下的功效
    for i, theta in enumerate(theta_values):
        power = 1 - stats.norm.cdf(c - theta)
        fig.add_trace(go.Scatter(
            x=[theta],
            y=[power],
            mode='markers',
            marker=dict(size=10, color=colors[i]),
            showlegend=False
        ), row=1, col=2)
    
    fig.update_xaxes(title_text='θ', row=1, col=2)
    fig.update_yaxes(title_text='功效 π(θ)', row=1, col=2)
    
    # 左下：单调似然比
    x_mlr = np.linspace(-3, 5, 200)
    theta1, theta2 = 1, 2
    
    # 似然比 L(theta2)/L(theta1)
    lr = np.exp((theta2 - theta1) * x_mlr - 0.5 * (theta2**2 - theta1**2))
    
    fig.add_trace(go.Scatter(
        x=x_mlr,
        y=lr,
        mode='lines',
        line=dict(color='#AF52DE', width=3),
        name=f'L(θ={theta2})/L(θ={theta1})',
        showlegend=True
    ), row=2, col=1)
    
    fig.update_xaxes(title_text='x', row=2, col=1)
    fig.update_yaxes(title_text='似然比', row=2, col=1)
    
    # 右下：比较不同检验
    # 检验1：UMP检验（拒绝X > c）
    # 检验2：另一个检验（如拒绝 |X| > c'，双侧检验）
    
    c_two_sided = 1.96  # 双侧alpha=0.05
    
    power_ump = 1 - stats.norm.cdf(1.645 - theta_range)
    power_two_sided = (1 - stats.norm.cdf(1.96 - theta_range)) + stats.norm.cdf(-1.96 - theta_range)
    
    fig.add_trace(go.Scatter(
        x=theta_range,
        y=power_ump,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='UMP（单侧）',
        showlegend=True
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=theta_range,
        y=power_two_sided,
        mode='lines',
        line=dict(color='#FF9500', width=2, dash='dash'),
        name='双侧检验',
        showlegend=True
    ), row=2, col=2)
    
    fig.update_xaxes(title_text='θ > 0', row=2, col=2)
    fig.update_yaxes(title_text='功效', row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='一致最优势（UMP）检验与单调似然比',
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
    
    save_and_compress(fig, f'{OUTPUT_DIR}/ump_example.png', width=1000, height=800)
    return fig


def plot_history_timeline():
    """图4：历史发展时间线"""
    fig = go.Figure()
    
    events = [
        (1900, '皮尔逊《科学的哲学》', '拟合优度检验', '#007AFF'),
        (1928, 'Neyman-Pearson引理', '最优检验理论', '#FF3B30'),
        (1933, 'Neyman-Pearson论文', '假设检验框架', '#FF3B30'),
        (1934, 'Karlin-Rubin定理', 'MLR与UMP', '#34C759'),
        (1937, 'Neyman置信区间', '对偶性', '#007AFF'),
        (1949, 'Wald序列分析', '序贯检验', '#FF9500'),
        (1950, 'Lehmann《检验统计假设》', '经典教科书', '#AF52DE'),
        (1988, 'Berger《统计决策理论》', '决策论视角', '#007AFF'),
    ]
    
    events.sort(key=lambda x: x[0])
    
    years = [e[0] for e in events]
    y_pos = list(range(len(events)))
    
    for i, (year, event, desc, color) in enumerate(events):
        fig.add_trace(go.Scatter(
            x=[year],
            y=[i],
            mode='markers',
            marker=dict(size=18, color=color, line=dict(color='white', width=2)),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'{year}: {event}'
        ))
        
        fig.add_trace(go.Scatter(
            x=[year + 2],
            y=[i],
            mode='text',
            text=[event],
            textposition='middle left',
            textfont=dict(size=11, color='#333'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[year + 2],
            y=[i - 0.35],
            mode='text',
            text=[desc],
            textposition='middle left',
            textfont=dict(size=9, color='#666'),
            showlegend=False
        ))
    
    fig.add_trace(go.Scatter(
        x=years,
        y=y_pos,
        mode='lines',
        line=dict(color='#ccc', width=2),
        showlegend=False
    ))
    
    # 添加时期标记
    fig.add_vrect(
        x0=1900, x1=1930,
        fillcolor='rgba(0, 122, 255, 0.1)',
        line_width=0,
        annotation_text='早期发展',
        annotation_position='top'
    )
    
    fig.add_vrect(
        x0=1928, x1=1950,
        fillcolor='rgba(255, 59, 48, 0.1)',
        line_width=0,
        annotation_text='NP框架形成',
        annotation_position='bottom'
    )
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1100,
        height=600,
        title=dict(
            text='Neyman-Pearson理论发展历程（1900-1988）',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='年份',
            tickmode='linear',
            dtick=20,
            range=[1895, 2000]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        margin=dict(l=50, r=350, t=100, b=60)
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/neyman_pearson_history.png', width=1100, height=600)
    return fig


def plot_applications():
    """图5：实际应用场景"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('雷达信号检测', '医学诊断检验',
                       'A/B测试样本量', '质量控制边界'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    import scipy.stats as stats
    np.random.seed(999)
    
    # 左上：雷达信号检测（SNR vs 检测概率）
    snr_db = np.linspace(-10, 20, 100)
    snr_linear = 10**(snr_db/10)
    
    # Neyman-Pearson检测器（高斯噪声中的已知信号）
    # 检测概率 = Q(Q^{-1}(PFA) - sqrt(SNR))
    PFA = 0.01  # 虚警概率
    Q_inv_PFA = stats.norm.ppf(1 - PFA)
    PD = 1 - stats.norm.cdf(Q_inv_PFA - np.sqrt(snr_linear))
    
    fig.add_trace(go.Scatter(
        x=snr_db,
        y=PD,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='NP检测器',
        showlegend=False
    ), row=1, col=1)
    
    fig.update_xaxes(title_text='信噪比 (dB)', row=1, col=1)
    fig.update_yaxes(title_text='检测概率 Pd', row=1, col=1)
    
    # 右上：ROC曲线（医学诊断）
    # 不同检验统计量的ROC
    alphas = np.linspace(0, 1, 100)
    
    # 好的检验（AUC ≈ 0.9）
    powers_good = alphas**0.25
    # 差的检验（AUC ≈ 0.6）
    powers_poor = alphas**2
    
    fig.add_trace(go.Scatter(
        x=alphas,
        y=powers_good,
        mode='lines',
        line=dict(color='#34C759', width=3),
        name='优秀检验',
        showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=alphas,
        y=powers_poor,
        mode='lines',
        line=dict(color='#FF3B30', width=3),
        name='较差检验',
        showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='#CCCCCC', width=2, dash='dash'),
        name='随机猜测',
        showlegend=True
    ), row=1, col=2)
    
    fig.update_xaxes(title_text='假阳性率', row=1, col=2)
    fig.update_yaxes(title_text='真阳性率', row=1, col=2)
    
    # 左下：A/B测试样本量计算
    # 功效分析
    effect_sizes = np.linspace(0.1, 0.5, 50)
    sample_sizes_80 = []
    sample_sizes_90 = []
    
    for d in effect_sizes:
        # 简化公式 n = 16/d^2 for 80% power, alpha=0.05
        n_80 = 16 / d**2
        n_90 = 21 / d**2
        sample_sizes_80.append(n_80)
        sample_sizes_90.append(n_90)
    
    fig.add_trace(go.Scatter(
        x=effect_sizes,
        y=sample_sizes_80,
        mode='lines',
        line=dict(color='#007AFF', width=2.5),
        name='功效80%',
        showlegend=True
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=effect_sizes,
        y=sample_sizes_90,
        mode='lines',
        line=dict(color='#FF9500', width=2.5),
        name='功效90%',
        showlegend=True
    ), row=2, col=1)
    
    fig.update_xaxes(title_text='效应量 (Cohen\'s d)', row=2, col=1)
    fig.update_yaxes(title_text='每组样本量 n', type='log', row=2, col=1)
    
    # 右下：质量控制边界
    n_samples = np.arange(5, 101)
    sigma = 1
    
    # 3-sigma控制限（随n变化）
    control_limit_3 = 3 * sigma / np.sqrt(n_samples)
    control_limit_2 = 2 * sigma / np.sqrt(n_samples)
    
    fig.add_trace(go.Scatter(
        x=n_samples,
        y=control_limit_3,
        mode='lines',
        line=dict(color='#FF3B30', width=2),
        name='3σ边界',
        showlegend=True
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=n_samples,
        y=control_limit_2,
        mode='lines',
        line=dict(color='#FF9500', width=2, dash='dash'),
        name='2σ边界',
        showlegend=True
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=n_samples,
        y=-control_limit_3,
        mode='lines',
        line=dict(color='#FF3B30', width=2),
        showlegend=False
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=n_samples,
        y=-control_limit_2,
        mode='lines',
        line=dict(color='#FF9500', width=2, dash='dash'),
        showlegend=False
    ), row=2, col=2)
    
    fig.update_xaxes(title_text='样本量 n', row=2, col=2)
    fig.update_yaxes(title_text='控制限', row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='Neyman-Pearson检验的实际应用',
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
    
    save_and_compress(fig, f'{OUTPUT_DIR}/neyman_pearson_applications.png', width=1000, height=800)
    return fig


if __name__ == '__main__':
    print("开始生成Neyman-Pearson引理配图...")
    
    print("\n1. 生成假设检验概念图...")
    plot_hypothesis_testing_concept()
    
    print("\n2. 生成似然比图...")
    plot_likelihood_ratio()
    
    print("\n3. 生成UMP检验示例图...")
    plot_ump_example()
    
    print("\n4. 生成历史发展时间线...")
    plot_history_timeline()
    
    print("\n5. 生成应用场景图...")
    plot_applications()
    
    print("\n✅ 所有配图生成完成！")
