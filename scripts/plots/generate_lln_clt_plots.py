#!/usr/bin/env python3
"""
生成大数定律和中心极限定理的配图
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


def plot_lln_demonstration():
    """图1：大数定律演示 - 频率收敛到概率"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('抛硬币：正面频率收敛到0.5', '样本均值收敛到期望',
                       '不同样本量的分布收缩', '收敛速度对比'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    np.random.seed(42)
    
    # 左上：抛硬币频率收敛
    n_max = 5000
    n_points = np.arange(1, n_max + 1)
    # 模拟抛硬币
    coin_flips = np.random.binomial(1, 0.5, n_max)
    cumulative_heads = np.cumsum(coin_flips)
    frequency = cumulative_heads / n_points
    
    # 采样显示（避免数据点过多）
    sample_idx = np.concatenate([
        np.arange(0, 100, 1),
        np.arange(100, 1000, 10),
        np.arange(1000, n_max, 50)
    ])
    
    fig.add_trace(go.Scatter(
        x=n_points[sample_idx],
        y=frequency[sample_idx],
        mode='lines',
        line=dict(color='#007AFF', width=1.5),
        name='正面频率',
        showlegend=False
    ), row=1, col=1)
    
    # 添加理论概率线
    fig.add_hline(y=0.5, line=dict(color='#FF3B30', width=2, dash='dash'),
                  row=1, col=1)
    
    # 添加收敛区间
    epsilon = 0.05
    fig.add_hline(y=0.5+epsilon, line=dict(color='#34C759', width=1, dash='dot'),
                  row=1, col=1)
    fig.add_hline(y=0.5-epsilon, line=dict(color='#34C759', width=1, dash='dot'),
                  row=1, col=1)
    
    # 右上：样本均值收敛（不同分布）
    sample_sizes = [10, 50, 100, 500, 1000, 5000]
    colors = ['#FF3B30', '#FF9500', '#FFCC00', '#34C759', '#007AFF', '#5856D6']
    
    for i, n in enumerate(sample_sizes):
        # 指数分布，理论均值=1
        samples = np.random.exponential(1, n)
        sample_means = np.cumsum(samples) / np.arange(1, n + 1)
        
        fig.add_trace(go.Scatter(
            x=np.arange(1, n + 1),
            y=sample_means,
            mode='lines',
            line=dict(color=colors[i], width=2),
            name=f'n={n}',
            showlegend=True
        ), row=1, col=2)
    
    fig.add_hline(y=1.0, line=dict(color='#000000', width=2, dash='dash'),
                  row=1, col=2)
    
    # 左下：样本均值的分布随样本量收缩
    sample_sizes_dist = [30, 100, 500, 2000]
    n_simulations = 1000
    
    for i, n in enumerate(sample_sizes_dist):
        means = []
        for _ in range(n_simulations):
            samples = np.random.uniform(0, 1, n)
            means.append(np.mean(samples))
        
        # 绘制直方图
        fig.add_trace(go.Histogram(
            x=means,
            nbinsx=30,
            opacity=0.6,
            name=f'n={n}',
            marker_color=colors[i],
            showlegend=True,
            histnorm='probability density'
        ), row=2, col=1)
    
    # 添加理论均值线
    fig.add_vline(x=0.5, line=dict(color='#000000', width=2, dash='dash'),
                  row=2, col=1)
    
    # 右下：收敛速度（对数坐标）
    n_range = np.logspace(1, 4, 50).astype(int)
    n_range = np.unique(n_range)
    
    # 计算95%置信区间半宽
    std_pop = 1.0  # 总体标准差
    margin_errors = 1.96 * std_pop / np.sqrt(n_range)
    
    fig.add_trace(go.Scatter(
        x=n_range,
        y=margin_errors,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='误差界限 (95%)',
        showlegend=False
    ), row=2, col=2)
    
    # 添加理论收敛速度参考线 O(1/sqrt(n))
    ref_line = 3.0 / np.sqrt(n_range)
    fig.add_trace(go.Scatter(
        x=n_range,
        y=ref_line,
        mode='lines',
        line=dict(color='#FF9500', width=2, dash='dash'),
        name='O(1/√n)',
        showlegend=False
    ), row=2, col=2)
    
    fig.update_xaxes(type='log', row=2, col=2)
    fig.update_yaxes(type='log', row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='大数定律：样本均值收敛到理论期望',
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
    
    save_and_compress(fig, f'{OUTPUT_DIR}/lln_demonstration.png', width=1000, height=800)
    return fig


def plot_clt_demonstration():
    """图2：中心极限定理演示 - 不同分布趋向正态"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('均匀分布 n=2', '指数分布 n=2', '伯努利分布 n=2',
                       '均匀分布 n=30', '指数分布 n=30', '伯努利分布 n=30'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    np.random.seed(123)
    n_simulations = 5000
    
    distributions = [
        ('uniform', lambda n: np.random.uniform(0, 1, n), 0.5, 1/np.sqrt(12)),
        ('exponential', lambda n: np.random.exponential(1, n), 1.0, 1.0),
        ('bernoulli', lambda n: np.random.binomial(1, 0.5, n), 0.5, 0.5)
    ]
    
    sample_sizes = [2, 30]
    colors = ['#007AFF', '#FF9500']
    
    for col, (dist_name, dist_func, mu, sigma) in enumerate(distributions, 1):
        for row, n in enumerate(sample_sizes, 1):
            # 生成样本均值
            sample_means = []
            for _ in range(n_simulations):
                samples = dist_func(n)
                sample_means.append(np.mean(samples))
            
            # 标准化
            standardized = (np.array(sample_means) - mu) / (sigma / np.sqrt(n))
            
            # 绘制直方图
            fig.add_trace(go.Histogram(
                x=standardized,
                nbinsx=40,
                opacity=0.7,
                marker_color=colors[row-1],
                showlegend=False,
                histnorm='probability density'
            ), row=row, col=col)
            
            # 叠加标准正态分布曲线
            x_range = np.linspace(-4, 4, 100)
            normal_curve = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_range**2)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_curve,
                mode='lines',
                line=dict(color='#FF3B30', width=2),
                showlegend=False
            ), row=row, col=col)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1200,
        height=800,
        title=dict(
            text='中心极限定理：不同分布的样本均值趋向正态分布',
            font=dict(size=18)
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/clt_demonstration.png', width=1200, height=800)
    return fig


def plot_clt_accuracy():
    """图3：CLT近似精度与样本量关系"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('正态近似误差随样本量变化', '不同置信水平下的样本量需求'),
        horizontal_spacing=0.12
    )
    
    np.random.seed(456)
    
    # 左图：Kolmogorov-Smirnov统计量随样本量变化
    sample_sizes = [5, 10, 20, 30, 50, 100, 200, 500, 1000]
    n_simulations = 2000
    
    ks_stats_uniform = []
    ks_stats_exponential = []
    
    for n in sample_sizes:
        # 均匀分布
        means_uniform = []
        for _ in range(n_simulations):
            means_uniform.append(np.mean(np.random.uniform(0, 1, n)))
        standardized_u = (np.array(means_uniform) - 0.5) / (1/np.sqrt(12*n))
        # 计算与标准正态的KS统计量
        from scipy import stats
        ks_stat_u, _ = stats.kstest(standardized_u, 'norm')
        ks_stats_uniform.append(ks_stat_u)
        
        # 指数分布
        means_exp = []
        for _ in range(n_simulations):
            means_exp.append(np.mean(np.random.exponential(1, n)))
        standardized_e = (np.array(means_exp) - 1.0) / (1/np.sqrt(n))
        ks_stat_e, _ = stats.kstest(standardized_e, 'norm')
        ks_stats_exponential.append(ks_stat_e)
    
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=ks_stats_uniform,
        mode='lines+markers',
        line=dict(color='#007AFF', width=2),
        marker=dict(size=8),
        name='均匀分布'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=ks_stats_exponential,
        mode='lines+markers',
        line=dict(color='#FF9500', width=2),
        marker=dict(size=8),
        name='指数分布'
    ), row=1, col=1)
    
    # 添加参考线 O(1/sqrt(n))
    ref_line = 0.5 / np.sqrt(np.array(sample_sizes))
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=ref_line,
        mode='lines',
        line=dict(color='#34C759', width=2, dash='dash'),
        name='O(1/√n)参考'
    ), row=1, col=1)
    
    fig.update_xaxes(type='log', row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=1)
    
    # 右图：不同置信水平和误差要求下的样本量
    confidence_levels = [0.90, 0.95, 0.99]
    margin_errors = np.linspace(0.01, 0.2, 50)
    
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    colors_conf = ['#007AFF', '#34C759', '#FF3B30']
    
    for i, conf in enumerate(confidence_levels):
        z = z_scores[conf]
        # 假设总体标准差为1
        n_required = (z / margin_errors) ** 2
        
        fig.add_trace(go.Scatter(
            x=margin_errors,
            y=n_required,
            mode='lines',
            line=dict(color=colors_conf[i], width=2),
            name=f'{int(conf*100)}% 置信度'
        ), row=1, col=2)
    
    fig.update_yaxes(type='log', row=1, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1100,
        height=500,
        title=dict(
            text='中心极限定理的实用精度分析',
            font=dict(size=18)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/clt_accuracy.png', width=1100, height=500)
    return fig


def plot_lln_clt_relationship():
    """图4：大数定律与中心极限定理的关系"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('收敛性对比：LLN vs CLT', '标准化后的分布演化',
                       '置信区间宽度变化', '两个定理的互补性'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    np.random.seed(789)
    
    # 左上：LLN（收敛到常数）vs CLT（收敛到分布）
    n_max = 2000
    n_trials = 100
    
    sample_sizes = np.logspace(1, np.log10(n_max), 20).astype(int)
    sample_sizes = np.unique(sample_sizes)
    
    means_distribution = []
    for n in sample_sizes:
        trial_means = []
        for _ in range(n_trials):
            samples = np.random.uniform(0, 1, n)
            trial_means.append(np.mean(samples))
        means_distribution.append(trial_means)
    
    # 绘制箱线图显示分布收缩
    fig.add_trace(go.Box(
        x=np.repeat(sample_sizes, n_trials),
        y=np.concatenate(means_distribution),
        name='样本均值分布',
        marker_color='#007AFF',
        boxmean=True,
        showlegend=False
    ), row=1, col=1)
    
    fig.add_hline(y=0.5, line=dict(color='#FF3B30', width=2, dash='dash'),
                  row=1, col=1)
    
    fig.update_xaxes(type='log', row=1, col=1)
    
    # 右上：标准化后的分布（CLT）
    n_values = [10, 50, 200]
    n_simulations = 2000
    
    for i, n in enumerate(n_values):
        means = []
        for _ in range(n_simulations):
            means.append(np.mean(np.random.uniform(0, 1, n)))
        
        # 标准化
        standardized = (np.array(means) - 0.5) / (1/np.sqrt(12*n))
        
        fig.add_trace(go.Histogram(
            x=standardized,
            nbinsx=40,
            opacity=0.5,
            name=f'n={n}',
            marker_color=['#007AFF', '#34C759', '#FF9500'][i],
            showlegend=True,
            histnorm='probability density'
        ), row=1, col=2)
    
    # 叠加标准正态
    x_range = np.linspace(-4, 4, 100)
    normal_curve = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_range**2)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_curve,
        mode='lines',
        line=dict(color='#FF3B30', width=3),
        name='标准正态',
        showlegend=True
    ), row=1, col=2)
    
    # 左下：置信区间宽度
    n_range = np.logspace(1, 4, 100)
    z_95 = 1.96
    sigma = 1.0
    
    ci_width = 2 * z_95 * sigma / np.sqrt(n_range)
    
    fig.add_trace(go.Scatter(
        x=n_range,
        y=ci_width,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='95% CI宽度',
        showlegend=False
    ), row=2, col=1)
    
    # LLN：区间收缩到0
    fig.add_hline(y=0, line=dict(color='#FF3B30', width=2, dash='dash'),
                  row=2, col=1)
    
    fig.update_xaxes(type='log', row=2, col=1)
    fig.update_yaxes(type='log', row=2, col=1)
    
    # 右下：互补性示意
    # 样本均值的分布随n变化
    n_display = [5, 20, 100]
    y_offset = [0, 0.5, 1.0]
    
    for i, (n, offset) in enumerate(zip(n_display, y_offset)):
        means = []
        for _ in range(3000):
            means.append(np.mean(np.random.uniform(0, 1, n)))
        
        # 为可视化添加偏移
        means_shifted = np.array(means) + offset
        
        # 核密度估计（简化版）
        hist, bins = np.histogram(means_shifted, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=hist + offset,
            mode='lines',
            fill='tozeroy',
            line=dict(color=['#007AFF', '#34C759', '#FF9500'][i], width=2),
            name=f'n={n}',
            showlegend=True
        ), row=2, col=2)
        
        # 标记期望位置
        fig.add_vline(x=0.5+offset, line=dict(color='#FF3B30', width=2, dash='dash'),
                      row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='大数定律与中心极限定理的关系',
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
    
    save_and_compress(fig, f'{OUTPUT_DIR}/lln_clt_relationship.png', width=1000, height=800)
    return fig


def plot_historical_timeline():
    """图5：历史发展时间线 - 全屏布局"""
    fig = go.Figure()
    
    events = [
        (1713, '伯努利《猜度术》', '弱大数定律', '#007AFF'),
        (1733, '棣莫弗-拉普拉斯', '二项分布正态近似', '#007AFF'),
        (1812, '拉普拉斯《分析概率论》', '一般CLT雏形', '#34C759'),
        (1837, '泊松大数定律', '独立不同分布', '#34C759'),
        (1867, '切比雪夫不等式', '概率论严格化', '#FF9500'),
        (1901, '李雅普诺夫CLT', '特征函数方法', '#FF9500'),
        (1909, '波莱尔强大数定律', '几乎必然收敛', '#AF52DE'),
        (1922, '林德伯格-莱维CLT', '独立同分布CLT', '#AF52DE'),
        (1933, '柯尔莫哥洛夫公理化', '现代概率论基础', '#FF3B30'),
        (1935, '林德伯格-费勒CLT', '充要条件', '#FF3B30'),
    ]
    
    # 按年份排序
    events.sort(key=lambda x: x[0])
    
    years = [e[0] for e in events]
    x_min, x_max = 1690, 1960
    
    # 分配y位置避免重叠：LLN在上(正)，CLT在下(负)，更大间距
    y_positions = [1.6, -1.2, 1.3, -0.9, 1.0, -0.6, 0.7, -1.5, 1.4, -1.8]
    
    for i, (year, event, desc, color) in enumerate(events):
        y_pos = y_positions[i]
        y_offset = 1 if y_pos > 0 else -1
        
        # 事件点（在时间线上）
        fig.add_trace(go.Scatter(
            x=[year],
            y=[0],
            mode='markers',
            marker=dict(size=28, color=color, line=dict(color='white', width=4)),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'{year}: {event}'
        ))
        
        # 年份标签
        fig.add_trace(go.Scatter(
            x=[year],
            y=[0.22 if y_offset > 0 else -0.22],
            mode='text',
            text=[str(year)],
            textposition='top center' if y_offset > 0 else 'bottom center',
            textfont=dict(size=20, color='#333', family='Arial Black'),
            showlegend=False
        ))
        
        # 连接线
        fig.add_trace(go.Scatter(
            x=[year, year],
            y=[0.1 if y_offset > 0 else -0.1, y_pos * 0.75],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False
        ))
        
        # 简化事件名称
        short_names = {
            '伯努利《猜度术》': '伯努利《猜度术》',
            '棣莫弗-拉普拉斯': '棣莫弗-拉普拉斯',
            '拉普拉斯《分析概率论》': '拉普拉斯《分析概率论》',
            '泊松大数定律': '泊松大数定律',
            '切比雪夫不等式': '切比雪夫不等式',
            '李雅普诺夫CLT': '李雅普诺夫CLT',
            '波莱尔强大数定律': '波莱尔强大数定律',
            '林德伯格-莱维CLT': '林德伯格-莱维CLT',
            '柯尔莫哥洛夫公理化': '柯尔莫哥洛夫公理化',
            '林德伯格-费勒CLT': '林德伯格-费勒CLT'
        }
        short_event = short_names.get(event, event)
        
        # 事件名称
        fig.add_trace(go.Scatter(
            x=[year],
            y=[y_pos],
            mode='text',
            text=[short_event],
            textposition='top center' if y_offset > 0 else 'bottom center',
            textfont=dict(size=18, color='#222', family='Arial'),
            showlegend=False
        ))
        
        # 描述
        fig.add_trace(go.Scatter(
            x=[year],
            y=[y_pos - 0.55 if y_offset > 0 else y_pos + 0.55],
            mode='text',
            text=[desc],
            textposition='top center' if y_offset > 0 else 'bottom center',
            textfont=dict(size=14, color='#666'),
            showlegend=False
        ))
    
    # 主时间线
    fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[0, 0],
        mode='lines',
        line=dict(color='#888', width=2.5),
        showlegend=False
    ))
    
    # 添加LLN和CLT区域标记
    fig.add_vrect(
        x0=1700, x1=1850,
        fillcolor='rgba(0, 122, 255, 0.06)',
        line_width=0,
        layer='below'
    )
    
    fig.add_vrect(
        x0=1700, x1=1940,
        fillcolor='rgba(52, 199, 89, 0.06)',
        line_width=0,
        layer='below'
    )
    
    # 阶段标签
    fig.add_annotation(
        x=1775, y=2.5,
        text='大数定律发展期',
        showarrow=False,
        font=dict(size=22, color='#007AFF'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#007AFF',
        borderwidth=3,
        borderpad=8
    )
    
    fig.add_annotation(
        x=1820, y=-2.5,
        text='中心极限定理发展期',
        showarrow=False,
        font=dict(size=22, color='#34C759'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#34C759',
        borderwidth=3,
        borderpad=8
    )
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=16),
        width=2400,
        height=1000,
        title=dict(
            text='大数定律与中心极限定理的发展历程（1713-1935）',
            font=dict(size=32, family='Arial'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='年份', font=dict(size=20)),
            tickmode='linear',
            dtick=20,
            range=[x_min, x_max],
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-2.8, 2.8]
        ),
        margin=dict(l=100, r=100, t=140, b=100),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/lln_clt_history.png', width=2400, height=1000)
    return fig


def plot_real_world_applications():
    """图6：实际应用场景"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('民意调查样本量计算', '蒙特卡洛积分收敛',
                       '质量控制中的均值检验', '保险风险聚合'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    np.random.seed(999)
    
    # 左上：民意调查 - 不同置信度和误差下的样本量
    confidence_levels = [0.90, 0.95, 0.99]
    margin_errors = np.linspace(0.01, 0.1, 50)
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    colors = ['#007AFF', '#34C759', '#FF3B30']
    
    for i, conf in enumerate(confidence_levels):
        z = z_scores[conf]
        # 假设p=0.5（最坏情况）
        p = 0.5
        n_required = (z**2 * p * (1-p)) / (margin_errors ** 2)
        
        fig.add_trace(go.Scatter(
            x=margin_errors * 100,  # 转换为百分比
            y=n_required,
            mode='lines',
            line=dict(color=colors[i], width=2.5),
            name=f'{int(conf*100)}% 置信度',
            showlegend=True
        ), row=1, col=1)
    
    # 标注常用点
    fig.add_annotation(x=3, y=1100, text='±3%, 95%', showarrow=True, arrowhead=2,
                      row=1, col=1)
    
    fig.update_yaxes(type='log', row=1, col=1)
    
    # 右上：蒙特卡洛积分
    true_value = np.pi / 4  # 单位圆面积的1/4
    sample_sizes_mc = np.logspace(2, 5, 50).astype(int)
    sample_sizes_mc = np.unique(sample_sizes_mc)
    
    estimates = []
    errors = []
    
    for n in sample_sizes_mc:
        # 估计 pi/4 = E[I(X^2+Y^2 <= 1)]
        x = np.random.uniform(0, 1, n)
        y = np.random.uniform(0, 1, n)
        estimate = np.mean(x**2 + y**2 <= 1)
        estimates.append(estimate)
        errors.append(abs(estimate - true_value))
    
    fig.add_trace(go.Scatter(
        x=sample_sizes_mc,
        y=errors,
        mode='markers',
        marker=dict(size=6, color='#007AFF'),
        name='估计误差',
        showlegend=False
    ), row=1, col=2)
    
    # 添加理论收敛速度
    ref_line = 0.5 / np.sqrt(sample_sizes_mc)
    fig.add_trace(go.Scatter(
        x=sample_sizes_mc,
        y=ref_line,
        mode='lines',
        line=dict(color='#FF3B30', width=2, dash='dash'),
        name='O(1/√n)',
        showlegend=False
    ), row=1, col=2)
    
    fig.update_xaxes(type='log', row=1, col=2)
    fig.update_yaxes(type='log', row=1, col=2)
    
    # 左下：质量控制 - 样本均值监控
    # 模拟生产过程
    n_batches = 50
    sample_size = 30
    target_mean = 100
    process_std = 5
    
    batch_means = []
    control_limits_upper = []
    control_limits_lower = []
    
    for batch in range(n_batches):
        # 偶尔加入偏移模拟异常
        if 20 <= batch < 25:
            offset = 3
        else:
            offset = 0
        
        samples = np.random.normal(target_mean + offset, process_std, sample_size)
        batch_means.append(np.mean(samples))
        
        # 控制限 (95% CLT)
        se = process_std / np.sqrt(sample_size)
        control_limits_upper.append(target_mean + 1.96 * se)
        control_limits_lower.append(target_mean - 1.96 * se)
    
    batch_numbers = list(range(1, n_batches + 1))
    
    fig.add_trace(go.Scatter(
        x=batch_numbers,
        y=batch_means,
        mode='lines+markers',
        line=dict(color='#007AFF', width=2),
        marker=dict(size=6),
        name='批次均值',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=batch_numbers,
        y=control_limits_upper,
        mode='lines',
        line=dict(color='#FF3B30', width=2, dash='dash'),
        name='控制上限',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=batch_numbers,
        y=control_limits_lower,
        mode='lines',
        line=dict(color='#FF3B30', width=2, dash='dash'),
        name='控制下限',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_hline(y=target_mean, line=dict(color='#34C759', width=2),
                  row=2, col=1)
    
    # 右下：保险风险聚合
    # 模拟大数定律在保险中的应用
    n_policies = np.logspace(2, 5, 30).astype(int)
    n_policies = np.unique(n_policies)
    
    expected_claim = 1000
    claim_std = 5000
    
    # 每份保单的风险（标准差）
    individual_risk = claim_std
    
    # 聚合风险（标准差）随保单数量变化
    aggregated_risk = claim_std / np.sqrt(n_policies)
    
    # 相对风险（风险/期望赔付）
    relative_risk = aggregated_risk / expected_claim
    
    fig.add_trace(go.Scatter(
        x=n_policies,
        y=relative_risk * 100,  # 转换为百分比
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='相对风险',
        showlegend=False
    ), row=2, col=2)
    
    fig.update_xaxes(type='log', row=2, col=2)
    
    fig.add_annotation(x=1000, y=15, text='风险分散效应',
                      showarrow=True, arrowhead=2, row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='大数定律与中心极限定理的实际应用',
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
    
    save_and_compress(fig, f'{OUTPUT_DIR}/lln_clt_applications.png', width=1000, height=800)
    return fig


if __name__ == '__main__':
    print("开始生成大数定律和中心极限定理配图...")
    
    print("\n1. 生成大数定律演示图...")
    plot_lln_demonstration()
    
    print("\n2. 生成中心极限定理演示图...")
    plot_clt_demonstration()
    
    print("\n3. 生成CLT精度分析图...")
    plot_clt_accuracy()
    
    print("\n4. 生成LLN与CLT关系图...")
    plot_lln_clt_relationship()
    
    print("\n5. 生成历史发展时间线...")
    plot_historical_timeline()
    
    print("\n6. 生成实际应用场景图...")
    plot_real_world_applications()
    
    print("\n✅ 所有配图生成完成！")
