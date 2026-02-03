import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import subprocess
import os

# 确保目录存在
os.makedirs('static/images/plots', exist_ok=True)

def save_and_compress(fig, filepath, width=900, height=600):
    """保存并压缩图片"""
    fig.write_image(filepath, width=width, height=height, scale=2)
    
    # 压缩
    if filepath.endswith('.png') and os.path.exists('/opt/homebrew/bin/pngquant'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存: {filepath}")

# 图1: 条件期望直观解释
def plot_conditional_expectation_intuition():
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('无条件分布', '条件分布'),
                        horizontal_spacing=0.12)
    
    # 骰子结果
    outcomes = np.array([1, 2, 3, 4, 5, 6])
    probs = np.ones(6) / 6
    
    # 左图：无条件分布
    fig.add_trace(go.Bar(x=outcomes, y=probs, 
                         marker_color='#007AFF',
                         name='P(X=x)',
                         showlegend=False), row=1, col=1)
    
    # 添加无条件期望线
    mean_uncond = 3.5
    fig.add_vline(x=mean_uncond, line=dict(color='#FF3B30', dash='dash', width=2),
                  annotation=dict(text=f'E[X]=3.5', font=dict(color='#FF3B30')),
                  row=1, col=1)
    
    # 右图：条件分布（奇数）
    odd_outcomes = np.array([1, 3, 5])
    odd_probs = np.ones(3) / 3
    
    fig.add_trace(go.Bar(x=odd_outcomes, y=odd_probs,
                         marker_color='#34C759',
                         name='P(X=x|奇数)',
                         showlegend=False), row=1, col=2)
    
    # 添加条件期望线
    mean_cond = 3
    fig.add_vline(x=mean_cond, line=dict(color='#FF3B30', dash='dash', width=2),
                  annotation=dict(text=f'E[X|奇数]=3', font=dict(color='#FF3B30')),
                  row=1, col=2)
    
    fig.update_layout(
        title='条件期望：以骰子为例',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text='结果', row=1, col=1)
    fig.update_xaxes(title_text='结果（给定奇数）', row=1, col=2)
    fig.update_yaxes(title_text='概率', row=1, col=1)
    fig.update_yaxes(title_text='条件概率', row=1, col=2)
    
    save_and_compress(fig, 'static/images/plots/conditional-expectation-intuition.png')

# 图2: 二元正态分布的条件期望
def plot_bivariate_normal_conditional():
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('联合分布等高线与回归线', '条件分布'),
                        horizontal_spacing=0.12)
    
    # 二元正态分布参数
    mu_x, mu_y = 0, 0
    sigma_x, sigma_y = 1, 1
    rho = 0.7
    
    # 生成网格
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # 计算联合密度
    Z = stats.multivariate_normal.pdf(np.dstack([X, Y]), 
                                       mean=[mu_x, mu_y],
                                       cov=[[sigma_x**2, rho*sigma_x*sigma_y],
                                            [rho*sigma_x*sigma_y, sigma_y**2]])
    
    # 左图：等高线
    contour = go.Contour(x=x, y=y, z=Z, 
                         colorscale='Blues',
                         showscale=False,
                         contours=dict(coloring='fill'))
    fig.add_trace(contour, row=1, col=1)
    
    # 添加回归线 E[Y|X=x]
    x_line = np.linspace(-3, 3, 100)
    y_line = mu_y + rho * (sigma_y/sigma_x) * (x_line - mu_x)
    fig.add_trace(go.Scatter(x=x_line, y=y_line, 
                             mode='lines', 
                             line=dict(color='#FF3B30', width=3),
                             name='E[Y|X=x]',
                             showlegend=False), row=1, col=2)
    
    # 右图：条件分布（给定 X=1）
    x0 = 1
    y_cond = np.linspace(-3, 3, 100)
    mu_cond = mu_y + rho * (sigma_y/sigma_x) * (x0 - mu_x)
    sigma_cond = sigma_y * np.sqrt(1 - rho**2)
    
    density = stats.norm.pdf(y_cond, mu_cond, sigma_cond)
    
    fig.add_trace(go.Scatter(x=density, y=y_cond,
                             mode='lines',
                             fill='tozerox',
                             fillcolor='rgba(0, 122, 255, 0.3)',
                             line=dict(color='#007AFF', width=2),
                             showlegend=False), row=1, col=2)
    
    # 添加条件均值线
    fig.add_hline(y=mu_cond, line=dict(color='#FF3B30', dash='dash', width=2),
                  annotation=dict(text=f'E[Y|X=1]={mu_cond:.2f}'),
                  row=1, col=2)
    
    fig.update_layout(
        title='二元正态分布的条件期望',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text='X', row=1, col=1)
    fig.update_xaxes(title_text='条件密度', row=1, col=2)
    fig.update_yaxes(title_text='Y', row=1, col=1)
    fig.update_yaxes(title_text='Y', row=1, col=2)
    
    save_and_compress(fig, 'static/images/plots/bivariate-normal-conditional.png')

# 图3: 方差分解
def plot_variance_decomposition():
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('数据分布与组间差异', '方差分解示意'),
                        horizontal_spacing=0.15)
    
    np.random.seed(42)
    
    # 生成三组数据（对应不同X值）
    n_per_group = 100
    group_means = [2, 5, 8]
    group_stds = [0.8, 1.0, 0.6]
    
    all_y = []
    all_x = []
    colors = ['#007AFF', '#34C759', '#FF9500']
    
    for i, (mean, std) in enumerate(zip(group_means, group_stds)):
        y = np.random.normal(mean, std, n_per_group)
        x = np.random.normal(i+1, 0.1, n_per_group)
        all_y.extend(y)
        all_x.extend([i+1] * n_per_group)
        
        # 左图：散点
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(color=colors[i], size=6, opacity=0.6),
                                 showlegend=False), row=1, col=1)
        
        # 添加组均值线
        fig.add_hline(y=mean, line=dict(color=colors[i], dash='dash', width=2),
                      row=1, col=1)
    
    # 添加总均值线
    total_mean = np.mean(all_y)
    fig.add_hline(y=total_mean, line=dict(color='#FF3B30', width=3),
                  annotation=dict(text=f'总均值={total_mean:.2f}'),
                  row=1, col=1)
    
    # 右图：方差分解饼图
    # 计算方差分解
    within_var = np.mean([std**2 for std in group_stds])
    between_var = np.var(group_means)
    
    labels = ['组内方差\n(Within)', '组间方差\n(Between)']
    values = [within_var, between_var]
    
    fig.add_trace(go.Pie(labels=labels, values=values,
                         marker=dict(colors=['#007AFF', '#FF9500']),
                         textinfo='label+percent',
                         showlegend=False), row=1, col=2)
    
    fig.update_layout(
        title='方差分解：信息X的价值',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text='组别（X值）', row=1, col=1)
    fig.update_yaxes(title_text='Y值', row=1, col=1)
    
    save_and_compress(fig, 'static/images/plots/variance-decomposition.png')

# 图4: Rao-Blackwell 定理
def plot_rao_blackwell():
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('原始估计量', 'Rao-Blackwell改进'),
                        horizontal_spacing=0.12)
    
    np.random.seed(42)
    
    # 模拟数据
    n = 200
    # 充分统计量 T（x轴）
    T = np.random.normal(5, 1, n)
    
    # 原始估计量（有噪声）
    theta_hat = T + np.random.normal(0, 0.8, n)
    
    # Rao-Blackwell改进（对T取条件期望，即投影到T的函数）
    # 简化为：theta_hat* = E[theta_hat | T] = T（假设T是充分统计量）
    theta_hat_star = T
    
    # 左图：原始估计量
    fig.add_trace(go.Scatter(x=T, y=theta_hat, mode='markers',
                             marker=dict(color='#007AFF', size=6, opacity=0.6),
                             showlegend=False), row=1, col=1)
    
    # 添加真实值线
    fig.add_hline(y=5, line=dict(color='#FF3B30', width=2),
                  annotation=dict(text='真实值'),
                  row=1, col=1)
    
    # 右图：改进估计量
    fig.add_trace(go.Scatter(x=T, y=theta_hat_star, mode='markers',
                             marker=dict(color='#34C759', size=6, opacity=0.6),
                             showlegend=False), row=1, col=2)
    
    fig.add_hline(y=5, line=dict(color='#FF3B30', width=2),
                  annotation=dict(text='真实值'),
                  row=1, col=2)
    
    # 添加方差标注
    var_orig = np.var(theta_hat - 5)
    var_improved = np.var(theta_hat_star - 5)
    
    fig.add_annotation(x=0.95, y=0.95, xref='paper', yref='paper',
                       text=f'方差: {var_orig:.3f}',
                       showarrow=False, bgcolor='rgba(255,255,255,0.8)',
                       row=1, col=1)
    
    fig.add_annotation(x=0.95, y=0.95, xref='paper', yref='paper',
                       text=f'方差: {var_improved:.3f}',
                       showarrow=False, bgcolor='rgba(255,255,255,0.8)',
                       row=1, col=2)
    
    fig.update_layout(
        title='Rao-Blackwell定理：条件期望降低方差',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text='充分统计量 T', row=1, col=1)
    fig.update_xaxes(title_text='充分统计量 T', row=1, col=2)
    fig.update_yaxes(title_text='估计值', row=1, col=1)
    fig.update_yaxes(title_text='改进估计值', row=1, col=2)
    
    save_and_compress(fig, 'static/images/plots/rao-blackwell-theorem.png')

# 图5: 回归作为条件期望估计
def plot_regression_conditional_expectation():
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('线性回归', '非参数回归（核平滑）'),
                        horizontal_spacing=0.12)
    
    np.random.seed(42)
    
    # 生成非线性数据
    n = 300
    x = np.random.uniform(-3, 3, n)
    # 真实条件期望：正弦函数
    y_true = np.sin(x) + 0.5 * x
    y = y_true + np.random.normal(0, 0.5, n)
    
    # 排序以便绘制
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    y_true_sorted = y_true[idx]
    
    # 左图：线性回归
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                             marker=dict(color='#007AFF', size=5, opacity=0.4),
                             name='数据',
                             showlegend=False), row=1, col=1)
    
    # 线性拟合
    coeffs = np.polyfit(x, y, 1)
    y_linear = np.polyval(coeffs, x_sorted)
    fig.add_trace(go.Scatter(x=x_sorted, y=y_linear, mode='lines',
                             line=dict(color='#FF3B30', width=2),
                             name='线性回归',
                             showlegend=False), row=1, col=1)
    
    # 右图：核平滑
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                             marker=dict(color='#007AFF', size=5, opacity=0.4),
                             name='数据',
                             showlegend=False), row=1, col=2)
    
    # Nadaraya-Watson 核回归
    from scipy.stats import norm
    
    def kernel_regression(x_query, x_data, y_data, bandwidth=0.5):
        weights = norm.pdf((x_query[:, None] - x_data[None, :]) / bandwidth)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights @ y_data
    
    x_grid = np.linspace(-3, 3, 200)
    y_kernel = kernel_regression(x_grid, x, y, bandwidth=0.5)
    
    fig.add_trace(go.Scatter(x=x_grid, y=y_kernel, mode='lines',
                             line=dict(color='#34C759', width=2),
                             name='核回归',
                             showlegend=False), row=1, col=2)
    
    # 添加真实条件期望
    y_true_grid = np.sin(x_grid) + 0.5 * x_grid
    fig.add_trace(go.Scatter(x=x_grid, y=y_true_grid, mode='lines',
                             line=dict(color='#FF9500', width=2, dash='dash'),
                             name='真实E[Y|X]',
                             showlegend=False), row=1, col=2)
    
    fig.update_layout(
        title='回归分析：估计条件期望 E[Y|X]',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text='X', row=1, col=1)
    fig.update_xaxes(title_text='X', row=1, col=2)
    fig.update_yaxes(title_text='Y', row=1, col=1)
    fig.update_yaxes(title_text='Y', row=1, col=2)
    
    save_and_compress(fig, 'static/images/plots/regression-conditional-expectation.png')

# 图6: VAE中的条件期望
def plot_vae_conditional_expectation():
    fig = go.Figure()
    
    # VAE结构示意图
    # 输入层
    fig.add_trace(go.Scatter(x=[1], y=[0], mode='markers+text',
                             marker=dict(size=60, color='#007AFF'),
                             text=['输入 x'],
                             textposition='middle center',
                             textfont=dict(size=11, color='white'),
                             showlegend=False))
    
    # 编码器
    fig.add_trace(go.Scatter(x=[2.5], y=[0], mode='markers+text',
                             marker=dict(size=60, color='#34C759'),
                             text=['编码器'],
                             textposition='middle center',
                             textfont=dict(size=11, color='white'),
                             showlegend=False))
    
    # 潜在变量分布（参数化条件期望）
    fig.add_trace(go.Scatter(x=[4], y=[0.5], mode='markers+text',
                             marker=dict(size=55, color='#FF9500'),
                             text=['μ(x)'],
                             textposition='middle center',
                             textfont=dict(size=10, color='white'),
                             showlegend=False))
    
    fig.add_trace(go.Scatter(x=[4], y=[-0.5], mode='markers+text',
                             marker=dict(size=55, color='#FF9500'),
                             text=['σ(x)'],
                             textposition='middle center',
                             textfont=dict(size=10, color='white'),
                             showlegend=False))
    
    # 采样
    fig.add_trace(go.Scatter(x=[5.5], y=[0], mode='markers+text',
                             marker=dict(size=55, color='#AF52DE'),
                             text=['采样 z'],
                             textposition='middle center',
                             textfont=dict(size=10, color='white'),
                             showlegend=False))
    
    # 解码器
    fig.add_trace(go.Scatter(x=[7], y=[0], mode='markers+text',
                             marker=dict(size=60, color='#34C759'),
                             text=['解码器'],
                             textposition='middle center',
                             textfont=dict(size=11, color='white'),
                             showlegend=False))
    
    # 输出（重构）
    fig.add_trace(go.Scatter(x=[8.5], y=[0], mode='markers+text',
                             marker=dict(size=60, color='#007AFF'),
                             text=['重构 x̂'],
                             textposition='middle center',
                             textfont=dict(size=11, color='white'),
                             showlegend=False))
    
    # 添加箭头连接
    arrows = [
        (1.5, 0, 2, 0),  # 输入->编码器
        (3, 0, 3.7, 0.4),  # 编码器->μ
        (3, 0, 3.7, -0.4),  # 编码器->σ
        (4.5, 0.5, 5, 0.1),  # μ->采样
        (4.5, -0.5, 5, -0.1),  # σ->采样
        (6, 0, 6.5, 0),  # 采样->解码器
        (7.5, 0, 8, 0),  # 解码器->输出
    ]
    
    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(x=x0, y=y0, ax=x1, ay=y1,
                           xref='data', yref='data',
                           axref='data', ayref='data',
                           showarrow=True,
                           arrowhead=2, arrowsize=1.5, arrowwidth=1.5,
                           arrowcolor='#8E8E93')
    
    # 添加注释
    fig.add_annotation(x=4, y=1.5, text='q(z|x) = N(μ(x), σ²(x))',
                       showarrow=False, font=dict(size=10))
    
    fig.add_annotation(x=8.5, y=-1.5, text='p(x|z) = N(解码器(z), σ²)',
                       showarrow=False, font=dict(size=10))
    
    fig.update_layout(
        title='变分自编码器（VAE）中的条件期望',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        xaxis=dict(range=[0, 9.5], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-2, 2], showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(l=30, r=30, t=80, b=30),
        height=450
    )
    
    save_and_compress(fig, 'static/images/plots/vae-conditional-expectation.png', width=950, height=450)

# 图7: 强化学习中的值函数
def plot_rl_value_function():
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('状态值函数 V(s)', '动作值函数 Q(s,a)'),
                        horizontal_spacing=0.12,
                        specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    
    # 创建网格
    s = np.linspace(0, 10, 50)
    a = np.linspace(-5, 5, 50)
    S, A = np.meshgrid(s, a)
    
    # 状态值函数（简单模型）
    V = 10 * np.sin(S * 0.5) + 5
    
    # 动作值函数
    Q = 10 * np.sin(S * 0.5) - 0.5 * A**2 + 2 * A * np.cos(S * 0.3) + 5
    
    # 左图：V(s)
    fig.add_surface(x=S, y=np.zeros_like(S), z=V,
                    colorscale='Viridis',
                    showscale=False,
                    opacity=0.8,
                    row=1, col=1)
    
    # 右图：Q(s,a)
    fig.add_surface(x=S, y=A, z=Q,
                    colorscale='Plasma',
                    showscale=False,
                    opacity=0.8,
                    row=1, col=2)
    
    fig.update_layout(
        title='强化学习中的值函数（条件期望）',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=50, r=50, t=80, b=50),
        scene=dict(xaxis_title='状态 s', yaxis_title='', zaxis_title='V(s)'),
        scene2=dict(xaxis_title='状态 s', yaxis_title='动作 a', zaxis_title='Q(s,a)')
    )
    
    save_and_compress(fig, 'static/images/plots/rl-value-function.png', width=950, height=500)

# 运行所有绘图函数
if __name__ == '__main__':
    print("开始生成配图...")
    plot_conditional_expectation_intuition()
    plot_bivariate_normal_conditional()
    plot_variance_decomposition()
    plot_rao_blackwell()
    plot_regression_conditional_expectation()
    plot_vae_conditional_expectation()
    plot_rl_value_function()
    print("所有配图生成完成！")
