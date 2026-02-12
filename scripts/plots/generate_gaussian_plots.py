import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import os

os.makedirs('static/images/plots', exist_ok=True)

def apply_apple_style(fig, title, width=800, height=500):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family='Arial, sans-serif')),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=width,
        height=height,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# ============ 图1: 棣莫弗-拉普拉斯极限定理 ============
print("生成图1: 二项分布收敛到正态分布...")

fig1 = go.Figure()

n_values = [10, 30, 100]
colors = ['#FF9500', '#34C759', '#007AFF']

for n, color in zip(n_values, colors):
    p = 0.5
    k = np.arange(0, n+1)
    binom_pmf = stats.binom.pmf(k, n, p)
    
    mu = n * p
    sigma = np.sqrt(n * p * (1-p))
    z = (k - mu) / sigma
    
    fig1.add_trace(go.Scatter(
        x=z, y=binom_pmf * sigma,
        mode='markers+lines',
        name=f'n={n}',
        marker=dict(size=6, color=color),
        line=dict(width=2, color=color)
    ))

x = np.linspace(-3.5, 3.5, 200)
normal_pdf = stats.norm.pdf(x, 0, 1)

fig1.add_trace(go.Scatter(
    x=x, y=normal_pdf,
    mode='lines',
    name='标准正态分布',
    line=dict(width=3, color='#FF3B30', dash='solid')
))

fig1.update_layout(
    xaxis_title='标准化变量 z',
    yaxis_title='概率密度',
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)
apply_apple_style(fig1, '棣莫弗-拉普拉斯极限定理：二项分布收敛到正态分布', 900, 500)
fig1.write_image('static/images/plots/gaussian-de-moivre-limit.png', scale=2)

# ============ 图2: 中心极限定理 - 骰子求和 ============
print("生成图2: 中心极限定理演示...")

fig2 = make_subplots(rows=2, cols=2, 
                     subplot_titles=('1个骰子 (均匀分布)', '2个骰子之和', 
                                   '5个骰子之和', '10个骰子之和'))

n_dice_list = [1, 2, 5, 10]
positions = [(1,1), (1,2), (2,1), (2,2)]

for n_dice, pos in zip(n_dice_list, positions):
    np.random.seed(42)
    samples = np.random.randint(1, 7, size=(10000, n_dice))
    sums = np.sum(samples, axis=1)
    
    counts, bins = np.histogram(sums, bins=range(n_dice, 6*n_dice+2), density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    fig2.add_trace(go.Bar(
        x=bin_centers, y=counts,
        marker_color='#007AFF',
        opacity=0.7,
        showlegend=False,
        width=0.8
    ), row=pos[0], col=pos[1])
    
    if n_dice > 1:
        mu = 3.5 * n_dice
        sigma = np.sqrt(35/12 * n_dice)
        x_theory = np.linspace(n_dice, 6*n_dice, 100)
        y_theory = stats.norm.pdf(x_theory, mu, sigma)
        
        fig2.add_trace(go.Scatter(
            x=x_theory, y=y_theory,
            mode='lines',
            line=dict(color='#FF3B30', width=3),
            showlegend=False
        ), row=pos[0], col=pos[1])

apply_apple_style(fig2, '', 900, 700)
fig2.write_image('static/images/plots/gaussian-clt-dice.png', scale=2)

# ============ 图3: 正态分布密度函数与68-95-99.7规则 ============
print("生成图3: 正态分布与经验法则...")

fig3 = go.Figure()

x = np.linspace(-4, 4, 500)
y = stats.norm.pdf(x, 0, 1)

fig3.add_trace(go.Scatter(
    x=x, y=y,
    mode='lines',
    name='标准正态分布',
    line=dict(width=3, color='#007AFF'),
    fill='tozeroy',
    fillcolor='rgba(0,122,255,0.1)'
))

x_68 = np.linspace(-1, 1, 100)
y_68 = stats.norm.pdf(x_68, 0, 1)
fig3.add_trace(go.Scatter(
    x=x_68, y=y_68,
    mode='lines',
    line=dict(width=0),
    fill='tozeroy',
    fillcolor='rgba(0,122,255,0.4)',
    name='68% (±1σ)',
    showlegend=True
))

x_95 = np.linspace(-2, 2, 100)
y_95 = stats.norm.pdf(x_95, 0, 1)
fig3.add_trace(go.Scatter(
    x=x_95, y=y_95,
    mode='lines',
    line=dict(width=0),
    fill='tozeroy',
    fillcolor='rgba(52,199,89,0.3)',
    name='95% (±2σ)',
    showlegend=True
))

fig3.add_annotation(x=0, y=0.42, text='68%', showarrow=False, font=dict(size=16, color='#007AFF'))
fig3.add_annotation(x=0, y=0.08, text='95%', showarrow=False, font=dict(size=14, color='#34C759'))
fig3.add_annotation(x=0, y=0.02, text='99.7%', showarrow=False, font=dict(size=12, color='#FF9500'))

for sigma in [1, 2, 3]:
    fig3.add_vline(x=sigma, line_dash="dash", line_color="gray", opacity=0.3)
    fig3.add_vline(x=-sigma, line_dash="dash", line_color="gray", opacity=0.3)

fig3.update_layout(
    xaxis_title='标准差 (σ)',
    yaxis_title='概率密度',
    showlegend=True,
    legend=dict(x=0.7, y=0.95)
)
apply_apple_style(fig3, '正态分布的68-95-99.7经验法则', 900, 500)
fig3.write_image('static/images/plots/gaussian-empirical-rule.png', scale=2)

# ============ 图4: 误差分布对比 ============
print("生成图4: 误差分布对比...")

fig4 = go.Figure()

x = np.linspace(-4, 4, 500)

y_gauss = stats.norm.pdf(x, 0, 1)
fig4.add_trace(go.Scatter(
    x=x, y=y_gauss,
    mode='lines',
    name='正态分布 (高斯误差)',
    line=dict(width=3, color='#007AFF')
))

y_laplace = stats.laplace.pdf(x, 0, 1/np.sqrt(2))
fig4.add_trace(go.Scatter(
    x=x, y=y_laplace,
    mode='lines',
    name='拉普拉斯分布',
    line=dict(width=3, color='#FF9500', dash='dash')
))

y_uniform = np.where(np.abs(x) <= np.sqrt(3), 1/(2*np.sqrt(3)), 0)
fig4.add_trace(go.Scatter(
    x=x, y=y_uniform,
    mode='lines',
    name='均匀分布',
    line=dict(width=3, color='#34C759', dash='dot')
))

fig3.update_layout(
    xaxis_title='误差大小',
    yaxis_title='概率密度',
    showlegend=True,
    legend=dict(x=0.6, y=0.95)
)
apply_apple_style(fig4, '不同误差分布的比较', 900, 500)
fig4.write_image('static/images/plots/gaussian-error-distributions.png', scale=2)

# ============ 图5: 最大熵原理演示 ============
print("生成图5: 最大熵原理...")

fig5 = go.Figure()

x = np.linspace(-5, 5, 500)

y_normal = stats.norm.pdf(x, 0, 1.5)
fig5.add_trace(go.Scatter(
    x=x, y=y_normal,
    mode='lines',
    name='正态分布 (最大熵)',
    line=dict(width=3, color='#007AFF'),
    fill='tozeroy',
    fillcolor='rgba(0,122,255,0.2)'
))

y_bimodal = 0.5 * stats.norm.pdf(x, -1.5, 0.7) + 0.5 * stats.norm.pdf(x, 1.5, 0.7)
fig5.add_trace(go.Scatter(
    x=x, y=y_bimodal,
    mode='lines',
    name='双峰分布 (较低熵)',
    line=dict(width=3, color='#FF9500', dash='dash')
))

y_uniform_bounded = np.where(np.abs(x) <= 2.6, 1/(2*2.6), 0)
fig5.add_trace(go.Scatter(
    x=x, y=y_uniform_bounded,
    mode='lines',
    name='均匀分布 (较低熵)',
    line=dict(width=3, color='#34C759', dash='dot')
))

fig5.add_annotation(x=2.8, y=0.25, text='熵 ≈ 2.15', showarrow=False, 
                   font=dict(size=14, color='#007AFF'))
fig5.add_annotation(x=2.8, y=0.20, text='熵 ≈ 1.89', showarrow=False, 
                   font=dict(size=14, color='#FF9500'))
fig5.add_annotation(x=2.8, y=0.15, text='熵 ≈ 1.63', showarrow=False, 
                   font=dict(size=14, color='#34C759'))

fig5.update_layout(
    xaxis_title='x',
    yaxis_title='概率密度',
    showlegend=True,
    legend=dict(x=0.02, y=0.95)
)
apply_apple_style(fig5, '最大熵原理：相同方差下正态分布的熵最大', 900, 500)
fig5.write_image('static/images/plots/gaussian-maximum-entropy.png', scale=2)

print("所有图形生成完成!")
