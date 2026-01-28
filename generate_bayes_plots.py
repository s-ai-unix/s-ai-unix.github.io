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

# ============ 图1: 疾病诊断的贝叶斯更新 ============
print("生成图1: 疾病诊断的贝叶斯更新...")

fig1 = make_subplots(rows=1, cols=3, 
                     subplot_titles=('先验概率', '似然 (测试准确率)', '后验概率'),
                     specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]])

# 先验概率
categories = ['患病', '健康']
prior_probs = [0.01, 0.99]
colors_prior = ['#FF3B30', '#34C759']

fig1.add_trace(go.Bar(
    x=categories, y=prior_probs,
    marker_color=colors_prior,
    text=[f'{p:.1%}' for p in prior_probs],
    textposition='auto',
    showlegend=False
), row=1, col=1)

# 似然 - 测试准确率
likelihood_categories = ['真阳性\n(患病+)', '假阳性\n(健康+)']
likelihood_vals = [0.95, 0.05]
colors_likelihood = ['#007AFF', '#FF9500']

fig1.add_trace(go.Bar(
    x=likelihood_categories, y=likelihood_vals,
    marker_color=colors_likelihood,
    text=[f'{p:.0%}' for p in likelihood_vals],
    textposition='auto',
    showlegend=False
), row=1, col=2)

# 后验概率 - 真正患病的概率
posterior_categories = ['真阳性\n(确实患病)', '假阳性\n(实际健康)']
posterior_vals = [0.0095, 0.0495]  # 未归一化的联合概率
posterior_normalized = [p/sum(posterior_vals) for p in posterior_vals]

fig1.add_trace(go.Bar(
    x=posterior_categories, y=posterior_normalized,
    marker_color=['#007AFF', '#FF9500'],
    text=[f'{p:.1%}' for p in posterior_normalized],
    textposition='auto',
    showlegend=False
), row=1, col=3)

# 添加注释
fig1.add_annotation(x=0.5, y=-0.15, 
                   text='真正患病概率 = 9.5/(9.5+49.5) ≈ 16.1%',
                   showarrow=False, xref='paper', yref='paper',
                   font=dict(size=14, color='#007AFF'))

apply_apple_style(fig1, '', 1000, 450)
fig1.write_image('static/images/plots/bayes-disease-diagnosis.png', scale=2)

# ============ 图2: Beta-二项共轭先验的更新过程 ============
print("生成图2: Beta-二项共轭先验更新...")

fig2 = make_subplots(rows=2, cols=2,
                     subplot_titles=('先验: Beta(1,1) = 均匀', '更新1: 观察到3正7负 -> Beta(4,8)', 
                                   '更新2: 观察到10正5负 -> Beta(14,13)', '最终: 观察到30正20负 -> Beta(31,21)'),
                     specs=[[{'type': 'xy'}, {'type': 'xy'}],
                            [{'type': 'xy'}, {'type': 'xy'}]])

theta = np.linspace(0, 1, 500)

# 四个阶段的 Beta 分布
params = [(1, 1), (4, 8), (14, 13), (31, 21)]
positions = [(1,1), (1,2), (2,1), (2,2)]
colors_beta = ['#007AFF', '#34C759', '#FF9500', '#FF3B30']

for (alpha, beta), pos, color in zip(params, positions, colors_beta):
    y = stats.beta.pdf(theta, alpha, beta)
    
    fig2.add_trace(go.Scatter(
        x=theta, y=y,
        mode='lines',
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.3])}',
        line=dict(width=3, color=color),
        showlegend=False
    ), row=pos[0], col=pos[1])
    
    # 标记均值
    mean = alpha / (alpha + beta)
    fig2.add_vline(x=mean, line_dash="dash", line_color="gray", opacity=0.5,
                  row=pos[0], col=pos[1])
    fig2.add_annotation(x=mean, y=max(y)*0.9, text=f'均值={mean:.2f}',
                       showarrow=False, row=pos[0], col=pos[1],
                       font=dict(size=12))

apply_apple_style(fig2, '', 900, 700)
fig2.write_image('static/images/plots/bayes-beta-conjugate.png', scale=2)

# ============ 图3: 贝叶斯信念更新过程 ============
print("生成图3: 贝叶斯信念更新过程...")

fig3 = go.Figure()

# 模拟硬币偏差估计
true_bias = 0.7
n_trials = np.arange(0, 101, 1)

# 生成观测数据
np.random.seed(42)
observations = np.random.binomial(1, true_bias, size=100)
cumulative_heads = np.cumsum(observations)
cumulative_bias = cumulative_heads / np.arange(1, 101)

# 计算贝叶斯后验均值 (Beta先验)
# 先验 Beta(1,1) = 均匀
alpha_post = 1 + cumulative_heads
beta_post = 1 + np.arange(1, 101) - cumulative_heads
posterior_mean = alpha_post / (alpha_post + beta_post)

# 计算95%可信区间
from scipy.special import betaincinv
ci_lower = []
ci_upper = []
for a, b in zip(alpha_post, beta_post):
    ci_lower.append(stats.beta.ppf(0.025, a, b))
    ci_upper.append(stats.beta.ppf(0.975, a, b))

# 绘制
fig3.add_trace(go.Scatter(
    x=n_trials[1:], y=posterior_mean,
    mode='lines',
    name='后验均值',
    line=dict(width=3, color='#007AFF')
))

fig3.add_trace(go.Scatter(
    x=n_trials[1:], y=ci_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))

fig3.add_trace(go.Scatter(
    x=n_trials[1:], y=ci_lower,
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(0,122,255,0.2)',
    name='95% 可信区间',
    showlegend=True
))

# 真实值
fig3.add_hline(y=true_bias, line_dash="dash", line_color="#FF3B30", 
              annotation_text="真实偏差=0.7", annotation_position="right")

# 先验
fig3.add_hline(y=0.5, line_dash="dot", line_color="#34C759",
              annotation_text="先验均值=0.5", annotation_position="left")

fig3.update_layout(
    xaxis_title='观测次数',
    yaxis_title='硬币偏差估计 θ',
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)
apply_apple_style(fig3, '贝叶斯更新：硬币偏差估计的收敛过程', 900, 500)
fig3.write_image('static/images/plots/bayes-belief-update.png', scale=2)

# ============ 图4: 频率学派 vs 贝叶斯学派对比 ============
print("生成图4: 频率学派 vs 贝叶斯学派...")

fig4 = make_subplots(rows=1, cols=2,
                     subplot_titles=('频率学派观点', '贝叶斯学派观点'),
                     specs=[[{'type': 'xy'}, {'type': 'xy'}]])

# 左图: 频率学派 - 固定参数，随机数据
theta_true = 0.6
n_samples = 30
np.random.seed(42)
samples = np.random.binomial(1, theta_true, n_samples)
estimates = np.cumsum(samples) / np.arange(1, n_samples + 1)

fig4.add_trace(go.Scatter(
    x=list(range(1, n_samples+1)), y=estimates,
    mode='lines+markers',
    name='MLE估计',
    line=dict(width=2, color='#007AFF'),
    marker=dict(size=6)
), row=1, col=1)

fig4.add_hline(y=theta_true, line_dash="dash", line_color="#FF3B30",
              row=1, col=1)

fig4.add_annotation(x=15, y=0.65, text='参数 θ 是固定值', showarrow=False,
                   row=1, col=1, font=dict(size=14))

# 右图: 贝叶斯学派 - 参数是随机变量
theta_range = np.linspace(0, 1, 200)
# 不同数据量下的后验
for i, (n, color) in enumerate([(5, '#FF9500'), (15, '#34C759'), (30, '#007AFF')]):
    heads = sum(samples[:n])
    posterior = stats.beta.pdf(theta_range, 1+heads, 1+n-heads)
    
    fig4.add_trace(go.Scatter(
        x=theta_range, y=posterior,
        mode='lines',
        name=f'n={n}',
        line=dict(width=2, color=color),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}'
    ), row=1, col=2)

fig4.add_vline(x=theta_true, line_dash="dash", line_color="#FF3B30",
              row=1, col=2)

fig4.add_annotation(x=0.5, y=4, text='参数 θ 是随机变量', showarrow=False,
                   row=1, col=2, font=dict(size=14))

apply_apple_style(fig4, '', 1000, 450)
fig4.write_image('static/images/plots/bayes-frequentist-comparison.png', scale=2)

print("所有图形生成完成!")
