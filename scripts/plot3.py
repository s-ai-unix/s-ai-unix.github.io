import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

fig = make_subplots(rows=1, cols=2, subplot_titles=('数据分布与组间差异', '方差分解示意'), horizontal_spacing=0.15)

np.random.seed(42)
n_per_group = 100
group_means = [2, 5, 8]
group_stds = [0.8, 1.0, 0.6]
colors = ['#007AFF', '#34C759', '#FF9500']
all_y, all_x = [], []

for i, (mean, std) in enumerate(zip(group_means, group_stds)):
    y = np.random.normal(mean, std, n_per_group)
    x = np.random.normal(i+1, 0.1, n_per_group)
    all_y.extend(y)
    all_x.extend([i+1] * n_per_group)
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color=colors[i], size=6, opacity=0.6), showlegend=False), row=1, col=1)
    fig.add_hline(y=mean, line=dict(color=colors[i], dash='dash', width=2), row=1, col=1)

total_mean = np.mean(all_y)
fig.add_hline(y=total_mean, line=dict(color='#FF3B30', width=3), annotation=dict(text=f'总均值={total_mean:.2f}'), row=1, col=1)

within_var = np.mean([std**2 for std in group_stds])
between_var = np.var(group_means)

# 使用柱状图代替饼图
fig.add_trace(go.Bar(x=['组内方差\n(Within)', '组间方差\n(Between)'], 
                     y=[within_var, between_var],
                     marker_color=['#007AFF', '#FF9500'],
                     text=[f'{within_var:.2f}', f'{between_var:.2f}'],
                     textposition='auto',
                     showlegend=False), row=1, col=2)

fig.update_layout(title='方差分解：信息X的价值', template='plotly_white', font=dict(family='Arial', size=12))
fig.update_xaxes(title_text='组别（X值）', row=1, col=1)
fig.update_yaxes(title_text='Y值', row=1, col=1)
fig.update_xaxes(title_text='方差来源', row=1, col=2)
fig.update_yaxes(title_text='方差大小', row=1, col=2)

fig.write_image('static/images/plots/variance-decomposition.png', width=900, height=500, scale=2)
print("图3完成")
