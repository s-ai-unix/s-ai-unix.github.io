import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

fig = make_subplots(rows=1, cols=2, subplot_titles=('状态值函数 V(s)', '动作值函数 Q(s,a)'), horizontal_spacing=0.12)

# 状态值函数
s = np.linspace(0, 10, 100)
V = 10 * np.sin(s * 0.5) + 5

fig.add_trace(go.Scatter(x=s, y=V, mode='lines', line=dict(color='#007AFF', width=3), fill='tozeroy', 
                         fillcolor='rgba(0, 122, 255, 0.3)', showlegend=False), row=1, col=1)

# 动作值函数（等高线图）
s = np.linspace(0, 10, 50)
a = np.linspace(-5, 5, 50)
S, A = np.meshgrid(s, a)
Q = 10 * np.sin(S * 0.5) - 0.5 * A**2 + 2 * A * np.cos(S * 0.3) + 5

fig.add_trace(go.Contour(x=s, y=a, z=Q, colorscale='Viridis', showscale=True,
                         colorbar=dict(title='Q值', x=0.97, len=0.8)), row=1, col=2)

fig.update_layout(title='强化学习中的值函数（条件期望）', template='plotly_white', font=dict(family='Arial', size=12))
fig.update_xaxes(title_text='状态 s', row=1, col=1)
fig.update_xaxes(title_text='状态 s', row=1, col=2)
fig.update_yaxes(title_text='V(s)', row=1, col=1)
fig.update_yaxes(title_text='动作 a', row=1, col=2)

fig.write_image('static/images/plots/rl-value-function.png', width=950, height=450, scale=2)
print("图7完成")
