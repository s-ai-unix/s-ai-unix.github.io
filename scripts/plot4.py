import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

fig = make_subplots(rows=1, cols=2, subplot_titles=('原始估计量', 'Rao-Blackwell改进'), horizontal_spacing=0.12)

np.random.seed(42)
n = 200
T = np.random.normal(5, 1, n)
theta_hat = T + np.random.normal(0, 0.8, n)
theta_hat_star = T

fig.add_trace(go.Scatter(x=T, y=theta_hat, mode='markers', marker=dict(color='#007AFF', size=6, opacity=0.6), showlegend=False), row=1, col=1)
fig.add_hline(y=5, line=dict(color='#FF3B30', width=2), annotation=dict(text='真实值'), row=1, col=1)

fig.add_trace(go.Scatter(x=T, y=theta_hat_star, mode='markers', marker=dict(color='#34C759', size=6, opacity=0.6), showlegend=False), row=1, col=2)
fig.add_hline(y=5, line=dict(color='#FF3B30', width=2), annotation=dict(text='真实值'), row=1, col=2)

var_orig = np.var(theta_hat - 5)
var_improved = np.var(theta_hat_star - 5)

fig.add_annotation(x=0.95, y=0.95, xref='paper', yref='paper', text=f'方差: {var_orig:.3f}', 
                   showarrow=False, bgcolor='rgba(255,255,255,0.8)', row=1, col=1)
fig.add_annotation(x=0.95, y=0.95, xref='paper', yref='paper', text=f'方差: {var_improved:.3f}', 
                   showarrow=False, bgcolor='rgba(255,255,255,0.8)', row=1, col=2)

fig.update_layout(title='Rao-Blackwell定理：条件期望降低方差', template='plotly_white', font=dict(family='Arial', size=12))
fig.update_xaxes(title_text='充分统计量 T', row=1, col=1)
fig.update_xaxes(title_text='充分统计量 T', row=1, col=2)
fig.update_yaxes(title_text='估计值', row=1, col=1)
fig.update_yaxes(title_text='改进估计值', row=1, col=2)

fig.write_image('static/images/plots/rao-blackwell-theorem.png', width=900, height=500, scale=2)
print("图4完成")
