import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('无条件分布', '条件分布'),
                    horizontal_spacing=0.12)

outcomes = np.array([1, 2, 3, 4, 5, 6])
probs = np.ones(6) / 6

fig.add_trace(go.Bar(x=outcomes, y=probs, marker_color='#007AFF', showlegend=False), row=1, col=1)
fig.add_vline(x=3.5, line=dict(color='#FF3B30', dash='dash', width=2), row=1, col=1)

odd_outcomes = np.array([1, 3, 5])
odd_probs = np.ones(3) / 3
fig.add_trace(go.Bar(x=odd_outcomes, y=odd_probs, marker_color='#34C759', showlegend=False), row=1, col=2)
fig.add_vline(x=3, line=dict(color='#FF3B30', dash='dash', width=2), row=1, col=2)

fig.update_layout(title='条件期望：以骰子为例', template='plotly_white', font=dict(family='Arial', size=12))
fig.update_xaxes(title_text='结果', row=1, col=1)
fig.update_xaxes(title_text='结果（给定奇数）', row=1, col=2)
fig.update_yaxes(title_text='概率', row=1, col=1)
fig.update_yaxes(title_text='条件概率', row=1, col=2)

fig.write_image('static/images/plots/conditional-expectation-intuition.png', width=900, height=500, scale=2)
print("图1完成")
