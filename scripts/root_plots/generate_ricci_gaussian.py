import plotly.graph_objects as go
import numpy as np

t_2d = np.linspace(0, 0.45, 200)
K_positive = 1 / (1 - 2 * t_2d + 0.01)
K_negative = -1 / (1 + 2 * t_2d)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t_2d, y=K_positive,
    mode='lines',
    name='正曲率区域 (K→+∞)',
    line=dict(color='#FF3B30', width=3)
))

fig.add_trace(go.Scatter(
    x=t_2d, y=K_negative,
    mode='lines',
    name='负曲率区域 (K→0)',
    line=dict(color='#34C759', width=3)
))

fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)

fig.update_layout(
    title=dict(text='2D流形的高斯曲率演化', font=dict(size=18, family='Arial, sans-serif')),
    xaxis_title='时间 t',
    yaxis_title='高斯曲率 K(t)',
    template='plotly_white',
    font=dict(family='Arial, sans-serif', size=12),
    width=900,
    height=500,
    showlegend=True
)

fig.write_image('../static/images/plots/ricci-flow-gaussian-curvature.png', scale=2)
print("完成!")
