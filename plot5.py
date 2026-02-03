import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import norm

fig = make_subplots(rows=1, cols=2, subplot_titles=('线性回归', '非参数回归（核平滑）'), horizontal_spacing=0.12)

np.random.seed(42)
n = 300
x = np.random.uniform(-3, 3, n)
y_true = np.sin(x) + 0.5 * x
y = y_true + np.random.normal(0, 0.5, n)

idx = np.argsort(x)
x_sorted = x[idx]
y_sorted = y[idx]

fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#007AFF', size=5, opacity=0.4), showlegend=False), row=1, col=1)
coeffs = np.polyfit(x, y, 1)
y_linear = np.polyval(coeffs, x_sorted)
fig.add_trace(go.Scatter(x=x_sorted, y=y_linear, mode='lines', line=dict(color='#FF3B30', width=2), showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#007AFF', size=5, opacity=0.4), showlegend=False), row=1, col=2)

def kernel_regression(x_query, x_data, y_data, bandwidth=0.5):
    weights = norm.pdf((x_query[:, None] - x_data[None, :]) / bandwidth)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ y_data

x_grid = np.linspace(-3, 3, 200)
y_kernel = kernel_regression(x_grid, x, y, bandwidth=0.5)
fig.add_trace(go.Scatter(x=x_grid, y=y_kernel, mode='lines', line=dict(color='#34C759', width=2), showlegend=False), row=1, col=2)

y_true_grid = np.sin(x_grid) + 0.5 * x_grid
fig.add_trace(go.Scatter(x=x_grid, y=y_true_grid, mode='lines', line=dict(color='#FF9500', width=2, dash='dash'), showlegend=False), row=1, col=2)

fig.update_layout(title='回归分析：估计条件期望 E[Y|X]', template='plotly_white', font=dict(family='Arial', size=12))
fig.update_xaxes(title_text='X', row=1, col=1)
fig.update_xaxes(title_text='X', row=1, col=2)
fig.update_yaxes(title_text='Y', row=1, col=1)
fig.update_yaxes(title_text='Y', row=1, col=2)

fig.write_image('static/images/plots/regression-conditional-expectation.png', width=900, height=500, scale=2)
print("图5完成")
