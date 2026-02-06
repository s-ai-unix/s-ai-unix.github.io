import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

fig = make_subplots(rows=1, cols=2, subplot_titles=('联合分布等高线与回归线', '条件分布'), horizontal_spacing=0.12)

mu_x, mu_y, sigma_x, sigma_y, rho = 0, 0, 1, 1, 0.7
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

Z = stats.multivariate_normal.pdf(np.dstack([X, Y]), mean=[mu_x, mu_y],
                                   cov=[[sigma_x**2, rho*sigma_x*sigma_y],
                                        [rho*sigma_x*sigma_y, sigma_y**2]])

fig.add_trace(go.Contour(x=x, y=y, z=Z, colorscale='Blues', showscale=False, contours=dict(coloring='fill')), row=1, col=1)

x_line = np.linspace(-3, 3, 100)
y_line = mu_y + rho * (sigma_y/sigma_x) * (x_line - mu_x)
fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', line=dict(color='#FF3B30', width=3), showlegend=False), row=1, col=1)

x0 = 1
y_cond = np.linspace(-3, 3, 100)
mu_cond = mu_y + rho * (sigma_y/sigma_x) * (x0 - mu_x)
sigma_cond = sigma_y * np.sqrt(1 - rho**2)
density = stats.norm.pdf(y_cond, mu_cond, sigma_cond)

fig.add_trace(go.Scatter(x=density, y=y_cond, mode='lines', fill='tozerox', fillcolor='rgba(0, 122, 255, 0.3)',
                         line=dict(color='#007AFF', width=2), showlegend=False), row=1, col=2)
fig.add_hline(y=mu_cond, line=dict(color='#FF3B30', dash='dash', width=2), row=1, col=2)

fig.update_layout(title='二元正态分布的条件期望', template='plotly_white', font=dict(family='Arial', size=12))
fig.update_xaxes(title_text='X', row=1, col=1)
fig.update_xaxes(title_text='条件密度', row=1, col=2)
fig.update_yaxes(title_text='Y', row=1, col=1)
fig.update_yaxes(title_text='Y', row=1, col=2)

fig.write_image('static/images/plots/bivariate-normal-conditional.png', width=900, height=500, scale=2)
print("图2完成")
