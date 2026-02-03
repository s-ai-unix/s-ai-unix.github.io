import numpy as np
import plotly.graph_objects as go

fig2 = go.Figure()

x, y = np.meshgrid(np.linspace(-3, 3, 20),
                   np.linspace(-3, 3, 20))

u = x / (x**2 + y**2)
v = -y / (x**2 + y**2)

mask = np.sqrt(x**2 + y**2) > 0.3
u = np.where(mask, u, 0)
v = np.where(mask, v, 0)

theta = np.linspace(0, 2*np.pi, 500)
r = 2
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)

fig2.add_trace(go.Scatter(
    x=x_circle, y=y_circle,
    mode='lines',
    line=dict(color='#1f77b4', width=3),
    name='积分路径 C',
    showlegend=True
))

fig2.add_trace(go.Scatter(
    x=[0], y=[0],
    mode='markers',
    marker=dict(color='#ff3b30', size=12),
    name='奇点 $z_0$',
    showlegend=True
))

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if mask[i, j]:
            fig2.add_trace(go.Scatter(
                x=[x[i, j], x[i, j] + u[i, j]*0.3],
                y=[y[i, j], y[i, j] + v[i, j]*0.3],
                mode='lines+markers',
                line=dict(color='#34c759', width=1),
                marker=dict(size=3, color='#34c759'),
                showlegend=False,
                hoverinfo='none'
            ))

fig2.update_layout(
    title='向量场 $1/(z-z_0)$',
    xaxis_title='实部 Re($z$)',
    yaxis_title='虚部 Im($z$)',
    width=700,
    height=600,
    template='plotly_white',
    font=dict(size=14),
    plot_bgcolor='white',
    xaxis=dict(
        gridcolor='#e0e0e0',
        zerolinecolor='#333',
        scaleanchor="y",
        scaleratio=1
    ),
    yaxis=dict(
        gridcolor='#e0e0e0',
        zerolinecolor='#333'
    )
)

fig2.write_image('../static/images/math/vector-field-1-z.png', scale=2)
print("Figure 2 created")
