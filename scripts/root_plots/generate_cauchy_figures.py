#!/usr/bin/env python3
import numpy as np
import plotly.graph_objects as go
import os

os.makedirs('../static/images/math', exist_ok=True)

theta = np.linspace(0, 2*np.pi, 500)
r = 2
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=x_circle, y=y_circle,
    mode='lines',
    line=dict(color='#1f77b4', width=3),
    name='积分路径 C',
    showlegend=True
))

fig1.add_trace(go.Scatter(
    x=[0], y=[0],
    mode='markers',
    marker=dict(color='#ff3b30', size=15),
    name='中心点 $z_0$',
    showlegend=True
))

fig1.add_trace(go.Scatter(
    x=[0.5], y=[0.5],
    mode='markers',
    marker=dict(color='#34c759', size=10),
    name='内部点 $z$',
    showlegend=True
))

arrow_idx = 0
fig1.add_annotation(
    x=x_circle[arrow_idx], y=y_circle[arrow_idx],
    ax=x_circle[arrow_idx+10], ay=y_circle[arrow_idx+10],
    axref='x', ayref='y',
    xref='x', yref='y',
    showarrow=True,
    arrowhead=3,
    arrowsize=1.5,
    arrowwidth=2,
    arrowcolor='#1f77b4'
)

fig1.update_layout(
    title='复平面上的积分路径',
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
    ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

fig1.write_image('../static/images/math/complex-plane-contour.png', scale=2)
print("Figure 1 created: complex-plane-contour.png")

fig2 = go.Figure()

x, y = np.meshgrid(np.linspace(-3, 3, 20),
                   np.linspace(-3, 3, 20))

u = x / (x**2 + y**2)
v = -y / (x**2 + y**2)

mask = np.sqrt(x**2 + y**2) > 0.3
u = np.where(mask, u, 0)
v = np.where(mask, v, 0)

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
print("Figure 2 created: vector-field-1-z.png")

fig3 = go.Figure()

radii = [2, 1.5, 1, 0.5]
colors = ['#1f77b4', '#34c759', '#ff9500', '#ff3b30']

for i, (r, color) in enumerate(zip(radii, colors)):
    theta = np.linspace(0, 2*np.pi, 500)
    x_c = r * np.cos(theta)
    y_c = r * np.sin(theta)

    fig3.add_trace(go.Scatter(
        x=x_c, y=y_c,
        mode='lines',
        line=dict(color=color, width=2 if i < 3 else 3),
        name=f'r={r}',
        showlegend=True
    ))

fig3.add_trace(go.Scatter(
    x=[0], y=[0],
    mode='markers',
    marker=dict(color='#333', size=10),
    name='点 $z$',
    showlegend=True
))

fig3.update_layout(
    title='柯西积分公式的几何直观：圆周收缩到点',
    xaxis_title='实部 Re($\\zeta$)',
    yaxis_title='虚部 Im($\\zeta$)',
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

fig3.write_image('../static/images/math/circles-shrinking.png', scale=2)
print("Figure 3 created: circles-shrinking.png")

print("\nAll figures created successfully!")
