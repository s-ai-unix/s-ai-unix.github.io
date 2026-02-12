import numpy as np
import plotly.graph_objects as go

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
print("Figure 1 created")
