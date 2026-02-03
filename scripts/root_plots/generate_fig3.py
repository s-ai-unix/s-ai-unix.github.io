import numpy as np
import plotly.graph_objects as go

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
print("Figure 3 created")
