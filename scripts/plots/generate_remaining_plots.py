import plotly.graph_objects as go
import numpy as np
import os

os.makedirs('static/images/plots', exist_ok=True)

def apply_apple_style(fig, title, width=800, height=600):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family='Arial, sans-serif')),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=width,
        height=height,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# 图: 椭球面主曲率
print("生成椭球面主曲率...")
u_ell = np.linspace(0, 2*np.pi, 60)
v_ell = np.linspace(0, np.pi, 40)
U_ell, V_ell = np.meshgrid(u_ell, v_ell)

a, b, c = 2, 1.5, 1
X_ell = a * np.sin(V_ell) * np.cos(U_ell)
Y_ell = b * np.sin(V_ell) * np.sin(U_ell)
Z_ell = c * np.cos(V_ell)

fig = go.Figure()
fig.add_trace(go.Surface(
    x=X_ell, y=Y_ell, z=Z_ell,
    colorscale='Greens',
    opacity=0.6,
    showscale=False
))

# 在顶点处的两个主方向
t = np.linspace(-0.3, 0.3, 50)
x1 = a * np.sin(np.pi/2 + t)
y1 = np.zeros_like(t)
z1 = c * np.cos(np.pi/2 + t)

fig.add_trace(go.Scatter3d(
    x=x1, y=y1, z=z1,
    mode='lines',
    line=dict(color='#007AFF', width=5),
    name='κ₁ 方向'
))

x2 = np.zeros_like(t)
y2 = b * np.sin(np.pi/2 + t)
z2 = c * np.cos(np.pi/2 + t)

fig.add_trace(go.Scatter3d(
    x=x2, y=y2, z=z2,
    mode='lines',
    line=dict(color='#34C759', width=5),
    name='κ₂ 方向'
))

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[c],
    mode='markers',
    marker=dict(size=8, color='#FF9500'),
    name='顶点'
))

fig.update_layout(scene=dict(aspectmode='cube'), showlegend=True)
apply_apple_style(fig, '椭球面的主曲率方向', 900, 700)
fig.write_image('static/images/plots/principal_curvature.png', scale=2)
print("完成!")
