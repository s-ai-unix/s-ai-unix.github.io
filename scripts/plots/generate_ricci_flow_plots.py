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

# ============ 图1: 球面 Ricci Flow - 半径收缩 ============
print("生成图1: 球面 Ricci Flow 半径收缩...")

t = np.linspace(0, 0.49, 200)
n = 3  # 3维球面
R0 = 1.0  # 初始半径
T_max = 1 / (2 * (n - 1))
R_t = R0 * np.sqrt(1 - 2 * (n - 1) * t)

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=t, y=R_t,
    mode='lines',
    name='球面半径 R(t)',
    line=dict(color='#007AFF', width=3)
))

fig1.add_trace(go.Scatter(
    x=[T_max], y=[0],
    mode='markers',
    name='奇点 (t = 0.25)',
    marker=dict(size=12, color='#FF3B30', symbol='x')
))

fig1.add_vline(x=T_max, line_dash="dash", line_color="#FF3B30", opacity=0.5)

fig1.update_layout(
    xaxis_title='时间 t',
    yaxis_title='半径 R(t)',
    showlegend=True
)
apply_apple_style(fig1, '球面 Ricci Flow: 半径随时间收缩 (n=3)', 900, 500)
fig1.write_image('static/images/plots/ricci-flow-sphere-radius.png', scale=2)

# ============ 图2: 标量曲率演化 ============
print("生成图2: 标量曲率演化...")

fig2 = go.Figure()

# 球面标量曲率演化
t_sphere = np.linspace(0, 0.4, 200)
R_scalar_sphere = n * (n - 1) / (1 - 2 * (n - 1) * t_sphere)

fig2.add_trace(go.Scatter(
    x=t_sphere, y=R_scalar_sphere,
    mode='lines',
    name='球面 (正曲率, K→+∞)',
    line=dict(color='#007AFF', width=3)
))

# 负曲率流形演化
t_neg = np.linspace(0, 2, 200)
R0_neg = -6
R_scalar_neg = R0_neg / (1 + 0.5 * t_neg)

fig2.add_trace(go.Scatter(
    x=t_neg, y=R_scalar_neg,
    mode='lines',
    name='负曲率流形 (趋向平坦)',
    line=dict(color='#34C759', width=3)
))

# 平坦流形
fig2.add_trace(go.Scatter(
    x=[0, 2], y=[0, 0],
    mode='lines',
    name='平坦流形 (R=0)',
    line=dict(color='#FF9500', width=2, dash='dash')
))

fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)

fig2.update_layout(
    xaxis_title='时间 t',
    yaxis_title='标量曲率 R(t)',
    showlegend=True
)
apply_apple_style(fig2, '不同流形的标量曲率演化', 900, 500)
fig2.write_image('static/images/plots/ricci-flow-scalar-curvature.png', scale=2)

# ============ 图3: 球面演化序列 (t=0, 0.1, 0.2) ============
print("生成图3: 球面演化序列...")

u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 30)
U, V = np.meshgrid(u, v)

# t = 0
R0 = 1.0
X1 = R0 * np.sin(V) * np.cos(U)
Y1 = R0 * np.sin(V) * np.sin(U)
Z1 = R0 * np.cos(V)

fig3a = go.Figure()
fig3a.add_trace(go.Surface(
    x=X1, y=Y1, z=Z1,
    colorscale='Blues',
    opacity=0.9,
    showscale=False
))
fig3a.update_layout(scene=dict(aspectmode='cube'))
apply_apple_style(fig3a, 't=0 (初始)', 600, 500)
fig3a.write_image('static/images/plots/ricci-flow-sphere-t0.png', scale=2)

# t = 0.1
R1 = np.sqrt(1 - 2 * 2 * 0.1)
X2 = R1 * np.sin(V) * np.cos(U)
Y2 = R1 * np.sin(V) * np.sin(U)
Z2 = R1 * np.cos(V)

fig3b = go.Figure()
fig3b.add_trace(go.Surface(
    x=X2, y=Y2, z=Z2,
    colorscale='Greens',
    opacity=0.9,
    showscale=False
))
fig3b.update_layout(scene=dict(aspectmode='cube'))
apply_apple_style(fig3b, 't=0.1 (收缩中)', 600, 500)
fig3b.write_image('static/images/plots/ricci-flow-sphere-t1.png', scale=2)

# t = 0.2
R2 = np.sqrt(1 - 2 * 2 * 0.2)
X3 = R2 * np.sin(V) * np.cos(U)
Y3 = R2 * np.sin(V) * np.sin(U)
Z3 = R2 * np.cos(V)

fig3c = go.Figure()
fig3c.add_trace(go.Surface(
    x=X3, y=Y3, z=Z3,
    colorscale='Oranges',
    opacity=0.9,
    showscale=False
))
fig3c.update_layout(scene=dict(aspectmode='cube'))
apply_apple_style(fig3c, 't=0.2 (接近奇点)', 600, 500)
fig3c.write_image('static/images/plots/ricci-flow-sphere-t2.png', scale=2)

# ============ 图4: Perelman 熵的单调性 ============
print("生成图4: Perelman 熵单调性...")

t_entropy = np.linspace(0.1, 2, 200)
W_entropy = 5 * np.log(t_entropy) + 2 * np.sin(2*t_entropy) + 10

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=t_entropy, y=W_entropy,
    mode='lines',
    name='Perelman 熵 W(t)',
    line=dict(color='#007AFF', width=3)
))

# 添加单调性箭头
fig4.add_annotation(
    x=1.5, y=W_entropy[140],
    ax=0.8, ay=W_entropy[70],
    xref='x', yref='y',
    axref='x', ayref='y',
    showarrow=True,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=2,
    arrowcolor='#FF9500'
)

fig4.add_annotation(
    x=1.2, y=W_entropy[140]+1,
    text='单调递增 →',
    showarrow=False,
    font=dict(size=14, color='#FF9500')
)

fig4.update_layout(
    xaxis_title='时间 t',
    yaxis_title='Perelman 熵 W(t)',
    showlegend=True
)
apply_apple_style(fig4, 'Perelman 熵在 Ricci Flow 下的单调性', 900, 500)
fig4.write_image('static/images/plots/ricci-flow-perelman-entropy.png', scale=2)

# ============ 图5: 2D高斯曲率演化 ============
print("生成图5: 2D高斯曲率演化...")

fig5 = go.Figure()

t_2d = np.linspace(0, 0.9, 200)
K_positive = 1 / (1 - 2 * t_2d + 0.01)
K_negative = -1 / (1 + 2 * t_2d)

fig5.add_trace(go.Scatter(
    x=t_2d, y=K_positive,
    mode='lines',
    name='正曲率区域 (K→+∞)',
    line=dict(color='#FF3B30', width=3)
))

fig5.add_trace(go.Scatter(
    x=t_2d, y=K_negative,
    mode='lines',
    name='负曲率区域 (K→0)',
    line=dict(color='#34C759', width=3)
))

fig5.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)

fig5.update_layout(
    xaxis_title='时间 t',
    yaxis_title='高斯曲率 K(t)',
    showlegend=True
)
apply_apple_style(fig5, '2D流形的高斯曲率演化', 900, 500)
fig5.write_image('static/images/plots/ricci-flow-gaussian-curvature.png', scale=2)

print("所有图形生成完成!")
