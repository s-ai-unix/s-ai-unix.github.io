#!/usr/bin/env python3
"""
向量微积分三大公式配图生成
包含：向量场、散度旋度、格林公式、高斯公式、斯托克斯公式可视化
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

os.makedirs('static/images/plots', exist_ok=True)

template = 'plotly_white'
primary_color = '#007AFF'
secondary_color = '#34C759'
accent_color = '#FF9500'

def save_fig(fig, filename):
    fig.write_image(f'static/images/plots/{filename}', scale=2)
    print(f'✅ {filename} 已保存')

print("开始生成向量微积分配图...")

# ========== 图1: 二维向量场可视化 ==========
print("生成图1: 向量场...")

fig1 = go.Figure()

# 绘制向量场 F = (-y, x)
x = np.linspace(-2, 2, 15)
y = np.linspace(-2, 2, 15)
X, Y = np.meshgrid(x, y)
U = -Y
V = X

# 只绘制部分箭头避免过度绘制
for i in range(0, len(x), 2):
    for j in range(0, len(y), 2):
        xi, yi = X[i,j], Y[i,j]
        ui, vi = U[i,j]*0.2, V[i,j]*0.2
        fig1.add_trace(go.Scatter(
            x=[xi, xi+ui], y=[yi, yi+vi],
            mode='lines+markers',
            line=dict(color=primary_color, width=2),
            marker=dict(size=3),
            showlegend=False,
            hoverinfo='skip'
        ))

# 添加一些流线
t = np.linspace(0, 2*np.pi, 100)
for r in [0.5, 1.0, 1.5]:
    x_circle = r * np.cos(t)
    y_circle = r * np.sin(t)
    fig1.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color='rgba(200,200,200,0.5)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

fig1.update_layout(
    title='向量场：F = (-y, x) 的旋转场',
    xaxis_title='x',
    yaxis_title='y',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=600,
    height=600,
    yaxis=dict(scaleanchor='x', scaleratio=1),
    showlegend=False
)

save_fig(fig1, 'vector_field_rotation.png')

# ========== 图2: 散度可视化（源和汇） ==========
print("生成图2: 散度...")

fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('散度 > 0（源）', '散度 < 0（汇）'),
    specs=[[{'type': 'scene'}, {'type': 'scene'}]]
)

# 简化的散度可视化 - 使用箭头
for i, (title, factor) in enumerate([('源', 1), ('汇', -1)]):
    col = i + 1
    
    # 绘制中心点
    fig2.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='red' if factor > 0 else 'blue'),
        showlegend=False
    ), row=1, col=col)
    
    # 绘制箭头
    directions = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (0.7, 0.7, 0), (-0.7, 0.7, 0), (0.7, -0.7, 0), (-0.7, -0.7, 0)
    ]
    
    for dx, dy, dz in directions:
        if factor > 0:  # 源 - 向外
            x_vals = [0, dx*0.8]
            y_vals = [0, dy*0.8]
            z_vals = [0, dz*0.8]
        else:  # 汇 - 向内
            x_vals = [dx*0.8, 0]
            y_vals = [dy*0.8, 0]
            z_vals = [dz*0.8, 0]
        
        fig2.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines+markers',
            line=dict(color='red' if factor > 0 else 'blue', width=3),
            marker=dict(size=2),
            showlegend=False
        ), row=1, col=col)

fig2.update_layout(
    title='散度的物理意义：源与汇',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=900,
    height=450
)

save_fig(fig2, 'divergence_source_sink.png')

# ========== 图3: 格林公式示意 ==========
print("生成图3: 格林公式...")

fig3 = go.Figure()

# 绘制区域 D（单位圆）
theta = np.linspace(0, 2*np.pi, 100)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

fig3.add_trace(go.Scatter(
    x=x_circle, y=y_circle,
    fill='toself',
    fillcolor='rgba(0,122,255,0.1)',
    line=dict(color=primary_color, width=2),
    name='区域 D'
))

# 绘制边界方向（逆时针箭头）
for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
    x_pos = np.cos(angle)
    y_pos = np.sin(angle)
    dx = -0.15 * np.sin(angle)
    dy = 0.15 * np.cos(angle)
    fig3.add_annotation(
        x=x_pos + dx, y=y_pos + dy,
        ax=x_pos, ay=y_pos,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red'
    )

# 添加简化的向量场
angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
for r in [0.3, 0.6]:
    for angle in angles:
        xi = r * np.cos(angle)
        yi = r * np.sin(angle)
        ui = -yi * 0.15
        vi = xi * 0.15
        fig3.add_trace(go.Scatter(
            x=[xi, xi+ui], y=[yi, yi+vi],
            mode='lines',
            line=dict(color='rgba(128,128,128,0.6)', width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ))

fig3.update_layout(
    title='格林公式：边界环流 = 内部旋度之和',
    xaxis_title='x',
    yaxis_title='y',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=600,
    height=600,
    yaxis=dict(scaleanchor='x', scaleratio=1),
    showlegend=True
)

save_fig(fig3, 'greens_formula.png')

# ========== 图4: 高斯公式示意 ==========
print("生成图4: 高斯公式...")

fig4 = go.Figure()

# 绘制球面
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 20)
U, V = np.meshgrid(u, v)

X_sphere = np.cos(U) * np.sin(V)
Y_sphere = np.sin(U) * np.sin(V)
Z_sphere = np.cos(V)

fig4.add_trace(go.Surface(
    x=X_sphere, y=Y_sphere, z=Z_sphere,
    opacity=0.3,
    colorscale='Blues',
    showscale=False,
    name='闭合曲面'
))

# 添加法向量（向外）
for phi in [np.pi/4, np.pi/2, 3*np.pi/4]:
    for theta in [0, np.pi/2, np.pi, 3*np.pi/2]:
        x_pos = np.cos(theta) * np.sin(phi)
        y_pos = np.sin(theta) * np.sin(phi)
        z_pos = np.cos(phi)
        fig4.add_trace(go.Scatter3d(
            x=[x_pos, x_pos*1.3], y=[y_pos, y_pos*1.3], z=[z_pos, z_pos*1.3],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))

fig4.update_layout(
    title='高斯公式：曲面上的通量 = 体积内的散度之和',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1))
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=700,
    height=600
)

save_fig(fig4, 'gauss_formula.png')

# ========== 图5: 斯托克斯公式示意 ==========
print("生成图5: 斯托克斯公式...")

fig5 = go.Figure()

# 绘制半球面
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi/2, 20)
U, V = np.meshgrid(u, v)

X_hemi = np.cos(U) * np.sin(V)
Y_hemi = np.sin(U) * np.sin(V)
Z_hemi = np.cos(V)

fig5.add_trace(go.Surface(
    x=X_hemi, y=Y_hemi, z=Z_hemi,
    opacity=0.4,
    colorscale='Greens',
    showscale=False,
    name='曲面 S'
))

# 绘制边界曲线（赤道）
theta = np.linspace(0, 2*np.pi, 100)
x_bound = np.cos(theta)
y_bound = np.sin(theta)
z_bound = np.zeros_like(theta)

fig5.add_trace(go.Scatter3d(
    x=x_bound, y=y_bound, z=z_bound,
    mode='lines',
    line=dict(color='red', width=4),
    name='边界曲线'
))

# 添加边界方向箭头
for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
    x_pos = np.cos(angle)
    y_pos = np.sin(angle)
    z_pos = 0
    dx = -0.2 * np.sin(angle)
    dy = 0.2 * np.cos(angle)
    fig5.add_trace(go.Scatter3d(
        x=[x_pos, x_pos+dx], y=[y_pos, y_pos+dy], z=[z_pos, z_pos],
        mode='lines+markers',
        line=dict(color='orange', width=3),
        marker=dict(size=4),
        showlegend=False
    ))

fig5.update_layout(
    title='斯托克斯公式：边界环流 = 曲面上旋度的通量',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1))
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=700,
    height=600
)

save_fig(fig5, 'stokes_formula.png')

# ========== 图6: 三大公式关系图 ==========
print("生成图6: 三大公式关系...")

fig6 = go.Figure()

# 节点位置
nodes = {
    '格林': (0, 0),
    '斯托克斯': (2, 1),
    '高斯': (2, -1),
    '统一': (4, 0)
}

# 绘制边
edges = [('格林', '斯托克斯'), ('格林', '高斯'), ('斯托克斯', '统一'), ('高斯', '统一')]

for start, end in edges:
    x0, y0 = nodes[start]
    x1, y1 = nodes[end]
    fig6.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode='lines',
        line=dict(color='gray', width=2),
        showlegend=False
    ))

# 绘制节点
node_colors = {
    '格林': primary_color,
    '斯托克斯': secondary_color,
    '高斯': accent_color,
    '统一': '#9B59B6'
}

for name, (x, y) in nodes.items():
    fig6.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(size=60, color=node_colors[name]),
        text=[name],
        textposition='middle center',
        textfont=dict(size=14, color='white'),
        showlegend=False
    ))

# 添加维度标注
annotations = [
    (0, -0.8, '2D 平面'),
    (2, 0.3, '3D 空间'),
    (2, -1.3, '3D 空间'),
    (4, -0.8, '高维推广')
]

for x, y, text in annotations:
    fig6.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(size=10))

fig6.update_layout(
    title='三大公式的层次关系',
    xaxis=dict(showgrid=False, showticklabels=False, range=[-1, 5]),
    yaxis=dict(showgrid=False, showticklabels=False, range=[-2, 2]),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=800,
    height=400
)

save_fig(fig6, 'three_formulas_relation.png')

print("\n✅ 所有向量微积分配图生成完成！")
