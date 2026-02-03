#!/usr/bin/env python3
"""
微分几何与深度学习配图生成
包含：流形学习可视化、测地线对比、曲率演化等
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

# 确保输出目录存在
os.makedirs('static/images/plots', exist_ok=True)

# 设置默认模板和颜色
template = 'plotly_white'
primary_color = '#007AFF'
secondary_color = '#34C759'
accent_color = '#FF9500'

print("开始生成微分几何与深度学习配图...")

# ========== 图1: 流形学习概念图 ==========
print("生成图1: 流形学习概念图...")

# 创建瑞士卷流形数据
def make_swiss_roll(n_samples=1000, noise=0.1):
    t = 3 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = 30 * np.random.rand(n_samples)
    X = np.column_stack([x, y, z])
    return X, t

X, t = make_swiss_roll(n_samples=800, noise=0.05)

fig1 = go.Figure()

# 原始高维数据（瑞士卷）
fig1.add_trace(go.Scatter3d(
    x=X[:, 0], y=X[:, 1], z=X[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=t,
        colorscale='Blues',
        opacity=0.7
    ),
    name='高维流形数据'
))

# 添加展开后的2D平面示意
x_plane = np.linspace(-10, 10, 20)
y_plane = np.linspace(0, 30, 20)
X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
Z_plane = np.full_like(X_plane, -15)

fig1.add_trace(go.Surface(
    x=X_plane, y=Y_plane, z=Z_plane,
    colorscale=[[0, 'rgba(200,200,200,0.3)'], [1, 'rgba(200,200,200,0.3)']],
    showscale=False,
    name='低维嵌入空间'
))

# 添加投影箭头
for i in range(0, len(X), 100):
    fig1.add_trace(go.Scatter3d(
        x=[X[i, 0], X[i, 0]],
        y=[X[i, 1], X[i, 2]],
        z=[X[i, 2], -15],
        mode='lines',
        line=dict(color='rgba(255,149,0,0.3)', width=1),
        showlegend=False
    ))

fig1.update_layout(
    title=dict(
        text='流形学习：从高维空间到低维嵌入',
        font=dict(size=16)
    ),
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=800,
    height=600,
    showlegend=True
)

fig1.write_image('static/images/plots/manifold_learning_concept.png', scale=2)
print("✅ 图1已保存")

# ========== 图2: 欧氏距离 vs 测地距离 ==========
print("生成图2: 欧氏距离 vs 测地距离...")

# 创建弯曲流形上的两个点
theta = np.linspace(0, np.pi, 100)
# 半圆形流形
r = 5
x_manifold = r * np.cos(theta)
y_manifold = r * np.sin(theta)

# 两个点
p1_idx, p2_idx = 20, 80
p1 = np.array([x_manifold[p1_idx], y_manifold[p1_idx]])
p2 = np.array([x_manifold[p2_idx], y_manifold[p2_idx]])

fig2 = go.Figure()

# 流形曲线
fig2.add_trace(go.Scatter(
    x=x_manifold, y=y_manifold,
    mode='lines',
    line=dict(color=primary_color, width=3),
    name='数据流形'
))

# 欧氏距离（直线）
fig2.add_trace(go.Scatter(
    x=[p1[0], p2[0]], y=[p1[1], p2[1]],
    mode='lines+markers',
    line=dict(color=accent_color, width=2, dash='dash'),
    marker=dict(size=10),
    name='欧氏距离（直线）'
))

# 测地距离（沿流形）
fig2.add_trace(go.Scatter(
    x=x_manifold[p1_idx:p2_idx+1], y=y_manifold[p1_idx:p2_idx+1],
    mode='lines',
    line=dict(color=secondary_color, width=3),
    name='测地距离（沿流形）'
))

# 标记点
fig2.add_trace(go.Scatter(
    x=[p1[0], p2[0]], y=[p1[1], p2[1]],
    mode='markers',
    marker=dict(size=15, color='red', symbol='star'),
    name='数据点',
    showlegend=False
))

# 添加距离标注
mid_euclidean = (p1 + p2) / 2
fig2.add_annotation(
    x=mid_euclidean[0], y=mid_euclidean[1] + 0.8,
    text='欧氏距离',
    showarrow=False,
    font=dict(size=12, color=accent_color)
)

mid_geodesic = np.array([x_manifold[(p1_idx+p2_idx)//2], y_manifold[(p1_idx+p2_idx)//2]])
fig2.add_annotation(
    x=mid_geodesic[0] - 1.5, y=mid_geodesic[1] + 0.5,
    text='测地距离',
    showarrow=False,
    font=dict(size=12, color=secondary_color)
)

fig2.update_layout(
    title='欧氏距离 vs 测地距离',
    xaxis_title='x',
    yaxis_title='y',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=700,
    height=500,
    yaxis=dict(scaleanchor='x', scaleratio=1)
)

fig2.write_image('static/images/plots/euclidean_vs_geodesic.png', scale=2)
print("✅ 图2已保存")

# ========== 图3: 黎曼梯度下降示意 ==========
print("生成图3: 黎曼梯度下降...")

# 球面上的优化示意
phi = np.linspace(0, np.pi, 50)
theta = np.linspace(0, 2*np.pi, 50)
phi, theta = np.meshgrid(phi, theta)

r = 1
x_sphere = r * np.sin(phi) * np.cos(theta)
y_sphere = r * np.sin(phi) * np.sin(theta)
z_sphere = r * np.cos(phi)

fig3 = go.Figure()

# 绘制球面（半透明）
fig3.add_trace(go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    colorscale=[[0, 'rgba(0,122,255,0.2)'], [1, 'rgba(0,122,255,0.2)']],
    showscale=False,
    name='约束流形（球面）'
))

# 优化路径（从某点开始沿测地线下降）
t_path = np.linspace(0, 1.2, 50)
# 从点 (0.7, 0, 0.7) 沿测地线向 (0, 0, 1) 移动
start_point = np.array([0.7, 0, 0.7])
start_point = start_point / np.linalg.norm(start_point)
end_point = np.array([0, 0, 1])

# 球面插值（SLERP）
omega = np.arccos(np.clip(np.dot(start_point, end_point), -1, 1))
path_points = []
for t in np.linspace(0, 1, 30):
    if omega < 1e-6:
        point = start_point
    else:
        point = (np.sin((1-t)*omega) * start_point + np.sin(t*omega) * end_point) / np.sin(omega)
    path_points.append(point)
path_points = np.array(path_points)

fig3.add_trace(go.Scatter3d(
    x=path_points[:, 0], y=path_points[:, 1], z=path_points[:, 2],
    mode='lines+markers',
    line=dict(color=accent_color, width=4),
    marker=dict(size=4),
    name='黎曼梯度下降路径'
))

# 起点和终点
fig3.add_trace(go.Scatter3d(
    x=[start_point[0]], y=[start_point[1]], z=[start_point[2]],
    mode='markers',
    marker=dict(size=12, color='red', symbol='diamond'),
    name='初始点'
))

fig3.add_trace(go.Scatter3d(
    x=[end_point[0]], y=[end_point[1]], z=[end_point[2]],
    mode='markers',
    marker=dict(size=12, color='green', symbol='square'),
    name='最优点'
))

# 切平面示意（在起点处）
tangent_scale = 0.3
# 切向量方向
v1 = np.array([0, 1, 0])
v2 = np.cross(start_point, v1)
v2 = v2 / np.linalg.norm(v2)

# 切平面上的网格
tangent_u = np.linspace(-1, 1, 10)
tangent_v = np.linspace(-1, 1, 10)
U, V = np.meshgrid(tangent_u, tangent_v)
tangent_x = start_point[0] + tangent_scale * (U * v1[0] + V * v2[0])
tangent_y = start_point[1] + tangent_scale * (U * v1[1] + V * v2[1])
tangent_z = start_point[2] + tangent_scale * (U * v1[2] + V * v2[2])

fig3.add_trace(go.Surface(
    x=tangent_x, y=tangent_y, z=tangent_z,
    colorscale=[[0, 'rgba(255,149,0,0.3)'], [1, 'rgba(255,149,0,0.3)']],
    showscale=False,
    name='切空间'
))

fig3.update_layout(
    title='黎曼梯度下降：在切空间中更新，沿测地线移动',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=800,
    height=600
)

fig3.write_image('static/images/plots/riemannian_gradient_descent.png', scale=2)
print("✅ 图3已保存")

# ========== 图4: 不同流形学习方法对比 ==========
print("生成图4: 流形学习方法对比...")

# 生成二维流形嵌入数据（S曲线）
from sklearn.datasets import make_s_curve
X_scurve, color = make_s_curve(n_samples=500, random_state=42)

fig4 = make_subplots(
    rows=1, cols=3,
    subplot_titles=('原始数据（3D）', 'PCA（线性投影）', '流形学习（非线性展开）'),
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}, {'type': 'scatter'}]]
)

# 原始3D数据
fig4.add_trace(go.Scatter3d(
    x=X_scurve[:, 0], y=X_scurve[:, 1], z=X_scurve[:, 2],
    mode='markers',
    marker=dict(size=3, color=color, colorscale='Viridis'),
    showlegend=False
), row=1, col=1)

# PCA投影（简单取前两个维度作为示意）
pca_x = X_scurve[:, 0]
pca_y = X_scurve[:, 1]
fig4.add_trace(go.Scatter(
    x=pca_x, y=pca_y,
    mode='markers',
    marker=dict(size=4, color=color, colorscale='Viridis'),
    showlegend=False
), row=1, col=2)

# 流形学习结果（用颜色保持结构示意）
# 用弧长参数化来模拟流形展开
unfolded_x = color
unfolded_y = X_scurve[:, 1] + 0.1 * np.random.randn(len(X_scurve))
fig4.add_trace(go.Scatter(
    x=unfolded_x, y=unfolded_y,
    mode='markers',
    marker=dict(size=4, color=color, colorscale='Viridis'),
    showlegend=False
), row=1, col=3)

fig4.update_layout(
    title='PCA vs 流形学习：线性与非线性降维',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=900,
    height=350
)

fig4.update_xaxes(title_text='成分1', row=1, col=2)
fig4.update_yaxes(title_text='成分2', row=1, col=2)
fig4.update_xaxes(title_text='成分1', row=1, col=3)
fig4.update_yaxes(title_text='成分2', row=1, col=3)

fig4.write_image('static/images/plots/manifold_methods_comparison.png', scale=2)
print("✅ 图4已保存")

# ========== 图5: 图神经网络的几何解释 ==========
print("生成图5: 图神经网络的几何解释...")

# 创建一个简单图结构
np.random.seed(42)
n_nodes = 8

# 节点位置（圆环布局）
angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n_nodes)}

# 边连接（带重边的图）
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
         (0, 4), (1, 5), (2, 6), (3, 7)]

fig5 = go.Figure()

# 绘制边
for edge in edges:
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    fig5.add_trace(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode='lines',
        line=dict(color='rgba(150,150,150,0.5)', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))

# 节点特征值（颜色）
node_features = np.random.rand(n_nodes)

# 绘制节点
fig5.add_trace(go.Scatter(
    x=[pos[i][0] for i in range(n_nodes)],
    y=[pos[i][1] for i in range(n_nodes)],
    mode='markers+text',
    marker=dict(
        size=30,
        color=node_features,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title='特征值')
    ),
    text=[str(i) for i in range(n_nodes)],
    textposition='middle center',
    textfont=dict(size=12, color='white'),
    name='节点'
))

# 添加邻居聚合示意（突出显示节点0及其邻居）
neighbors_0 = [1, 7, 4]
for neighbor in neighbors_0:
    x0, y0 = pos[0]
    x1, y1 = pos[neighbor]
    fig5.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode='lines',
        line=dict(color=accent_color, width=4),
        showlegend=False
    ))

# 高亮中心节点
fig5.add_trace(go.Scatter(
    x=[pos[0][0]], y=[pos[0][1]],
    mode='markers',
    marker=dict(size=40, color='rgba(255,149,0,0.5)', line=dict(width=2, color='red')),
    showlegend=False
))

fig5.update_layout(
    title='图神经网络：消息传递作为几何聚合',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=600,
    height=500,
    showlegend=False
)

fig5.write_image('static/images/plots/gnn_geometric_interpretation.png', scale=2)
print("✅ 图5已保存")

# ========== 图6: 神经网络的损失景观几何 ==========
print("生成图6: 损失景观的几何结构...")

# 创建一个简化的损失函数景观
def loss_landscape(x, y):
    """模拟一个具有多个局部最小值的损失函数"""
    return (x**2 + y**2) * 0.5 + 0.5 * np.sin(3*x) * np.cos(3*y) + 0.3 * (x - 1)**2 * (y + 1)**2

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X_grid, Y_grid = np.meshgrid(x, y)
Z = loss_landscape(X_grid, Y_grid)

# 梯度
dx, dy = np.gradient(Z)

fig6 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('损失函数景观', '梯度向量场'),
    specs=[[{'type': 'surface'}, {'type': 'contour'}]]
)

# 3D表面
fig6.add_trace(go.Surface(
    x=X_grid, y=Y_grid, z=Z,
    colorscale='Viridis',
    showscale=False,
    name='损失函数'
), row=1, col=1)

# 添加优化路径示意
path_x = np.linspace(1.5, 0, 20)
path_y = np.linspace(1.5, 0, 20)
path_z = loss_landscape(path_x, path_y)

fig6.add_trace(go.Scatter3d(
    x=path_x, y=path_y, z=path_z + 0.1,
    mode='lines+markers',
    line=dict(color='red', width=4),
    marker=dict(size=3),
    name='优化路径',
    showlegend=False
), row=1, col=1)

# 等高线图带梯度场
fig6.add_trace(go.Contour(
    x=x, y=y, z=Z,
    colorscale='Viridis',
    showscale=False,
    contours=dict(coloring='heatmap'),
    name='等高线'
), row=1, col=2)

# 采样显示梯度向量
skip = 8
fig6.add_trace(go.Quiver(
    x=X_grid[::skip, ::skip].flatten(),
    y=Y_grid[::skip, ::skip].flatten(),
    u=-dx[::skip, ::skip].flatten() * 0.1,
    v=-dy[::skip, ::skip].flatten() * 0.1,
    scale=1,
    line=dict(color='white', width=1),
    showlegend=False
), row=1, col=2)

fig6.update_layout(
    title='神经网络的损失景观：曲率与优化的几何',
    scene=dict(
        xaxis_title='参数 θ₁',
        yaxis_title='参数 θ₂',
        zaxis_title='损失 L',
        camera=dict(eye=dict(x=1.5, y=-1.5, z=1))
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=900,
    height=450
)

fig6.update_xaxes(title_text='参数 θ₁', row=1, col=2)
fig6.update_yaxes(title_text='参数 θ₂', row=1, col=2)

fig6.write_image('static/images/plots/loss_landscape_geometry.png', scale=2)
print("✅ 图6已保存")

# ========== 图7: 曲率与泛化能力关系 ==========
print("生成图7: 曲率与泛化...")

# 模拟曲率与泛化误差的关系
curvature = np.linspace(0, 5, 100)
generalization_error = 0.1 + 0.3 * curvature**1.5 / (1 + curvature**1.5) + 0.05 * np.random.randn(100)
training_error = 0.05 * np.ones_like(curvature) + 0.02 * np.random.randn(100)

fig7 = go.Figure()

fig7.add_trace(go.Scatter(
    x=curvature, y=training_error,
    mode='lines',
    line=dict(color=secondary_color, width=2),
    name='训练误差'
))

fig7.add_trace(go.Scatter(
    x=curvature, y=generalization_error,
    mode='lines',
    line=dict(color=accent_color, width=2),
    name='泛化误差'
))

fig7.add_vline(x=2, line=dict(color='red', dash='dash', width=2))
fig7.add_annotation(
    x=2, y=0.35,
    text='平坦最小值区域',
    showarrow=False,
    font=dict(size=12, color='red')
)

fig7.update_layout(
    title='损失景观曲率与模型泛化能力',
    xaxis_title='损失景观曲率 κ',
    yaxis_title='误差',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=600,
    height=400,
    legend=dict(x=0.02, y=0.98)
)

fig7.write_image('static/images/plots/curvature_generalization.png', scale=2)
print("✅ 图7已保存")

print("\n✅ 所有配图生成完成！")
print("图片保存在 static/images/plots/ 目录下")
