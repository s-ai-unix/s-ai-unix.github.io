#!/usr/bin/env python3
"""
极小曲面配图生成
包含：悬链面、螺旋面、Scherk曲面、Enneper曲面等经典极小曲面可视化
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

print("开始生成极小曲面配图...")

# ========== 图1: 悬链面（Catenoid）==========
print("生成图1: 悬链面...")

u = np.linspace(-2, 2, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u, v)

# 悬链面参数方程
a = 1
x_cat = a * np.cosh(U) * np.cos(V)
y_cat = a * np.cosh(U) * np.sin(V)
z_cat = a * U

fig1 = go.Figure()

fig1.add_trace(go.Surface(
    x=x_cat, y=y_cat, z=z_cat,
    colorscale='Blues',
    showscale=False,
    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.4),
    name='悬链面'
))

# 添加边界圆环
u_bound = 2
theta = np.linspace(0, 2*np.pi, 100)
x_bound = a * np.cosh(u_bound) * np.cos(theta)
y_bound = a * np.cosh(u_bound) * np.sin(theta)
z_bound_top = a * u_bound * np.ones_like(theta)
z_bound_bottom = -a * u_bound * np.ones_like(theta)

fig1.add_trace(go.Scatter3d(
    x=x_bound, y=y_bound, z=z_bound_top,
    mode='lines',
    line=dict(color='red', width=4),
    name='上边界'
))

fig1.add_trace(go.Scatter3d(
    x=x_bound, y=y_bound, z=z_bound_bottom,
    mode='lines',
    line=dict(color='red', width=4),
    name='下边界'
))

fig1.update_layout(
    title='悬链面：旋转悬链线形成的极小曲面',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        aspectratio=dict(x=1, y=1, z=2)
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=800,
    height=700
)

fig1.write_image('static/images/plots/catenoid.png', scale=2)
print("✅ 图1已保存")

# ========== 图2: 螺旋面（Helicoid）==========
print("生成图2: 螺旋面...")

u = np.linspace(-2, 2, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u, v)

# 螺旋面参数方程
x_hel = U * np.cos(V)
y_hel = U * np.sin(V)
z_hel = V

fig2 = go.Figure()

fig2.add_trace(go.Surface(
    x=x_hel, y=y_hel, z=z_hel,
    colorscale='Greens',
    showscale=False,
    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.4)
))

# 添加螺旋线
u_spiral = np.linspace(-2, 2, 50)
v_spiral = np.linspace(0, 4*np.pi, 200)
U_sp, V_sp = np.meshgrid(u_spiral, v_spiral)
x_sp = U_sp * np.cos(V_sp)
y_sp = U_sp * np.sin(V_sp)
z_sp = V_sp

fig2.add_trace(go.Scatter3d(
    x=x_sp.flatten(), y=y_sp.flatten(), z=z_sp.flatten(),
    mode='markers',
    marker=dict(size=1, color='rgba(255,255,255,0.3)'),
    showlegend=False
))

fig2.update_layout(
    title='螺旋面：螺旋上升的极小曲面',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        aspectratio=dict(x=1, y=1, z=1.5)
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=800,
    height=700
)

fig2.write_image('static/images/plots/helicoid.png', scale=2)
print("✅ 图2已保存")

# ========== 图3: Scherk曲面 ==========
print("生成图3: Scherk曲面...")

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Scherk第一曲面: z = ln(cos(x)/cos(y))
Z = np.log(np.abs(np.cos(X) / np.cos(Y)))
# 处理无穷大值
Z[np.isinf(Z)] = np.nan
Z[np.isnan(Z)] = 0
Z = np.clip(Z, -3, 3)

fig3 = go.Figure()

fig3.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    colorscale='Oranges',
    showscale=False,
    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.4)
))

fig3.update_layout(
    title='Scherk曲面：周期性极小曲面',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=800,
    height=700
)

fig3.write_image('static/images/plots/scherk_surface.png', scale=2)
print("✅ 图3已保存")

# ========== 图4: Enneper曲面 ==========
print("生成图4: Enneper曲面...")

u = np.linspace(-2, 2, 100)
v = np.linspace(-2, 2, 100)
U, V = np.meshgrid(u, v)

# Enneper曲面参数方程
x_enn = U - (U**3)/3 + U*V**2
y_enn = V - (V**3)/3 + V*U**2
z_enn = U**2 - V**2

fig4 = go.Figure()

fig4.add_trace(go.Surface(
    x=x_enn, y=y_enn, z=z_enn,
    colorscale='Purples',
    showscale=False,
    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.4)
))

fig4.update_layout(
    title='Enneper曲面：自相交的极小曲面',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=800,
    height=700
)

fig4.write_image('static/images/plots/enneper_surface.png', scale=2)
print("✅ 图4已保存")

# ========== 图5: 平均曲率示意 ==========
print("生成图5: 平均曲率示意...")

# 创建一个曲面并显示其曲率
u = np.linspace(-1, 1, 50)
v = np.linspace(-1, 1, 50)
U, V = np.meshgrid(u, v)

# 椭圆抛物面（非极小）
a, b = 1, 0.5
Z_elliptic = (U**2)/a**2 + (V**2)/b**2

# 双曲抛物面（ saddle）
Z_hyperbolic = U**2 - V**2

fig5 = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=('椭圆抛物面（H > 0）', '双曲抛物面（H 可正可负）')
)

fig5.add_trace(go.Surface(
    x=U, y=V, z=Z_elliptic,
    colorscale='Reds',
    showscale=False
), row=1, col=1)

fig5.add_trace(go.Surface(
    x=U, y=V, z=Z_hyperbolic,
    colorscale='Blues',
    showscale=False
), row=1, col=2)

fig5.update_layout(
    title='平均曲率：曲面在每一点的弯曲程度',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=900,
    height=450
)

fig5.write_image('static/images/plots/mean_curvature.png', scale=2)
print("✅ 图5已保存")

# ========== 图6: 肥皂膜实验示意 ==========
print("生成图6: 肥皂膜实验...")

# 两个圆环之间的肥皂膜（悬链面近似）
theta = np.linspace(0, 2*np.pi, 100)
z_levels = np.linspace(-1, 1, 20)

fig6 = go.Figure()

# 绘制多个水平截面的圆环
for z in z_levels:
    r = np.cosh(z)  # 悬链面的截面半径
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    z_circle = z * np.ones_like(theta)
    
    fig6.add_trace(go.Scatter3d(
        x=x_circle, y=y_circle, z=z_circle,
        mode='lines',
        line=dict(color='rgba(100,149,237,0.5)', width=2),
        showlegend=False
    ))

# 上下边界圆环（金属丝框架）
r_bound = np.cosh(1)
x_bound = r_bound * np.cos(theta)
y_bound = r_bound * np.sin(theta)

fig6.add_trace(go.Scatter3d(
    x=x_bound, y=y_bound, z=np.ones_like(theta),
    mode='lines',
    line=dict(color='gray', width=6),
    name='上圆环'
))

fig6.add_trace(go.Scatter3d(
    x=x_bound, y=y_bound, z=-np.ones_like(theta),
    mode='lines',
    line=dict(color='gray', width=6),
    name='下圆环'
))

fig6.update_layout(
    title='肥皂膜实验：两个圆环之间的极小曲面',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=800,
    height=700
)

fig6.write_image('static/images/plots/soap_film.png', scale=2)
print("✅ 图6已保存")

# ========== 图7: 变分原理示意 ==========
print("生成图7: 变分原理...")

x = np.linspace(0, 1, 100)

# 几条不同的曲线
y1 = 0.5 * np.ones_like(x)  # 直线
y2 = 0.5 + 0.3 * np.sin(np.pi * x)  # 波动曲线
y3 = 0.5 + 0.2 * np.sin(2 * np.pi * x)  # 高频波动

fig7 = go.Figure()

fig7.add_trace(go.Scatter(
    x=x, y=y1,
    mode='lines',
    line=dict(color=primary_color, width=3),
    name='极小曲面（稳定平衡）'
))

fig7.add_trace(go.Scatter(
    x=x, y=y2,
    mode='lines',
    line=dict(color=accent_color, width=2, dash='dash'),
    name='扰动曲面'
))

fig7.add_trace(go.Scatter(
    x=x, y=y3,
    mode='lines',
    line=dict(color='gray', width=1),
    name='更大扰动'
))

# 添加面积变化的示意
fig7.add_annotation(
    x=0.5, y=0.8,
    text='面积 A = ∫∫ dS',
    showarrow=False,
    font=dict(size=14)
)

fig7.update_layout(
    title='变分原理：极小曲面是面积泛函的临界点',
    xaxis_title='x',
    yaxis_title='y',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=700,
    height=450
)

fig7.write_image('static/images/plots/variational_principle.png', scale=2)
print("✅ 图7已保存")

# ========== 图8: 高斯映射与Weierstrass表示 ==========
print("生成图8: 高斯映射示意...")

# 单位球面
phi = np.linspace(0, np.pi, 50)
theta = np.linspace(0, 2*np.pi, 50)
PHI, THETA = np.meshgrid(phi, theta)

X_sphere = np.sin(PHI) * np.cos(THETA)
Y_sphere = np.sin(PHI) * np.sin(THETA)
Z_sphere = np.cos(PHI)

fig8 = go.Figure()

# 绘制球面的一部分（上半球）
mask = Z_sphere >= 0
X_half = np.where(mask, X_sphere, np.nan)
Y_half = np.where(mask, Y_sphere, np.nan)
Z_half = np.where(mask, Z_sphere, np.nan)

fig8.add_trace(go.Surface(
    x=X_half, y=Y_half, z=Z_half,
    colorscale='Blues',
    opacity=0.5,
    showscale=False,
    name='高斯球面'
))

# 添加几个法向量点
points_theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
points_phi = np.array([np.pi/4, np.pi/4, np.pi/4, np.pi/4])

for i in range(len(points_theta)):
    x_pt = np.sin(points_phi[i]) * np.cos(points_theta[i])
    y_pt = np.sin(points_phi[i]) * np.sin(points_theta[i])
    z_pt = np.cos(points_phi[i])
    
    fig8.add_trace(go.Scatter3d(
        x=[0, x_pt], y=[0, y_pt], z=[0, z_pt],
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=8),
        showlegend=False
    ))

fig8.update_layout(
    title='高斯映射：曲面法向量在球面上的表示',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=700,
    height=600
)

fig8.write_image('static/images/plots/gauss_map.png', scale=2)
print("✅ 图8已保存")

print("\n✅ 所有极小曲面配图生成完成！")
