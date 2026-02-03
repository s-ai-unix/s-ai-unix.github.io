import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# 确保输出目录存在
os.makedirs('static/images/plots', exist_ok=True)

# 设置统一的样式
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

# ============ 图1: 参数化曲面与切平面 ============
print("生成图1: 参数化曲面与切平面...")

# 生成一个鞍面 (双曲抛物面)
u = np.linspace(-2, 2, 50)
v = np.linspace(-2, 2, 50)
U, V = np.meshgrid(u, v)
X = U
Y = V
Z = U**2 - V**2  # 鞍面

# 在某点计算切平面
u0, v0 = 0.5, 0.5
x0, y0, z0 = u0, v0, u0**2 - v0**2

# 偏导数
ru = np.array([1, 0, 2*u0])  # r_u
rv = np.array([0, 1, -2*v0])  # r_v

# 切平面上的点
uu, vv = np.meshgrid(np.linspace(-0.5, 0.5, 10), np.linspace(-0.5, 0.5, 10))
X_tangent = x0 + uu * ru[0] + vv * rv[0]
Y_tangent = y0 + uu * ru[1] + vv * rv[1]
Z_tangent = z0 + uu * ru[2] + vv * rv[2]

fig1 = go.Figure()

# 曲面
fig1.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    colorscale='Blues',
    opacity=0.7,
    showscale=False,
    name='曲面 S'
))

# 切平面
fig1.add_trace(go.Surface(
    x=X_tangent, y=Y_tangent, z=Z_tangent,
    colorscale='Oranges',
    opacity=0.5,
    showscale=False,
    name='切平面'
))

# 切点
fig1.add_trace(go.Scatter3d(
    x=[x0], y=[y0], z=[z0],
    mode='markers',
    marker=dict(size=8, color='#FF9500'),
    name='点 P'
))

# 坐标曲线
u_curve = np.linspace(-2, 2, 100)
v_const = v0
fig1.add_trace(go.Scatter3d(
    x=u_curve, y=v_const*np.ones_like(u_curve), 
    z=u_curve**2 - v_const**2,
    mode='lines',
    line=dict(color='#007AFF', width=4),
    name='u-曲线 (v=v₀)'
))

v_curve = np.linspace(-2, 2, 100)
u_const = u0
fig1.add_trace(go.Scatter3d(
    x=u_const*np.ones_like(v_curve), y=v_curve, 
    z=u_const**2 - v_curve**2,
    mode='lines',
    line=dict(color='#34C759', width=4),
    name='v-曲线 (u=u₀)'
))

fig1.update_layout(
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        aspectmode='cube'
    ),
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)
apply_apple_style(fig1, '参数化曲面与切平面', 900, 700)
fig1.write_image('static/images/plots/surface_parametrization.png', scale=2)

# ============ 图2: 第一基本型的度量意义 ============
print("生成图2: 第一基本型的度量意义...")

# 展示曲面上的弧长元素
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 50)
U, V = np.meshgrid(u, v)

# 单位球面
X_sphere = np.sin(V) * np.cos(U)
Y_sphere = np.sin(V) * np.sin(U)
Z_sphere = np.cos(V)

fig2 = go.Figure()

# 球面
fig2.add_trace(go.Surface(
    x=X_sphere, y=Y_sphere, z=Z_sphere,
    colorscale='Blues',
    opacity=0.4,
    showscale=False
))

# 画两条路径对比
# 路径1: 沿纬线 (短路径)
t1 = np.linspace(0.3, 1.8, 100)
v0_path = np.pi/3  # 30度纬度
x1 = np.sin(v0_path) * np.cos(t1)
y1 = np.sin(v0_path) * np.sin(t1)
z1 = np.cos(v0_path) * np.ones_like(t1)

fig2.add_trace(go.Scatter3d(
    x=x1, y=y1, z=z1,
    mode='lines',
    line=dict(color='#007AFF', width=6),
    name='路径 γ₁ (曲面上)'
))

# 路径2: 沿经线
t2 = np.linspace(0.5, 1.5, 100)
u0_path = np.pi/4
x2 = np.sin(t2) * np.cos(u0_path)
y2 = np.sin(t2) * np.sin(u0_path)
z2 = np.cos(t2)

fig2.add_trace(go.Scatter3d(
    x=x2, y=y2, z=z2,
    mode='lines',
    line=dict(color='#34C759', width=6),
    name='路径 γ₂ (曲面上)'
))

# 起点和终点标记
fig2.add_trace(go.Scatter3d(
    x=[x1[0], x1[-1]], y=[y1[0], y1[-1]], z=[z1[0], z1[-1]],
    mode='markers',
    marker=dict(size=8, color='#FF9500'),
    name='端点'
))

fig2.update_layout(
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        aspectmode='cube'
    ),
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)
apply_apple_style(fig2, '曲面上的弧长度量', 900, 700)
fig2.write_image('static/images/plots/first_fundamental_form.png', scale=2)

# ============ 图3: 法曲率与主曲率（圆柱面） ============
print("生成图3: 法曲率与主曲率...")

# 圆柱面展示不同方向的曲率
u_cyl = np.linspace(0, 2*np.pi, 100)
v_cyl = np.linspace(-1, 1, 50)
U_cyl, V_cyl = np.meshgrid(u_cyl, v_cyl)

R = 1  # 圆柱半径
X_cyl = R * np.cos(U_cyl)
Y_cyl = R * np.sin(U_cyl)
Z_cyl = V_cyl

fig3 = go.Figure()

fig3.add_trace(go.Surface(
    x=X_cyl, y=Y_cyl, z=Z_cyl,
    colorscale='Blues',
    opacity=0.5,
    showscale=False,
    name='圆柱面'
))

# 在一点处的法曲率示意
P = np.array([R, 0, 0])

# 方向1: 沿母线（直线，曲率为0）
t_dir1 = np.linspace(-0.5, 0.5, 50)
x_dir1 = np.ones_like(t_dir1) * R
y_dir1 = np.zeros_like(t_dir1)
z_dir1 = t_dir1

fig3.add_trace(go.Scatter3d(
    x=x_dir1, y=y_dir1, z=z_dir1,
    mode='lines',
    line=dict(color='#34C759', width=5),
    name='方向1: κ=0 (母线)'
))

# 方向2: 沿圆周（曲率最大）
t_dir2 = np.linspace(-0.3, 0.3, 50)
x_dir2 = R * np.cos(t_dir2)
y_dir2 = R * np.sin(t_dir2)
z_dir2 = np.zeros_like(t_dir2)

fig3.add_trace(go.Scatter3d(
    x=x_dir2, y=y_dir2, z=z_dir2,
    mode='lines',
    line=dict(color='#007AFF', width=5),
    name='方向2: κ=1/R (圆周)'
))

# 标记点
fig3.add_trace(go.Scatter3d(
    x=[R], y=[0], z=[0],
    mode='markers',
    marker=dict(size=8, color='#FF9500'),
    name='点 P'
))

fig3.update_layout(
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        aspectmode='cube'
    ),
    showlegend=True,
    legend=dict(x=0.02, y=0.98)
)
apply_apple_style(fig3, '圆柱面上不同方向的法曲率', 900, 700)
fig3.write_image('static/images/plots/normal_curvature_cylinder.png', scale=2)

# ============ 图4: 高斯曲率分类 - 椭圆点 ============
print("生成图4: 高斯曲率分类...")

fig4a = go.Figure()
u_sph = np.linspace(0, 2*np.pi, 50)
v_sph = np.linspace(0, np.pi/2, 50)
U_sph, V_sph = np.meshgrid(u_sph, v_sph)
X_sph = np.sin(V_sph) * np.cos(U_sph)
Y_sph = np.sin(V_sph) * np.sin(U_sph)
Z_sph = np.cos(V_sph)

fig4a.add_trace(go.Surface(
    x=X_sph, y=Y_sph, z=Z_sph,
    colorscale='Reds',
    opacity=0.9,
    showscale=False,
    name='球面 K>0'
))
fig4a.update_layout(scene=dict(aspectmode='cube'))
apply_apple_style(fig4a, '椭圆点: K>0', 600, 500)
fig4a.write_image('static/images/plots/elliptic_point.png', scale=2)

# 双曲点
fig4b = go.Figure()
u_hyp = np.linspace(-1.5, 1.5, 50)
v_hyp = np.linspace(-1.5, 1.5, 50)
U_hyp, V_hyp = np.meshgrid(u_hyp, v_hyp)
X_hyp = U_hyp
Y_hyp = V_hyp
Z_hyp = U_hyp**2 - V_hyp**2

fig4b.add_trace(go.Surface(
    x=X_hyp, y=Y_hyp, z=Z_hyp,
    colorscale='Blues',
    opacity=0.9,
    showscale=False,
    name='鞍面 K<0'
))
fig4b.update_layout(scene=dict(aspectmode='cube'))
apply_apple_style(fig4b, '双曲点: K<0', 600, 500)
fig4b.write_image('static/images/plots/hyperbolic_point.png', scale=2)

# 抛物点
fig4c = go.Figure()
fig4c.add_trace(go.Surface(
    x=X_cyl, y=Y_cyl, z=Z_cyl,
    colorscale='Greens',
    opacity=0.9,
    showscale=False,
    name='圆柱 K=0'
))
fig4c.update_layout(scene=dict(aspectmode='cube'))
apply_apple_style(fig4c, '抛物点: K=0', 600, 500)
fig4c.write_image('static/images/plots/parabolic_point.png', scale=2)

# ============ 图5: 可展曲面与高斯绝妙定理 ============
print("生成图5: 高斯绝妙定理...")

fig5 = go.Figure()

# 圆锥
u_cone = np.linspace(0, 2*np.pi, 50)
v_cone = np.linspace(0, 1, 30)
U_cone, V_cone = np.meshgrid(u_cone, v_cone)
h = 1  # 高度
r_base = 0.8
X_cone = r_base * (1 - V_cone) * np.cos(U_cone)
Y_cone = r_base * (1 - V_cone) * np.sin(U_cone)
Z_cone = h * V_cone

fig5.add_trace(go.Surface(
    x=X_cone, y=Y_cone, z=Z_cone,
    colorscale='Oranges',
    opacity=0.8,
    showscale=False,
    name='圆锥 (可展曲面)'
))

fig5.update_layout(scene=dict(aspectmode='cube'))
apply_apple_style(fig5, '可展曲面: 圆锥', 800, 600)
fig5.write_image('static/images/plots/developable_surface.png', scale=2)

# ============ 图6: 曲面上的测地线 ============
print("生成图6: 曲面上的测地线...")

fig6 = go.Figure()

# 球面
X_sphere_full = np.sin(V_sph) * np.cos(U_sph)
Y_sphere_full = np.sin(V_sph) * np.sin(U_sph)
Z_sphere_full = np.cos(V_sph)

fig6.add_trace(go.Surface(
    x=X_sphere_full, y=Y_sphere_full, z=Z_sphere_full,
    colorscale='Blues',
    opacity=0.4,
    showscale=False
))

# 测地线（大圆）
t_geo = np.linspace(0, 2*np.pi, 200)
# 通过球心的平面与球面的交线就是测地线
# 选择一个倾斜的平面
phi = np.pi/6  # 倾斜角
x_geo = np.cos(t_geo)
y_geo = np.sin(phi) * np.sin(t_geo)
z_geo = np.cos(phi) * np.sin(t_geo)

fig6.add_trace(go.Scatter3d(
    x=x_geo, y=y_geo, z=z_geo,
    mode='lines',
    line=dict(color='#FF9500', width=5),
    name='测地线 (大圆)'
))

# 另一个测地线
psi = np.pi/3
x_geo2 = np.cos(psi) * np.cos(t_geo)
y_geo2 = np.sin(t_geo)
z_geo2 = np.sin(psi) * np.cos(t_geo)

fig6.add_trace(go.Scatter3d(
    x=x_geo2, y=y_geo2, z=z_geo2,
    mode='lines',
    line=dict(color='#007AFF', width=5),
    name='测地线2'
))

fig6.update_layout(
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        aspectmode='cube'
    ),
    showlegend=True
)
apply_apple_style(fig6, '球面上的测地线', 900, 700)
fig6.write_image('static/images/plots/geodesic.png', scale=2)

print("所有图形生成完成!")
