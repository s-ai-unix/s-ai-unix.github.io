import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

os.makedirs('static/images/plots', exist_ok=True)

def apply_apple_style(fig, title, width=800, height=500):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family='Arial, sans-serif')),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=width,
        height=height,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# ============ 图1: SE(3) 李群结构 ============
print("生成图1: SE(3) 李群结构...")

fig1 = make_subplots(rows=1, cols=2,
                     subplot_titles=('李代数 se(3)', '李群 SE(3)'),
                     specs=[[{'type': 'xy'}, {'type': 'xy'}]])

# 左图：李代数元素（螺旋运动）
theta = np.linspace(0, 2*np.pi, 100)
# 角速度方向
fig1.add_trace(go.Scatter(
    x=[0, 1], y=[0, 0],
    mode='lines+markers',
    line=dict(color='#007AFF', width=4),
    marker=dict(size=8, color='#007AFF'),
    name='角速度 ω'
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=[0, 0], y=[0, 0.8],
    mode='lines+markers',
    line=dict(color='#34C759', width=4),
    marker=dict(size=8, color='#34C759'),
    name='线速度 v'
), row=1, col=1)

# 螺旋表示
x_spiral = 0.2 * theta
y_spiral = 0.5 * np.sin(theta)
fig1.add_trace(go.Scatter(
    x=x_spiral, y=y_spiral,
    mode='lines',
    line=dict(color='#FF9500', width=2),
    name='螺旋运动 ξ'
), row=1, col=1)

fig1.add_annotation(x=0.5, y=-0.15, 
                   text='ξ = (ω, v) ∈ se(3)',
                   showarrow=False, xref='paper', yref='paper',
                   font=dict(size=14, color='#007AFF'))

# 右图：李群元素（刚体变换）
# 原始坐标系
origin = np.array([0, 0])
x_axis = np.array([1, 0])
y_axis = np.array([0, 1])

# 绘制原始坐标系（虚线）
fig1.add_trace(go.Scatter(
    x=[origin[0], x_axis[0]], y=[origin[1], x_axis[1]],
    mode='lines',
    line=dict(color='#CCCCCC', width=2, dash='dash'),
    showlegend=False
), row=1, col=2)

fig1.add_trace(go.Scatter(
    x=[origin[0], y_axis[0]], y=[origin[1], y_axis[1]],
    mode='lines',
    line=dict(color='#CCCCCC', width=2, dash='dash'),
    showlegend=False
), row=1, col=2)

# 旋转后的坐标系（30度旋转 + 平移）
angle = np.pi/6
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])
translation = np.array([2, 0.5])

x_axis_rot = R @ x_axis + translation
y_axis_rot = R @ y_axis + translation
origin_rot = translation

fig1.add_trace(go.Scatter(
    x=[origin_rot[0], x_axis_rot[0]], y=[origin_rot[1], x_axis_rot[1]],
    mode='lines+markers',
    line=dict(color='#FF3B30', width=4),
    marker=dict(size=6, color='#FF3B30'),
    name='x轴'
), row=1, col=2)

fig1.add_trace(go.Scatter(
    x=[origin_rot[0], y_axis_rot[0]], y=[origin_rot[1], y_axis_rot[1]],
    mode='lines+markers',
    line=dict(color='#34C759', width=4),
    marker=dict(size=6, color='#34C759'),
    name='y轴'
), row=1, col=2)

fig1.add_trace(go.Scatter(
    x=[origin_rot[0]], y=[origin_rot[1]],
    mode='markers',
    marker=dict(size=10, color='#007AFF'),
    name='原点'
), row=1, col=2)

# 添加变换箭头
fig1.add_annotation(x=1, y=0.25,
                   ax=0, ay=0,
                   xref='x2', yref='y2',
                   axref='x2', ayref='y2',
                   showarrow=True,
                   arrowhead=2, arrowsize=1, arrowwidth=2,
                   arrowcolor='#FF9500')

fig1.add_annotation(x=0.5, y=-0.15, 
                   text='T = (R, t) ∈ SE(3)',
                   showarrow=False, xref='paper', yref='paper',
                   font=dict(size=14, color='#007AFF'))

fig1.update_xaxes(range=[-0.5, 3], row=1, col=1)
fig1.update_yaxes(range=[-1, 1.5], row=1, col=1)
fig1.update_xaxes(range=[-0.5, 3.5], row=1, col=2)
fig1.update_yaxes(range=[-0.5, 2], row=1, col=2)

apply_apple_style(fig1, '李群 SE(3) 与李代数 se(3)', 900, 450)
fig1.write_image('static/images/plots/robotics-se3-lie-group.png', scale=2)

# ============ 图2: 指数映射 ============
print("生成图2: 指数映射...")

fig2 = go.Figure()

# 展示从李代数到李群的映射
theta_range = np.linspace(0, 2*np.pi, 200)

# 左轴：李代数（角速度大小）
# 右轴：李群（旋转角度）

# 指数映射可视化
omega_vals = np.linspace(-np.pi, np.pi, 100)
# 对于 SO(3)，指数映射直接对应：旋转角度 = |ω|
rotation_angle = np.abs(omega_vals)

fig2.add_trace(go.Scatter(
    x=omega_vals, y=rotation_angle,
    mode='lines',
    line=dict(color='#007AFF', width=3),
    name='指数映射: θ = |ω|'
))

# 添加参考线
fig2.add_trace(go.Scatter(
    x=[-np.pi, np.pi], y=[np.pi, np.pi],
    mode='lines',
    line=dict(color='#FF3B30', width=2, dash='dash'),
    name='2π 边界'
))

# 标记几个关键点
key_points = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
for pt in key_points:
    fig2.add_trace(go.Scatter(
        x=[pt], y=[abs(pt)],
        mode='markers',
        marker=dict(size=10, color='#34C759'),
        showlegend=False
    ))

fig2.update_layout(
    xaxis_title='李代数元素 ω (角速度)',
    yaxis_title='李群元素 θ (旋转角度)',
    yaxis=dict(range=[0, np.pi+0.5])
)

apply_apple_style(fig2, '指数映射: so(3) → SO(3)', 800, 500)
fig2.write_image('static/images/plots/robotics-exponential-map.png', scale=2)

# ============ 图3: 雅可比矩阵映射 ============
print("生成图3: 雅可比矩阵映射...")

fig3 = make_subplots(rows=2, cols=2,
                     subplot_titles=('关节空间速度 θ̇', '任务空间速度 V',
                                   '雅可比矩阵 J(θ)', '速度椭球'),
                     specs=[[{'type': 'xy'}, {'type': 'xy'}],
                            [{'type': 'xy'}, {'type': 'xy'}]])

# 关节空间速度示例
joint_vel_samples = [
    [0.5, 0.3],
    [-0.3, 0.6],
    [0.7, -0.2],
    [-0.4, -0.5]
]

colors = ['#007AFF', '#34C759', '#FF9500', '#FF3B30']

for vel, color in zip(joint_vel_samples, colors):
    fig3.add_trace(go.Scatter(
        x=[0, vel[0]], y=[0, vel[1]],
        mode='lines+markers',
        line=dict(color=color, width=3),
        marker=dict(size=[4, 8], color=color),
        showlegend=False
    ), row=1, col=1)

fig3.update_xaxes(range=[-1, 1], title='θ̇₁', row=1, col=1)
fig3.update_yaxes(range=[-1, 1], title='θ̇₂', row=1, col=1)

# 简化的雅可比映射（2x2矩阵）
J = np.array([[0.8, 0.3], [0.2, 0.9]])

# 任务空间速度
for vel, color in zip(joint_vel_samples, colors):
    task_vel = J @ np.array(vel)
    fig3.add_trace(go.Scatter(
        x=[0, task_vel[0]], y=[0, task_vel[1]],
        mode='lines+markers',
        line=dict(color=color, width=3),
        marker=dict(size=[4, 8], color=color),
        showlegend=False
    ), row=1, col=2)

fig3.update_xaxes(range=[-1, 1], title='vₓ', row=1, col=2)
fig3.update_yaxes(range=[-1, 1], title='vᵧ', row=1, col=2)

# 雅可比矩阵可视化
fig3.add_trace(go.Heatmap(
    z=J,
    colorscale='Blues',
    showscale=True,
    text=[[f'{J[0,0]:.2f}', f'{J[0,1]:.2f}'],
          [f'{J[1,0]:.2f}', f'{J[1,1]:.2f}']],
    texttemplate='%{text}',
    textfont=dict(size=16)
), row=2, col=1)

fig3.update_xaxes(showticklabels=False, row=2, col=1)
fig3.update_yaxes(showticklabels=False, row=2, col=1)

# 速度椭球（可操作性）
theta_ellipse = np.linspace(0, 2*np.pi, 100)
# 单位圆通过 J 映射为椭球
unit_circle = np.array([np.cos(theta_ellipse), np.sin(theta_ellipse)])
ellipse = J @ unit_circle

fig3.add_trace(go.Scatter(
    x=ellipse[0, :], y=ellipse[1, :],
    mode='lines',
    line=dict(color='#007AFF', width=3),
    fill='toself',
    fillcolor='rgba(0,122,255,0.2)',
    showlegend=False
), row=2, col=2)

# 主轴
U, S, Vt = np.svd(J)
for i in range(2):
    fig3.add_trace(go.Scatter(
        x=[0, S[i]*U[0, i]], y=[0, S[i]*U[1, i]],
        mode='lines',
        line=dict(color='#FF3B30', width=2, dash='dash'),
        showlegend=False
    ), row=2, col=2)

fig3.update_xaxes(range=[-1.5, 1.5], title='vₓ', row=2, col=2)
fig3.update_yaxes(range=[-1.5, 1.5], title='vᵧ', row=2, col=2)

apply_apple_style(fig3, '', 900, 800)
fig3.write_image('static/images/plots/robotics-jacobian.png', scale=2)

# ============ 图4: 测地线 vs 线性插值 ============
print("生成图4: 测地线 vs 线性插值...")

fig4 = make_subplots(rows=1, cols=2,
                     subplot_titles=('线性插值 (错误)', '测地线插值 SLERP (正确)'),
                     specs=[[{'type': 'xy'}, {'type': 'xy'}]])

# 起点和终点旋转（用2D旋转表示）
angle1 = 0
angle2 = 3*np.pi/4  # 135度

# 线性插值（直接在角度上插值）
t = np.linspace(0, 1, 50)
# 线性插值角度
linear_angle = (1-t) * angle1 + t * angle2
# 单位圆上的点
x_linear = np.cos(linear_angle)
y_linear = np.sin(linear_angle)

fig4.add_trace(go.Scatter(
    x=x_linear, y=y_linear,
    mode='lines',
    line=dict(color='#FF3B30', width=3),
    name='线性路径'
), row=1, col=1)

# 添加起点和终点
fig4.add_trace(go.Scatter(
    x=[np.cos(angle1), np.cos(angle2)],
    y=[np.sin(angle1), np.sin(angle2)],
    mode='markers',
    marker=dict(size=12, color=['#34C759', '#FF9500']),
    showlegend=False
), row=1, col=1)

# 单位圆
unit_circle_x = np.cos(np.linspace(0, 2*np.pi, 100))
unit_circle_y = np.sin(np.linspace(0, 2*np.pi, 100))
fig4.add_trace(go.Scatter(
    x=unit_circle_x, y=unit_circle_y,
    mode='lines',
    line=dict(color='#CCCCCC', width=1),
    showlegend=False
), row=1, col=1)

# 测地线插值（SLERP）- 在圆上最短路径
geodesic_angle = np.where(linear_angle > np.pi, 
                          linear_angle - 2*np.pi, 
                          linear_angle)
x_geodesic = np.cos(geodesic_angle)
y_geodesic = np.sin(geodesic_angle)

fig4.add_trace(go.Scatter(
    x=x_geodesic, y=y_geodesic,
    mode='lines',
    line=dict(color='#34C759', width=3),
    name='测地线路径'
), row=1, col=2)

fig4.add_trace(go.Scatter(
    x=[np.cos(angle1), np.cos(angle2)],
    y=[np.sin(angle1), np.sin(angle2)],
    mode='markers',
    marker=dict(size=12, color=['#34C759', '#FF9500']),
    showlegend=False
), row=1, col=2)

fig4.add_trace(go.Scatter(
    x=unit_circle_x, y=unit_circle_y,
    mode='lines',
    line=dict(color='#CCCCCC', width=1),
    showlegend=False
), row=1, col=2)

for i in [1, 2]:
    fig4.update_xaxes(range=[-1.5, 1.5], aspectscale=1, row=1, col=i)
    fig4.update_yaxes(range=[-1.5, 1.5], aspectscale=1, row=1, col=i)

apply_apple_style(fig4, 'SO(2) 上的插值对比', 900, 450)
fig4.write_image('static/images/plots/robotics-interpolation.png', scale=2)

# ============ 图5: 机器人动力学能量 ============
print("生成图5: 动力学能量曲面...")

fig5 = go.Figure()

# 简化的双摆势能
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# 势能函数：双势阱
V = 2*(X**2 - 1)**2 + 0.5*Y**2 + 0.3*X*Y

# 绘制等高线
contours = fig5.add_trace(go.Contour(
    x=x, y=y, z=V,
    colorscale='Blues',
    contours=dict(start=0, end=8, size=0.5),
    colorbar=dict(title='势能')
))

# 添加轨迹示例
t_traj = np.linspace(0, 3, 200)
x_traj = 0.9 * np.cos(t_traj) * np.exp(-0.1*t_traj)
y_traj = 0.5 * np.sin(1.5*t_traj) * np.exp(-0.1*t_traj)

fig5.add_trace(go.Scatter(
    x=x_traj, y=y_traj,
    mode='lines',
    line=dict(color='#FF3B30', width=3),
    name='轨迹'
))

# 标记平衡点
fig5.add_trace(go.Scatter(
    x=[-1, 1], y=[0, 0],
    mode='markers',
    marker=dict(size=12, color='#34C759', symbol='star'),
    name='稳定平衡点'
))

fig5.add_trace(go.Scatter(
    x=[0], y=[0],
    mode='markers',
    marker=dict(size=12, color='#FF9500', symbol='x'),
    name='不稳定平衡点'
))

fig5.update_layout(
    xaxis_title='θ₁',
    yaxis_title='θ₂',
    xaxis=dict(scaleanchor='y')
)

apply_apple_style(fig5, '关节空间势能等高线与运动轨迹', 800, 600)
fig5.write_image('static/images/plots/robotics-dynamics.png', scale=2)

print("所有图形生成完成!")
