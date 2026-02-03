#!/usr/bin/env python3
"""
为微分几何在自动驾驶中的应用文章生成 Plotly 图形
输出为 PNG 图片格式
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# 设置 Kaleido 超时
os.environ['KALEIDO_TIMEOUT'] = '300'

OUTPUT_DIR = 'static/images/math'

# 苹果风格配色
APPLE_BLUE = "#007AFF"
APPLE_GREEN = "#34C759"
APPLE_ORANGE = "#FF9500"
APPLE_RED = "#FF3B30"
APPLE_GRAY = "#8E8E93"


def save_plotly_as_png(fig, filename, width=800, height=600, scale=2):
    """将 Plotly 图形保存为 PNG 图片"""
    filepath = f'{OUTPUT_DIR}/{filename}'
    fig.write_image(filepath, width=width, height=height, scale=scale)
    print(f"✅ 已生成: {filepath}")
    return filepath


def plot_curvature_speed_relation():
    """图1: 曲率与安全速度的关系"""
    fig = go.Figure()
    
    # 曲率范围
    kappa = np.linspace(0.01, 0.5, 100)  # 曲率 (1/m)
    
    # 不同摩擦系数下的最大安全速度
    mu_values = [0.3, 0.5, 0.7, 0.9]  # 摩擦系数
    colors = [APPLE_RED, APPLE_ORANGE, APPLE_GREEN, APPLE_BLUE]
    
    g = 9.8  # 重力加速度
    
    for mu, color in zip(mu_values, colors):
        v_max = np.sqrt(mu * g / kappa) * 3.6  # 转换为 km/h
        fig.add_trace(go.Scatter(
            x=kappa, y=v_max,
            mode='lines',
            line=dict(color=color, width=2.5),
            name=f'μ = {mu}'
        ))
    
    # 标注典型弯道的曲率
    typical_curvatures = [
        (0.05, "高速公路弯道"),
        (0.15, "城市主干道"),
        (0.30, "急转弯")
    ]
    
    for k, label in typical_curvatures:
        fig.add_vline(x=k, line=dict(color=APPLE_GRAY, width=1, dash='dash'))
        fig.add_annotation(x=k, y=120, text=label, showarrow=False, 
                          font=dict(size=10, color=APPLE_GRAY), textangle=90)
    
    fig.update_layout(
        title=dict(text='道路曲率与最大安全速度的关系', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='曲率 κ (1/m)',
        yaxis_title='最大安全速度 v_max (km/h)',
        template='plotly_white',
        width=800, height=500,
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.7, y=0.95)
    )
    
    save_plotly_as_png(fig, 'autodriving-curvature-speed.png', width=800, height=500)


def plot_dubins_paths():
    """图2: Dubins 曲线示例"""
    fig = go.Figure()
    
    # 定义起点和终点
    start = np.array([0, 0])
    start_theta = 0  # 起点朝向（弧度）
    
    end = np.array([8, 3])
    end_theta = np.pi / 3  # 终点朝向
    
    # 绘制起点和终点
    fig.add_trace(go.Scatter(
        x=[start[0]], y=[start[1]],
        mode='markers+text',
        marker=dict(size=15, color=APPLE_GREEN, symbol='triangle-up'),
        text=['起点'],
        textposition='bottom center',
        name='起点'
    ))
    
    fig.add_trace(go.Scatter(
        x=[end[0]], y=[end[1]],
        mode='markers+text',
        marker=dict(size=15, color=APPLE_RED, symbol='triangle-up'),
        text=['终点'],
        textposition='top center',
        name='终点'
    ))
    
    # 绘制朝向箭头
    arrow_len = 1.5
    fig.add_annotation(
        x=start[0] + arrow_len * np.cos(start_theta),
        y=start[1] + arrow_len * np.sin(start_theta),
        ax=start[0], ay=start[1],
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        arrowcolor=APPLE_GREEN
    )
    
    fig.add_annotation(
        x=end[0] + arrow_len * np.cos(end_theta),
        y=end[1] + arrow_len * np.sin(end_theta),
        ax=end[0], ay=end[1],
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        arrowcolor=APPLE_RED
    )
    
    # 绘制简化的 Dubins 路径（LSL 类型）
    R = 1.5  # 转弯半径
    
    # 左转圆弧（起点）
    t1 = np.linspace(start_theta + np.pi/2, start_theta + np.pi, 50)
    center1 = start - R * np.array([np.cos(start_theta + np.pi/2), np.sin(start_theta + np.pi/2)])
    arc1_x = center1[0] + R * np.cos(t1)
    arc1_y = center1[1] + R * np.sin(t1)
    
    fig.add_trace(go.Scatter(
        x=arc1_x, y=arc1_y,
        mode='lines',
        line=dict(color=APPLE_BLUE, width=3),
        name='左转 (L)'
    ))
    
    # 直线段
    line_start = np.array([arc1_x[-1], arc1_y[-1]])
    line_end = end - R * np.array([np.cos(end_theta - np.pi/2), np.sin(end_theta - np.pi/2)])
    fig.add_trace(go.Scatter(
        x=[line_start[0], line_end[0]],
        y=[line_start[1], line_end[1]],
        mode='lines',
        line=dict(color=APPLE_ORANGE, width=3, dash='dash'),
        name='直行 (S)'
    ))
    
    # 右转圆弧（终点）
    center2 = end - R * np.array([np.cos(end_theta - np.pi/2), np.sin(end_theta - np.pi/2)])
    t2 = np.linspace(end_theta, end_theta - np.pi/2, 50)
    arc2_x = center2[0] + R * np.cos(t2)
    arc2_y = center2[1] + R * np.sin(t2)
    
    fig.add_trace(go.Scatter(
        x=arc2_x, y=arc2_y,
        mode='lines',
        line=dict(color=APPLE_GREEN, width=3),
        name='右转 (R)'
    ))
    
    fig.update_layout(
        title=dict(text='Dubins 路径示例 (LSL 类型)', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        template='plotly_white',
        width=800, height=500,
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.02, y=0.98),
        xaxis=dict(range=[-2, 10]),
        yaxis=dict(range=[-2, 6], scaleanchor='x', scaleratio=1)
    )
    
    save_plotly_as_png(fig, 'autodriving-dubins-path.png', width=800, height=500)


def plot_bezier_curve():
    """图3: 贝塞尔曲线轨迹规划"""
    fig = go.Figure()
    
    # 控制点
    P0 = np.array([0, 0])
    P1 = np.array([2, 4])
    P2 = np.array([6, 4])
    P3 = np.array([8, 1])
    
    control_points = np.array([P0, P1, P2, P3])
    
    # 计算三次贝塞尔曲线
    t = np.linspace(0, 1, 100)
    curve = np.zeros((len(t), 2))
    
    for i, ti in enumerate(t):
        curve[i] = ((1-ti)**3 * P0 + 3*(1-ti)**2*ti * P1 + 
                   3*(1-ti)*ti**2 * P2 + ti**3 * P3)
    
    # 绘制控制多边形
    fig.add_trace(go.Scatter(
        x=control_points[:, 0], y=control_points[:, 1],
        mode='lines+markers',
        line=dict(color=APPLE_GRAY, width=1, dash='dash'),
        marker=dict(size=10, color=APPLE_GRAY),
        name='控制多边形'
    ))
    
    # 绘制贝塞尔曲线
    fig.add_trace(go.Scatter(
        x=curve[:, 0], y=curve[:, 1],
        mode='lines',
        line=dict(color=APPLE_BLUE, width=4),
        name='贝塞尔曲线'
    ))
    
    # 标注控制点
    for i, (x, y) in enumerate(control_points):
        fig.add_annotation(x=x, y=y+0.3, text=f'P{i}', showarrow=False,
                          font=dict(size=12, color=APPLE_GRAY))
    
    # 绘制切向量
    # 起点切向量
    tangent_start = 3 * (P1 - P0)
    fig.add_annotation(
        x=P0[0] + tangent_start[0]*0.3, y=P0[1] + tangent_start[1]*0.3,
        ax=P0[0], ay=P0[1],
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        arrowcolor=APPLE_RED
    )
    fig.add_annotation(x=P0[0]+0.5, y=P0[1]+0.8, text='切线方向', 
                      showarrow=False, font=dict(size=10, color=APPLE_RED))
    
    # 终点切向量
    tangent_end = 3 * (P3 - P2)
    fig.add_annotation(
        x=P3[0] + tangent_end[0]*0.3, y=P3[1] + tangent_end[1]*0.3,
        ax=P3[0], ay=P3[1],
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        arrowcolor=APPLE_RED
    )
    
    fig.update_layout(
        title=dict(text='三次贝塞尔曲线用于轨迹规划', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        template='plotly_white',
        width=800, height=500,
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.65, y=0.95),
        xaxis=dict(range=[-1, 9]),
        yaxis=dict(range=[-1, 5], scaleanchor='x', scaleratio=1)
    )
    
    save_plotly_as_png(fig, 'autodriving-bezier-curve.png', width=800, height=500)


def plot_point_cloud_registration():
    """图4: 点云配准示意图"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('配准前', '配准后 (ICP)'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    np.random.seed(42)
    
    # 生成目标点云（L 形）
    n_points = 50
    target = np.vstack([
        np.column_stack([np.linspace(0, 5, n_points//2), np.zeros(n_points//2)]),
        np.column_stack([np.zeros(n_points//2), np.linspace(0, 3, n_points//2)])
    ])
    target += np.random.normal(0, 0.1, target.shape)
    
    # 源点云（旋转平移后的版本）
    theta = np.pi / 6  # 30度旋转
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    t = np.array([2, 1])
    
    source_before = (target @ R.T) + t + np.random.normal(0, 0.15, target.shape)
    
    # 配准后（近似）
    source_after = target + np.random.normal(0, 0.05, target.shape)
    
    # 左图：配准前
    fig.add_trace(go.Scatter(
        x=target[:, 0], y=target[:, 1],
        mode='markers',
        marker=dict(size=6, color=APPLE_BLUE, opacity=0.7),
        name='目标点云'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=source_before[:, 0], y=source_before[:, 1],
        mode='markers',
        marker=dict(size=6, color=APPLE_RED, opacity=0.7),
        name='源点云'
    ), row=1, col=1)
    
    # 添加对应线
    for i in range(0, len(target), 5):
        fig.add_trace(go.Scatter(
            x=[target[i, 0], source_before[i, 0]],
            y=[target[i, 1], source_before[i, 1]],
            mode='lines',
            line=dict(color=APPLE_GRAY, width=0.5),
            showlegend=False
        ), row=1, col=1)
    
    # 右图：配准后
    fig.add_trace(go.Scatter(
        x=target[:, 0], y=target[:, 1],
        mode='markers',
        marker=dict(size=6, color=APPLE_BLUE, opacity=0.7),
        name='目标点云'
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=source_after[:, 0], y=source_after[:, 1],
        mode='markers',
        marker=dict(size=6, color=APPLE_GREEN, opacity=0.7),
        name='配准后源点云'
    ), row=1, col=2)
    
    fig.update_layout(
        title=dict(text='点云配准：ICP 算法示意图', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white',
        width=1000, height=450,
        margin=dict(l=60, r=40, t=70, b=50),
        showlegend=True
    )
    
    fig.update_xaxes(title_text='x (m)')
    fig.update_yaxes(title_text='y (m)')
    
    save_plotly_as_png(fig, 'autodriving-pointcloud-registration.png', width=1000, height=450)


def plot_configuration_space():
    """图5: 配置空间与障碍物"""
    fig = go.Figure()
    
    # 创建网格表示配置空间 (x, y)
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # 定义障碍物（在配置空间中）
    # 障碍物1：圆形
    obstacle1 = (X - 3)**2 + (Y - 3)**2 < 1.5**2
    # 障碍物2：矩形
    obstacle2 = (X > 6) & (X < 8) & (Y > 2) & (Y < 5)
    # 障碍物3：圆形
    obstacle3 = (X - 7)**2 + (Y - 8)**2 < 1**2
    
    obstacles = obstacle1 | obstacle2 | obstacle3
    
    # 绘制自由空间
    Z = np.where(obstacles, 1, 0)
    
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale=[[0, '#34C759'], [1, '#FF3B30']],
        showscale=False,
        contours=dict(coloring='fill'),
        opacity=0.3
    ))
    
    # 绘制路径
    path_x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    path_y = [2, 3, 4, 5, 5.5, 6, 7, 8, 8.5]
    
    fig.add_trace(go.Scatter(
        x=path_x, y=path_y,
        mode='lines+markers',
        line=dict(color=APPLE_BLUE, width=4),
        marker=dict(size=8, color=APPLE_BLUE),
        name='规划路径'
    ))
    
    # 起点和终点
    fig.add_trace(go.Scatter(
        x=[path_x[0]], y=[path_y[0]],
        mode='markers+text',
        marker=dict(size=15, color=APPLE_GREEN, symbol='circle'),
        text=['起点'],
        textposition='bottom center',
        name='起点'
    ))
    
    fig.add_trace(go.Scatter(
        x=[path_x[-1]], y=[path_y[-1]],
        mode='markers+text',
        marker=dict(size=15, color=APPLE_RED, symbol='star'),
        text=['终点'],
        textposition='top center',
        name='终点'
    ))
    
    # 添加图例说明
    fig.add_annotation(x=0.02, y=0.98, xref='paper', yref='paper',
                       text='绿色: 自由空间<br>红色: 配置空间障碍物',
                       showarrow=False, font=dict(size=11),
                       bgcolor='rgba(255,255,255,0.9)', align='left')
    
    fig.update_layout(
        title=dict(text='配置空间 (C-space) 与运动规划', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        template='plotly_white',
        width=700, height=600,
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.65, y=0.95),
        xaxis=dict(range=[0, 10]),
        yaxis=dict(range=[0, 10], scaleanchor='x', scaleratio=1)
    )
    
    save_plotly_as_png(fig, 'autodriving-configuration-space.png', width=700, height=600)


if __name__ == '__main__':
    print("开始生成微分几何与自动驾驶文章的 Plotly 图形...")
    print("=" * 60)
    
    plot_curvature_speed_relation()
    plot_dubins_paths()
    plot_bezier_curve()
    plot_point_cloud_registration()
    plot_configuration_space()
    
    print("=" * 60)
    print("所有图形生成完成！")
