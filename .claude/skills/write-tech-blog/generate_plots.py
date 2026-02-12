#!/usr/bin/env python3
"""
Plotly 图形生成工具
用于技术博客文章中的数学、物理图形
"""

import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import numpy as np
import math
from pathlib import Path

# 确保输出目录存在
PLOT_DIR = Path("static/images/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 苹果风格配色
APPLE_BLUE = "#007AFF"
APPLE_GREEN = "#34C759"
APPLE_ORANGE = "#FF9500"
APPLE_GRAY = "#8E8E93"
APPLE_WHITE = "#FFFFFF"

def plot_ricci_flow_examples():
    """生成 Ricci Flow 相关图形"""
    print("生成 Ricci Flow 图形...")

    # 1. 球面 Ricci Flow 演化
    fig1 = go.Figure()
    t = np.linspace(0, 2, 100)

    # 不同维度的球面
    for n in [2, 3, 4]:
        radius = np.exp(-2 * (n-1) * t)
        fig1.add_trace(go.Scatter(
            x=t, y=radius,
            mode='lines',
            name=f'{n}维球面',
            line=dict(color=APPLE_BLUE if n==2 else APPLE_GREEN if n==3 else APPLE_ORANGE,
                     width=3)
        ))

    fig1.update_layout(
        title='Ricci Flow 下球面半��演化',
        xaxis_title='时间 t',
        yaxis_title='半径 R(t)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    fig1.write_html(PLOT_DIR / "ricci-flow-evolution.html")

    # 2. Ricci 曲率热方程
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    # 初始曲率分布（高斯）
    R0 = np.exp(-(X**2 + Y**2))

    fig2 = go.Figure(data=[
        go.Surface(
            z=R0,
            colorscale='Blues',
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
            )
        )
    ])

    fig2.update_layout(
        title='二维流形初始曲率分布',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='曲率 R'
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14)
    )
    fig2.write_html(PLOT_DIR / "ricci-curvature-initial.html")

    return fig1, fig2

def plot_geodesics_and_curvature():
    """生成测地线和曲率图形"""
    print("生成测地线和曲率图形...")

    # 1. 球面上的测地线
    fig1 = go.Figure()

    # 绘制球面
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig1.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Greys',
        opacity=0.3,
        showscale=False
    ))

    # 添加测地线
    for theta in np.linspace(0, np.pi, 5):
        geodesic_x = np.cos(theta) * np.linspace(-1, 1, 50)
        geodesic_y = np.sin(theta) * np.linspace(-1, 1, 50)
        geodesic_z = np.zeros_like(geodesic_x)

        # 转换到球面坐标
        for i in range(len(geodesic_x)):
            r = np.sqrt(geodesic_x[i]**2 + geodesic_y[i]**2)
            if r > 0:
                phi = np.arctan2(geodesic_y[i], geodesic_x[i])
                theta_geo = np.arcsin(r)
                geodesic_x[i] = np.cos(phi) * np.sin(theta_geo)
                geodesic_y[i] = np.sin(phi) * np.sin(theta_geo)
                geodesic_z[i] = np.cos(theta_geo)

        fig1.add_trace(go.Scatter3d(
            x=geodesic_x, y=geodesic_y, z=geodesic_z,
            mode='lines',
            line=dict(color=APPLE_BLUE, width=4),
            name=f'测地线 θ={theta:.1f}'
        ))

    fig1.update_layout(
        title='球面上的测地线',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14)
    )
    fig1.write_html(PLOT_DIR / "geodesics-sphere.html")

    # 2. 高斯曲率可视化
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # 双曲面马鞍形
    Z = X**2 - Y**2
    K = -4 / (1 + 4*X**2 + 4*Y**2)**2  # 高斯曲率

    fig2 = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=K,
            colorscale='RdBu_r',
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
            )
        )
    ])

    fig2.update_layout(
        title='双曲面及其高斯曲率分布',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z = x² - y²'
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        coloraxis=dict(colorbar=dict(title='高斯曲率 K'))
    )
    fig2.write_html(PLOT_DIR / "gaussian-curvature.html")

    return fig1, fig2

def plot_einstein_field_equations():
    """生成爱因斯坦场方程相关图形"""
    print("生成爱因斯坦场方程图形...")

    # 时空曲率图示
    fig = go.Figure()

    # 绘制质量周围的时空弯曲（简化模型）
    r = np.linspace(1, 5, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    R, Theta = np.meshgrid(r, theta)

    # 史瓦西度规的曲率影响
    curvature = 1 - 1/R

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = curvature

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )
    ))

    # 添加中心质量
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color=APPLE_ORANGE),
        name='中心质量'
    ))

    fig.update_layout(
        title='爱因斯坦场方程：时空弯曲示意',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='时空曲率'
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14)
    )
    fig.write_html(PLOT_DIR / "einstein-field-equations.html")

    return fig

def plot_heat_equation_comparison():
    """生成热方程与 Ricci Flow 的对比图"""
    print("生成热方程对比图形...")

    # 创建子图
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=('热方程：∂u/∂t = Δu', 'Ricci Flow：∂g/∂t = -2Ric'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )

    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    # 热方程解
    t_heat = 0.5
    u_heat = np.exp(-(X**2 + Y**2) / (4*t_heat)) / (4*np.pi*t_heat)

    fig.add_trace(go.Surface(
        z=u_heat,
        colorscale='Blues',
        showscale=False
    ), row=1, col=1)

    # Ricci Flow "熨平"效果
    initial_R = np.exp(-(X**2 + Y**2))
    t_ricci = 0.5
    R_ricci = initial_R * np.exp(-2*t_ricci)

    fig.add_trace(go.Surface(
        z=R_ricci,
        colorscale='Reds',
        showscale=False
    ), row=1, col=2)

    fig.update_layout(
        title='热方程与 Ricci Flow 的相似性',
        font=dict(family='Arial, sans-serif', size=14),
        template='plotly_white'
    )

    fig.update_scenes(
        xaxis_title='x', yaxis_title='y',
        zaxis_title='值',
        row=1, col=1
    )
    fig.update_scenes(
        xaxis_title='x', yaxis_title='y',
        zaxis_title='曲率',
        row=1, col=2
    )

    fig.write_html(PLOT_DIR / "heat-equation-comparison.html")

    return fig

def main():
    """主函数：生成所有图形"""
    print("开始生成技术博客所需图形...")

    # 生成所有图形
    plot_ricci_flow_examples()
    plot_geodesics_and_curvature()
    plot_einstein_field_equations()
    plot_heat_equation_comparison()

    print(f"\n所有图形已生成在 {PLOT_DIR} 目录")
    print("在文章中引用格式：")
    print('<div class="plot-container">')
    print('  <iframe src="/images/plots/图形文件名.html" width="100%" height="500" frameborder="0"></iframe>')
    print('</div>')

if __name__ == "__main__":
    main()