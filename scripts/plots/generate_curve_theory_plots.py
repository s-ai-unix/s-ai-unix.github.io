#!/usr/bin/env python3
"""
生成微分几何曲线论相关的 Plotly 图形，输出为 PNG 图片
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# 设置输出目录
OUTPUT_DIR = 'static/images/math'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure_as_png(fig, filename, width=800, height=600, scale=2):
    """保存 Plotly 图形为 PNG 图片"""
    filepath = f'{OUTPUT_DIR}/{filename}'
    fig.write_image(filepath, width=width, height=height, scale=scale)
    print(f"✅ 生成: {filepath}")
    return filepath


def plot_parametric_curves():
    """各种参数曲线示例"""
    t = np.linspace(0, 2*np.pi, 200)
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '直线: r(t)=(t,2t)',
            '圆: r(t)=(cos t,sin t)',
            '椭圆: r(t)=(2cos t,sin t)',
            '抛物线: r(t)=(t,t²)',
            '双曲线',
            '摆线'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # 直线
    t_line = np.linspace(-2, 2, 100)
    fig.add_trace(go.Scatter(x=t_line, y=2*t_line, mode='lines', 
                             line=dict(color='#007AFF', width=2), showlegend=False), row=1, col=1)
    
    # 圆
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode='lines',
                             line=dict(color='#34C759', width=2), showlegend=False), row=1, col=2)
    
    # 椭圆
    fig.add_trace(go.Scatter(x=2*np.cos(t), y=np.sin(t), mode='lines',
                             line=dict(color='#FF9500', width=2), showlegend=False), row=1, col=3)
    
    # 抛物线
    t_para = np.linspace(-2, 2, 100)
    fig.add_trace(go.Scatter(x=t_para, y=t_para**2, mode='lines',
                             line=dict(color='#AF52DE', width=2), showlegend=False), row=2, col=1)
    
    # 双曲线
    t_hyp = np.linspace(-2, 2, 100)
    x_hyp, y_hyp = np.cosh(t_hyp), np.sinh(t_hyp)
    fig.add_trace(go.Scatter(x=x_hyp, y=y_hyp, mode='lines',
                             line=dict(color='#FF3B30', width=2), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=-x_hyp, y=-y_hyp, mode='lines',
                             line=dict(color='#FF3B30', width=2), showlegend=False), row=2, col=2)
    
    # 摆线
    x_cyc, y_cyc = t - np.sin(t), 1 - np.cos(t)
    fig.add_trace(go.Scatter(x=x_cyc, y=y_cyc, mode='lines',
                             line=dict(color='#5AC8FA', width=2), showlegend=False), row=2, col=3)
    
    fig.update_layout(
        title=dict(text='各种参数曲线示例', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=10),
        width=900,
        height=600,
        margin=dict(l=50, r=40, t=80, b=50)
    )
    
    for i in range(1, 3):
        for j in range(1, 4):
            fig.update_xaxes(zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0', row=i, col=j)
            fig.update_yaxes(zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0', row=i, col=j, scaleanchor='x', scaleratio=1)
    
    save_figure_as_png(fig, 'curve-parametric-examples.png', width=900, height=600)


def plot_tangent_normal_vectors():
    """切向量和法向量"""
    t = np.linspace(0, 2*np.pi, 200)
    x_ellipse = 2 * np.cos(t)
    y_ellipse = np.sin(t)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode='lines',
                             line=dict(color='#007AFF', width=2), name='椭圆'))
    
    # 几个点上的切向量和法向量
    t_points = [0, np.pi/2, np.pi, 3*np.pi/2]
    colors = ['#FF3B30', '#FF9500', '#34C759', '#AF52DE']
    
    for t0, color in zip(t_points, colors):
        x0 = 2 * np.cos(t0)
        y0 = np.sin(t0)
        tx = -2 * np.sin(t0)
        ty = np.cos(t0)
        norm = np.sqrt(tx**2 + ty**2)
        tx, ty = tx/norm * 0.6, ty/norm * 0.6
        nx, ny = -ty, tx  # 法向量
        
        fig.add_trace(go.Scatter(x=[x0], y=[y0], mode='markers',
                                 marker=dict(color=color, size=8), showlegend=False))
        fig.add_trace(go.Scatter(x=[x0, x0+tx], y=[y0, y0+ty], mode='lines+markers',
                                 line=dict(color=color, width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[x0, x0+nx], y=[y0, y0+ny], mode='lines+markers',
                                 line=dict(color=color, width=2, dash='dash'), showlegend=False))
    
    fig.update_layout(
        title=dict(text='椭圆的切向量(实线)和法向量(虚线)', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white',
        width=700, height=550,
        margin=dict(l=60, r=40, t=70, b=50)
    )
    fig.update_xaxes(zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0')
    fig.update_yaxes(zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0', scaleanchor='x', scaleratio=1)
    
    save_figure_as_png(fig, 'curve-tangent-normal.png', width=700, height=550)


def plot_curvature_circle():
    """曲率圆"""
    t = np.linspace(0, 2*np.pi, 200)
    x_ellipse = 2 * np.cos(t)
    y_ellipse = np.sin(t)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode='lines',
                             line=dict(color='#007AFF', width=2)))
    
    t_points = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    colors = ['#FF3B30', '#FF9500', '#34C759', '#AF52DE']
    
    for t0, color in zip(t_points, colors):
        x0 = 2 * np.cos(t0)
        y0 = np.sin(t0)
        dx, dy = -2 * np.sin(t0), np.cos(t0)
        ddx, ddy = -2 * np.cos(t0), -np.sin(t0)
        
        curvature = abs(dx * ddy - ddx * dy) / (dx**2 + dy**2)**1.5
        radius = 1 / curvature
        
        tangent = np.array([dx, dy]) / np.sqrt(dx**2 + dy**2)
        normal = np.array([-tangent[1], tangent[0]])
        center = np.array([x0, y0]) + normal * radius
        
        theta = np.linspace(0, 2*np.pi, 50)
        fig.add_trace(go.Scatter(x=center[0] + radius * np.cos(theta),
                                 y=center[1] + radius * np.sin(theta),
                                 mode='lines', line=dict(color=color, width=1, dash='dash'), showlegend=False))
        fig.add_trace(go.Scatter(x=[x0], y=[y0], mode='markers',
                                 marker=dict(color=color, size=8), showlegend=False))
    
    fig.update_layout(
        title=dict(text='椭圆的曲率圆(密切圆)', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white', width=700, height=550,
        margin=dict(l=60, r=40, t=70, b=50)
    )
    fig.update_xaxes(zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0')
    fig.update_yaxes(zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0', scaleanchor='x', scaleratio=1)
    
    save_figure_as_png(fig, 'curve-curvature-circle.png', width=700, height=550)


def plot_frenet_frame():
    """Frenet标架"""
    t = np.linspace(0, 2*np.pi, 200)
    x_cyc = t - np.sin(t)
    y_cyc = 1 - np.cos(t)
    
    fig = go.Figure()
    mask = (t >= 0) & (t <= 2*np.pi)
    fig.add_trace(go.Scatter(x=x_cyc[mask], y=y_cyc[mask], mode='lines',
                             line=dict(color='#007AFF', width=2)))
    
    t_points = [np.pi/3, np.pi, 5*np.pi/3]
    colors = ['#FF3B30', '#34C759', '#FF9500']
    
    for t0, color in zip(t_points, colors):
        x0 = t0 - np.sin(t0)
        y0 = 1 - np.cos(t0)
        tx, ty = 1 - np.cos(t0), np.sin(t0)
        norm = np.sqrt(tx**2 + ty**2)
        tx, ty = tx/norm * 0.6, ty/norm * 0.6
        nx, ny = -ty, tx
        
        fig.add_trace(go.Scatter(x=[x0], y=[y0], mode='markers',
                                 marker=dict(color=color, size=8), showlegend=False))
        fig.add_trace(go.Scatter(x=[x0, x0+tx], y=[y0, y0+ty], mode='lines+markers',
                                 line=dict(color=color, width=3), showlegend=False))
        fig.add_trace(go.Scatter(x=[x0, x0+nx], y=[y0, y0+ny], mode='lines+markers',
                                 line=dict(color=color, width=3, dash='dash'), showlegend=False))
    
    fig.update_layout(
        title=dict(text='摆线的Frenet标架:切向量T(实线)和法向量N(虚线)', font=dict(size=14, color='#1d1d1f')),
        template='plotly_white', width=800, height=450,
        margin=dict(l=60, r=40, t=70, b=50)
    )
    fig.update_xaxes(zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0')
    fig.update_yaxes(zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0')
    
    save_figure_as_png(fig, 'curve-frenet-frame.png', width=800, height=450)


def plot_curvature_torsion():
    """曲率和挠率"""
    t = np.linspace(0, 4*np.pi, 200)
    a, b = 1, 0.3
    curvature = a / (a**2 + b**2) * np.ones_like(t)
    torsion = b / (a**2 + b**2) * np.ones_like(t)
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=('曲率 κ(t)', '挠率 τ(t)'), vertical_spacing=0.15)
    
    fig.add_trace(go.Scatter(x=t, y=curvature, mode='lines',
                             line=dict(color='#007AFF', width=2), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=torsion, mode='lines',
                             line=dict(color='#34C759', width=2), showlegend=False), row=2, col=1)
    
    fig.update_layout(
        title=dict(text='圆柱螺旋线的曲率和挠率(均为常数)', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white', width=700, height=500,
        margin=dict(l=60, r=40, t=70, b=50)
    )
    fig.update_xaxes(title_text='参数 t', gridcolor='#f0f0f0', row=1, col=1)
    fig.update_xaxes(title_text='参数 t', gridcolor='#f0f0f0', row=2, col=1)
    fig.update_yaxes(title_text='曲率 κ', gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(title_text='挠率 τ', gridcolor='#f0f0f0', row=2, col=1)
    
    save_figure_as_png(fig, 'curve-curvature-torsion.png', width=700, height=500)


def plot_helix_3d():
    """3D螺旋线"""
    t = np.linspace(0, 4*np.pi, 150)
    x = np.cos(t)
    y = np.sin(t)
    z = 0.3 * t
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                line=dict(color='#007AFF', width=4), name='螺旋线'))
    fig.add_trace(go.Scatter3d(x=x, y=y, z=np.zeros_like(z), mode='lines',
                                line=dict(color='#007AFF', width=2, dash='dash'),
                                opacity=0.5, name='投影'))
    
    # 切向量
    t0_idx = 75
    x0, y0, z0 = x[t0_idx], y[t0_idx], z[t0_idx]
    tx, ty, tz = -np.sin(t[t0_idx]), np.cos(t[t0_idx]), 0.3
    norm = np.sqrt(tx**2 + ty**2 + tz**2)
    tx, ty, tz = tx/norm * 0.4, ty/norm * 0.4, tz/norm * 0.4
    
    fig.add_trace(go.Scatter3d(x=[x0, x0+tx], y=[y0, y0+ty], z=[z0, z0+tz],
                                mode='lines+markers', line=dict(color='#FF3B30', width=4),
                                marker=dict(size=4), name='切向量'))
    
    fig.update_layout(
        title=dict(text='圆柱螺旋线及其切向量', font=dict(size=16, color='#1d1d1f')),
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='cube'),
        template='plotly_white', width=700, height=550,
        margin=dict(l=60, r=40, t=70, b=50)
    )
    
    save_figure_as_png(fig, 'curve-helix-3d.png', width=700, height=550)


if __name__ == '__main__':
    print("开始生成曲线论图形...")
    plot_parametric_curves()
    plot_tangent_normal_vectors()
    plot_curvature_circle()
    plot_frenet_frame()
    plot_curvature_torsion()
    plot_helix_3d()
    print("\n所有图形生成完成！")
