#!/usr/bin/env python3
"""
生成曲线积分与曲面积分文章所需的Plotly图形
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import subprocess
import os

def save_and_compress(fig, filepath, width=800, height=600):
    """保存并压缩图片"""
    fig.write_image(filepath, scale=2, width=width, height=height)
    
    # 压缩PNG
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force',
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")

# 设置输出目录
output_dir = 'static/images/plots'
os.makedirs(output_dir, exist_ok=True)

# 苹果风格配色
APPLE_BLUE = '#007AFF'
APPLE_GREEN = '#34C759'
APPLE_ORANGE = '#FF9500'
APPLE_RED = '#FF3B30'
APPLE_PURPLE = '#AF52DE'
APPLE_GRAY = '#8E8E93'

def plot_line_integral_type1():
    """图1：第一类曲线积分示意图"""
    fig = go.Figure()
    
    # 绘制曲线（抛物线）
    t = np.linspace(0, 2, 100)
    x = t
    y = t**2 / 2
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color=APPLE_BLUE, width=3),
        name='曲线C',
        showlegend=False
    ))
    
    # 在曲线上添加若干点表示分割
    n_points = 6
    t_points = np.linspace(0.2, 1.8, n_points)
    x_points = t_points
    y_points = t_points**2 / 2
    
    fig.add_trace(go.Scatter(
        x=x_points, y=y_points,
        mode='markers',
        marker=dict(size=10, color=APPLE_ORANGE, symbol='circle'),
        showlegend=False
    ))
    
    # 添加弧长微元示意（在第二个点上）
    i = 1
    dx = 0.3
    dy = ((t_points[i]+0.15)**2 - t_points[i]**2) / 2
    fig.add_annotation(x=x_points[i]+dx/2, y=y_points[i]+dy/2, ax=x_points[i], ay=y_points[i],
                      xref='x', yref='y', axref='x', ayref='y',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                      arrowcolor=APPLE_GREEN)
    
    # 标注
    fig.add_trace(go.Scatter(x=[x_points[1]+0.2], y=[y_points[1]+0.3], mode='text',
                            text=['$ds$'], textfont=dict(size=14, color=APPLE_GREEN),
                            showlegend=False))
    fig.add_trace(go.Scatter(x=[x_points[1]+0.15], y=[y_points[1]-0.15], mode='text',
                            text=['$f(\\xi_i, \\eta_i)$'], textfont=dict(size=11, color=APPLE_ORANGE),
                            showlegend=False))
    
    # 添加公式标注
    fig.add_trace(go.Scatter(x=[1], y=[-0.3], mode='text',
                            text=['$\\int_C f(x,y) \\, ds$'],
                            textfont=dict(size=14, color='black'),
                            showlegend=False))
    
    fig.update_xaxes(range=[-0.3, 2.5], showgrid=True, gridcolor='lightgray', zeroline=True,
                     zerolinecolor='black', zerolinewidth=1, title='x')
    fig.update_yaxes(range=[-0.5, 2.5], showgrid=True, gridcolor='lightgray', zeroline=True,
                     zerolinecolor='black', zerolinewidth=1, title='y')
    
    fig.update_layout(
        title=dict(text='第一类曲线积分（对弧长）', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=800, height=600,
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    save_and_compress(fig, f'{output_dir}/line_integral_type1.png', 800, 600)

def plot_line_integral_type2():
    """图2：第二类曲线积分示意图（力场做功）"""
    fig = go.Figure()
    
    # 绘制曲线（螺旋线的一部分）
    t = np.linspace(0, 2*np.pi, 100)
    x = 0.5 + 0.4 * np.cos(t)
    y = 0.5 + 0.4 * np.sin(t)
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color=APPLE_BLUE, width=3),
        showlegend=False
    ))
    
    # 在几个点上绘制力向量
    n_arrows = 5
    t_arrows = np.linspace(0, 2*np.pi, n_arrows, endpoint=False)
    
    for t_i in t_arrows:
        x_i = 0.5 + 0.4 * np.cos(t_i)
        y_i = 0.5 + 0.4 * np.sin(t_i)
        
        # 力向量（指向中心但略有偏移）
        fx = -0.25 * np.cos(t_i) - 0.15 * np.sin(t_i)
        fy = -0.25 * np.sin(t_i) + 0.15 * np.cos(t_i)
        
        fig.add_annotation(x=x_i+fx, y=y_i+fy, ax=x_i, ay=y_i,
                          xref='x', yref='y', axref='x', ayref='y',
                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                          arrowcolor=APPLE_RED)
    
    # 绘制切向微元dr
    t_mid = np.pi
    x_mid = 0.5 + 0.4 * np.cos(t_mid)
    y_mid = 0.5 + 0.4 * np.sin(t_mid)
    dx = -0.15 * np.sin(t_mid)
    dy = 0.15 * np.cos(t_mid)
    
    fig.add_annotation(x=x_mid+dx, y=y_mid+dy, ax=x_mid, ay=y_mid,
                      xref='x', yref='y', axref='x', ayref='y',
                      showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5,
                      arrowcolor=APPLE_GREEN)
    
    # 标注
    fig.add_trace(go.Scatter(x=[x_mid+dx+0.08], y=[y_mid+dy+0.08], mode='text',
                            text=['$d\\mathbf{r}$'], textfont=dict(size=14, color=APPLE_GREEN),
                            showlegend=False))
    fig.add_trace(go.Scatter(x=[0.15], y=[0.9], mode='text',
                            text=['$\\mathbf{F}$'], textfont=dict(size=14, color=APPLE_RED),
                            showlegend=False))
    
    # 添加公式标注
    fig.add_trace(go.Scatter(x=[0.5], y=[-0.05], mode='text',
                            text=['$W = \\int_C \\mathbf{F} \\cdot d\\mathbf{r} = \\int_C P \\, dx + Q \\, dy$'],
                            textfont=dict(size=13, color='black'),
                            showlegend=False))
    
    fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor='lightgray', zeroline=False, title='x')
    fig.update_yaxes(range=[0, 1], showgrid=True, gridcolor='lightgray', zeroline=False, title='y')
    
    fig.update_layout(
        title=dict(text='第二类曲线积分（对坐标/做功）', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=800, height=650,
        margin=dict(l=60, r=40, t=80, b=80)
    )
    
    save_and_compress(fig, f'{output_dir}/line_integral_type2.png', 800, 650)

def plot_green_formula():
    """图3：Green公式示意图"""
    fig = go.Figure()
    
    # 绘制区域D（椭圆）
    theta = np.linspace(0, 2*np.pi, 100)
    a, b = 0.4, 0.3
    x_center, y_center = 0.5, 0.5
    x_ellipse = x_center + a * np.cos(theta)
    y_ellipse = y_center + b * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_ellipse, y=y_ellipse,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.15)',
        line=dict(color=APPLE_BLUE, width=2),
        name='区域D',
        showlegend=False
    ))
    
    # 绘制边界箭头（逆时针）
    n_arrows = 8
    for i in range(n_arrows):
        angle = 2 * np.pi * i / n_arrows
        x_b = x_center + a * np.cos(angle)
        y_b = y_center + b * np.sin(angle)
        
        # 切向方向（逆时针）
        dx = -0.06 * np.sin(angle)
        dy = 0.06 * np.cos(angle)
        
        fig.add_annotation(x=x_b+dx, y=y_b+dy, ax=x_b, ay=y_b,
                          xref='x', yref='y', axref='x', ayref='y',
                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                          arrowcolor=APPLE_ORANGE)
    
    # 标注区域和边界
    fig.add_trace(go.Scatter(x=[x_center], y=[y_center], mode='text',
                            text=['$D$'], textfont=dict(size=16, color=APPLE_BLUE),
                            showlegend=False))
    fig.add_trace(go.Scatter(x=[x_center+a+0.08], y=[y_center], mode='text',
                            text=['$C$'], textfont=dict(size=14, color=APPLE_ORANGE),
                            showlegend=False))
    
    # 添加公式
    fig.add_trace(go.Scatter(x=[0.5], y=[0.02], mode='text',
                            text=['$\\displaystyle \\oint_C P \\, dx + Q \\, dy = \\iint_D \\left(\\frac{\\partial Q}{\\partial x} - \\frac{\\partial P}{\\partial y}\\right) dx \\, dy$'],
                            textfont=dict(size=11, color='black'),
                            showlegend=False))
    
    fig.update_xaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='Green公式', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=800, height=700,
        margin=dict(l=40, r=40, t=80, b=100)
    )
    
    save_and_compress(fig, f'{output_dir}/green_formula.png', 800, 700)

def plot_surface_integral_type1():
    """图4：第一类曲面积分示意图"""
    fig = go.Figure()
    
    # 绘制曲面（抛物面的一部分）投影到2D
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, 0.5, 30)
    U, V = np.meshgrid(u, v)
    
    # 圆锥面投影示意
    R = 0.3 + V
    X = 0.5 + R * np.cos(U)
    Y = 0.5 + R * np.sin(U)
    
    # 绘制曲面边界（两个圆）
    theta = np.linspace(0, 2*np.pi, 100)
    r1, r2 = 0.3, 0.8
    x1 = 0.5 + r1 * np.cos(theta)
    y1 = 0.5 + r1 * np.sin(theta)
    x2 = 0.5 + r2 * np.cos(theta)
    y2 = 0.5 + r2 * np.sin(theta)
    
    # 填充环形区域表示曲面投影
    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.1)',
        line=dict(color=APPLE_BLUE, width=2),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=x1, y=y1,
        fill='toself',
        fillcolor='white',
        line=dict(color=APPLE_BLUE, width=2),
        showlegend=False
    ))
    
    # 添加网格线表示曲面分割
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        x_line = [0.5 + r1 * np.cos(angle), 0.5 + r2 * np.cos(angle)]
        y_line = [0.5 + r1 * np.sin(angle), 0.5 + r2 * np.sin(angle)]
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color=APPLE_GRAY, width=1, dash='dot'),
            showlegend=False
        ))
    
    # 添加同心圆
    for r in [0.4, 0.5, 0.6, 0.7]:
        x_c = 0.5 + r * np.cos(theta)
        y_c = 0.5 + r * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x_c, y=y_c,
            mode='lines',
            line=dict(color=APPLE_GRAY, width=1, dash='dot'),
            showlegend=False
        ))
    
    # 标注dS在一个小区域
    angle_mid = np.pi / 4
    r_mid = 0.55
    x_mid = 0.5 + r_mid * np.cos(angle_mid)
    y_mid = 0.5 + r_mid * np.sin(angle_mid)
    
    fig.add_trace(go.Scatter(
        x=[x_mid], y=[y_mid],
        mode='markers',
        marker=dict(size=12, color=APPLE_ORANGE, symbol='square'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(x=[x_mid+0.08], y=[y_mid+0.06], mode='text',
                            text=['$dS$'], textfont=dict(size=12, color=APPLE_ORANGE),
                            showlegend=False))
    
    # 添加公式
    fig.add_trace(go.Scatter(x=[0.5], y=[0.02], mode='text',
                            text=['$\\displaystyle \\iint_S f(x,y,z) \\, dS$'],
                            textfont=dict(size=14, color='black'),
                            showlegend=False))
    
    fig.update_xaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='第一类曲面积分（对面积）', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=700, height=700,
        margin=dict(l=40, r=40, t=80, b=80)
    )
    
    save_and_compress(fig, f'{output_dir}/surface_integral_type1.png', 700, 700)

def plot_surface_integral_type2():
    """图5：第二类曲面积分示意图（流量）"""
    fig = go.Figure()
    
    # 绘制曲面（半球面投影为圆）
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.35
    x_center, y_center = 0.5, 0.5
    x_circle = x_center + r * np.cos(theta)
    y_circle = y_center + r * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.15)',
        line=dict(color=APPLE_BLUE, width=2),
        showlegend=False
    ))
    
    # 绘制法向量（向上）
    n_arrows = 6
    for i in range(n_arrows):
        angle = 2 * np.pi * i / n_arrows
        x_b = x_center + 0.25 * np.cos(angle)
        y_b = y_center + 0.25 * np.sin(angle)
        
        # 法向量（径向向外）
        nx = 0.08 * np.cos(angle)
        ny = 0.08 * np.sin(angle)
        
        fig.add_annotation(x=x_b+nx, y=y_b+ny, ax=x_b, ay=y_b,
                          xref='x', yref='y', axref='x', ayref='y',
                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                          arrowcolor=APPLE_GREEN)
    
    # 绘制流速向量（从左上流向右下）
    for i in range(n_arrows):
        angle = 2 * np.pi * i / n_arrows + np.pi/n_arrows
        x_b = x_center + 0.25 * np.cos(angle)
        y_b = y_center + 0.25 * np.sin(angle)
        
        # 流速（统一方向）
        vx = 0.1
        vy = -0.05
        
        fig.add_annotation(x=x_b+vx, y=b+vy, ax=x_b, ay=y_b,
                          xref='x', yref='y', axref='x', ayref='y',
                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                          arrowcolor=APPLE_RED)
    
    # 标注
    fig.add_trace(go.Scatter(x=[x_center-0.15], y=[y_center+0.2], mode='text',
                            text=['$\\mathbf{n}$'], textfont=dict(size=13, color=APPLE_GREEN),
                            showlegend=False))
    fig.add_trace(go.Scatter(x=[x_center+0.25], y=[y_center-0.2], mode='text',
                            text=['$\\mathbf{v}$'], textfont=dict(size=13, color=APPLE_RED),
                            showlegend=False))
    
    # 添加公式
    fig.add_trace(go.Scatter(x=[0.5], y=[0.02], mode='text',
                            text=['$\\Phi = \\displaystyle \\iint_S \\mathbf{v} \\cdot \\mathbf{n} \\, dS = \\iint_S P \\, dy \\, dz + Q \\, dz \\, dx + R \\, dx \\, dy$'],
                            textfont=dict(size=10, color='black'),
                            showlegend=False))
    
    fig.update_xaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='第二类曲面积分（对坐标/流量）', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=900, height=700,
        margin=dict(l=40, r=40, t=80, b=100)
    )
    
    save_and_compress(fig, f'{output_dir}/surface_integral_type2.png', 900, 700)

def plot_integral_theorems():
    """图6：三大积分公式关系图"""
    fig = make_subplots(1, 3, 
        subplot_titles=('Green公式', 'Gauss公式', 'Stokes公式'),
        horizontal_spacing=0.1)
    
    # Green公式（左）
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.35
    x_c, y_c = 0.5, 0.5
    x_ellipse = x_c + r * np.cos(theta)
    y_ellipse = y_c + 0.7*r * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_ellipse, y=y_ellipse,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.15)',
        line=dict(color=APPLE_BLUE, width=2),
        showlegend=False
    ), row=1, col=1)
    
    # 边界箭头
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x_b = x_c + r * np.cos(angle)
        y_b = y_c + 0.7*r * np.sin(angle)
        dx = -0.05 * np.sin(angle)
        dy = 0.05 * 0.7 * np.cos(angle)
        fig.add_annotation(x=x_b+dx, y=y_b+dy, ax=x_b, ay=y_b,
                          xref='x', yref='y', axref='x', ayref='y',
                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                          arrowcolor=APPLE_ORANGE, row=1, col=1)
    
    fig.add_trace(go.Scatter(x=[0.5], y=[0.15], mode='text',
                            text=['2D区域'],
                            textfont=dict(size=10, color=APPLE_BLUE), showlegend=False), row=1, col=1)
    
    fig.update_xaxes(range=[0, 1], row=1, col=1, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 1], row=1, col=1, showgrid=False, zeroline=False, showticklabels=False)
    
    # Gauss公式（中）- 球体剖面
    x_sphere_outer = 0.5 + 0.4 * np.cos(theta)
    y_sphere_outer = 0.5 + 0.4 * np.sin(theta)
    x_sphere_inner = 0.5 + 0.15 * np.cos(theta)
    y_sphere_inner = 0.5 + 0.15 * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_sphere_outer, y=y_sphere_outer,
        fill='toself',
        fillcolor='rgba(52, 199, 89, 0.15)',
        line=dict(color=APPLE_GREEN, width=2),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=x_sphere_inner, y=y_sphere_inner,
        fill='toself',
        fillcolor='white',
        line=dict(color=APPLE_GREEN, width=1, dash='dot'),
        showlegend=False
    ), row=1, col=2)
    
    # 外法向箭头
    for i in range(4):
        angle = 2 * np.pi * i / 4
        x_b = 0.5 + 0.4 * np.cos(angle)
        y_b = 0.5 + 0.4 * np.sin(angle)
        nx = 0.08 * np.cos(angle)
        ny = 0.08 * np.sin(angle)
        fig.add_annotation(x=x_b+nx, y=y_b+ny, ax=x_b, ay=y_b,
                          xref='x2', yref='y2', axref='x2', ayref='y2',
                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                          arrowcolor=APPLE_ORANGE, row=1, col=2)
    
    fig.add_trace(go.Scatter(x=[0.5], y=[0.15], mode='text',
                            text=['3D体积'],
                            textfont=dict(size=10, color=APPLE_GREEN), showlegend=False), row=1, col=2)
    
    fig.update_xaxes(range=[0, 1], row=1, col=2, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 1], row=1, col=2, showgrid=False, zeroline=False, showticklabels=False)
    
    # Stokes公式（右）- 曲面带边界
    # 绘制曲面形状（类似扭曲的带子）
    t_surf = np.linspace(0, 2*np.pi, 100)
    x_surf = 0.5 + 0.35 * np.cos(t_surf)
    y_surf = 0.5 + 0.2 * np.sin(2*t_surf)
    
    fig.add_trace(go.Scatter(
        x=x_surf, y=y_surf,
        fill='toself',
        fillcolor='rgba(175, 82, 222, 0.15)',
        line=dict(color=APPLE_PURPLE, width=2),
        showlegend=False
    ), row=1, col=3)
    
    # 边界曲线（两个椭圆表示）
    x_bound1 = 0.5 + 0.1 * np.cos(theta)
    y_bound1 = 0.65 + 0.05 * np.sin(theta)
    x_bound2 = 0.5 + 0.1 * np.cos(theta)
    y_bound2 = 0.35 + 0.05 * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_bound1, y=y_bound1,
        mode='lines',
        line=dict(color=APPLE_ORANGE, width=2),
        showlegend=False
    ), row=1, col=3)
    
    fig.add_trace(go.Scatter(
        x=x_bound2, y=y_bound2,
        mode='lines',
        line=dict(color=APPLE_ORANGE, width=2),
        showlegend=False
    ), row=1, col=3)
    
    # 边界箭头
    fig.add_annotation(x=0.6, y=0.65, ax=0.6, ay=0.72,
                      xref='x3', yref='y3', axref='x3', ayref='y3',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                      arrowcolor=APPLE_ORANGE, row=1, col=3)
    fig.add_annotation(x=0.6, y=0.35, ax=0.6, ay=0.28,
                      xref='x3', yref='y3', axref='x3', ayref='y3',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                      arrowcolor=APPLE_ORANGE, row=1, col=3)
    
    fig.add_trace(go.Scatter(x=[0.5], y=[0.15], mode='text',
                            text=['曲面+边界'],
                            textfont=dict(size=10, color=APPLE_PURPLE), showlegend=False), row=1, col=3)
    
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='三大积分公式：Green、Gauss、Stokes', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000, height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    save_and_compress(fig, f'{output_dir}/integral_theorems.png', 1000, 450)

def plot_integral_evolution():
    """图7：四种积分的演化关系"""
    fig = go.Figure()
    
    # 绘制2x2网格布局
    positions = {
        '定积分': (0.25, 0.75),
        '第一类曲线': (0.75, 0.75),
        '第一类曲面': (0.25, 0.25),
        '第二类曲线': (0.75, 0.25),
    }
    
    colors = {
        '定积分': APPLE_BLUE,
        '第一类曲线': APPLE_GREEN,
        '第一类曲面': APPLE_ORANGE,
        '第二类曲线': APPLE_PURPLE,
    }
    
    # 绘制节点
    for name, (x, y) in positions.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=70, color=colors[name], line=dict(color='white', width=2)),
            text=[name],
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial'),
            showlegend=False
        ))
    
    # 添加第二类曲面积分（右上角上方）
    fig.add_trace(go.Scatter(
        x=[0.75], y=[0.9],
        mode='markers+text',
        marker=dict(size=70, color=APPLE_RED, line=dict(color='white', width=2)),
        text=['第二类曲面'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial'),
        showlegend=False
    ))
    
    # 绘制箭头连接
    # 定积分 -> 第一类曲线
    fig.add_annotation(x=0.55, y=0.75, ax=0.4, ay=0.75,
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                      arrowcolor='gray')
    fig.add_trace(go.Scatter(x=[0.475], y=[0.82], mode='text',
                            text=['曲线化'],
                            textfont=dict(size=9, color='gray'), showlegend=False))
    
    # 定积分 -> 第一类曲面
    fig.add_annotation(x=0.25, y=0.45, ax=0.25, ay=0.6,
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                      arrowcolor='gray')
    fig.add_trace(go.Scatter(x=[0.32], y=[0.525], mode='text',
                            text=['曲面化'],
                            textfont=dict(size=9, color='gray'), showlegend=False))
    
    # 第一类曲线 -> 第二类曲线
    fig.add_annotation(x=0.75, y=0.55, ax=0.75, ay=0.65,
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                      arrowcolor='gray')
    fig.add_trace(go.Scatter(x=[0.82], y=[0.6], mode='text',
                            text=['向量化'],
                            textfont=dict(size=9, color='gray'), showlegend=False))
    
    # 第一类曲面 -> 第二类曲面
    fig.add_annotation(x=0.75, y=0.35, ax=0.75, ay=0.4,
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                      arrowcolor='gray')
    
    # 第二类曲线 -> 第二类曲面
    fig.add_annotation(x=0.75, y=0.82, ax=0.75, ay=0.72,
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                      arrowcolor='gray')
    
    fig.update_xaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='四种积分的演化关系', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=900, height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    save_and_compress(fig, f'{output_dir}/integral_evolution.png', 900, 600)

if __name__ == '__main__':
    print("🎨 开始生成曲线积分与曲面积分文章图形...")
    
    print("\n1. 生成第一类曲线积分示意图...")
    plot_line_integral_type1()
    
    print("\n2. 生成第二类曲线积分示意图...")
    plot_line_integral_type2()
    
    print("\n3. 生成Green公式示意图...")
    plot_green_formula()
    
    print("\n4. 生成第一类曲面积分示意图...")
    plot_surface_integral_type1()
    
    print("\n5. 生成第二类曲面积分示意图...")
    plot_surface_integral_type2()
    
    print("\n6. 生成三大公式关系图...")
    plot_integral_theorems()
    
    print("\n7. 生成四种积分演化关系图...")
    plot_integral_evolution()
    
    print("\n✅ 所有图形生成完成！")
