#!/usr/bin/env python3
"""
生成Poincaré体积元定向文章所需的Plotly图形
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

def plot_orientation_comparison():
    """图1：体积元定向对比"""
    fig = make_subplots(1, 2, subplot_titles=('标准定向 ($dx \\wedge dy$)', '反转定向 ($dy \\wedge dx = -dx \\wedge dy$)'))
    
    # 左图：标准定向
    # 绘制一个平行四边形，表示dx ^ dy
    x1 = [0, 1, 1.3, 0.3, 0]
    y1 = [0, 0, 1, 1, 0]
    
    fig.add_trace(go.Scatter(
        x=x1, y=y1,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.3)',
        line=dict(color=APPLE_BLUE, width=2),
        name='标准定向',
        showlegend=False
    ), row=1, col=1)
    
    # 添加向量箭头
    fig.add_annotation(x=1, y=0, ax=0, ay=0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                       arrowcolor=APPLE_BLUE, row=1, col=1)
    fig.add_annotation(x=0.3, y=1, ax=0, ay=0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                       arrowcolor=APPLE_GREEN, row=1, col=1)
    
    # 添加文字标注
    fig.add_trace(go.Scatter(x=[0.5], y=[-0.15], mode='text',
                             text=['$dx$'], textfont=dict(size=16, color=APPLE_BLUE),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[-0.15], y=[0.5], mode='text',
                             text=['$dy$'], textfont=dict(size=16, color=APPLE_GREEN),
                             showlegend=False), row=1, col=1)
    
    # 添加定向标记（逆时针箭头）
    fig.add_annotation(x=0.65, y=0.5, text='↺',
                       showarrow=False, font=dict(size=30, color=APPLE_BLUE),
                       row=1, col=1)
    
    # 右图：反转定向
    # 同样的平行四边形，但标注不同
    x2 = [0, 1, 1.3, 0.3, 0]
    y2 = [0, 0, 1, 1, 0]
    
    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        fill='toself',
        fillcolor='rgba(255, 59, 48, 0.3)',
        line=dict(color=APPLE_RED, width=2),
        name='反转定向',
        showlegend=False
    ), row=1, col=2)
    
    # 交换箭头方向
    fig.add_annotation(x=0.3, y=1, ax=0, ay=0,
                       xref='x2', yref='y2', axref='x2', ayref='y2',
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                       arrowcolor=APPLE_GREEN, row=1, col=2)
    fig.add_annotation(x=1, y=0, ax=0, ay=0,
                       xref='x2', yref='y2', axref='x2', ayref='y2',
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                       arrowcolor=APPLE_BLUE, row=1, col=2)
    
    fig.add_trace(go.Scatter(x=[-0.15], y=[0.5], mode='text',
                             text=['$dx$'], textfont=dict(size=16, color=APPLE_BLUE),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0.5], y=[-0.15], mode='text',
                             text=['$dy$'], textfont=dict(size=16, color=APPLE_GREEN),
                             showlegend=False), row=1, col=2)
    
    # 顺时针箭头
    fig.add_annotation(x=0.65, y=0.5, text='↻',
                       showarrow=False, font=dict(size=30, color=APPLE_RED),
                       row=1, col=2)
    
    # 添加符号标注
    fig.add_trace(go.Scatter(x=[0.65], y=[1.3], mode='text',
                             text=['$dx \\wedge dy > 0$'], 
                             textfont=dict(size=14, color=APPLE_BLUE),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0.65], y=[1.3], mode='text',
                             text=['$dy \\wedge dx = -dx \\wedge dy < 0$'], 
                             textfont=dict(size=14, color=APPLE_RED),
                             showlegend=False), row=1, col=2)
    
    fig.update_xaxes(range=[-0.5, 1.8], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-0.5, 1.8], showgrid=False, zeroline=False, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='2D平面上的定向', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=900, height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    save_and_compress(fig, f'{output_dir}/poincare_orientation_comparison.png', 900, 450)

def plot_coordinate_transform():
    """图2：坐标变换对比"""
    fig = make_subplots(1, 2, 
        subplot_titles=('定向保持 ($J > 0$)', '定向反转 ($J < 0$)'))
    
    # 左图：定向保持（旋转）
    theta = np.pi/6
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # 原始正方形
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    
    # 绘制原始正方形（虚线）
    fig.add_trace(go.Scatter(
        x=square[0], y=square[1],
        mode='lines', line=dict(color='gray', dash='dot', width=1.5),
        name='原坐标',
        showlegend=False
    ), row=1, col=1)
    
    # 绘制变换后的正方形
    transformed = R @ square
    fig.add_trace(go.Scatter(
        x=transformed[0], y=transformed[1],
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.3)',
        line=dict(color=APPLE_BLUE, width=2),
        name='变换后',
        showlegend=False
    ), row=1, col=1)
    
    # 添加箭头表示变换
    fig.add_annotation(x=0.87, y=0.5, ax=0.5, ay=0.5,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
                       arrowcolor=APPLE_ORANGE, row=1, col=1)
    
    fig.add_trace(go.Scatter(x=[0.7], y=[0.65], mode='text',
                             text=['$J = \\cos\\theta > 0$'], 
                             textfont=dict(size=12, color=APPLE_BLUE),
                             showlegend=False), row=1, col=1)
    
    # 右图：定向反转（反射）
    # 关于y轴反射
    Refl = np.array([[-1, 0], [0, 1]])
    
    # 绘制原始正方形（虚线）
    fig.add_trace(go.Scatter(
        x=square[0], y=square[1],
        mode='lines', line=dict(color='gray', dash='dot', width=1.5),
        name='原坐标',
        showlegend=False
    ), row=1, col=2)
    
    # 绘制变换后的正方形
    transformed_refl = Refl @ square
    fig.add_trace(go.Scatter(
        x=transformed_refl[0], y=transformed_refl[1],
        fill='toself',
        fillcolor='rgba(255, 59, 48, 0.3)',
        line=dict(color=APPLE_RED, width=2),
        name='变换后',
        showlegend=False
    ), row=1, col=2)
    
    # 添加翻转符号
    fig.add_trace(go.Scatter(x=[-0.5], y=[0.5], mode='text',
                             text=['↔'], 
                             textfont=dict(size=40, color=APPLE_ORANGE),
                             showlegend=False), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=[-0.5], y=[1.3], mode='text',
                             text=['$J = -1 < 0$'], 
                             textfont=dict(size=12, color=APPLE_RED),
                             showlegend=False), row=1, col=2)
    
    fig.update_xaxes(range=[-1.5, 1.5], showgrid=False, zeroline=True, 
                     zerolinecolor='lightgray', zerolinewidth=1, showticklabels=False)
    fig.update_yaxes(range=[-0.5, 1.8], showgrid=False, zeroline=True,
                     zerolinecolor='lightgray', zerolinewidth=1, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='坐标变换下的体积元行为', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=900, height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    save_and_compress(fig, f'{output_dir}/poincare_coordinate_transform.png', 900, 450)

def plot_exterior_derivative():
    """图3：外微分示意图"""
    fig = go.Figure()
    
    # 绘制上链复形图
    levels = ['0-形式\\n(函数)', '1-形式', '2-形式', '3-形式', '...']
    y_positions = [4, 3, 2, 1, 0]
    colors = [APPLE_BLUE, APPLE_GREEN, APPLE_ORANGE, APPLE_PURPLE, 'gray']
    
    for i, (level, y, color) in enumerate(zip(levels, y_positions, colors)):
        # 绘制节点
        fig.add_trace(go.Scatter(
            x=[0], y=[y],
            mode='markers+text',
            marker=dict(size=60, color=color, line=dict(color='white', width=2)),
            text=[level],
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial'),
            showlegend=False
        ))
        
        # 绘制d箭头
        if i < len(levels) - 1:
            fig.add_annotation(
                x=0, y=y-0.15,
                ax=0, ay=y_positions[i+1]+0.15,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                arrowcolor='black'
            )
            
            # 添加d标签
            fig.add_trace(go.Scatter(
                x=[0.15], y=[(y + y_positions[i+1])/2],
                mode='text',
                text=['$d$'],
                textfont=dict(size=16, color='black'),
                showlegend=False
            ))
    
    # 添加关键性质标注
    fig.add_trace(go.Scatter(
        x=[0.8], y=[2.5],
        mode='text',
        text=['$d^2 = 0$\\n(关键性质)'],
        textfont=dict(size=12, color=APPLE_RED),
        showlegend=False
    ))
    
    fig.add_annotation(x=0.4, y=2.5, ax=0.15, ay=2.5,
                       showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                       arrowcolor=APPLE_RED)
    
    fig.update_xaxes(range=[-1, 2], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-0.5, 4.5], showgrid=False, zeroline=False, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='外微分 $d$ 将 $k$-形式提升到 $(k+1)$-形式', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=700, height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    save_and_compress(fig, f'{output_dir}/poincare_exterior_derivative.png', 700, 600)

def plot_stokes_unification():
    """图4：Stokes定理统一框架 - 使用2D示意图"""
    fig = make_subplots(2, 2, 
        subplot_titles=('n=1: Newton-Leibniz', 'n=2: Green公式', 
                       'n=3: Gauss定理', 'n=3: Stokes定理'),
        vertical_spacing=0.15, horizontal_spacing=0.1)
    
    # n=1: 区间
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0],
        mode='lines+markers',
        line=dict(color=APPLE_BLUE, width=4),
        marker=dict(size=12, color=[APPLE_GREEN, APPLE_RED]),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=[0], y=[0.2], mode='text', text=['$a$'],
                            textfont=dict(size=14), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1], y=[0.2], mode='text', text=['$b$'],
                            textfont=dict(size=14), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0.5], y=[-0.3], mode='text', 
                            text=['$\\int_a^b df = f(b) - f(a)$'],
                            textfont=dict(size=10), showlegend=False), row=1, col=1)
    
    fig.update_xaxes(range=[-0.3, 1.3], row=1, col=1, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-0.5, 0.5], row=1, col=1, showgrid=False, zeroline=False, showticklabels=False)
    
    # n=2: 圆盘
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.4
    x_circle = r * np.cos(theta) + 0.5
    y_circle = r * np.sin(theta) + 0.5
    
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        line=dict(color=APPLE_BLUE, width=2),
        showlegend=False
    ), row=1, col=2)
    
    # 边界箭头（逆时针）
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        x_arrow = r * np.cos(angle) + 0.5
        y_arrow = r * np.sin(angle) + 0.5
        dx = -0.08 * np.sin(angle)
        dy = 0.08 * np.cos(angle)
        fig.add_annotation(x=x_arrow+dx, y=y_arrow+dy, ax=x_arrow, ay=y_arrow,
                          xref='x2', yref='y2', axref='x2', ayref='y2',
                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                          arrowcolor=APPLE_ORANGE, row=1, col=2)
    
    fig.add_trace(go.Scatter(x=[0.5], y=[-0.05], mode='text',
                            text=['$\\iint_D d\\omega = \\oint_{\\partial D} \\omega$'],
                            textfont=dict(size=10), showlegend=False), row=1, col=2)
    
    fig.update_xaxes(range=[0, 1], row=1, col=2, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 1], row=1, col=2, showgrid=False, zeroline=False, showticklabels=False)
    
    # n=3 Gauss: 用圆表示球体（剖面图）
    # 外圆表示球面
    r_outer = 0.35
    x_outer = r_outer * np.cos(theta) + 0.5
    y_outer = r_outer * np.sin(theta) + 0.5
    
    fig.add_trace(go.Scatter(
        x=x_outer, y=y_outer,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.15)',
        line=dict(color=APPLE_BLUE, width=2),
        showlegend=False
    ), row=2, col=1)
    
    # 内圆（表示球的内部结构）
    r_inner = 0.15
    x_inner = r_inner * np.cos(theta) + 0.5
    y_inner = r_inner * np.sin(theta) + 0.5
    
    fig.add_trace(go.Scatter(
        x=x_inner, y=y_inner,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.3)',
        line=dict(color=APPLE_BLUE, width=1, dash='dot'),
        showlegend=False
    ), row=2, col=1)
    
    # 外法向箭头
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        x_base = r_outer * np.cos(angle) + 0.5
        y_base = r_outer * np.sin(angle) + 0.5
        x_tip = x_base + 0.08 * np.cos(angle)
        y_tip = y_base + 0.08 * np.sin(angle)
        fig.add_annotation(x=x_tip, y=y_tip, ax=x_base, ay=y_base,
                          xref='x3', yref='y3', axref='x3', ayref='y3',
                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                          arrowcolor=APPLE_ORANGE, row=2, col=1)
    
    # 添加公式
    fig.add_trace(go.Scatter(x=[0.5], y=[-0.05], mode='text',
                            text=['$\\iiint_V d\\omega = \\iint_{\\partial V} \\omega$'],
                            textfont=dict(size=9), showlegend=False), row=2, col=1)
    
    # 标注
    fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode='text', text=['体积$V$'],
                            textfont=dict(size=10, color=APPLE_BLUE), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0.88], y=[0.5], mode='text', text=['边界$\\partial V$'],
                            textfont=dict(size=9, color=APPLE_ORANGE), showlegend=False), row=2, col=1)
    
    fig.update_xaxes(range=[0, 1], row=2, col=1, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 1], row=2, col=1, showgrid=False, zeroline=False, showticklabels=False)
    
    # n=3 Stokes: 曲面带边界
    # 主曲线（曲面投影）
    t = np.linspace(0, 2*np.pi, 100)
    x_surf = 0.3 * np.cos(t) + 0.5
    y_surf = 0.15 * np.sin(2*t) + 0.5
    
    fig.add_trace(go.Scatter(
        x=x_surf, y=y_surf,
        fill='toself',
        fillcolor='rgba(175, 82, 222, 0.2)',
        line=dict(color=APPLE_PURPLE, width=2),
        showlegend=False
    ), row=2, col=2)
    
    # 边界（两个椭圆）
    # 上边界
    theta_ell = np.linspace(0, 2*np.pi, 50)
    r_ell_x = 0.12
    r_ell_y = 0.06
    x_top = r_ell_x * np.cos(theta_ell) + 0.5
    y_top = r_ell_y * np.sin(theta_ell) + 0.65
    
    fig.add_trace(go.Scatter(
        x=x_top, y=y_top,
        mode='lines',
        line=dict(color=APPLE_ORANGE, width=2),
        showlegend=False
    ), row=2, col=2)
    
    # 下边界
    x_bot = r_ell_x * np.cos(theta_ell) + 0.5
    y_bot = r_ell_y * np.sin(theta_ell) + 0.35
    
    fig.add_trace(go.Scatter(
        x=x_bot, y=y_bot,
        mode='lines',
        line=dict(color=APPLE_ORANGE, width=2),
        showlegend=False
    ), row=2, col=2)
    
    # 边界箭头
    fig.add_annotation(x=0.62, y=0.65, ax=0.62, ay=0.72,
                      xref='x4', yref='y4', axref='x4', ayref='y4',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                      arrowcolor=APPLE_ORANGE, row=2, col=2)
    fig.add_annotation(x=0.62, y=0.35, ax=0.62, ay=0.28,
                      xref='x4', yref='y4', axref='x4', ayref='y4',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                      arrowcolor=APPLE_ORANGE, row=2, col=2)
    
    # 添加公式
    fig.add_trace(go.Scatter(x=[0.5], y=[0.05], mode='text',
                            text=['$\\iint_S d\\omega = \\oint_{\\partial S} \\omega$'],
                            textfont=dict(size=9), showlegend=False), row=2, col=2)
    
    # 标注
    fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode='text', text=['曲面$S$'],
                            textfont=dict(size=10, color=APPLE_PURPLE), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[0.75], y=[0.72], mode='text', text=['$\\partial S$'],
                            textfont=dict(size=9, color=APPLE_ORANGE), showlegend=False), row=2, col=2)
    
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='Stokes定理的统一框架', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=900, height=800,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    save_and_compress(fig, f'{output_dir}/poincare_stokes_unification.png', 900, 800)

def plot_poincare_lemma():
    """图5：Poincaré引理示意"""
    fig = go.Figure()
    
    # 绘制星形区域
    theta = np.linspace(0, 2*np.pi, 100)
    # 创建星形（5角星）
    r_base = 0.3 + 0.15 * np.cos(5*theta)
    x_star = r_base * np.cos(theta) + 0.5
    y_star = r_base * np.sin(theta) + 0.5
    
    fig.add_trace(go.Scatter(
        x=x_star, y=y_star,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        line=dict(color=APPLE_BLUE, width=2),
        showlegend=False
    ))
    
    # 添加中心点（星形区域的中心）
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5],
        mode='markers',
        marker=dict(size=10, color=APPLE_RED),
        showlegend=False
    ))
    
    # 添加文字标注
    fig.add_trace(go.Scatter(x=[0.5], y=[0.95], mode='text',
                            text=['星形区域 $U$'],
                            textfont=dict(size=14, color=APPLE_BLUE),
                            showlegend=False))
    
    # 添加omega和eta的示意
    fig.add_trace(go.Scatter(x=[0.2], y=[0.2], mode='text',
                            text=['$\\omega$ (闭形式)'],
                            textfont=dict(size=12, color=APPLE_GREEN),
                            showlegend=False))
    
    # 添加箭头
    fig.add_annotation(x=0.35, y=0.3, ax=0.25, ay=0.25,
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                      arrowcolor=APPLE_ORANGE)
    
    fig.add_trace(go.Scatter(x=[0.5], y=[0.35], mode='text',
                            text=['$\\omega = d\\eta$'],
                            textfont=dict(size=11, color=APPLE_ORANGE),
                            showlegend=False))
    
    # 添加定理说明
    fig.add_trace(go.Scatter(x=[0.5], y=[0.05], mode='text',
                            text=['Poincaré引理: $d\\omega = 0 \\Rightarrow \\omega = d\\eta$ (在星形区域上)'],
                            textfont=dict(size=11, color='black'),
                            showlegend=False))
    
    fig.update_xaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='Poincaré引理', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=700, height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    save_and_compress(fig, f'{output_dir}/poincare_lemma.png', 700, 600)

def plot_de_rham():
    """图6：de Rham上同调示意"""
    fig = make_subplots(1, 2, 
        subplot_titles=('闭形式空间 $Z^k$', 'de Rham上同调 $H^k_{\\text{dR}} = Z^k / B^k$'))
    
    # 左图：闭形式空间包含恰当形式
    # 绘制大圆（闭形式）
    theta = np.linspace(0, 2*np.pi, 100)
    r_z = 0.4
    x_z = r_z * np.cos(theta) + 0.5
    y_z = r_z * np.sin(theta) + 0.5
    
    fig.add_trace(go.Scatter(
        x=x_z, y=y_z,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        line=dict(color=APPLE_BLUE, width=2),
        showlegend=False
    ), row=1, col=1)
    
    # 绘制小圆（恰当形式）
    r_b = 0.2
    x_b = r_b * np.cos(theta) + 0.5
    y_b = r_b * np.sin(theta) + 0.5
    
    fig.add_trace(go.Scatter(
        x=x_b, y=y_b,
        fill='toself',
        fillcolor='rgba(52, 199, 89, 0.3)',
        line=dict(color=APPLE_GREEN, width=2),
        showlegend=False
    ), row=1, col=1)
    
    # 添加标注
    fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode='text', text=['$B^k$'],
                            textfont=dict(size=12, color=APPLE_GREEN), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0.8], y=[0.5], mode='text', text=['$Z^k$'],
                            textfont=dict(size=12, color=APPLE_BLUE), showlegend=False), row=1, col=1)
    
    # 添加$d\omega=0$标注
    fig.add_trace(go.Scatter(x=[0.5], y=[0.85], mode='text', text=['$d\\omega = 0$'],
                            textfont=dict(size=11, color=APPLE_BLUE), showlegend=False), row=1, col=1)
    
    # 右图：商空间示意（环形）
    fig.add_trace(go.Scatter(
        x=x_z, y=y_z,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.1)',
        line=dict(color=APPLE_BLUE, width=2),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=x_b, y=y_b,
        fill='toself',
        fillcolor='white',
        line=dict(color=APPLE_GREEN, width=2),
        showlegend=False
    ), row=1, col=2)
    
    # 在环形区域画一个代表元
    fig.add_trace(go.Scatter(
        x=[0.65], y=[0.5],
        mode='markers',
        marker=dict(size=15, color=APPLE_ORANGE, symbol='diamond'),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=[0.65], y=[0.35], mode='text', text=['$[\\omega]$'],
                            textfont=dict(size=12, color=APPLE_ORANGE), showlegend=False), row=1, col=2)
    
    # 添加等价的其他点（用虚线连接）
    angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
    for angle in angles:
        r_point = 0.3
        x_p = r_point * np.cos(angle) + 0.5
        y_p = r_point * np.sin(angle) + 0.5
        fig.add_trace(go.Scatter(
            x=[x_p], y=[y_p],
            mode='markers',
            marker=dict(size=10, color=APPLE_ORANGE, symbol='diamond'),
            showlegend=False
        ), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=[0.5], y=[0.05], mode='text',
                            text=['$[\\omega]$ = 等价类 $\\omega + B^k$'],
                            textfont=dict(size=10, color='black'), showlegend=False), row=1, col=2)
    
    fig.update_xaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
    
    fig.update_layout(
        title=dict(text='de Rham上同调', x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=900, height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    save_and_compress(fig, f'{output_dir}/poincare_de_rham.png', 900, 450)

if __name__ == '__main__':
    print("🎨 开始生成Poincaré体积元定向文章图形...")
    
    print("\n1. 生成体积元定向对比图...")
    plot_orientation_comparison()
    
    print("\n2. 生成坐标变换对比图...")
    plot_coordinate_transform()
    
    print("\n3. 生成外微分示意图...")
    plot_exterior_derivative()
    
    print("\n4. 生成Stokes定理统一框架图...")
    plot_stokes_unification()
    
    print("\n5. 生成Poincaré引理示意图...")
    plot_poincare_lemma()
    
    print("\n6. 生成de Rham上同调示意图...")
    plot_de_rham()
    
    print("\n✅ 所有图形生成完成！")
