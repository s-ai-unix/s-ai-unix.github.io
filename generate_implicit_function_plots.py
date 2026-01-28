#!/usr/bin/env python3
"""
生成隐函数定理相关的 Plotly 图形，输出为 PNG 图片
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 设置输出目录
OUTPUT_DIR = 'static/images/math'


def save_figure_as_png(fig, filename, width=800, height=600, scale=2):
    """保存 Plotly 图形为 PNG 图片"""
    filepath = f'{OUTPUT_DIR}/{filename}'
    fig.write_image(filepath, width=width, height=height, scale=scale)
    print(f"✅ 生成: {filepath}")
    return filepath


def plot_unit_circle_implicit():
    """
    绘制单位圆，展示隐函数与显函数的关系
    上半圆: y = sqrt(1-x^2)
    下半圆: y = -sqrt(1-x^2)
    """
    theta = np.linspace(0, 2*np.pi, 500)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    # 上半圆（显式表示）
    x_upper = np.linspace(-1, 1, 200)
    y_upper = np.sqrt(1 - x_upper**2)
    
    # 下半圆（显式表示）
    y_lower = -np.sqrt(1 - x_upper**2)
    
    fig = go.Figure()
    
    # 完整的单位圆（隐式）
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        name='单位圆 x² + y² = 1',
        line=dict(color='#007AFF', width=3),
    ))
    
    # 上半圆点
    fig.add_trace(go.Scatter(
        x=x_upper, y=y_upper,
        mode='lines',
        name='上半圆 y = √(1-x²)',
        line=dict(color='#34C759', width=2, dash='dash'),
    ))
    
    # 下半圆点
    fig.add_trace(go.Scatter(
        x=x_upper, y=y_lower,
        mode='lines',
        name='下半圆 y = -√(1-x²)',
        line=dict(color='#FF9500', width=2, dash='dash'),
    ))
    
    # 标记点 (0.6, 0.8)
    fig.add_trace(go.Scatter(
        x=[0.6], y=[0.8],
        mode='markers+text',
        name='点 P(0.6, 0.8)',
        marker=dict(color='#FF3B30', size=12, symbol='circle'),
        text=['P(0.6, 0.8)'],
        textposition='top right',
        textfont=dict(size=12, color='#FF3B30'),
    ))
    
    # 切线
    x_tangent = np.linspace(0.3, 0.9, 100)
    y_tangent = -0.75 * x_tangent + 1.25  # 切线方程: 0.6x + 0.8y = 1
    fig.add_trace(go.Scatter(
        x=x_tangent, y=y_tangent,
        mode='lines',
        name='切线: 0.6x + 0.8y = 1',
        line=dict(color='#AF52DE', width=2, dash='dot'),
    ))
    
    fig.update_layout(
        title=dict(
            text='单位圆的隐函数表示',
            font=dict(family='Arial, sans-serif', size=18, color='#1d1d1f')
        ),
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#d2d2d7',
            borderwidth=1
        ),
        xaxis=dict(
            range=[-1.5, 1.5],
            zeroline=True,
            zerolinecolor='#d2d2d7',
            gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            range=[-1.5, 1.5],
            zeroline=True,
            zerolinecolor='#d2d2d7',
            gridcolor='#f0f0f0',
            scaleanchor='x',
            scaleratio=1
        ),
        width=800,
        height=600,
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    save_figure_as_png(fig, 'implicit-unit-circle.png')
    return fig


def plot_implicit_derivative_geometric():
    """
    展示隐函数导数的几何意义：法向量与切线方向
    """
    # 隐函数曲线 F(x,y) = x² + y² - 1 = 0（单位圆）
    theta = np.linspace(0, 2*np.pi, 500)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    fig = go.Figure()
    
    # 单位圆
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        name='隐函数曲线 F(x,y)=0',
        line=dict(color='#007AFF', width=2),
    ))
    
    # 选择几个点展示梯度方向
    points = [
        (0.6, 0.8, 'P₁'),
        (-0.8, 0.6, 'P₂'),
        (0, -1, 'P₃'),
        (1, 0, 'P₄')
    ]
    
    colors = ['#FF3B30', '#34C759', '#FF9500', '#AF52DE']
    
    for i, (x0, y0, label) in enumerate(points):
        # 梯度方向（法向量）: ∇F = (2x, 2y)
        grad_x, grad_y = 2*x0, 2*y0
        
        # 切线方向（垂直于梯度）
        tangent_x, tangent_y = -grad_y, grad_x
        
        # 绘制点
        fig.add_trace(go.Scatter(
            x=[x0], y=[y0],
            mode='markers+text',
            name=f'{label}({x0:.1f}, {y0:.1f})',
            marker=dict(color=colors[i], size=10),
            text=[label],
            textposition='top center',
            textfont=dict(size=11, color=colors[i]),
        ))
        
        # 梯度向量（法向量）
        fig.add_trace(go.Scatter(
            x=[x0, x0 + grad_x*0.3], y=[y0, y0 + grad_y*0.3],
            mode='lines+markers',
            name=f'{label} 梯度 ∇F',
            line=dict(color=colors[i], width=2),
            marker=dict(size=6),
        ))
        
        # 切线向量
        fig.add_trace(go.Scatter(
            x=[x0 - tangent_x*0.3, x0 + tangent_x*0.3],
            y=[y0 - tangent_y*0.3, y0 + tangent_y*0.3],
            mode='lines',
            name=f'{label} 切线',
            line=dict(color=colors[i], width=1, dash='dash'),
        ))
    
    fig.update_layout(
        title=dict(
            text='隐函数导数的几何意义：梯度与切线',
            font=dict(family='Arial, sans-serif', size=18, color='#1d1d1f')
        ),
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=False,
        xaxis=dict(
            range=[-1.8, 1.8],
            zeroline=True,
            zerolinecolor='#d2d2d7',
            gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            range=[-1.8, 1.8],
            zeroline=True,
            zerolinecolor='#d2d2d7',
            gridcolor='#f0f0f0',
            scaleanchor='x',
            scaleratio=1
        ),
        width=800,
        height=600,
        margin=dict(l=60, r=40, t=80, b=60),
        annotations=[
            dict(x=0.02, y=0.98, xref='paper', yref='paper',
                 text='<b>图例:</b><br>— 隐函数曲线<br>→ 梯度∇F<br>- - 切线方向',
                 showarrow=False, font=dict(size=10), align='left',
                 bgcolor='rgba(255,255,255,0.9)', bordercolor='#d2d2d7', borderwidth=1)
        ]
    )
    
    save_figure_as_png(fig, 'implicit-derivative-geometric.png')
    return fig


def plot_applications_comparison():
    """
    比较不同隐函数的应用
    """
    x = np.linspace(-2, 2, 400)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '椭圆: x²/4 + y² = 1',
            '双曲线: x² - y² = 1',
            '抛物线: y - x² = 0',
            '笛卡尔叶形线: x³ + y³ - 3xy = 0'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # 椭圆
    theta = np.linspace(0, 2*np.pi, 400)
    x_ellipse = 2 * np.cos(theta)
    y_ellipse = np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_ellipse, y=y_ellipse,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='椭圆',
        showlegend=False
    ), row=1, col=1)
    
    # 双曲线
    x_hyp = np.linspace(1, 3, 200)
    y_hyp_pos = np.sqrt(x_hyp**2 - 1)
    y_hyp_neg = -np.sqrt(x_hyp**2 - 1)
    x_hyp2 = np.linspace(-3, -1, 200)
    y_hyp2_pos = np.sqrt(x_hyp2**2 - 1)
    y_hyp2_neg = -np.sqrt(x_hyp2**2 - 1)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_hyp, x_hyp[::-1]]),
        y=np.concatenate([y_hyp_pos, y_hyp_neg[::-1]]),
        mode='lines',
        line=dict(color='#34C759', width=2),
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_hyp2, x_hyp2[::-1]]),
        y=np.concatenate([y_hyp2_pos, y_hyp2_neg[::-1]]),
        mode='lines',
        line=dict(color='#34C759', width=2),
        showlegend=False
    ), row=1, col=2)
    
    # 抛物线
    y_para = x**2
    fig.add_trace(go.Scatter(
        x=x, y=y_para,
        mode='lines',
        line=dict(color='#FF9500', width=2),
        showlegend=False
    ), row=2, col=1)
    
    # 笛卡尔叶形线
    t = np.linspace(-10, 10, 1000)
    x_folium = 3*t / (1 + t**3)
    y_folium = 3*t**2 / (1 + t**3)
    # 移除无穷大点
    mask = np.isfinite(x_folium) & np.isfinite(y_folium) & (np.abs(x_folium) < 10) & (np.abs(y_folium) < 10)
    fig.add_trace(go.Scatter(
        x=x_folium[mask], y=y_folium[mask],
        mode='lines',
        line=dict(color='#AF52DE', width=2),
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        title=dict(
            text='各类隐函数曲线示例',
            font=dict(family='Arial, sans-serif', size=18, color='#1d1d1f')
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=11),
        width=900,
        height=800,
        margin=dict(l=60, r=40, t=100, b=60)
    )
    
    # 更新所有子图的轴
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(range=[-3, 3], row=i, col=j, gridcolor='#f0f0f0', zerolinecolor='#d2d2d7')
            fig.update_yaxes(range=[-3, 3], row=i, col=j, gridcolor='#f0f0f0', zerolinecolor='#d2d2d7')
    
    save_figure_as_png(fig, 'implicit-functions-comparison.png', width=900, height=800)
    return fig


def plot_newton_method_visualization():
    """
    展示隐函数求导在牛顿迭代法中的应用
    """
    x = np.linspace(-2, 2, 400)
    
    # 隐函数 F(x,y) = x² + y² - 2 = 0 （半径为√2的圆）
    # 求解 F(x0, y) = 0 对于给定的 x0
    
    fig = go.Figure()
    
    # 完整的圆
    theta = np.linspace(0, 2*np.pi, 400)
    x_circle = np.sqrt(2) * np.cos(theta)
    y_circle = np.sqrt(2) * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        name='隐函数曲线 F(x,y)=x²+y²-2=0',
        line=dict(color='#007AFF', width=2),
    ))
    
    # 牛顿迭代过程
    x0 = 1.0  # 固定 x = 1
    # 初始猜测 y0 = 0.5
    y_current = 0.5
    iterations = [y_current]
    
    colors = ['#FF3B30', '#FF9500', '#34C759', '#007AFF']
    
    for i in range(4):
        # F(x0, y) = x0² + y² - 2
        # F_y = 2y
        F_val = x0**2 + y_current**2 - 2
        F_y = 2 * y_current
        
        y_next = y_current - F_val / F_y
        
        # 绘制迭代点
        fig.add_trace(go.Scatter(
            x=[x0], y=[y_current],
            mode='markers+text',
            name=f'迭代 {i}: y_{i} = {y_current:.4f}',
            marker=dict(color=colors[i % len(colors)], size=10),
            text=[f'y_{i}={y_current:.3f}'],
            textposition='middle right',
            textfont=dict(size=10),
        ))
        
        y_current = y_next
        iterations.append(y_current)
    
    # 真实解
    y_true = np.sqrt(2 - x0**2)
    fig.add_trace(go.Scatter(
        x=[x0], y=[y_true],
        mode='markers+text',
        name=f'真实解: y = {y_true:.4f}',
        marker=dict(color='#AF52DE', size=12, symbol='star'),
        text=['真实解'],
        textposition='top right',
        textfont=dict(size=11, color='#AF52DE'),
    ))
    
    fig.update_layout(
        title=dict(
            text='牛顿迭代法求解隐函数',
            font=dict(family='Arial, sans-serif', size=18, color='#1d1d1f')
        ),
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#d2d2d7',
            borderwidth=1
        ),
        xaxis=dict(range=[-2, 2], zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0'),
        yaxis=dict(range=[-2, 2], zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0', scaleanchor='x', scaleratio=1),
        width=800,
        height=600,
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    save_figure_as_png(fig, 'implicit-newton-method.png')
    return fig


def plot_sphere_3d_projection():
    """
    球面的2D投影展示（因为3D图在文章中不好展示，用2D等高线图表示）
    """
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    
    # 球面函数 F(x,y,z) = x² + y² + z² - 1 = 0
    # 等高线图表示 z = ±√(1-x²-y²)
    Z_pos = np.sqrt(np.maximum(0, 1 - X**2 - Y**2))
    Z_neg = -Z_pos
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('上半球 z = √(1-x²-y²)', '下半球 z = -√(1-x²-y²)'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )
    
    # 实际上用2D等高线更好展示
    fig2 = go.Figure()
    
    # 用等高线表示球面
    levels = np.linspace(-1, 1, 11)
    for i, z_level in enumerate([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]):
        if abs(z_level) < 1:
            r = np.sqrt(1 - z_level**2)
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = r * np.cos(theta)
            y_circle = r * np.sin(theta)
            color_intensity = (z_level + 1) / 2
            color = f'rgba({int(0 + 0*color_intensity)}, {int(122 + 133*color_intensity)}, {int(255 - 100*color_intensity)}, 0.7)'
            
            fig2.add_trace(go.Scatter(
                x=x_circle, y=y_circle,
                mode='lines',
                line=dict(color=color, width=2),
                name=f'z = {z_level:.1f}',
                showlegend=False
            ))
    
    # 标记几个点
    points = [(0.5, 0.5, np.sqrt(0.5)), (0.8, 0, 0.6), (0, 0.8, 0.6)]
    point_colors = ['#FF3B30', '#34C759', '#FF9500']
    
    for i, (px, py, pz) in enumerate(points):
        fig2.add_trace(go.Scatter(
            x=[px], y=[py],
            mode='markers+text',
            marker=dict(color=point_colors[i], size=10),
            text=[f'P{i+1}'],
            textposition='top right',
            textfont=dict(size=10),
        ))
    
    fig2.update_layout(
        title=dict(
            text='球面隐函数的水平集投影',
            font=dict(family='Arial, sans-serif', size=18, color='#1d1d1f')
        ),
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        xaxis=dict(range=[-1.2, 1.2], zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0'),
        yaxis=dict(range=[-1.2, 1.2], zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0', scaleanchor='x', scaleratio=1),
        width=700,
        height=600,
        margin=dict(l=60, r=40, t=80, b=60),
        annotations=[
            dict(x=0.02, y=0.98, xref='paper', yref='paper',
                 text='同心圆表示不同 z 值的水平集',
                 showarrow=False, font=dict(size=10), align='left',
                 bgcolor='rgba(255,255,255,0.9)', bordercolor='#d2d2d7', borderwidth=1)
        ]
    )
    
    save_figure_as_png(fig2, 'implicit-sphere-contours.png', width=700, height=600)
    return fig2


if __name__ == '__main__':
    print("开始生成隐函数定理相关的 Plotly 图形（PNG格式）...")
    print()
    
    plot_unit_circle_implicit()
    plot_implicit_derivative_geometric()
    plot_sphere_3d_projection()
    plot_applications_comparison()
    plot_newton_method_visualization()
    
    print()
    print("所有图形生成完成！")
