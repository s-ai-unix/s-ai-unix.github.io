#!/usr/bin/env python3
"""
为雅可比矩阵与黑塞矩阵文章生成 Plotly 图形
输出为 PNG 图片格式
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# 设置 Kaleido 超时（秒）
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
    fig.write_image(filepath, width=width, height=height, scale=scale, engine="kaleido")
    print(f"✅ 已生成: {filepath}")
    return filepath


def plot_gradient_vector_field():
    """图1: 梯度向量场"""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # 梯度 ∇f = (2x, 2y)
    U = 2 * X
    V = 2 * Y
    
    fig = go.Figure()
    
    # 绘制等高线背景
    Z = X**2 + Y**2
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Blues',
        showscale=False,
        contours=dict(coloring='fill'),
        opacity=0.6
    ))
    
    # 绘制梯度向量场
    arrow_x, arrow_y = [], []
    for i in range(0, len(x), 2):
        for j in range(0, len(y), 2):
            xi, yi = X[i,j], Y[i,j]
            ui, vi = U[i,j]*0.15, V[i,j]*0.15
            arrow_x.extend([xi, xi+ui, None])
            arrow_y.extend([yi, yi+vi, None])
    
    fig.add_trace(go.Scatter(
        x=arrow_x, y=arrow_y,
        mode='lines',
        line=dict(color=APPLE_RED, width=1.5),
        showlegend=False
    ))
    
    # 添加箭头
    for i in range(0, len(x), 2):
        for j in range(0, len(y), 2):
            xi, yi = X[i,j], Y[i,j]
            ui, vi = U[i,j]*0.15, V[i,j]*0.15
            if abs(ui) > 0.01 or abs(vi) > 0.01:
                fig.add_annotation(
                    x=xi+ui, y=yi+vi,
                    ax=xi, ay=yi,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2, arrowsize=1, arrowwidth=1,
                    arrowcolor=APPLE_RED
                )
    
    fig.update_layout(
        title=dict(text='Gradient Vector Field: ∇f(x,y) = (2x, 2y)', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        width=700, height=600,
        margin=dict(l=60, r=40, t=70, b=50)
    )
    
    save_plotly_as_png(fig, 'gradient-vector-field.png', width=700, height=600)


def plot_jacobian_transformation():
    """图2: 雅可比矩阵的线性变换"""
    theta = np.linspace(0, 2*np.pi, 100)
    x, y = np.cos(theta), np.sin(theta)
    
    # 雅可比矩阵
    J = np.array([[2, 1], [1, 3]])
    xy = np.vstack([x, y])
    transformed = J @ xy
    x_new, y_new = transformed[0], transformed[1]
    
    fig = go.Figure()
    
    # 原始单位圆
    fig.add_trace(go.Scatter(
        x=x, y=y, 
        mode='lines', 
        line=dict(color=APPLE_BLUE, width=2), 
        name='Unit Circle',
        fill='toself',
        fillcolor='rgba(0,122,255,0.1)'
    ))
    
    # 变换后的椭圆
    fig.add_trace(go.Scatter(
        x=x_new, y=y_new, 
        mode='lines', 
        line=dict(color=APPLE_RED, width=2), 
        name='Transformed Ellipse',
        fill='toself',
        fillcolor='rgba(255,59,48,0.1)'
    ))
    
    # 原始基向量
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode='lines+markers', 
                             line=dict(color=APPLE_BLUE, width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, 1], mode='lines+markers', 
                             line=dict(color=APPLE_BLUE, width=2), showlegend=False))
    
    # 变换后的基向量
    e1_new = J @ np.array([1, 0])
    e2_new = J @ np.array([0, 1])
    fig.add_trace(go.Scatter(x=[0, e1_new[0]], y=[0, e1_new[1]], mode='lines+markers',
                             line=dict(color=APPLE_RED, width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=[0, e2_new[0]], y=[0, e2_new[1]], mode='lines+markers',
                             line=dict(color=APPLE_RED, width=2), showlegend=False))
    
    # 标注
    fig.add_annotation(x=1.15, y=0.1, text='e₁', showarrow=False, font=dict(size=14))
    fig.add_annotation(x=0.1, y=1.15, text='e₂', showarrow=False, font=dict(size=14))
    fig.add_annotation(x=e1_new[0]+0.25, y=e1_new[1]+0.1, text='Je₁', showarrow=False, font=dict(size=12, color=APPLE_RED))
    fig.add_annotation(x=e2_new[0]+0.1, y=e2_new[1]+0.25, text='Je₂', showarrow=False, font=dict(size=12, color=APPLE_RED))
    
    # 矩阵标注
    fig.add_annotation(x=0.02, y=0.98, xref='paper', yref='paper',
                       text='J = [[2, 1], [1, 3]]',
                       showarrow=False, font=dict(size=13),
                       bgcolor='rgba(255,255,255,0.9)', bordercolor=APPLE_GRAY, borderwidth=1,
                       align='left')
    
    fig.update_layout(
        title=dict(text='Jacobian Linear Transformation', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white', 
        width=700, height=600,
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.02, y=0.85)
    )
    fig.update_xaxes(range=[-4, 4], zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0')
    fig.update_yaxes(range=[-4, 4], zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0', scaleanchor='x', scaleratio=1)
    
    save_plotly_as_png(fig, 'jacobian-linear-transform.png', width=700, height=600)


def plot_hessian_curvature():
    """图3: 黑塞矩阵与曲面曲率"""
    x = np.linspace(-2, 2, 80)
    y = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2
    
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=('Surface: f(x,y)=x²-y²', 'Contour with Principal Directions'),
        specs=[[{'type': 'surface'}, {'type': 'xy'}]]
    )
    
    # 3D 曲面
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='RdBu', showscale=False, opacity=0.9), row=1, col=1)
    
    # 等高线图
    fig.add_trace(go.Contour(x=x, y=y, z=Z, colorscale='RdBu', showscale=False, contours=dict(coloring='fill')), row=1, col=2)
    
    # 主曲率方向
    fig.add_trace(go.Scatter(x=[-1.5, 1.5], y=[0, 0], mode='lines', 
                             line=dict(color=APPLE_RED, width=3), name='λ₁ > 0'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 0], y=[-1.5, 1.5], mode='lines', 
                             line=dict(color=APPLE_GREEN, width=3), name='λ₂ < 0'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=10, color='black'), name='Saddle Point'), row=1, col=2)
    
    fig.update_layout(
        title=dict(text='Hessian Matrix and Surface Curvature', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white', 
        width=1000, height=500, 
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.update_scenes(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='cube')
    
    save_plotly_as_png(fig, 'hessian-curvature.png', width=1000, height=500)


def plot_optimization_landscape():
    """图4: 优化景观"""
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Rosenbrock 函数
    Z = (1-X)**2 + 100*(Y-X**2)**2
    Z = np.log(Z + 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Contour(x=x, y=y, z=Z, colorscale='Viridis', 
                             contours=dict(coloring='fill'), 
                             colorbar=dict(title=dict(text='log(1+f)', side='right'))))
    
    fig.add_trace(go.Scatter(x=[1], y=[1], mode='markers', 
                             marker=dict(color=APPLE_RED, size=15, symbol='star'), name='Global Minimum'))
    
    # 优化路径
    path_x = np.linspace(-1.5, 1, 30)
    path_y = path_x**2
    fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines+markers', 
                             line=dict(color=APPLE_ORANGE, width=2), marker=dict(size=4), name='Optimization Path'))
    
    fig.update_layout(
        title=dict(text='Optimization Landscape: Rosenbrock Function', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        width=700, height=600, margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.02, y=0.98)
    )
    
    save_plotly_as_png(fig, 'optimization-landscape.png', width=700, height=600)


def plot_newton_vs_gradient():
    """图5: 牛顿法 vs 梯度下降"""
    H = np.array([[2, 1], [1, 3]])
    b = np.array([1, 2])
    x_star = np.linalg.solve(H, b)
    
    x = np.linspace(-0.5, 1.5, 80)
    y = np.linspace(-0.5, 1.5, 80)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (2*X**2 + 2*X*Y + 3*Y**2) - X - 2*Y
    
    fig = go.Figure()
    
    fig.add_trace(go.Contour(x=x, y=y, z=Z, colorscale='Viridis', 
                             contours=dict(coloring='fill'), showscale=False))
    
    # 梯度下降路径
    x_gd, y_gd = [0], [0]
    lr, x_curr = 0.15, np.array([0.0, 0.0])
    for _ in range(15):
        grad = H @ x_curr - b
        x_curr = x_curr - lr * grad
        x_gd.append(x_curr[0])
        y_gd.append(x_curr[1])
    
    fig.add_trace(go.Scatter(x=x_gd, y=y_gd, mode='lines+markers', 
                             line=dict(color=APPLE_ORANGE, width=2), marker=dict(size=6), name='Gradient Descent (15 steps)'))
    
    fig.add_trace(go.Scatter(x=[0, x_star[0]], y=[0, x_star[1]], mode='lines+markers',
                             line=dict(color=APPLE_RED, width=3, dash='dash'), marker=dict(size=8), name='Newton Method (1 step)'))
    
    fig.add_trace(go.Scatter(x=[x_star[0]], y=[x_star[1]], mode='markers',
                             marker=dict(color=APPLE_GREEN, size=15, symbol='star'), name='Optimal'))
    
    fig.update_layout(
        title=dict(text='Newton Method vs Gradient Descent', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        width=700, height=600, margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.02, y=0.98)
    )
    
    save_plotly_as_png(fig, 'newton-vs-gradient.png', width=700, height=600)


def plot_jacobian_chain_rule():
    """图6: 链式法则可视化"""
    t = np.linspace(0, 2*np.pi, 100)
    
    # 内函数 g: R -> R^2 (螺旋)
    r = 1 + 0.3 * t
    g_x = r * np.cos(t)
    g_y = r * np.sin(t)
    
    # 外函数 f: R^2 -> R
    f_z = g_x**2 + g_y**2
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Inner Function g(t): Spiral', 'Composite Function (f∘g)(t)'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # 左图: 平面螺旋
    fig.add_trace(go.Scatter(x=g_x, y=g_y, mode='lines', line=dict(color=APPLE_BLUE, width=2), name='g(t)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[g_x[0]], y=[g_y[0]], mode='markers', marker=dict(size=10, color=APPLE_GREEN), name='Start'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[g_x[-1]], y=[g_y[-1]], mode='markers', marker=dict(size=10, color=APPLE_RED), name='End'), row=1, col=1)
    
    # 右图: 复合函数
    fig.add_trace(go.Scatter(x=t, y=f_z, mode='lines', line=dict(color=APPLE_ORANGE, width=2), name='(f∘g)(t)'), row=1, col=2)
    
    fig.update_layout(
        title=dict(text='Chain Rule: d(f∘g)/dt = ∇f · g\'(t)', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white',
        width=1000, height=450, margin=dict(l=60, r=40, t=70, b=50),
        showlegend=True
    )
    
    fig.update_xaxes(title_text='x', row=1, col=1)
    fig.update_yaxes(title_text='y', row=1, col=1)
    fig.update_xaxes(title_text='t', row=1, col=2)
    fig.update_yaxes(title_text='(f∘g)(t)', row=1, col=2)
    
    save_plotly_as_png(fig, 'jacobian-chain-rule.png', width=1000, height=450)


if __name__ == '__main__':
    print("开始生成雅可比矩阵与黑塞矩阵文章的 Plotly 图形...")
    print("=" * 60)
    
    plot_gradient_vector_field()
    plot_jacobian_transformation()
    plot_hessian_curvature()
    plot_optimization_landscape()
    plot_newton_vs_gradient()
    plot_jacobian_chain_rule()
    
    print("=" * 60)
    print("所有图形生成完成！")
