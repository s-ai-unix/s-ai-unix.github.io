#!/usr/bin/env python3
"""
为雅可比矩阵与黑塞矩阵文章生成 Plotly 图形
输出为 PNG 图片格式
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import io

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
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    img = Image.open(io.BytesIO(img_bytes))
    img.save(filepath)
    print(f"✅ 已生成: {filepath}")
    return filepath


def plot_gradient_vector_field():
    """图1: 梯度向量场 - 展示多元函数的变化方向"""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # 函数 f(x,y) = x^2 + y^2，梯度 ∇f = (2x, 2y)
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
    arrow_x, arrow_y, arrow_u, arrow_v = [], [], [], []
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
        showlegend=False,
        name='梯度方向'
    ))
    
    # 添加箭头标记
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
        title=dict(text='梯度向量场: $\\nabla f(x,y) = (2x, 2y)$', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='$x$',
        yaxis_title='$y$',
        template='plotly_white',
        width=700, height=600,
        margin=dict(l=60, r=40, t=70, b=50),
        showlegend=True
    )
    
    save_plotly_as_png(fig, 'gradient-vector-field.png', width=700, height=600)


def plot_jacobian_transformation():
    """图2: 雅可比矩阵的线性变换效果 - 展示局部线性近似"""
    theta = np.linspace(0, 2*np.pi, 100)
    x, y = np.cos(theta), np.sin(theta)
    
    # 雅可比矩阵 J = [[2, 1], [1, 3]]
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
        name='原始单位圆',
        fill='toself',
        fillcolor='rgba(0,122,255,0.1)'
    ))
    
    # 变换后的椭圆
    fig.add_trace(go.Scatter(
        x=x_new, y=y_new, 
        mode='lines', 
        line=dict(color=APPLE_RED, width=2), 
        name='变换后椭圆',
        fill='toself',
        fillcolor='rgba(255,59,48,0.1)'
    ))
    
    # 原始基向量
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], 
        mode='lines+markers', 
        line=dict(color=APPLE_BLUE, width=2),
        marker=dict(size=6, color=APPLE_BLUE),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1], 
        mode='lines+markers', 
        line=dict(color=APPLE_BLUE, width=2),
        marker=dict(size=6, color=APPLE_BLUE),
        showlegend=False
    ))
    
    # 变换后的基向量
    e1_new = J @ np.array([1, 0])
    e2_new = J @ np.array([0, 1])
    fig.add_trace(go.Scatter(
        x=[0, e1_new[0]], y=[0, e1_new[1]], 
        mode='lines+markers', 
        line=dict(color=APPLE_RED, width=2),
        marker=dict(size=6, color=APPLE_RED),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0, e2_new[0]], y=[0, e2_new[1]], 
        mode='lines+markers', 
        line=dict(color=APPLE_RED, width=2),
        marker=dict(size=6, color=APPLE_RED),
        showlegend=False
    ))
    
    # 添加标注
    fig.add_annotation(x=1.1, y=0.1, text='$\\mathbf{e}_1$', showarrow=False, font=dict(size=14))
    fig.add_annotation(x=0.1, y=1.1, text='$\\mathbf{e}_2$', showarrow=False, font=dict(size=14))
    fig.add_annotation(x=e1_new[0]+0.2, y=e1_new[1]+0.1, text="$J\\mathbf{e}_1$", showarrow=False, font=dict(size=12, color=APPLE_RED))
    fig.add_annotation(x=e2_new[0]+0.1, y=e2_new[1]+0.2, text="$J\\mathbf{e}_2$", showarrow=False, font=dict(size=12, color=APPLE_RED))
    
    fig.update_layout(
        title=dict(text='雅可比矩阵的线性变换: $J = \\begin{pmatrix} 2 & 1 \\ 1 & 3 \\end{pmatrix}$', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white', 
        width=700, height=600,
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.02, y=0.98)
    )
    fig.update_xaxes(zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0', range=[-4, 4])
    fig.update_yaxes(zeroline=True, zerolinecolor='#d2d2d7', gridcolor='#f0f0f0', scaleanchor='x', scaleratio=1, range=[-4, 4])
    
    save_plotly_as_png(fig, 'jacobian-linear-transform.png', width=700, height=600)


def plot_hessian_curvature():
    """图3: 黑塞矩阵与曲面曲率 - 展示鞍点结构"""
    x = np.linspace(-2, 2, 80)
    y = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2  # 鞍点函数
    
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=('函数曲面 $f(x,y)=x^2-y^2$', '等高线与主曲率方向'),
        specs=[[{'type': 'surface'}, {'type': 'xy'}]]
    )
    
    # 3D 曲面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, 
        colorscale='RdBu', 
        showscale=False,
        opacity=0.9
    ), row=1, col=1)
    
    # 等高线图
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z, 
        colorscale='RdBu', 
        showscale=False, 
        contours=dict(coloring='fill')
    ), row=1, col=2)
    
    # 主曲率方向
    fig.add_trace(go.Scatter(
        x=[-1.5, 1.5], y=[0, 0], 
        mode='lines', 
        line=dict(color=APPLE_RED, width=3),
        name='$\\lambda_1 > 0$ (向上凸)'
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-1.5, 1.5], 
        mode='lines', 
        line=dict(color=APPLE_GREEN, width=3),
        name='$\\lambda_2 < 0$ (向下凹)'
    ), row=1, col=2)
    
    # 标记鞍点
    fig.add_trace(go.Scatter(
        x=[0], y=[0], 
        mode='markers',
        marker=dict(size=10, color='black'),
        name='鞍点'
    ), row=1, col=2)
    
    fig.update_layout(
        title=dict(text='黑塞矩阵与曲面曲率: 鞍点示例', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white', 
        width=1000, height=500, 
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.update_scenes(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        aspectmode='cube'
    )
    
    save_plotly_as_png(fig, 'hessian-curvature.png', width=1000, height=500)


def plot_optimization_landscape():
    """图4: 优化景观 - 展示损失函数的复杂地形"""
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Rosenbrock 函数 (香蕉函数)
    Z = (1-X)**2 + 100*(Y-X**2)**2
    Z = np.log(Z + 1)  # 对数缩放以便可视化
    
    fig = go.Figure()
    
    # 等高线填充
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z, 
        colorscale='Viridis', 
        contours=dict(coloring='fill'), 
        colorbar=dict(title=dict(text='$\\log(1+f)$', side='right'))
    ))
    
    # 全局最小值
    fig.add_trace(go.Scatter(
        x=[1], y=[1], 
        mode='markers', 
        marker=dict(color=APPLE_RED, size=15, symbol='star'),
        name='全局最小值 $(1,1)$'
    ))
    
    # 优化路径示意
    path_x = np.linspace(-1.5, 1, 20)
    path_y = path_x**2  # 沿抛物线接近
    fig.add_trace(go.Scatter(
        x=path_x, y=path_y, 
        mode='lines+markers', 
        line=dict(color=APPLE_ORANGE, width=2), 
        marker=dict(size=4),
        name='优化路径'
    ))
    
    fig.update_layout(
        title=dict(text='优化景观: Rosenbrock函数', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='$x$',
        yaxis_title='$y$',
        template='plotly_white',
        width=700, 
        height=600, 
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.02, y=0.98)
    )
    
    save_plotly_as_png(fig, 'optimization-landscape.png', width=700, height=600)


def plot_newton_vs_gradient():
    """图5: 牛顿法 vs 梯度下降 - 展示二阶信息的价值"""
    # 定义二次函数: f(x,y) = x^2 + xy + 1.5*y^2 - x - 2y
    # 黑塞矩阵 H = [[2, 1], [1, 3]]
    H = np.array([[2, 1], [1, 3]])
    b = np.array([1, 2])
    x_star = np.linalg.solve(H, b)
    
    x = np.linspace(-0.5, 1.5, 80)
    y = np.linspace(-0.5, 1.5, 80)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (2*X**2 + 2*X*Y + 3*Y**2) - X - 2*Y
    
    fig = go.Figure()
    
    # 等高线背景
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z, 
        colorscale='Viridis', 
        contours=dict(coloring='fill'), 
        showscale=False
    ))
    
    # 梯度下降路径
    x_gd, y_gd = [0], [0]
    lr, x_curr = 0.15, np.array([0.0, 0.0])
    for _ in range(15):
        grad = H @ x_curr - b
        x_curr = x_curr - lr * grad
        x_gd.append(x_curr[0])
        y_gd.append(x_curr[1])
    
    fig.add_trace(go.Scatter(
        x=x_gd, y=y_gd, 
        mode='lines+markers', 
        line=dict(color=APPLE_ORANGE, width=2), 
        marker=dict(size=6),
        name='梯度下降 (15步)'
    ))
    
    # 牛顿法 - 一步直达
    fig.add_trace(go.Scatter(
        x=[0, x_star[0]], y=[0, x_star[1]], 
        mode='lines+markers',
        line=dict(color=APPLE_RED, width=3, dash='dash'), 
        marker=dict(size=8),
        name='牛顿法 (1步)'
    ))
    
    # 最优解
    fig.add_trace(go.Scatter(
        x=[x_star[0]], y=[x_star[1]], 
        mode='markers',
        marker=dict(color=APPLE_GREEN, size=15, symbol='star'),
        name='最优解'
    ))
    
    fig.update_layout(
        title=dict(text='牛顿法 vs 梯度下降: 二阶信息的优势', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='$x$',
        yaxis_title='$y$',
        template='plotly_white',
        width=700, 
        height=600, 
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.02, y=0.98)
    )
    
    save_plotly_as_png(fig, 'newton-vs-gradient.png', width=700, height=600)


def plot_jacobian_chain_rule():
    """图6: 链式法则的可视化 - 复合函数的雅可比"""
    # 展示 f(g(x)) 的分解
    t = np.linspace(0, 2*np.pi, 100)
    
    # 内函数 g: R -> R^2 (螺旋)
    r = 1 + 0.3 * t
    g_x = r * np.cos(t)
    g_y = r * np.sin(t)
    
    # 外函数 f: R^2 -> R (高度函数)
    f_z = g_x**2 + g_y**2
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('内函数 $\\mathbf{g}(t)$: 平面螺旋', '复合函数 $(f \\circ \\mathbf{g})(t)$: 高度变化'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # 左图: 内函数 (平面螺旋)
    fig.add_trace(go.Scatter(
        x=g_x, y=g_y,
        mode='lines',
        line=dict(color=APPLE_BLUE, width=2),
        name='$\\mathbf{g}(t)$'
    ), row=1, col=1)
    
    # 标记起点和终点
    fig.add_trace(go.Scatter(
        x=[g_x[0]], y=[g_y[0]],
        mode='markers',
        marker=dict(size=10, color=APPLE_GREEN),
        name='起点'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[g_x[-1]], y=[g_y[-1]],
        mode='markers',
        marker=dict(size=10, color=APPLE_RED),
        name='终点'
    ), row=1, col=1)
    
    # 右图: 复合函数 (高度随参数变化)
    fig.add_trace(go.Scatter(
        x=t, y=f_z,
        mode='lines',
        line=dict(color=APPLE_ORANGE, width=2),
        name='$(f \\circ \\mathbf{g})(t)$'
    ), row=1, col=2)
    
    fig.update_layout(
        title=dict(text='链式法则: 复合函数 $\\frac{d(f \\circ \\mathbf{g})}{dt} = \\nabla f \\cdot \\mathbf{g}^{\\prime}(t)$', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white',
        width=1000,
        height=450,
        margin=dict(l=60, r=40, t=70, b=50),
        showlegend=True
    )
    
    fig.update_xaxes(title_text='$x$', row=1, col=1)
    fig.update_yaxes(title_text='$y$', row=1, col=1)
    fig.update_xaxes(title_text='$t$', row=1, col=2)
    fig.update_yaxes(title_text='$(f \\circ \\mathbf{g})(t)$', row=1, col=2)
    
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
    print(f"输出目录: {OUTPUT_DIR}")
