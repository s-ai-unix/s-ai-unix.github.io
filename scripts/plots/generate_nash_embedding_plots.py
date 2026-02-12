#!/usr/bin/env python3
"""
为纳什嵌入定理文章生成 Plotly 图形
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


def plot_gauss_theorema_egregium():
    """图1: 高斯绝妙定理 - 橘子皮无法展平"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('球面（正曲率）', '平面（零曲率）'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )
    
    # 左图: 球面
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi/2, 25)  # 半球
    U, V = np.meshgrid(u, v)
    R = 1
    X1 = R * np.sin(V) * np.cos(U)
    Y1 = R * np.sin(V) * np.sin(U)
    Z1 = R * np.cos(V)
    
    fig.add_trace(go.Surface(
        x=X1, y=Y1, z=Z1,
        colorscale='Reds',
        showscale=False,
        opacity=0.9,
        name='球面'
    ), row=1, col=1)
    
    # 添加"橘子皮"纹理线
    for i in range(0, len(u), 5):
        fig.add_trace(go.Scatter3d(
            x=X1[:, i], y=Y1[:, i], z=Z1[:, i],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.5)', width=1),
            showlegend=False
        ), row=1, col=1)
    
    # 右图: 尝试展平的平面（有撕裂）
    x2 = np.linspace(-2, 2, 50)
    y2 = np.linspace(-2, 2, 50)
    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = np.zeros_like(X2)
    
    # 添加"撕裂"效果 - 缝隙
    mask = (X2 > 0.5) | (X2 < -0.5) | (Y2 > 0.5) | (Y2 < -0.5)
    Z2_masked = np.where(mask, Z2, np.nan)
    
    fig.add_trace(go.Surface(
        x=X2, y=Y2, z=Z2_masked,
        colorscale='Blues',
        showscale=False,
        opacity=0.7,
        name='展平后的平面（有缝隙）'
    ), row=1, col=2)
    
    # 添加撕裂缝隙的标记
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-0.5, 0.5], z=[0.1, 0.1],
        mode='lines',
        line=dict(color=APPLE_RED, width=4),
        showlegend=False,
        name='撕裂处'
    ), row=1, col=2)
    
    fig.update_layout(
        title=dict(text="Gauss's Theorema Egregium: Curvature is Intrinsic", font=dict(size=16, color='#1d1d1f')),
        template='plotly_white',
        width=1000, height=500,
        margin=dict(l=60, r=40, t=70, b=50),
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            aspectmode='cube'
        ),
        scene2=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            aspectmode='cube'
        )
    )
    
    save_plotly_as_png(fig, 'nash-gauss-theorema.png', width=1000, height=500)


def plot_local_vs_global_embedding():
    """图2: 局部嵌入 vs 全局嵌入对比"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Local Embedding (Easy)', 'Global Embedding (Hard)'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # 左图: 局部嵌入 - 圆柱面
    theta = np.linspace(0, 2*np.pi, 100)
    z_cyl = np.linspace(0, 1, 50)
    Theta, Z_cyl = np.meshgrid(theta, z_cyl)
    R = 1
    X_cyl = R * np.cos(Theta)
    Y_cyl = R * np.sin(Theta)
    
    # 只显示一部分（局部）
    fig.add_trace(go.Scatter(
        x=X_cyl[:, :30].flatten(), y=Y_cyl[:, :30].flatten(),
        mode='markers',
        marker=dict(size=3, color=APPLE_BLUE, opacity=0.6),
        name='Local Patch'
    ), row=1, col=1)
    
    # 添加局部坐标框
    fig.add_shape(type="rect", x0=0.5, y0=-0.5, x1=1.5, y1=0.5,
                  line=dict(color=APPLE_RED, width=2), row=1, col=1)
    fig.add_annotation(x=1, y=0, text="Local patch<br>embeds easily", 
                       showarrow=False, font=dict(size=10), row=1, col=1)
    
    # 右图: 全局嵌入 - 尝试闭合环面但有问题
    # 克莱因瓶的示意（自相交）
    t = np.linspace(0, 2*np.pi, 200)
    # 8字形曲线（自相交示意）
    x_klein = np.sin(t) * (2 + np.cos(t/2))
    y_klein = np.cos(t) * (2 + np.cos(t/2))
    
    fig.add_trace(go.Scatter(
        x=x_klein, y=y_klein,
        mode='lines',
        line=dict(color=APPLE_ORANGE, width=2),
        name='Global embedding with self-intersection'
    ), row=1, col=2)
    
    # 标记自相交点
    fig.add_trace(go.Scatter(
        x=[0], y=[3],
        mode='markers',
        marker=dict(size=15, color=APPLE_RED, symbol='x'),
        name='Self-intersection'
    ), row=1, col=2)
    
    fig.add_annotation(x=0, y=3.5, text="Self-intersection!<br>Global embedding fails", 
                       showarrow=False, font=dict(size=10, color=APPLE_RED), row=1, col=2)
    
    fig.update_xaxes(title_text='x', row=1, col=1)
    fig.update_yaxes(title_text='y', row=1, col=1)
    fig.update_xaxes(title_text='x', row=1, col=2)
    fig.update_yaxes(title_text='y', row=1, col=2)
    
    fig.update_layout(
        title=dict(text='Local vs Global Embedding', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white',
        width=1000, height=450,
        margin=dict(l=60, r=40, t=70, b=50),
        showlegend=False
    )
    
    save_plotly_as_png(fig, 'nash-local-global-embedding.png', width=1000, height=450)


def plot_manifold_embedding():
    """图3: 流形嵌入到高维空间的示意图"""
    fig = go.Figure()
    
    # 2D 环面在 3D 中的嵌入
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, 2*np.pi, 40)
    U, V = np.meshgrid(u, v)
    
    R = 2  # 大半径
    r = 0.8  # 小半径
    
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        showscale=False,
        opacity=0.9,
        name='Torus embedded in R³'
    ))
    
    # 添加网格线表示坐标
    for i in range(0, len(u), 10):
        fig.add_trace(go.Scatter3d(
            x=X[:, i], y=Y[:, i], z=Z[:, i],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.3)', width=1),
            showlegend=False
        ))
    
    # 添加一个点及其切平面示意
    u0, v0 = np.pi/4, np.pi/4
    x0 = (R + r * np.cos(v0)) * np.cos(u0)
    y0 = (R + r * np.cos(v0)) * np.sin(u0)
    z0 = r * np.sin(v0)
    
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=10, color=APPLE_RED),
        name='Point p'
    ))
    
    # 添加标注
    fig.add_annotation(x=0.5, y=0.95, xref='paper', yref='paper',
                       text='2D Torus → R³<br>Nash: n-dim manifold → R^N',
                       showarrow=False, font=dict(size=12),
                       bgcolor='rgba(255,255,255,0.9)', bordercolor=APPLE_GRAY)
    
    fig.update_layout(
        title=dict(text='Manifold Embedding: 2D Torus in 3D Space', font=dict(size=16, color='#1d1d1f')),
        template='plotly_white',
        width=800, height=600,
        margin=dict(l=60, r=40, t=70, b=50),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='cube'
        )
    )
    
    save_plotly_as_png(fig, 'nash-manifold-embedding.png', width=800, height=600)


def plot_nash_iteration():
    """图4: 纳什迭代修正过程（误差收敛曲线）"""
    fig = go.Figure()
    
    # 模拟纳什迭代的二次收敛
    iterations = np.arange(0, 10)
    # 二次收敛：误差平方递减
    error = 1.0 * (0.5 ** (2 ** iterations))
    error = np.maximum(error, 1e-15)  # 防止下溢
    
    fig.add_trace(go.Scatter(
        x=iterations, y=error,
        mode='lines+markers',
        line=dict(color=APPLE_BLUE, width=3),
        marker=dict(size=10, color=APPLE_BLUE),
        name='Nash Iteration Error'
    ))
    
    # 添加线性收敛对比
    linear_error = 1.0 * (0.5 ** iterations)
    fig.add_trace(go.Scatter(
        x=iterations, y=linear_error,
        mode='lines',
        line=dict(color=APPLE_GRAY, width=2, dash='dash'),
        marker=dict(size=6),
        name='Linear Convergence (for comparison)'
    ))
    
    # 标注二次收敛
    fig.add_annotation(x=5, y=error[5]*10,
                       text='Quadratic convergence:<br>Error decreases as ε²',
                       showarrow=True, arrowhead=2,
                       font=dict(size=11))
    
    fig.update_layout(
        title=dict(text="Nash Iteration: Quadratic Convergence", font=dict(size=16, color='#1d1d1f')),
        xaxis_title='Iteration k',
        yaxis_title='Error ||E^(k)||',
        template='plotly_white',
        width=700, height=500,
        margin=dict(l=60, r=40, t=70, b=50),
        yaxis=dict(type='log', title='Error (log scale)'),
        legend=dict(x=0.65, y=0.95)
    )
    
    save_plotly_as_png(fig, 'nash-iteration-convergence.png', width=700, height=500)


def plot_dimension_comparison():
    """图5: 不同维度嵌入对比"""
    fig = go.Figure()
    
    n_values = np.arange(1, 11)
    
    # 不同嵌入的维数要求
    janet_cartan = n_values * (n_values + 1) / 2  # n(n+1)/2
    nash_compact = n_values * (3*n_values + 11) / 2  # n(3n+11)/2
    nash_noncompact = n_values * (n_values + 1) * (3*n_values + 11) / 2  # n(n+1)(3n+11)/2
    whitney = 2 * n_values  # 2n (拓扑嵌入)
    
    fig.add_trace(go.Scatter(
        x=n_values, y=janet_cartan,
        mode='lines+markers',
        line=dict(color=APPLE_GREEN, width=2),
        marker=dict(size=8),
        name='Janet-Cartan (analytic): n(n+1)/2'
    ))
    
    fig.add_trace(go.Scatter(
        x=n_values, y=nash_compact,
        mode='lines+markers',
        line=dict(color=APPLE_BLUE, width=3),
        marker=dict(size=10),
        name='Nash (compact): n(3n+11)/2'
    ))
    
    fig.add_trace(go.Scatter(
        x=n_values, y=nash_noncompact,
        mode='lines+markers',
        line=dict(color=APPLE_RED, width=2),
        marker=dict(size=8),
        name='Nash (non-compact): n(n+1)(3n+11)/2'
    ))
    
    fig.add_trace(go.Scatter(
        x=n_values, y=whitney,
        mode='lines',
        line=dict(color=APPLE_GRAY, width=2, dash='dash'),
        marker=dict(size=6),
        name='Whitney (topological only): 2n'
    ))
    
    # 标注 n=3 的点
    n = 3
    fig.add_annotation(x=n, y=n*(3*n+11)/2,
                       text=f'n=3 (spacetime)<br>Nash: {int(n*(3*n+11)/2)} dim',
                       showarrow=True, arrowhead=2, ay=-40,
                       font=dict(size=10))
    
    fig.update_layout(
        title=dict(text='Embedding Dimension Requirements', font=dict(size=16, color='#1d1d1f')),
        xaxis_title='Manifold dimension n',
        yaxis_title='Embedding dimension N',
        template='plotly_white',
        width=800, height=550,
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(x=0.02, y=0.98)
    )
    
    save_plotly_as_png(fig, 'nash-dimension-comparison.png', width=800, height=550)


if __name__ == '__main__':
    print("开始生成纳什嵌入定理文章的 Plotly 图形...")
    print("=" * 60)
    
    plot_gauss_theorema_egregium()
    plot_local_vs_global_embedding()
    plot_manifold_embedding()
    plot_nash_iteration()
    plot_dimension_comparison()
    
    print("=" * 60)
    print("所有图形生成完成！")
    print(f"输出目录: {OUTPUT_DIR}")
