"""
生成拓扑学相关的配图
用于文章：从拓扑到微分几何：系统掌握大学微分几何所需的拓扑学前置知识
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os

def save_and_compress(fig, filepath, width=900, height=600, scale=2):
    """保存并压缩图片"""
    # 先保存
    fig.write_image(filepath, width=width, height=height, scale=scale)
    
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")


def plot_mobius_strip():
    """绘制莫比乌斯带 - 展示单侧曲面的拓扑特性"""
    # 莫比乌斯带参数方程
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(-0.3, 0.3, 30)
    U, V = np.meshgrid(u, v)
    
    # 莫比乌斯带的标准参数化
    X = (1 + V * np.cos(U/2)) * np.cos(U)
    Y = (1 + V * np.cos(U/2)) * np.sin(U)
    Z = V * np.sin(U/2)
    
    # 颜色映射：根据位置编码
    colors = U
    
    fig = go.Figure()
    
    # 添加表面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Blues',
        showscale=False,
        lighting=dict(ambient=0.7, diffuse=0.7, specular=0.5, roughness=0.5),
        name='莫比乌斯带'
    ))
    
    # 添加中心曲线
    u_center = np.linspace(0, 2*np.pi, 200)
    x_center = np.cos(u_center)
    y_center = np.sin(u_center)
    z_center = np.zeros_like(u_center)
    
    fig.add_trace(go.Scatter3d(
        x=x_center, y=y_center, z=z_center,
        mode='lines',
        line=dict(color='#FF3B30', width=4),
        name='中心曲线'
    ))
    
    # 添加箭头表示方向
    arrow_idx = [0, 50, 100, 150]
    for idx in arrow_idx:
        fig.add_trace(go.Scatter3d(
            x=[x_center[idx]], y=[y_center[idx]], z=[z_center[idx]],
            mode='markers',
            marker=dict(size=4, color='#FF9500'),
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(text='莫比乌斯带：单侧不可定向曲面', font=dict(size=16)),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        template='plotly_white',
        showlegend=True,
        width=800, height=600
    )
    
    return fig


def plot_klein_bottle():
    """绘制克莱因瓶 - 展示不可定向闭曲面的拓扑特性"""
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, 2*np.pi, 80)
    U, V = np.meshgrid(u, v)
    
    # 克莱因瓶的参数方程（"8字形"浸入版本）
    a = 2
    n = 2
    
    X = (a + np.cos(U/2)*np.sin(V) - np.sin(U/2)*np.sin(2*V)) * np.cos(U)
    Y = (a + np.cos(U/2)*np.sin(V) - np.sin(U/2)*np.sin(2*V)) * np.sin(U)
    Z = np.sin(U/2)*np.sin(V) + np.cos(U/2)*np.sin(2*V)
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        showscale=False,
        lighting=dict(ambient=0.7, diffuse=0.7, specular=0.3, roughness=0.5),
        name='克莱因瓶'
    ))
    
    fig.update_layout(
        title=dict(text='克莱因瓶：无边界不可定向闭曲面', font=dict(size=16)),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            camera=dict(eye=dict(x=2, y=2, z=1.5))
        ),
        template='plotly_white',
        width=800, height=600
    )
    
    return fig


def plot_topological_open_sets():
    """绘制拓扑空间中的开集概念示意图"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('拓扑空间中的开集', '开集的基本性质'),
        horizontal_spacing=0.15
    )
    
    # 左图：拓扑空间示意
    # 绘制包含空间 X
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 外边界（空间X）
    fig.add_trace(go.Scatter(
        x=3*np.cos(theta), y=3*np.sin(theta),
        mode='lines', line=dict(color='#8E8E93', width=2, dash='dash'),
        fill='toself', fillcolor='rgba(142, 142, 147, 0.1)',
        name='空间 X', showlegend=False
    ), row=1, col=1)
    
    # 开集 U
    fig.add_trace(go.Scatter(
        x=1.5*np.cos(theta) - 0.5, y=1.5*np.sin(theta) + 0.5,
        mode='lines', line=dict(color='#007AFF', width=3),
        fill='toself', fillcolor='rgba(0, 122, 255, 0.3)',
        name='开集 U', showlegend=False
    ), row=1, col=1)
    
    # 开集 V
    fig.add_trace(go.Scatter(
        x=1.2*np.cos(theta) + 1.2, y=1.2*np.sin(theta) - 0.8,
        mode='lines', line=dict(color='#34C759', width=3),
        fill='toself', fillcolor='rgba(52, 199, 89, 0.3)',
        name='开集 V', showlegend=False
    ), row=1, col=1)
    
    # 交点标记
    fig.add_trace(go.Scatter(
        x=[0.8], y=[0.2],
        mode='markers+text',
        marker=dict(size=8, color='#FF9500'),
        text=['$p$'], textposition='top right',
        textfont=dict(size=14),
        showlegend=False
    ), row=1, col=1)
    
    # 右图：开集性质
    # 任意并
    fig.add_trace(go.Scatter(
        x=2*np.cos(theta) - 1.5, y=2*np.sin(theta),
        mode='lines', line=dict(color='#007AFF', width=2),
        fill='toself', fillcolor='rgba(0, 122, 255, 0.2)',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=1.5*np.cos(theta) + 0.5, y=1.5*np.sin(theta) - 0.5,
        mode='lines', line=dict(color='#34C759', width=2),
        fill='toself', fillcolor='rgba(52, 199, 89, 0.2)',
        showlegend=False
    ), row=1, col=2)
    
    # 有限交
    fig.add_trace(go.Scatter(
        x=1.2*np.cos(theta), y=1.2*np.sin(theta) - 0.2,
        mode='lines', line=dict(color='#FF9500', width=2),
        fill='toself', fillcolor='rgba(255, 149, 0, 0.3)',
        showlegend=False
    ), row=1, col=2)
    
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, range=[-4, 4])
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, range=[-4, 4])
    
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        width=900, height=450
    )
    
    return fig


def plot_continuity_concept():
    """绘制连续性概念的拓扑解释"""
    x = np.linspace(-2, 2, 400)
    y_continuous = np.sin(x) + 0.3*x
    
    # 创建间断函数
    x_left = np.linspace(-2, -0.1, 100)
    x_right = np.linspace(0.1, 2, 100)
    y_left = x_left**2 / 2
    y_right = x_right**2 / 2 + 0.8  # 跳跃间断
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('连续函数', '间断函数（跳跃间断）'),
        horizontal_spacing=0.1
    )
    
    # 左图：连续函数
    fig.add_trace(go.Scatter(
        x=x, y=y_continuous,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='连续函数'
    ), row=1, col=1)
    
    # 添加 epsilon-delta 解释
    x0, y0 = 0.5, np.sin(0.5) + 0.3*0.5
    epsilon = 0.4
    delta = 0.3
    
    # epsilon 带
    fig.add_hrect(y0=y0-epsilon, y1=y0+epsilon, 
                  fillcolor='rgba(255, 149, 0, 0.2)', 
                  line_width=0, row=1, col=1)
    # delta 带
    fig.add_vrect(x0=x0-delta, x1=x0+delta, 
                  fillcolor='rgba(0, 122, 255, 0.15)', 
                  line_width=0, row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[x0], y=[y0],
        mode='markers',
        marker=dict(size=10, color='#FF3B30', symbol='diamond'),
        showlegend=False
    ), row=1, col=1)
    
    # 右图：间断函数
    fig.add_trace(go.Scatter(
        x=x_left, y=y_left,
        mode='lines',
        line=dict(color='#FF3B30', width=3),
        name='左极限'
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=x_right, y=y_right,
        mode='lines',
        line=dict(color='#FF9500', width=3),
        name='右极限'
    ), row=1, col=2)
    
    # 标记间断点
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=10, color='#FF3B30', symbol='circle-open'),
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[0], y=[0.8],
        mode='markers',
        marker=dict(size=10, color='#FF9500', symbol='circle'),
        showlegend=False
    ), row=1, col=2)
    
    # 添加跳跃标记
    fig.add_annotation(
        x=0, y=0.4, ax=0, ay=0.4,
        xref='x2', yref='y2',
        axref='x2', ayref='y2',
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowcolor='#8E8E93'
    )
    
    fig.update_xaxes(title_text='$x$', row=1, col=1)
    fig.update_xaxes(title_text='$x$', row=1, col=2)
    fig.update_yaxes(title_text='$f(x)$', row=1, col=1)
    fig.update_yaxes(title_text='$f(x)$', row=1, col=2)
    
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        width=900, height=450
    )
    
    return fig


def plot_homeomorphism():
    """绘制同胚映射示意图"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('空间 X', '空间 Y'),
        horizontal_spacing=0.1
    )
    
    # 左图：空间X（圆盘）
    theta = np.linspace(0, 2*np.pi, 100)
    r = 2
    
    fig.add_trace(go.Scatter(
        x=r*np.cos(theta), y=r*np.sin(theta),
        mode='lines', line=dict(color='#007AFF', width=3),
        fill='toself', fillcolor='rgba(0, 122, 255, 0.15)',
        name='空间 X'
    ), row=1, col=1)
    
    # 内部点
    np.random.seed(42)
    n_points = 20
    for _ in range(n_points):
        r_pt = np.random.uniform(0, 1.5)
        theta_pt = np.random.uniform(0, 2*np.pi)
        x_pt, y_pt = r_pt*np.cos(theta_pt), r_pt*np.sin(theta_pt)
        fig.add_trace(go.Scatter(
            x=[x_pt], y=[y_pt],
            mode='markers',
            marker=dict(size=5, color='#34C759'),
            showlegend=False
        ), row=1, col=1)
    
    # 右图：空间Y（正方形 - 通过同胚映射得到）
    # 正方形的边界
    square_x = [-2, 2, 2, -2, -2]
    square_y = [-2, -2, 2, 2, -2]
    
    fig.add_trace(go.Scatter(
        x=square_x, y=square_y,
        mode='lines', line=dict(color='#FF9500', width=3),
        fill='toself', fillcolor='rgba(255, 149, 0, 0.15)',
        name='空间 Y'
    ), row=1, col=2)
    
    # 对应的点（映射后）
    np.random.seed(42)
    for _ in range(n_points):
        r_pt = np.random.uniform(0, 1.5)
        theta_pt = np.random.uniform(0, 2*np.pi)
        # 将圆映射到正方形的同胚
        x_pt = r_pt * np.cos(theta_pt) * np.sqrt(2) / np.maximum(np.abs(np.cos(theta_pt)), np.abs(np.sin(theta_pt)))
        y_pt = r_pt * np.sin(theta_pt) * np.sqrt(2) / np.maximum(np.abs(np.cos(theta_pt)), np.abs(np.sin(theta_pt)))
        # 缩放
        x_pt *= 0.8
        y_pt *= 0.8
        fig.add_trace(go.Scatter(
            x=[x_pt], y=[y_pt],
            mode='markers',
            marker=dict(size=5, color='#34C759'),
            showlegend=False
        ), row=1, col=2)
    
    fig.update_xaxes(showgrid=False, zeroline=False, range=[-3, 3])
    fig.update_yaxes(showgrid=False, zeroline=False, range=[-3, 3])
    
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        width=800, height=450,
        title=dict(text='同胚映射：圆盘与正方形拓扑等价', font=dict(size=16))
    )
    
    return fig


def plot_compactness():
    """绘制紧致性概念示意图"""
    theta = np.linspace(0, 2*np.pi, 100)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('紧致：闭区间 [a,b]', '非紧致：开区间 (a,b)'),
        horizontal_spacing=0.1
    )
    
    # 左图：紧致（闭区间）
    x_closed = np.linspace(0, 4, 200)
    fig.add_trace(go.Scatter(
        x=x_closed, y=np.zeros_like(x_closed),
        mode='lines',
        line=dict(color='#007AFF', width=4),
        name='闭区间 [0,4]'
    ), row=1, col=1)
    
    # 端点
    fig.add_trace(go.Scatter(
        x=[0, 4], y=[0, 0],
        mode='markers',
        marker=dict(size=12, color='#007AFF', symbol='circle'),
        showlegend=False
    ), row=1, col=1)
    
    # 有限子覆盖示意
    cover_colors = ['#FF9500', '#34C759', '#AF52DE']
    intervals = [(0, 1.5), (1, 3), (2.5, 4)]
    for (a, b), color in zip(intervals, cover_colors):
        fig.add_trace(go.Scatter(
            x=[a, b], y=[0.15, 0.15],
            mode='lines',
            line=dict(color=color, width=8),
            opacity=0.5,
            showlegend=False
        ), row=1, col=1)
    
    fig.add_annotation(x=2, y=0.5, text='有限子覆盖存在', showarrow=False, row=1, col=1)
    
    # 右图：非紧致（开区间）
    x_open = np.linspace(0.1, 3.9, 200)
    fig.add_trace(go.Scatter(
        x=x_open, y=np.zeros_like(x_open),
        mode='lines',
        line=dict(color='#FF3B30', width=4),
        name='开区间 (0,4)'
    ), row=1, col=2)
    
    # 端点（空心）
    fig.add_trace(go.Scatter(
        x=[0, 4], y=[0, 0],
        mode='markers',
        marker=dict(size=12, color='#FF3B30', symbol='circle-open'),
        showlegend=False
    ), row=1, col=2)
    
    # 无限覆盖示意
    n_covers = 8
    for i in range(n_covers):
        a = 0.5 + i * 0.4
        b = a + 0.6
        if b < 4:
            fig.add_trace(go.Scatter(
                x=[a, b], y=[0.15, 0.15],
                mode='lines',
                line=dict(color='#FF9500', width=6),
                opacity=0.3,
                showlegend=False
            ), row=1, col=2)
    
    fig.add_annotation(x=2, y=0.5, text='不存在有限子覆盖', showarrow=False, row=1, col=2)
    
    fig.update_xaxes(showgrid=False, zeroline=False, range=[-0.5, 4.5])
    fig.update_yaxes(showgrid=False, zeroline=False, range=[-0.3, 0.8], visible=False)
    
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        width=900, height=400,
        title=dict(text='紧致性：从开覆盖角度理解', font=dict(size=16))
    )
    
    return fig


def plot_manifold_chart():
    """绘制流形的坐标卡概念"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('流形 M 的局部坐标卡', '坐标变换'),
        horizontal_spacing=0.15
    )
    
    # 左图：流形与坐标卡
    # 绘制一个环面（torus）的轮廓
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    U, V = np.meshgrid(u, v)
    
    R, r = 3, 1
    X = (R + r*np.cos(V)) * np.cos(U)
    Y = (R + r*np.cos(V)) * np.sin(U)
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.zeros_like(X),
        colorscale='Blues',
        showscale=False,
        opacity=0.3
    ), row=1, col=1)
    
    # 绘制局部坐标区域
    u_local = np.linspace(np.pi/4, 3*np.pi/4, 30)
    v_local = np.linspace(np.pi/4, 3*np.pi/4, 30)
    U_local, V_local = np.meshgrid(u_local, v_local)
    X_local = (R + r*np.cos(V_local)) * np.cos(U_local)
    Y_local = (R + r*np.cos(V_local)) * np.sin(U_local)
    
    fig.add_trace(go.Surface(
        x=X_local, y=Y_local, z=np.zeros_like(X_local) + 0.1,
        colorscale='Oranges',
        showscale=False,
        opacity=0.8
    ), row=1, col=1)
    
    # 右图：坐标变换
    # 欧氏空间中的映射
    x_euclid = np.linspace(-2, 2, 100)
    y_euclid = np.linspace(-2, 2, 100)
    X_e, Y_e = np.meshgrid(x_euclid, y_euclid)
    
    # 绘制欧氏开球
    mask = X_e**2 + Y_e**2 < 3
    Z_e = np.where(mask, 1, np.nan)
    
    fig.add_trace(go.Surface(
        x=X_e, y=Y_e, z=Z_e,
        colorscale='Greens',
        showscale=False,
        opacity=0.5
    ), row=1, col=2)
    
    # 坐标变换箭头
    fig.add_annotation(
        x=1.5, y=1.5, ax=-1.5, ay=-1.5,
        xref='x2', yref='y2',
        axref='x2', ayref='y2',
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowcolor='#007AFF',
        text='φ', font=dict(size=16)
    )
    
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        width=900, height=500,
        title=dict(text='流形的坐标卡：局部同胚于欧氏空间', font=dict(size=16))
    )
    
    return fig


def plot_connectedness():
    """绘制连通性示意图"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('连通空间', '道路连通', '不连通空间'),
        horizontal_spacing=0.1
    )
    
    # 左图：连通空间
    theta = np.linspace(0, 2*np.pi, 100)
    # 甜甜圈形状
    x_donut = 2*np.cos(theta)
    y_donut = 2*np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_donut, y=y_donut,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        fill='toself', fillcolor='rgba(0, 122, 255, 0.2)',
        showlegend=False
    ), row=1, col=1)
    
    # 绘制一条连接任意两点的曲线
    t = np.linspace(0, 1, 50)
    x_path = 1.5*np.cos(t*np.pi) - 0.5
    y_path = 1.5*np.sin(t*np.pi)
    fig.add_trace(go.Scatter(
        x=x_path, y=y_path,
        mode='lines',
        line=dict(color='#FF9500', width=2, dash='dash'),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[1.5, -0.5], y=[0, 0],
        mode='markers',
        marker=dict(size=8, color='#FF3B30'),
        showlegend=False
    ), row=1, col=1)
    
    # 中图：道路连通
    # 绘制从A到B的连续路径
    t = np.linspace(0, 1, 100)
    x_path = 2*t - 1
    y_path = 0.5*np.sin(3*np.pi*t)
    
    fig.add_trace(go.Scatter(
        x=x_path, y=y_path,
        mode='lines',
        line=dict(color='#34C759', width=3),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[-1], y=[0],
        mode='markers+text',
        marker=dict(size=12, color='#007AFF'),
        text=['A'], textposition='bottom center',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[1], y=[0],
        mode='markers+text',
        marker=dict(size=12, color='#007AFF'),
        text=['B'], textposition='bottom center',
        showlegend=False
    ), row=1, col=2)
    
    # 右图：不连通空间
    # 两个分离的圆
    circle1_x = 0.8*np.cos(theta) - 1.2
    circle1_y = 0.8*np.sin(theta)
    circle2_x = 0.8*np.cos(theta) + 1.2
    circle2_y = 0.8*np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=circle1_x, y=circle1_y,
        mode='lines',
        line=dict(color='#FF3B30', width=3),
        fill='toself', fillcolor='rgba(255, 59, 48, 0.2)',
        showlegend=False
    ), row=1, col=3)
    
    fig.add_trace(go.Scatter(
        x=circle2_x, y=circle2_y,
        mode='lines',
        line=dict(color='#FF3B30', width=3),
        fill='toself', fillcolor='rgba(255, 59, 48, 0.2)',
        showlegend=False
    ), row=1, col=3)
    
    # 标记分离
    fig.add_annotation(
        x=0, y=0, ax=0, ay=0,
        xref='x3', yref='y3',
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowcolor='#8E8E93',
        arrowwidth=2
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False, range=[-3, 3], visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, range=[-2, 2], visible=False)
    
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        width=1000, height=350,
        title=dict(text='连通性的不同层次', font=dict(size=16))
    )
    
    return fig


def main():
    """生成所有配图"""
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成拓扑学配图...")
    
    # 1. 莫比乌斯带
    print("1. 生成莫比乌斯带图...")
    fig1 = plot_mobius_strip()
    save_and_compress(fig1, f'{output_dir}/mobius-strip.png', width=800, height=600)
    
    # 2. 克莱因瓶
    print("2. 生成克莱因瓶图...")
    fig2 = plot_klein_bottle()
    save_and_compress(fig2, f'{output_dir}/klein-bottle.png', width=800, height=600)
    
    # 3. 开集概念
    print("3. 生成开集概念图...")
    fig3 = plot_topological_open_sets()
    save_and_compress(fig3, f'{output_dir}/topological-open-sets.png', width=900, height=450)
    
    # 4. 连续性概念
    print("4. 生成连续性概念图...")
    fig4 = plot_continuity_concept()
    save_and_compress(fig4, f'{output_dir}/continuity-concept.png', width=900, height=450)
    
    # 5. 同胚映射
    print("5. 生成同胚映射图...")
    fig5 = plot_homeomorphism()
    save_and_compress(fig5, f'{output_dir}/homeomorphism.png', width=800, height=450)
    
    # 6. 紧致性
    print("6. 生成紧致性概念图...")
    fig6 = plot_compactness()
    save_and_compress(fig6, f'{output_dir}/compactness.png', width=900, height=400)
    
    # 7. 流形坐标卡
    print("7. 生成流形坐标卡图...")
    fig7 = plot_manifold_chart()
    save_and_compress(fig7, f'{output_dir}/manifold-chart.png', width=900, height=500)
    
    # 8. 连通性
    print("8. 生成连通性图...")
    fig8 = plot_connectedness()
    save_and_compress(fig8, f'{output_dir}/connectedness.png', width=1000, height=350)
    
    print("\n✅ 所有配图生成完成！")


if __name__ == '__main__':
    main()
