#!/usr/bin/env python3
"""
生成微分几何前序知识综述文章的配图
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os

# 设置中文字体支持
import plotly.io as pio

OUTPUT_DIR = "static/images/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_and_compress(fig, filepath, width=900, height=600):
    """保存并压缩图片"""
    fig.write_image(filepath, width=width, height=height, scale=2)
    
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")


def plot_calculus_foundations():
    """图1：微积分基础概念的演变"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('极限概念：无穷小的量化', '导数：变化率的精确描述', 
                       '积分：无穷小量的累积', '微积分基本定理'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 左上图：极限概念
    x = np.linspace(0.1, 2, 200)
    y = np.sin(x) / x
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='sin(x)/x'
    ), row=1, col=1)
    
    # 添加趋近箭头和标记
    fig.add_annotation(
        x=0.3, y=0.98,
        ax=0.1, ay=0.98,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowcolor='#FF3B30',
        row=1, col=1
    )
    
    fig.add_trace(go.Scatter(
        x=[0], y=[1],
        mode='markers',
        marker=dict(size=12, color='#34C759', symbol='diamond'),
        name='极限值',
        showlegend=False
    ), row=1, col=1)
    
    # 右上图：导数概念
    x = np.linspace(-2, 2, 200)
    y = x**2
    tangent_x = np.linspace(0.5, 1.5, 50)
    tangent_y = 2 * 1 * (tangent_x - 1) + 1
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='f(x)=x²',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=tangent_x, y=tangent_y,
        mode='lines',
        line=dict(color='#FF9500', width=2, dash='dash'),
        name='切线',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[1], y=[1],
        mode='markers',
        marker=dict(size=10, color='#FF3B30'),
        name='切点',
        showlegend=False
    ), row=1, col=2)
    
    # 左下图：积分概念
    x = np.linspace(0, 3, 200)
    y = x**2
    x_fill = np.linspace(0, 2, 100)
    y_fill = x_fill**2
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='f(x)=x²',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([[0], x_fill, [2]]),
        y=np.concatenate([[0], y_fill, [0]]),
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        line=dict(width=0),
        showlegend=False
    ), row=2, col=1)
    
    # 添加矩形条
    for i in range(5):
        xi = 0.4 * i
        yi = (0.4 * i)**2
        fig.add_shape(
            type='rect',
            x0=xi, x1=xi+0.4,
            y0=0, y1=yi,
            fillcolor='rgba(255, 149, 0, 0.3)',
            line=dict(width=0),
            row=2, col=1
        )
    
    # 右下图：微积分基本定理
    x = np.linspace(0, 3, 200)
    f = x**2
    F = x**3 / 3
    
    fig.add_trace(go.Scatter(
        x=x, y=f,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name="f(x)=x²",
        showlegend=False
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=x, y=F,
        mode='lines',
        line=dict(color='#34C759', width=2),
        name="F(x)=x³/3",
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        showlegend=False,
        width=900,
        height=700,
        title=dict(
            text='微积分四大支柱',
            font=dict(size=18)
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/calculus_foundations.png')
    return fig


def plot_multivariable_calculus():
    """图2：多元微积分核心概念"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('偏导数：沿坐标轴的变化率', '梯度：最速上升方向', 
                       '方向导数：任意方向的变化率'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]]
    )
    
    # 创建曲面数据
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # 抛物面
    
    # 左图：偏导数示意
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Blues',
        showscale=False,
        opacity=0.8
    ), row=1, col=1)
    
    # 添加沿x方向的切线
    x_tangent = np.linspace(-1.5, 1.5, 20)
    y_tangent = np.zeros_like(x_tangent)
    z_tangent = 2 * 1 * (x_tangent - 1) + 1  # 在(1,0)处沿x方向的切线
    
    fig.add_trace(go.Scatter3d(
        x=x_tangent, y=y_tangent, z=z_tangent,
        mode='lines',
        line=dict(color='#FF3B30', width=4),
        showlegend=False
    ), row=1, col=1)
    
    # 中图：梯度示意
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Greens',
        showscale=False,
        opacity=0.8
    ), row=1, col=2)
    
    # 添加梯度向量
    fig.add_trace(go.Cone(
        x=[1], y=[0.5], z=[1.25],
        u=[2], v=[1], w=[-1],
        colorscale='Reds',
        showscale=False,
        sizemode='absolute',
        sizeref=1
    ), row=1, col=2)
    
    # 右图：方向导数
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Oranges',
        showscale=False,
        opacity=0.8
    ), row=1, col=3)
    
    # 添加不同方向的切线
    theta = np.pi/4
    t = np.linspace(-1, 1, 20)
    x_dir = 1 + t * np.cos(theta)
    y_dir = 0.5 + t * np.sin(theta)
    z_dir = 1.25 + t * (2*np.cos(theta) + np.sin(theta))
    
    fig.add_trace(go.Scatter3d(
        x=x_dir, y=y_dir, z=z_dir,
        mode='lines',
        line=dict(color='#FF9500', width=4),
        showlegend=False
    ), row=1, col=3)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1200,
        height=500,
        title=dict(
            text='多元微积分的三个核心概念',
            font=dict(size=16)
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/multivariable_calculus.png', width=1200, height=500)
    return fig


def plot_linear_algebra_foundations():
    """图3：线性代数基础概念"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('向量空间：n维空间的结构', '线性变换：保持结构的映射',
                       '特征分解：找到不变的方向', '内积空间：度量的引入'),
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # 左上：向量空间（3D示意）
    # 基向量
    fig.add_trace(go.Scatter3d(
        x=[0, 3], y=[0, 0], z=[0, 0],
        mode='lines+text',
        line=dict(color='#007AFF', width=4),
        text=['', 'e₁'],
        textposition='top center',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 3], z=[0, 0],
        mode='lines+text',
        line=dict(color='#34C759', width=4),
        text=['', 'e₂'],
        textposition='top center',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, 3],
        mode='lines+text',
        line=dict(color='#FF9500', width=4),
        text=['', 'e₃'],
        textposition='top center',
        showlegend=False
    ), row=1, col=1)
    
    # 任意向量
    fig.add_trace(go.Scatter3d(
        x=[0, 2], y=[0, 2.5], z=[0, 1.5],
        mode='lines+markers+text',
        line=dict(color='#FF3B30', width=4),
        marker=dict(size=6),
        text=['', 'v'],
        textposition='top center',
        showlegend=False
    ), row=1, col=1)
    
    # 右上：线性变换
    # 变换前的网格
    for i in range(-2, 3):
        fig.add_trace(go.Scatter3d(
            x=[-2, 2], y=[i, i], z=[0, 0],
            mode='lines',
            line=dict(color='rgba(0, 122, 255, 0.3)', width=1),
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter3d(
            x=[i, i], y=[-2, 2], z=[0, 0],
            mode='lines',
            line=dict(color='rgba(0, 122, 255, 0.3)', width=1),
            showlegend=False
        ), row=1, col=2)
    
    # 变换后的网格（剪切变换示例）
    for i in range(-2, 3):
        fig.add_trace(go.Scatter3d(
            x=[-2+i*0.5, 2+i*0.5], y=[i, i], z=[0.5, 0.5],
            mode='lines',
            line=dict(color='rgba(255, 59, 48, 0.6)', width=2),
            showlegend=False
        ), row=1, col=2)
    
    # 左下：特征分解
    theta = np.linspace(0, 2*np.pi, 100)
    # 单位圆
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    # 变换后的椭圆（对角化后的效果）
    x_ellipse = 3 * np.cos(theta)
    y_ellipse = np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='单位圆',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_ellipse, y=y_ellipse,
        mode='lines',
        line=dict(color='#FF3B30', width=2),
        name='变换后',
        showlegend=False
    ), row=2, col=1)
    
    # 特征向量方向
    fig.add_trace(go.Scatter(
        x=[-3.5, 3.5], y=[0, 0],
        mode='lines',
        line=dict(color='#34C759', width=3, dash='dash'),
        name='特征方向1',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-1.5, 1.5],
        mode='lines',
        line=dict(color='#34C759', width=3, dash='dash'),
        name='特征方向2',
        showlegend=False
    ), row=2, col=1)
    
    # 右下：内积空间
    # 正交投影示意
    fig.add_trace(go.Scatter(
        x=[0, 3], y=[0, 2],
        mode='lines+markers',
        line=dict(color='#007AFF', width=3),
        marker=dict(size=8),
        name='向量v',
        showlegend=False
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=[0, 3], y=[0, 0],
        mode='lines+markers',
        line=dict(color='#FF9500', width=3),
        marker=dict(size=8),
        name='投影',
        showlegend=False
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=[3, 3], y=[0, 2],
        mode='lines',
        line=dict(color='#FF3B30', width=2, dash='dash'),
        name='正交分量',
        showlegend=False
    ), row=2, col=2)
    
    # 直角标记
    fig.add_trace(go.Scatter(
        x=[2.7], y=[0.3],
        mode='markers',
        marker=dict(size=6, color='#34C759', symbol='square'),
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='线性代数的四大核心概念',
            font=dict(size=18)
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/linear_algebra_foundations.png', width=1000, height=800)
    return fig


def plot_differential_equations():
    """图4：微分方程基础"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('常微分方程：单变量变化规律', '解的演化：初值条件的影响',
                       '偏微分方程：多变量耦合', '特征线法：追踪信息传播'),
        vertical_spacing=0.15
    )
    
    # 左上：ODE示例 - 指数增长/衰减
    t = np.linspace(0, 5, 200)
    y_growth = np.exp(0.5 * t)
    y_decay = np.exp(-t)
    y_oscillate = np.sin(2*t) * np.exp(-0.3*t)
    
    fig.add_trace(go.Scatter(
        x=t, y=y_growth,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='增长'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=t, y=y_decay,
        mode='lines',
        line=dict(color='#FF3B30', width=2),
        name='衰减'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=t, y=y_oscillate,
        mode='lines',
        line=dict(color='#34C759', width=2),
        name='阻尼振荡'
    ), row=1, col=1)
    
    # 右上：不同初值的解
    for y0 in [0.5, 1.0, 1.5, 2.0]:
        y = y0 * np.exp(-t)
        fig.add_trace(go.Scatter(
            x=t, y=y,
            mode='lines',
            line=dict(width=2),
            name=f'y(0)={y0}',
            showlegend=False
        ), row=1, col=2)
    
    fig.add_annotation(
        x=0, y=2,
        text='初值决定<br>唯一解',
        showarrow=True,
        arrowhead=2,
        arrowcolor='#FF9500',
        row=1, col=2
    )
    
    # 左下：波动方程示意
    x = np.linspace(0, 10, 100)
    for t_val in [0, 0.5, 1.0, 1.5]:
        y = np.sin(x - 2*t_val)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(width=2),
            name=f't={t_val}',
            showlegend=False
        ), row=2, col=1)
    
    # 右下：特征线
    x = np.linspace(0, 5, 50)
    for c in np.linspace(-2, 2, 7):
        y = x + c
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='#007AFF', width=1.5),
            showlegend=False
        ), row=2, col=2)
    
    # 添加方向箭头
    fig.add_annotation(
        x=3, y=3,
        ax=2.5, ay=2.5,
        xref='x4', yref='y4',
        axref='x4', ayref='y4',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowcolor='#FF3B30'
    )
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=800,
        title=dict(
            text='微分方程：描述变化的数学语言',
            font=dict(size=18)
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/differential_equations.png', width=1000, height=800)
    return fig


def plot_analytic_geometry():
    """图5：解析几何基础"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('参数曲线：运动的轨迹', '参数曲面：二维流形',
                       '曲率：弯曲程度的度量', 'Frenet标架：局部坐标系'),
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'xy'}, {'type': 'scene'}]]
    )
    
    # 左上：参数曲线（螺旋线）
    t = np.linspace(0, 4*np.pi, 200)
    x = np.cos(t)
    y = np.sin(t)
    z = 0.2 * t
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='#007AFF', width=4),
        showlegend=False
    ), row=1, col=1)
    
    # 添加切向量
    for i in [50, 100, 150]:
        fig.add_trace(go.Cone(
            x=[x[i]], y=[y[i]], z=[z[i]],
            u=[-np.sin(t[i])], v=[np.cos(t[i])], w=[0.2],
            colorscale='Reds',
            showscale=False,
            sizemode='absolute',
            sizeref=0.3
        ), row=1, col=1)
    
    # 右上：参数曲面（环面）
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, 2*np.pi, 50)
    U, V = np.meshgrid(u, v)
    R, r = 3, 1
    X = (R + r*np.cos(V)) * np.cos(U)
    Y = (R + r*np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Greens',
        showscale=False,
        opacity=0.9
    ), row=1, col=2)
    
    # 左下：曲率示意
    theta = np.linspace(0, 2*np.pi, 200)
    
    # 圆（恒定曲率）
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='圆（恒定曲率）',
        showlegend=False
    ), row=2, col=1)
    
    # 椭圆（变化曲率）
    a, b = 2, 1
    x_ellipse = a * np.cos(theta)
    y_ellipse = b * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_ellipse, y=y_ellipse,
        mode='lines',
        line=dict(color='#FF9500', width=2),
        name='椭圆（变化曲率）',
        showlegend=False
    ), row=2, col=1)
    
    # 标记曲率最大和最小点
    fig.add_trace(go.Scatter(
        x=[-2, 2], y=[0, 0],
        mode='markers',
        marker=dict(size=12, color='#FF3B30', symbol='diamond'),
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-1, 1],
        mode='markers',
        marker=dict(size=12, color='#34C759', symbol='circle'),
        showlegend=False
    ), row=2, col=1)
    
    # 右下：Frenet标架
    t = np.linspace(0, 2*np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros_like(t)
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        showlegend=False
    ), row=2, col=2)
    
    # 在特定点绘制Frenet标架
    for i in [25, 50, 75]:
        # 切向量T
        fig.add_trace(go.Cone(
            x=[x[i]], y=[y[i]], z=[z[i]],
            u=[-np.sin(t[i])], v=[np.cos(t[i])], w=[0],
            colorscale='Reds',
            showscale=False,
            sizemode='absolute',
            sizeref=0.3
        ), row=2, col=2)
        
        # 法向量N
        fig.add_trace(go.Cone(
            x=[x[i]], y=[y[i]], z=[z[i]],
            u=[-np.cos(t[i])], v=[-np.sin(t[i])], w=[0],
            colorscale='Blues',
            showscale=False,
            sizemode='absolute',
            sizeref=0.3
        ), row=2, col=2)
        
        # 副法向量B
        fig.add_trace(go.Cone(
            x=[x[i]], y=[y[i]], z=[z[i]],
            u=[0], v=[0], w=[1],
            colorscale='Greens',
            showscale=False,
            sizemode='absolute',
            sizeref=0.3
        ), row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=900,
        title=dict(
            text='解析几何：从曲线到曲面',
            font=dict(size=18)
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/analytic_geometry.png', width=1000, height=900)
    return fig


def plot_knowledge_integration():
    """图6：知识融合进入微分几何"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('从欧氏空间到流形', '度量张量：距离的推广',
                       '协变导数：曲面上的平行移动', '曲率张量：内在几何的体现'),
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]]
    )
    
    # 左上：流形概念
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    U, V = np.meshgrid(u, v)
    
    # 球面
    R = 2
    X = R * np.sin(V) * np.cos(U)
    Y = R * np.sin(V) * np.sin(U)
    Z = R * np.cos(V)
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Blues',
        showscale=False,
        opacity=0.7
    ), row=1, col=1)
    
    # 局部坐标卡
    u_local = np.linspace(0.5, 1.5, 20)
    v_local = np.linspace(1.0, 2.0, 20)
    U_loc, V_loc = np.meshgrid(u_local, v_local)
    X_loc = R * np.sin(V_loc) * np.cos(U_loc)
    Y_loc = R * np.sin(V_loc) * np.sin(U_loc)
    Z_loc = R * np.cos(V_loc)
    
    fig.add_trace(go.Surface(
        x=X_loc, y=Y_loc, z=Z_loc,
        colorscale='Reds',
        showscale=False,
        opacity=0.9
    ), row=1, col=1)
    
    # 右上：度量张量
    # 展示平面和曲面上的距离差异
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (X**2 + Y**2)
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Greens',
        showscale=False,
        opacity=0.8
    ), row=1, col=2)
    
    # 绘制测地线（近似为直线）
    t_geo = np.linspace(-1.5, 1.5, 50)
    x_geo = t_geo
    y_geo = 0.5 * t_geo
    z_geo = 0.5 * (x_geo**2 + y_geo**2)
    
    fig.add_trace(go.Scatter3d(
        x=x_geo, y=y_geo, z=z_geo,
        mode='lines',
        line=dict(color='#FF3B30', width=4),
        showlegend=False
    ), row=1, col=2)
    
    # 左下：协变导数
    # 球面上的向量场
    theta = np.linspace(0.3, np.pi-0.3, 8)
    phi = np.linspace(0, 2*np.pi, 16)
    
    for th in theta[::2]:
        for ph in phi[::4]:
            x = R * np.sin(th) * np.cos(ph)
            y = R * np.sin(th) * np.sin(ph)
            z = R * np.cos(th)
            
            # 切向量
            vx = R * np.cos(th) * np.cos(ph) * 0.3
            vy = R * np.cos(th) * np.sin(ph) * 0.3
            vz = -R * np.sin(th) * 0.3
            
            fig.add_trace(go.Cone(
                x=[x], y=[y], z=[z],
                u=[vx], v=[vy], w=[vz],
                colorscale='Oranges',
                showscale=False,
                sizemode='absolute',
                sizeref=0.4
            ), row=2, col=1)
    
    # 绘制球面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Blues',
        showscale=False,
        opacity=0.3
    ), row=2, col=1)
    
    # 右下：曲率张量示意（高斯曲率）
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # 马鞍面（负曲率）
    Z = X**2 - Y**2
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='RdBu',
        showscale=True,
        colorbar=dict(title='曲率', x=0.95),
        opacity=0.9
    ), row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1100,
        height=1000,
        title=dict(
            text='从基础到微分几何：知识的融合',
            font=dict(size=18)
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/knowledge_integration.png', width=1100, height=1000)
    return fig


def plot_learning_pathway():
    """图7：学习路径图"""
    fig = go.Figure()
    
    # 定义节点位置
    nodes = {
        '微积分': (2, 8),
        '极限与连续': (1, 7),
        '导数与微分': (2, 7),
        '积分理论': (3, 7),
        '多元微积分': (2, 6),
        '线性代数': (6, 8),
        '向量空间': (5, 7),
        '矩阵理论': (6, 7),
        '特征值问题': (7, 7),
        '内积空间': (6, 6),
        '微分方程': (10, 8),
        '常微分方程': (9.5, 7),
        '偏微分方程': (10.5, 7),
        '解析几何': (14, 8),
        '参数曲线': (13, 7),
        '参数曲面': (14, 7),
        '曲线论': (13, 6),
        '曲面论': (14, 6),
        '微分几何': (8, 3),
        '流形理论': (7, 4),
        '黎曼几何': (8, 4),
        '张量分析': (9, 4),
        '曲率理论': (8, 2),
    }
    
    # 颜色映射
    colors = {
        '微积分': '#007AFF',
        '线性代数': '#34C759',
        '微分方程': '#FF9500',
        '解析几何': '#AF52DE',
        '微分几何': '#FF3B30',
    }
    
    # 绘制连接线
    connections = [
        ('微积分', '多元微积分'),
        ('多元微积分', '微分几何'),
        ('线性代数', '内积空间'),
        ('内积空间', '微分几何'),
        ('微分方程', '微分几何'),
        ('解析几何', '曲线论'),
        ('解析几何', '曲面论'),
        ('曲线论', '微分几何'),
        ('曲面论', '微分几何'),
        ('微分几何', '流形理论'),
        ('微分几何', '黎曼几何'),
        ('微分几何', '张量分析'),
        ('流形理论', '曲率理论'),
        ('黎曼几何', '曲率理论'),
        ('张量分析', '曲率理论'),
        # 内部连接
        ('微积分', '极限与连续'),
        ('微积分', '导数与微分'),
        ('微积分', '积分理论'),
        ('极限与连续', '导数与微分'),
        ('导数与微分', '积分理论'),
        ('导数与微分', '多元微积分'),
        ('线性代数', '向量空间'),
        ('线性代数', '矩阵理论'),
        ('向量空间', '矩阵理论'),
        ('矩阵理论', '特征值问题'),
        ('矩阵理论', '内积空间'),
        ('微分方程', '常微分方程'),
        ('微分方程', '偏微分方程'),
        ('解析几何', '参数曲线'),
        ('解析几何', '参数曲面'),
        ('参数曲线', '曲线论'),
        ('参数曲面', '曲面论'),
    ]
    
    for start, end in connections:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]
        
        # 确定颜色
        if '微积分' in start or start in ['极限与连续', '导数与微分', '积分理论', '多元微积分']:
            color = '#007AFF'
        elif '线性代数' in start or start in ['向量空间', '矩阵理论', '特征值问题', '内积空间']:
            color = '#34C759'
        elif '微分方程' in start or start in ['常微分方程', '偏微分方程']:
            color = '#FF9500'
        elif '解析几何' in start or start in ['参数曲线', '参数曲面', '曲线论', '曲面论']:
            color = '#AF52DE'
        else:
            color = '#FF3B30'
            
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color=color, width=1.5),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # 绘制节点
    for name, (x, y) in nodes.items():
        # 确定颜色
        if '微积分' in name or name in ['极限与连续', '导数与微分', '积分理论', '多元微积分']:
            color = '#007AFF'
        elif '线性代数' in name or name in ['向量空间', '矩阵理论', '特征值问题', '内积空间']:
            color = '#34C759'
        elif '微分方程' in name or name in ['常微分方程', '偏微分方程']:
            color = '#FF9500'
        elif '解析几何' in name or name in ['参数曲线', '参数曲面', '曲线论', '曲面论']:
            color = '#AF52DE'
        else:
            color = '#FF3B30'
        
        size = 40 if name in ['微积分', '线性代数', '微分方程', '解析几何', '微分几何'] else 30
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                line=dict(color='white', width=2)
            ),
            showlegend=False,
            hoverinfo='text',
            hovertext=name
        ))
        
        # 添加文字标签
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[name],
            textposition='middle center',
            textfont=dict(
                size=10 if len(name) <= 4 else 8,
                color='white',
                family='Arial, sans-serif'
            ),
            showlegend=False
        ))
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1200,
        height=700,
        title=dict(
            text='微分几何学习路径图',
            font=dict(size=18)
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white'
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/learning_pathway.png', width=1200, height=700)
    return fig


def plot_historical_development():
    """图8：微分几何发展历史时间线 - 全屏布局超大版"""
    fig = go.Figure()
    
    # 时间线数据
    events = [
        (1687, '牛顿《自然哲学的数学原理》', '经典力学基础', '#007AFF'),
        (1736, '欧拉解决哥尼斯堡七桥问题', '图论诞生', '#34C759'),
        (1827, '高斯《曲面的一般研究》', '现代微分几何起点', '#FF3B30'),
        (1854, '黎曼的就职演讲', '黎曼几何诞生', '#AF52DE'),
        (1869, '克里斯托费尔发展张量分析', '协变微分', '#FF9500'),
        (1900, '列维-奇维塔平行移动', '联络理论', '#007AFF'),
        (1915, '爱因斯坦广义相对论', '微分几何的物理应用', '#34C759'),
        (1950, '陈省身示性类理论', '整体微分几何', '#FF3B30'),
        (1982, '丘成桐证明卡拉比猜想', '微分几何里程碑', '#AF52DE'),
        (2002, '佩雷尔曼证明庞加莱猜想', '里奇流方法', '#FF9500'),
    ]
    
    events.sort(key=lambda x: x[0])
    years = [e[0] for e in events]
    x_min, x_max = 1680, 2020
    
    # 分配y位置避免重叠：交错分布，更大范围填充画布
    y_positions = [5.5, -4.0, 4.8, -3.2, 4.0, -2.5, 3.2, -5.5, 4.5, -6.2]
    
    for i, (year, event, desc, color) in enumerate(events):
        y_pos = y_positions[i]
        y_offset = 1 if y_pos > 0 else -1
        
        # 事件点（在时间线上）
        fig.add_trace(go.Scatter(
            x=[year],
            y=[0],
            mode='markers',
            marker=dict(size=48, color=color, line=dict(color='white', width=5)),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'{year}: {event}'
        ))
        
        # 年份标签
        fig.add_trace(go.Scatter(
            x=[year],
            y=[0.25 if y_offset > 0 else -0.25],
            mode='text',
            text=[str(year)],
            textposition='top center' if y_offset > 0 else 'bottom center',
            textfont=dict(size=32, color='#333', family='Arial Black'),
            showlegend=False
        ))
        
        # 连接线
        fig.add_trace(go.Scatter(
            x=[year, year],
            y=[0.15 if y_offset > 0 else -0.15, y_pos * 0.85],
            mode='lines',
            line=dict(color=color, width=6),
            showlegend=False
        ))
        
        # 简化事件名称
        short_names = {
            '牛顿《自然哲学的数学原理》': '牛顿原理',
            '欧拉解决哥尼斯堡七桥问题': '欧拉七桥问题',
            '高斯《曲面的一般研究》': '高斯《曲面研究》',
            '黎曼的就职演讲': '黎曼就职演讲',
            '克里斯托费尔发展张量分析': '克里斯托费尔',
            '列维-奇维塔平行移动': '列维-奇维塔',
            '爱因斯坦广义相对论': '爱因斯坦相对论',
            '陈省身示性类理论': '陈省身示性类',
            '丘成桐证明卡拉比猜想': '丘成桐卡拉比猜想',
            '佩雷尔曼证明庞加莱猜想': '佩雷尔曼庞加莱猜想'
        }
        short_event = short_names.get(event, event)
        
        # 事件名称
        fig.add_trace(go.Scatter(
            x=[year],
            y=[y_pos],
            mode='text',
            text=[short_event],
            textposition='top center' if y_offset > 0 else 'bottom center',
            textfont=dict(size=28, color='#222', family='Arial'),
            showlegend=False
        ))
        
        # 描述
        fig.add_trace(go.Scatter(
            x=[year],
            y=[y_pos - 0.6 if y_offset > 0 else y_pos + 0.6],
            mode='text',
            text=[desc],
            textposition='top center' if y_offset > 0 else 'bottom center',
            textfont=dict(size=20, color='#666'),
            showlegend=False
        ))
    
    # 主时间线
    fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[0, 0],
        mode='lines',
        line=dict(color='#888', width=6),
        showlegend=False
    ))
    
    # 添加时期背景
    fig.add_vrect(
        x0=1680, x1=1800,
        fillcolor='rgba(0, 122, 255, 0.06)',
        line_width=0,
        layer='below'
    )
    
    fig.add_vrect(
        x0=1800, x1=1950,
        fillcolor='rgba(175, 82, 222, 0.06)',
        line_width=0,
        layer='below'
    )
    
    fig.add_vrect(
        x0=1950, x1=2020,
        fillcolor='rgba(255, 59, 48, 0.06)',
        line_width=0,
        layer='below'
    )
    
    # 时期标签
    fig.add_annotation(
        x=1740, y=6.5,
        text='经典时期',
        showarrow=False,
        font=dict(size=32, color='#007AFF'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#007AFF',
        borderwidth=3,
        borderpad=8
    )
    
    fig.add_annotation(
        x=1875, y=-6.5,
        text='黎曼几何时期',
        showarrow=False,
        font=dict(size=32, color='#AF52DE'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#AF52DE',
        borderwidth=3,
        borderpad=8
    )
    
    fig.add_annotation(
        x=1985, y=6.5,
        text='现代发展',
        showarrow=False,
        font=dict(size=32, color='#FF3B30'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#FF3B30',
        borderwidth=3,
        borderpad=8
    )
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=16),
        width=2600,
        height=1100,
        title=dict(
            text='微分几何发展历程（1687-2002）',
            font=dict(size=48, family='Arial'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='年份', font=dict(size=28)),
            tickmode='linear',
            dtick=30,
            range=[x_min, x_max],
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=24)
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-7.2, 7.2]
        ),
        margin=dict(l=200, r=60, t=80, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/historical_development.png', width=2600, height=1100)
    return fig


if __name__ == '__main__':
    print("开始生成微分几何前序知识配图...")
    
    print("\n1. 生成微积分基础图...")
    plot_calculus_foundations()
    
    print("\n2. 生成多元微积分图...")
    plot_multivariable_calculus()
    
    print("\n3. 生成线性代数基础图...")
    plot_linear_algebra_foundations()
    
    print("\n4. 生成微分方程图...")
    plot_differential_equations()
    
    print("\n5. 生成解析几何图...")
    plot_analytic_geometry()
    
    print("\n6. 生成知识融合图...")
    plot_knowledge_integration()
    
    print("\n7. 生成学习路径图...")
    plot_learning_pathway()
    
    print("\n8. 生成发展历史图...")
    plot_historical_development()
    
    print("\n✅ 所有配图生成完成！")
