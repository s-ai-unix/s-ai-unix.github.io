#!/usr/bin/env python3
"""
生成微分几何前序知识综述文章的配图（简化版）
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os

OUTPUT_DIR = "../static/images/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_and_compress(fig, filepath, width=900, height=600):
    """保存并压缩图片"""
    fig.write_image(filepath, width=width, height=height, scale=2)
    
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")


def plot_multivariable_calculus_simple():
    """图2：多元微积分核心概念（简化版）"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('偏导数：沿坐标轴的变化率', '梯度：最速上升方向', 
                       '方向导数：任意方向的变化率')
    )
    
    # 创建等高线数据
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # 抛物面投影
    
    # 左图：偏导数示意
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Blues',
        showscale=False,
        contours=dict(start=0, end=8, size=0.5)
    ), row=1, col=1)
    
    # 添加沿x方向的箭头
    fig.add_annotation(
        x=0.5, y=0, ax=1.5, ay=0,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor='#FF3B30',
        row=1, col=1
    )
    
    # 中图：梯度示意
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Greens',
        showscale=False,
        contours=dict(start=0, end=8, size=0.5)
    ), row=1, col=2)
    
    # 添加梯度向量
    for xi in [-1.5, -0.5, 0.5, 1.5]:
        for yi in [-1.5, -0.5, 0.5, 1.5]:
            if xi**2 + yi**2 > 0.1:
                fig.add_annotation(
                    x=xi, y=yi, ax=xi-0.2*xi, ay=yi-0.2*yi,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1, arrowcolor='#FF3B30',
                    row=1, col=2
                )
    
    # 右图：方向导数
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Oranges',
        showscale=False,
        contours=dict(start=0, end=8, size=0.5)
    ), row=1, col=3)
    
    # 添加45度方向的线
    fig.add_trace(go.Scatter(
        x=[-1.5, 1.5], y=[-1.5, 1.5],
        mode='lines',
        line=dict(color='#FF3B30', width=3),
        showlegend=False
    ), row=1, col=3)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=400,
        title=dict(
            text='多元微积分的三个核心概念',
            font=dict(size=16)
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/multivariable_calculus.png', width=1000, height=400)
    return fig


def plot_analytic_geometry_simple():
    """图5：解析几何基础（简化版）"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('参数曲线：运动的轨迹', '曲率：弯曲程度的度量',
                       'Frenet标架：局部坐标系', '曲面参数化')
    )
    
    # 左上：参数曲线（螺旋线投影）
    t = np.linspace(0, 4*np.pi, 200)
    x = np.cos(t)
    y = np.sin(t)
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        showlegend=False
    ), row=1, col=1)
    
    # 添加参数点
    for ti in [np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
        fig.add_trace(go.Scatter(
            x=[np.cos(ti)], y=[np.sin(ti)],
            mode='markers',
            marker=dict(size=10, color='#FF3B30'),
            showlegend=False
        ), row=1, col=1)
    
    # 右上：曲率示意
    theta = np.linspace(0, 2*np.pi, 200)
    
    # 圆（恒定曲率）
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='圆',
        showlegend=False
    ), row=1, col=2)
    
    # 椭圆（变化曲率）
    a, b = 2, 1
    x_ellipse = a * np.cos(theta)
    y_ellipse = b * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_ellipse, y=y_ellipse,
        mode='lines',
        line=dict(color='#FF9500', width=2),
        name='椭圆',
        showlegend=False
    ), row=1, col=2)
    
    # 标记曲率最大和最小点
    fig.add_trace(go.Scatter(
        x=[-2, 2], y=[0, 0],
        mode='markers',
        marker=dict(size=12, color='#FF3B30', symbol='diamond'),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-1, 1],
        mode='markers',
        marker=dict(size=12, color='#34C759', symbol='circle'),
        showlegend=False
    ), row=1, col=2)
    
    # 左下：Frenet标架
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        showlegend=False
    ), row=2, col=1)
    
    # 在特定点绘制切向量和法向量
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        cx, cy = np.cos(angle), np.sin(angle)
        # 切向量
        fig.add_annotation(
            x=cx, y=cy, ax=cx-0.3*np.sin(angle), ay=cy+0.3*np.cos(angle),
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor='#FF3B30',
            row=2, col=1
        )
        # 法向量
        fig.add_annotation(
            x=cx, y=cy, ax=cx-0.3*np.cos(angle), ay=cy-0.3*np.sin(angle),
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor='#34C759',
            row=2, col=1
        )
    
    # 右下：曲面参数化示意（等高线表示）
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    U, V = np.meshgrid(u, v)
    
    # 用等高线表示球面投影
    X = np.sin(V) * np.cos(U)
    Y = np.sin(V) * np.sin(U)
    
    fig.add_trace(go.Contour(
        x=X.flatten(), y=Y.flatten(), z=V.flatten(),
        colorscale='Greens',
        showscale=False,
        contours=dict(start=0, end=np.pi, size=0.2)
    ), row=2, col=2)
    
    # 添加u=常数和v=常数的网格线
    for ui in np.linspace(0, 2*np.pi, 8):
        vi = np.linspace(0, np.pi, 50)
        xi = np.sin(vi) * np.cos(ui)
        yi = np.sin(vi) * np.sin(ui)
        fig.add_trace(go.Scatter(
            x=xi, y=yi,
            mode='lines',
            line=dict(color='#FF9500', width=1),
            showlegend=False
        ), row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=900,
        height=800,
        title=dict(
            text='解析几何：从曲线到曲面',
            font=dict(size=18)
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/analytic_geometry.png', width=900, height=800)
    return fig


def plot_knowledge_integration_simple():
    """图6：知识融合进入微分几何（简化版）"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('从欧氏空间到流形', '度量张量：距离的推广',
                       '协变导数：曲面上的导数', '曲率：内在几何的体现')
    )
    
    # 左上：流形概念 - 用等高线表示曲面
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Blues',
        showscale=False,
        contours=dict(start=-1, end=1, size=0.2)
    ), row=1, col=1)
    
    # 标记局部坐标卡
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5],
        mode='markers',
        marker=dict(size=50, color='rgba(255, 59, 48, 0.3)', symbol='circle'),
        showlegend=False
    ), row=1, col=1)
    
    # 右上：度量张量 - 展示距离差异
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (X**2 + Y**2)
    
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Greens',
        showscale=False,
        contours=dict(start=0, end=4, size=0.5)
    ), row=1, col=2)
    
    # 绘制测地线
    t_geo = np.linspace(-1.5, 1.5, 50)
    fig.add_trace(go.Scatter(
        x=t_geo, y=0.5*t_geo,
        mode='lines',
        line=dict(color='#FF3B30', width=4),
        showlegend=False
    ), row=1, col=2)
    
    # 左下：协变导数 - 向量场沿曲线
    theta = np.linspace(0, 2*np.pi, 50)
    x = np.cos(theta)
    y = np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        showlegend=False
    ), row=2, col=1)
    
    # 添加切向量场
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        cx, cy = np.cos(angle), np.sin(angle)
        # 切向量
        fig.add_annotation(
            x=cx, y=cy, ax=cx-0.25*np.sin(angle), ay=cy+0.25*np.cos(angle),
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor='#FF9500',
            row=2, col=1
        )
    
    # 右下：曲率示意（高斯曲率）- 马鞍面
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2
    
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='RdBu',
        showscale=True,
        contours=dict(start=-4, end=4, size=0.5),
        colorbar=dict(title='高度', x=0.95)
    ), row=2, col=2)
    
    # 标记鞍点
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=15, color='#FF3B30', symbol='diamond'),
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=900,
        title=dict(
            text='从基础到微分几何：知识的融合',
            font=dict(size=18)
        )
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/knowledge_integration.png', width=1000, height=900)
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
    colors = ['#007AFF', '#34C759', '#FF9500', '#FF3B30']
    for i, t_val in enumerate([0, 0.5, 1.0, 1.5]):
        y = np.sin(x - 2*t_val)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(width=2, color=colors[i]),
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


def plot_learning_pathway():
    """图7：学习路径图"""
    fig = go.Figure()
    
    # 定义节点位置
    nodes = {
        '微积分': (2, 8),
        '极限': (1, 7),
        '导数': (2, 7),
        '积分': (3, 7),
        '多元微积分': (2, 6),
        '线性代数': (6, 8),
        '向量空间': (5, 7),
        '矩阵': (6, 7),
        '特征值': (7, 7),
        '内积空间': (6, 6),
        '微分方程': (10, 8),
        'ODE': (9.5, 7),
        'PDE': (10.5, 7),
        '解析几何': (14, 8),
        '曲线': (13, 7),
        '曲面': (14, 7),
        '曲线论': (13, 6),
        '曲面论': (14, 6),
        '微分几何': (8, 3),
        '流形': (7, 4),
        '黎曼几何': (8, 4),
        '张量': (9, 4),
        '曲率': (8, 2),
    }
    
    # 颜色映射
    color_map = {
        '微积分': '#007AFF',
        '极限': '#007AFF',
        '导数': '#007AFF',
        '积分': '#007AFF',
        '多元微积分': '#007AFF',
        '线性代数': '#34C759',
        '向量空间': '#34C759',
        '矩阵': '#34C759',
        '特征值': '#34C759',
        '内积空间': '#34C759',
        '微分方程': '#FF9500',
        'ODE': '#FF9500',
        'PDE': '#FF9500',
        '解析几何': '#AF52DE',
        '曲线': '#AF52DE',
        '曲面': '#AF52DE',
        '曲线论': '#AF52DE',
        '曲面论': '#AF52DE',
        '微分几何': '#FF3B30',
        '流形': '#FF3B30',
        '黎曼几何': '#FF3B30',
        '张量': '#FF3B30',
        '曲率': '#FF3B30',
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
        ('微分几何', '流形'),
        ('微分几何', '黎曼几何'),
        ('微分几何', '张量'),
        ('流形', '曲率'),
        ('黎曼几何', '曲率'),
        ('张量', '曲率'),
        ('微积分', '极限'),
        ('微积分', '导数'),
        ('微积分', '积分'),
        ('极限', '导数'),
        ('导数', '积分'),
        ('导数', '多元微积分'),
        ('线性代数', '向量空间'),
        ('线性代数', '矩阵'),
        ('向量空间', '矩阵'),
        ('矩阵', '特征值'),
        ('矩阵', '内积空间'),
        ('微分方程', 'ODE'),
        ('微分方程', 'PDE'),
        ('解析几何', '曲线'),
        ('解析几何', '曲面'),
        ('曲线', '曲线论'),
        ('曲面', '曲面论'),
    ]
    
    for start, end in connections:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]
        color = color_map[start]
            
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color=color, width=1.5),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # 绘制节点
    for name, (x, y) in nodes.items():
        color = color_map[name]
        size = 45 if name in ['微积分', '线性代数', '微分方程', '解析几何', '微分几何'] else 35
        
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
                size=10 if len(name) <= 3 else 9,
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
    """图8：微分几何发展历史时间线"""
    fig = go.Figure()
    
    # 时间线数据
    events = [
        (1687, '牛顿《自然哲学的数学原理》', '经典力学基础', '#007AFF'),
        (1736, '欧拉解决哥尼斯堡七桥问题', '图论诞生', '#34C759'),
        (1827, '高斯《曲面的一般研究》', '现代微分几何起点', '#FF3B30'),
        (1854, '黎曼的就职演讲', '黎曼几何诞生', '#AF52DE'),
        (1869, '克里斯托费尔张量分析', '协变微分', '#FF9500'),
        (1900, '列维-奇维塔平行移动', '联络理论', '#007AFF'),
        (1915, '爱因斯坦广义相对论', '物理应用', '#34C759'),
        (1950, '陈省身示性类理论', '整体微分几何', '#FF3B30'),
        (1982, '丘成桐卡拉比猜想', '微分几何里程碑', '#AF52DE'),
        (2002, '佩雷尔曼庞加莱猜想', '里奇流方法', '#FF9500'),
    ]
    
    years = [e[0] for e in events]
    y_pos = list(range(len(events)))
    
    # 绘制时间线
    for i, (year, event, desc, color) in enumerate(events):
        # 点
        fig.add_trace(go.Scatter(
            x=[year], y=[i],
            mode='markers',
            marker=dict(
                size=20,
                color=color,
                line=dict(color='white', width=2)
            ),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'{year}: {event}'
        ))
        
        # 事件名称
        fig.add_trace(go.Scatter(
            x=[year + 5], y=[i],
            mode='text',
            text=[event],
            textposition='middle left',
            textfont=dict(size=11, color='#333'),
            showlegend=False
        ))
        
        # 描述
        fig.add_trace(go.Scatter(
            x=[year + 5], y=[i - 0.3],
            mode='text',
            text=[desc],
            textposition='middle left',
            textfont=dict(size=9, color='#666'),
            showlegend=False
        ))
    
    # 连接线
    fig.add_trace(go.Scatter(
        x=years,
        y=y_pos,
        mode='lines',
        line=dict(color='#ccc', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1100,
        height=600,
        title=dict(
            text='微分几何发展历程（1687-2002）',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='年份',
            tickmode='linear',
            dtick=50,
            range=[1680, 2020]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        margin=dict(l=50, r=300, t=80, b=60)
    )
    
    save_and_compress(fig, f'{OUTPUT_DIR}/historical_development.png', width=1100, height=600)
    return fig


if __name__ == '__main__':
    print("开始生成微分几何前序知识配图（简化版）...")
    
    print("\n1. 生成多元微积分图...")
    plot_multivariable_calculus_simple()
    
    print("\n2. 生成微分方程图...")
    plot_differential_equations()
    
    print("\n3. 生成解析几何图...")
    plot_analytic_geometry_simple()
    
    print("\n4. 生成知识融合图...")
    plot_knowledge_integration_simple()
    
    print("\n5. 生成学习路径图...")
    plot_learning_pathway()
    
    print("\n6. 生成发展历史图...")
    plot_historical_development()
    
    print("\n✅ 所有配图生成完成！")
