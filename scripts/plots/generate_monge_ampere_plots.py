#!/usr/bin/env python3
"""
生成蒙日-安培方程相关的 Plotly 图形
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import subprocess
import os

# 确保目录存在
os.makedirs('static/images/plots', exist_ok=True)

def save_and_compress(fig, filepath, width=800, height=600):
    """保存并压缩图片"""
    fig.write_image(filepath, width=width, height=height, scale=2)
    
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")


def plot_convex_function():
    """
    图1: 凸函数与Hessian矩阵示意
    展示凸函数的几何特性及其Hessian的正定性
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'xy'}]],
        subplot_titles=('凸函数曲面 z = u(x,y)', '等高线与次微分'),
        horizontal_spacing=0.1
    )
    
    # 创建凸函数数据
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # 凸函数：u(x,y) = x^2 + y^2 (碗形)
    Z = X**2 + Y**2
    
    # 3D 曲面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Blues',
        showscale=False,
        name='凸函数'
    ), row=1, col=1)
    
    # 等高线
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Blues',
        showscale=False,
        contours=dict(
            coloring='lines',
            showlabels=True,
            labelfont=dict(size=10, color='#333')
        )
    ), row=1, col=2)
    
    # 添加梯度向量场（次微分）
    x_vec = np.linspace(-1.5, 1.5, 8)
    y_vec = np.linspace(-1.5, 1.5, 8)
    X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
    
    # 梯度 grad(u) = (2x, 2y)
    U = 2 * X_vec * 0.3
    V = 2 * Y_vec * 0.3
    
    fig.add_trace(go.Scatter(
        x=X_vec.flatten(),
        y=Y_vec.flatten(),
        mode='markers',
        marker=dict(size=6, color='#FF9500'),
        showlegend=False
    ), row=1, col=2)
    
    # 添加箭头表示梯度
    for i in range(len(x_vec)):
        for j in range(len(y_vec)):
            fig.add_annotation(
                x=X_vec[j, i], y=Y_vec[j, i],
                ax=X_vec[j, i] + U[j, i], ay=Y_vec[j, i] + V[j, i],
                xref='x2', yref='y2', axref='x2', ayref='y2',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor='#007AFF'
            )
    
    fig.update_layout(
        title=dict(text='凸函数的几何特性', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=450
    )
    
    # 更新3D视角
    fig.update_scenes(
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'
    )
    
    fig.update_xaxes(title_text='x', row=1, col=2)
    fig.update_yaxes(title_text='y', row=1, col=2)
    
    save_and_compress(fig, 'static/images/plots/ma_convex_function.png', width=1000, height=450)
    return fig


def plot_gaussian_curvature():
    """
    图2: 高斯曲率与Monge-Ampere方程
    展示不同曲面的高斯曲率分布
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        subplot_titles=('球面 (K>0)', '双曲抛物面 (K<0)', '正曲率等高线', '负曲率等高线'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    x = np.linspace(-1.5, 1.5, 40)
    y = np.linspace(-1.5, 1.5, 40)
    X, Y = np.meshgrid(x, y)
    
    # 球面: z = sqrt(1 - x^2 - y^2)
    R = 1.5
    Z_sphere = np.sqrt(np.maximum(0, R**2 - X**2 - Y**2))
    
    # 双曲抛物面: z = x^2 - y^2
    Z_hyper = X**2 - Y**2
    
    # 球面（上排左）
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_sphere,
        colorscale='RdYlBu',
        showscale=False,
        cmin=0, cmax=1.5
    ), row=1, col=1)
    
    # 双曲抛物面（上排右）
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_hyper,
        colorscale='RdYlBu_r',
        showscale=False,
        cmin=-2, cmax=2
    ), row=1, col=2)
    
    # 球面等高线（下排左）
    K_sphere = 1 / R**2 * np.ones_like(X)  # 常正曲率
    fig.add_trace(go.Contour(
        x=x, y=y, z=K_sphere,
        colorscale='Reds',
        showscale=False,
        contours=dict(coloring='heatmap'),
        line=dict(width=0)
    ), row=2, col=1)
    
    # 双曲抛物面等高线（下排右）
    # 高斯曲率 K = -4 / (1 + 4x^2 + 4y^2)^2
    K_hyper = -4 / (1 + 4*X**2 + 4*Y**2)**2
    fig.add_trace(go.Contour(
        x=x, y=y, z=K_hyper,
        colorscale='Blues_r',
        showscale=False,
        contours=dict(coloring='heatmap'),
        line=dict(width=0)
    ), row=2, col=2)
    
    fig.update_layout(
        title=dict(text='高斯曲率与曲面类型', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=11),
        width=900,
        height=700
    )
    
    # 更新视角
    fig.update_scenes(
        camera=dict(eye=dict(x=1.5, y=1.5, z=1)),
        aspectmode='cube'
    )
    
    save_and_compress(fig, 'static/images/plots/ma_gaussian_curvature.png', width=900, height=700)
    return fig


def plot_optimal_transport():
    """
    图3: 最优传输问题示意
    展示从源分布到目标分布的传输映射
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('源分布 f(x)', '传输映射 T=∇u', '目标分布 g(y)'),
        horizontal_spacing=0.08
    )
    
    # 创建两个高斯分布
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    
    # 源分布：中心在 -1 的高斯
    mu = -1
    sigma = 0.8
    f_x = np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    
    # 目标分布：中心在 1 的高斯
    mu2 = 1
    sigma2 = 0.6
    g_y = np.exp(-0.5 * ((y - mu2) / sigma2)**2) / (sigma2 * np.sqrt(2 * np.pi))
    
    # 传输映射（线性变换示例）
    T = (x - mu) * (sigma2 / sigma) + mu2
    
    # 源分布
    fig.add_trace(go.Scatter(
        x=x, y=f_x,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(0, 122, 255, 0.3)',
        line=dict(color='#007AFF', width=2),
        name='源分布 f(x)'
    ), row=1, col=1)
    
    # 目标分布
    fig.add_trace(go.Scatter(
        x=y, y=g_y,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(52, 199, 89, 0.3)',
        line=dict(color='#34C759', width=2),
        name='目标分布 g(y)'
    ), row=1, col=3)
    
    # 传输映射
    fig.add_trace(go.Scatter(
        x=x, y=T,
        mode='lines',
        line=dict(color='#FF9500', width=2.5),
        name='传输映射 T(x)'
    ), row=1, col=2)
    
    # 添加映射箭头
    sample_points = [-2, -1.5, -1, -0.5, 0]
    for xp in sample_points:
        fp = np.exp(-0.5 * ((xp - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
        Tp = (xp - mu) * (sigma2 / sigma) + mu2
        fig.add_annotation(
            x=xp, y=0,
            ax=Tp, ay=0,
            xref='x2', yref='y2', axref='x2', ayref='y2',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='#007AFF',
            row=1, col=2
        )
    
    fig.update_layout(
        title=dict(text='最优传输问题：二次代价情形', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        showlegend=False,
        width=1000,
        height=350
    )
    
    fig.update_xaxes(title_text='x', row=1, col=1)
    fig.update_yaxes(title_text='密度 f(x)', row=1, col=1)
    fig.update_xaxes(title_text='x', row=1, col=2)
    fig.update_yaxes(title_text='T(x)', row=1, col=2)
    fig.update_xaxes(title_text='y', row=1, col=3)
    fig.update_yaxes(title_text='密度 g(y)', row=1, col=3)
    
    save_and_compress(fig, 'static/images/plots/ma_optimal_transport.png', width=1000, height=350)
    return fig


def plot_determinant_ellipticity():
    """
    图4: Hessian行列式与椭圆性
    展示行列式与特征值的关系
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('正定矩阵的椭圆', '行列式与特征值'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}]],
        horizontal_spacing=0.12
    )
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 示例矩阵1：强正定的椭圆
    A1 = np.array([[2, 0.5], [0.5, 1]])
    eigvals1, eigvecs1 = np.linalg.eig(A1)
    
    # 椭圆: x^T A x = 1
    # 参数化: x = A^{-1/2} * (cos t, sin t)
    A1_inv_sqrt = np.linalg.inv(np.linalg.cholesky(A1))
    ellipse1 = np.array([A1_inv_sqrt @ np.array([np.cos(t), np.sin(t)]) for t in theta])
    
    fig.add_trace(go.Scatter(
        x=ellipse1[:, 0], y=ellipse1[:, 1],
        mode='lines',
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        line=dict(color='#007AFF', width=2),
        name='椭圆1'
    ), row=1, col=1)
    
    # 示例矩阵2：较弱正定的椭圆
    A2 = np.array([[1.2, 0.3], [0.3, 0.8]])
    A2_inv_sqrt = np.linalg.inv(np.linalg.cholesky(A2))
    ellipse2 = np.array([A2_inv_sqrt @ np.array([np.cos(t), np.sin(t)]) for t in theta])
    
    fig.add_trace(go.Scatter(
        x=ellipse2[:, 0] + 3, y=ellipse2[:, 1],
        mode='lines',
        fill='toself',
        fillcolor='rgba(52, 199, 89, 0.2)',
        line=dict(color='#34C759', width=2),
        name='椭圆2'
    ), row=1, col=1)
    
    # 添加特征值方向
    for i, (eigval, eigvec) in enumerate(zip(eigvals1, eigvecs1.T)):
        scale = 1 / np.sqrt(eigval)
        fig.add_annotation(
            x=0, y=0,
            ax=eigvec[0]*scale*0.8, ay=eigvec[1]*scale*0.8,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#FF9500'
        )
    
    # 右图：行列式与特征值的关系
    lambda1_range = np.linspace(0.1, 3, 100)
    lambda2_values = [0.5, 1.0, 1.5, 2.0]
    colors = ['#FF3B30', '#FF9500', '#34C759', '#007AFF']
    
    for lambda2, color in zip(lambda2_values, colors):
        det = lambda1_range * lambda2
        fig.add_trace(go.Scatter(
            x=lambda1_range, y=det,
            mode='lines',
            line=dict(color=color, width=2),
            name=f'λ₂={lambda2}'
        ), row=1, col=2)
    
    fig.add_hline(y=0, line=dict(color='#999', width=1, dash='dash'), row=1, col=2)
    
    # 添加标注
    fig.add_annotation(x=2.5, y=2.5, text='椭圆性区域<br>(det>0)', showarrow=False,
                       font=dict(size=10, color='#34C759'), row=1, col=2)
    
    fig.update_layout(
        title=dict(text='Hessian行列式与椭圆性', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        showlegend=True,
        legend=dict(x=0.65, y=0.95),
        width=900,
        height=400
    )
    
    fig.update_xaxes(title_text='x₁', range=[-2, 5], row=1, col=1)
    fig.update_yaxes(title_text='x₂', range=[-2, 2], row=1, col=1)
    fig.update_xaxes(title_text='特征值 λ₁', row=1, col=2)
    fig.update_yaxes(title_text='行列式 det = λ₁λ₂', row=1, col=2)
    
    save_and_compress(fig, 'static/images/plots/ma_determinant_ellipticity.png', width=900, height=400)
    return fig


def plot_minkowski_problem():
    """
    图5: Minkowski问题示意图
    展示给定曲率求曲面的问题
    """
    fig = go.Figure()
    
    # 创建单位球面上的函数（给定高斯曲率）
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    PHI, THETA = np.meshgrid(phi, theta)
    
    # 原始球面（常曲率）
    r = 1.0
    X_sphere = r * np.sin(THETA) * np.cos(PHI)
    Y_sphere = r * np.sin(THETA) * np.sin(PHI)
    Z_sphere = r * np.cos(THETA)
    
    # 扰动后的曲面（变曲率）
    # K = 1 + 0.3 * sin(3*phi) * sin(theta)
    r_perturbed = 1 + 0.2 * np.sin(3*PHI) * np.sin(THETA)
    X_pert = r_perturbed * np.sin(THETA) * np.cos(PHI)
    Y_pert = r_perturbed * np.sin(THETA) * np.sin(PHI)
    Z_pert = r_perturbed * np.cos(THETA)
    
    # 添加球面
    fig.add_trace(go.Surface(
        x=X_sphere, y=Y_sphere, z=Z_sphere,
        opacity=0.3,
        colorscale='Blues',
        showscale=False,
        name='参考球面'
    ))
    
    # 添加扰动曲面
    fig.add_trace(go.Surface(
        x=X_pert, y=Y_pert, z=Z_pert,
        opacity=0.7,
        colorscale='RdYlBu',
        showscale=False,
        name='目标曲面'
    ))
    
    # 添加标注
    fig.add_annotation(
        x=0.02, y=0.95,
        xref='paper', yref='paper',
        text='Minkowski问题：<br>给定曲率 K(n)<br>求凸曲面',
        showarrow=False,
        font=dict(size=12, color='#333'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#007AFF',
        borderwidth=1
    )
    
    fig.update_layout(
        title=dict(text='Minkowski问题：给定曲率求凸曲面', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=700,
        height=600,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='cube'
        )
    )
    
    save_and_compress(fig, 'static/images/plots/ma_minkowski_problem.png', width=700, height=600)
    return fig


def plot_regularity_theory():
    """
    图6: 正则性理论示意
    展示不同条件下的解的光滑性
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('严格凸解 (C^2,α)', '非严格凸解 (奇异)', '边界正则性', '内部正则性'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    x = np.linspace(-2, 2, 200)
    
    # 左上: 严格凸光滑解
    y1 = x**2 + 0.1 * np.sin(5*x)  # 光滑扰动
    fig.add_trace(go.Scatter(
        x=x, y=y1,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        name='C²,α 解'
    ), row=1, col=1)
    
    # 添加导数
    dy1 = 2*x + 0.5 * np.cos(5*x)
    fig.add_trace(go.Scatter(
        x=x, y=dy1*0.3,
        mode='lines',
        line=dict(color='#FF9500', width=1.5, dash='dash'),
        name='一阶导数'
    ), row=1, col=1)
    
    # 右上: 非严格凸解（有角点）
    y2 = np.where(x < 0, 0.5*x**2, x**2)
    fig.add_trace(go.Scatter(
        x=x, y=y2,
        mode='lines',
        line=dict(color='#FF3B30', width=2),
        name='非严格凸'
    ), row=1, col=2)
    
    # 标记角点
    fig.add_vline(x=0, line=dict(color='#FF3B30', width=1, dash='dot'), row=1, col=2)
    fig.add_annotation(x=0, y=0.5, text='角点', showarrow=False,
                       font=dict(size=10, color='#FF3B30'), row=1, col=2)
    
    # 左下: 边界正则性
    x_bdry = np.linspace(0, 1, 100)
    y_bdry = x_bdry**2 * (1 - x_bdry)**2  # 在边界为0
    fig.add_trace(go.Scatter(
        x=x_bdry, y=y_bdry,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(52, 199, 89, 0.2)',
        line=dict(color='#34C759', width=2),
        name='边界正则'
    ), row=2, col=1)
    
    # 右下: 内部正则性（放大显示光滑性）
    x_zoom = np.linspace(-0.5, 0.5, 100)
    y_zoom = x_zoom**4 - x_zoom**2  # 更光滑的函数
    fig.add_trace(go.Scatter(
        x=x_zoom, y=y_zoom,
        mode='lines',
        line=dict(color='#AF52DE', width=2),
        name='内部光滑'
    ), row=2, col=2)
    
    # 添加二阶导数
    d2y_zoom = 12*x_zoom**2 - 2
    fig.add_trace(go.Scatter(
        x=x_zoom, y=d2y_zoom*0.1,
        mode='lines',
        line=dict(color='#FF9500', width=1.5, dash='dash'),
        name='二阶导数'
    ), row=2, col=2)
    
    fig.update_layout(
        title=dict(text='Caffarelli正则性理论', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=11),
        showlegend=False,
        width=900,
        height=700
    )
    
    fig.update_xaxes(title_text='x', row=1, col=1)
    fig.update_yaxes(title_text='u(x)', row=1, col=1)
    fig.update_xaxes(title_text='x', row=1, col=2)
    fig.update_yaxes(title_text='u(x)', row=1, col=2)
    fig.update_xaxes(title_text='x', row=2, col=1)
    fig.update_yaxes(title_text='u(x)', row=2, col=1)
    fig.update_xaxes(title_text='x (放大)', row=2, col=2)
    fig.update_yaxes(title_text='u(x)', row=2, col=2)
    
    save_and_compress(fig, 'static/images/plots/ma_regularity_theory.png', width=900, height=700)
    return fig


def plot_history_timeline():
    """
    图7: 历史发展时间线
    """
    fig = go.Figure()
    
    # 历史事件数据
    events = [
        (1771, '蒙日\n《Memoire》', '蒙日几何', '#007AFF'),
        (1820, '安培\n解析研究', '偏微分方程', '#34C759'),
        (1903, '闵可夫斯基\n《体积与表面积》', '凸几何', '#FF9500'),
        (1950, 'Alexandrov\n弱解理论', '非线性PDE', '#AF52DE'),
        (1976, 'Cheng-Yau\n高维Minkowski', '正则性理论', '#FF3B30'),
        (1987, 'Brenier\n极值分解', '最优传输', '#5AC8FA'),
        (1991, 'Caffarelli\n正则性理论', '完全非线性', '#007AFF'),
        (2010, 'Figalli\n最优传输应用', '交叉学科', '#34C759'),
        (2018, 'Fields Medal\nFigalli', '现代发展', '#FF9500'),
    ]
    
    years = [e[0] for e in events]
    labels = [e[1] for e in events]
    categories = [e[2] for e in events]
    colors = [e[3] for e in events]
    
    # 时间线
    fig.add_trace(go.Scatter(
        x=years,
        y=[0]*len(years),
        mode='lines+markers',
        line=dict(color='#333', width=3),
        marker=dict(size=15, color=colors, line=dict(color='white', width=2)),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 交替放置标签
    for i, (year, label, cat, color) in enumerate(events):
        y_offset = 0.5 if i % 2 == 0 else -0.5
        
        # 连接线
        fig.add_trace(go.Scatter(
            x=[year, year],
            y=[0, y_offset*0.7],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))
        
        # 文字标注
        fig.add_annotation(
            x=year, y=y_offset,
            text=f'<b>{year}</b><br>{label}<br><span style="font-size:9px;color:#666">{cat}</span>',
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=color,
            borderwidth=1,
            borderpad=4
        )
    
    fig.update_layout(
        title=dict(text='蒙日-安培方程的历史发展', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=400,
        xaxis=dict(
            title='年份',
            range=[1760, 2030],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            visible=False,
            range=[-1, 1]
        ),
        plot_bgcolor='white'
    )
    
    save_and_compress(fig, 'static/images/plots/ma_history_timeline.png', width=1000, height=400)
    return fig


if __name__ == '__main__':
    print("生成蒙日-安培方程相关图形...")
    
    print("\n1. 生成凸函数示意图...")
    plot_convex_function()
    
    print("\n2. 生成高斯曲率示意图...")
    plot_gaussian_curvature()
    
    print("\n3. 生成最优传输示意图...")
    plot_optimal_transport()
    
    print("\n4. 生成Hessian行列式与椭圆性图...")
    plot_determinant_ellipticity()
    
    print("\n5. 生成Minkowski问题示意图...")
    plot_minkowski_problem()
    
    print("\n6. 生成正则性理论示意图...")
    plot_regularity_theory()
    
    print("\n7. 生成历史发展时间线...")
    plot_history_timeline()
    
    print("\n✅ 所有图形生成完成！")
    
    # 验证文件
    print("\n生成的文件:")
    for f in os.listdir('static/images/plots'):
        if f.startswith('ma_'):
            filepath = os.path.join('static/images/plots', f)
            size = os.path.getsize(filepath)
            print(f"  - {f}: {size/1024:.1f} KB")
