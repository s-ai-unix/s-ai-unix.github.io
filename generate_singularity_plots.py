#!/usr/bin/env python3
"""
生成彭罗斯-霍金奇点定理相关的 Plotly 图形
"""

import plotly.graph_objects as go
import plotly.express as px
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


def plot_light_cone_structure():
    """
    图1: 光锥结构示意图
    展示时空中光锥的因果结构
    """
    fig = go.Figure()
    
    # 时间轴 (垂直)
    t = np.linspace(-2, 2, 100)
    
    # 光锥边界 (45度线，c=1)
    x_light = np.linspace(0, 2, 50)
    t_future_upper = x_light
    t_future_lower = -x_light
    t_past_upper = -x_light
    t_past_lower = x_light
    
    # 未来光锥
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_light, x_light[::-1]]),
        y=np.concatenate([t_future_upper, t_future_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        line=dict(color='#007AFF', width=2),
        name='未来光锥'
    ))
    
    # 过去光锥
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_light, x_light[::-1]]),
        y=np.concatenate([t_past_upper, t_past_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(52, 199, 89, 0.2)',
        line=dict(color='#34C759', width=2),
        name='过去光锥'
    ))
    
    # 时间轴
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-2.5, 2.5],
        mode='lines',
        line=dict(color='#333333', width=3),
        name='时间轴'
    ))
    
    # 空间轴
    fig.add_trace(go.Scatter(
        x=[-2.5, 2.5], y=[0, 0],
        mode='lines',
        line=dict(color='#333333', width=3),
        name='空间轴'
    ))
    
    # 标记原点
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=12, color='#FF9500'),
        text=['事件P'],
        textposition='bottom right',
        textfont=dict(size=12, color='#333333'),
        showlegend=False
    ))
    
    # 添加因果区域标注
    fig.add_annotation(x=1.2, y=1.2, text='未来', showarrow=False, 
                       font=dict(size=14, color='#007AFF'))
    fig.add_annotation(x=1.2, y=-1.2, text='过去', showarrow=False, 
                       font=dict(size=14, color='#34C759'))
    fig.add_annotation(x=1.8, y=0.3, text='类空区域', showarrow=False, 
                       font=dict(size=12, color='#666666'))
    fig.add_annotation(x=-1.8, y=0.3, text='类空区域', showarrow=False, 
                       font=dict(size=12, color='#666666'))
    
    fig.update_layout(
        title=dict(text='时空中的光锥结构', font=dict(size=16)),
        xaxis_title='空间 x',
        yaxis_title='时间 ct',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        width=800,
        height=600,
        xaxis=dict(range=[-2.5, 2.5], zeroline=False),
        yaxis=dict(range=[-2.5, 2.5], zeroline=False)
    )
    
    save_and_compress(fig, 'static/images/plots/light_cone_structure.png')
    return fig


def plot_trapped_surface():
    """
    图2: Trapped Surface 示意图
    展示 trapped null surface 的概念：内向和外向光线都汇聚
    """
    fig = go.Figure()
    
    # 中心坍缩区域
    theta = np.linspace(0, 2*np.pi, 100)
    r_center = 0.3
    x_center = r_center * np.cos(theta)
    y_center = r_center * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_center, y=y_center,
        fill='toself',
        fillcolor='rgba(255, 149, 0, 0.3)',
        line=dict(color='#FF9500', width=2),
        name='坍缩区域'
    ))
    
    # Trapped surface (闭合 trapped surface)
    r_trapped = 1.0
    x_trapped = r_trapped * np.cos(theta)
    y_trapped = r_trapped * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_trapped, y=y_trapped,
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.1)',
        line=dict(color='#007AFF', width=3, dash='solid'),
        name='Trapped Surface'
    ))
    
    # 外向光线（在 trapped surface 内部，本应向外但实际汇聚）
    angles_out = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for angle in angles_out:
        # 起点在 trapped surface 上
        x_start = r_trapped * np.cos(angle)
        y_start = r_trapped * np.sin(angle)
        
        # 光线方向（向内汇聚）
        t = np.linspace(0, 0.6, 20)
        x_ray = x_start + t * (-np.cos(angle) * 0.8 + np.sin(angle) * 0.1)
        y_ray = y_start + t * (-np.sin(angle) * 0.8 - np.cos(angle) * 0.1)
        
        fig.add_trace(go.Scatter(
            x=x_ray, y=y_ray,
            mode='lines',
            line=dict(color='#FF3B30', width=2),
            showlegend=False
        ))
        
        # 箭头
        fig.add_annotation(
            x=x_ray[-1], y=y_ray[-1],
            ax=x_ray[-3], ay=y_ray[-3],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#FF3B30'
        )
    
    # 添加标注
    fig.add_annotation(x=0, y=0, text='奇点', showarrow=False, 
                       font=dict(size=12, color='white', family='Arial'))
    fig.add_annotation(x=1.3, y=0.3, text='θ=0', showarrow=False, 
                       font=dict(size=10, color='#007AFF'))
    
    # 添加说明文字
    fig.add_annotation(
        x=2.2, y=1.5,
        text='<b>Trapped Surface 特征：</b><br>• 内向光线汇聚 ✓<br>• 外向光线也汇聚 ✓<br>• 面积不断减小',
        showarrow=False,
        font=dict(size=11, color='#333333'),
        align='left',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#007AFF',
        borderwidth=1
    )
    
    fig.update_layout(
        title=dict(text='闭合捕获面 (Closed Trapped Surface)', font=dict(size=16)),
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        width=800,
        height=600,
        xaxis=dict(range=[-2.5, 3], zeroline=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[-2, 2], zeroline=False)
    )
    
    save_and_compress(fig, 'static/images/plots/trapped_surface.png')
    return fig


def plot_raychaudhuri_focusing():
    """
    图3: Raychaudhuri方程 - 测地线聚焦效应
    展示光线如何在引力作用下汇聚
    """
    fig = go.Figure()
    
    # 模拟一组平行的零测地线（光射线）
    n_rays = 5
    y_positions = np.linspace(-1, 1, n_rays)
    
    # 参数：仿射参数 λ
    lambda_vals = np.linspace(0, 2, 100)
    
    # 模拟聚焦效应：横截面积随仿射参数减小
    # A(λ) = A_0 - α * λ^2 (简化模型)
    alpha = 0.3  # 聚焦强度
    
    for i, y0 in enumerate(y_positions):
        # 初始平行光线，由于引力作用逐渐汇聚
        # 使用简化的聚焦模型
        y = y0 * (1 - alpha * lambda_vals**2)
        
        # 避免穿过奇点
        y = np.where(np.abs(y) < 0.05, np.nan, y)
        
        color = '#007AFF' if i == n_rays // 2 else '#5AC8FA'
        width = 3 if i == n_rays // 2 else 1.5
        
        fig.add_trace(go.Scatter(
            x=lambda_vals, y=y,
            mode='lines',
            line=dict(color=color, width=width),
            showlegend=False
        ))
        
        # 添加方向箭头
        if i == n_rays // 2:
            fig.add_annotation(
                x=1.5, y=y[75],
                ax=1.3, ay=y[65],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#007AFF'
            )
    
    # 标记焦散点（caustic）
    fig.add_trace(go.Scatter(
        x=[1.8], y=[0],
        mode='markers+text',
        marker=dict(size=15, color='#FF9500', symbol='x'),
        text=['焦散点'],
        textposition='top center',
        textfont=dict(size=12, color='#FF9500'),
        showlegend=False
    ))
    
    # 添加横截面积变化曲线（右侧Y轴）
    area = 1 - 0.25 * lambda_vals**2
    area = np.maximum(area, 0)
    
    fig.add_trace(go.Scatter(
        x=lambda_vals, y=area * 0.8 - 1.5,
        mode='lines',
        line=dict(color='#34C759', width=3, dash='dash'),
        name='横截面积 A(λ)',
        yaxis='y2'
    ))
    
    # 添加标注
    fig.add_annotation(x=0.5, y=1.3, text='光线束', showarrow=False, 
                       font=dict(size=12, color='#007AFF'))
    fig.add_annotation(x=1.0, y=-1.3, text='横截面积 A(λ)', showarrow=False, 
                       font=dict(size=12, color='#34C759'))
    
    fig.update_layout(
        title=dict(text='测地线聚焦效应 (Raychaudhuri方程)', font=dict(size=16)),
        xaxis_title='仿射参数 λ',
        yaxis_title='横向位置',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=True,
        legend=dict(x=0.65, y=0.95),
        width=900,
        height=500,
        xaxis=dict(range=[-0.1, 2.2]),
        yaxis=dict(range=[-1.8, 1.8], zeroline=True),
        yaxis2=dict(
            overlaying='y',
            side='right',
            showticklabels=False,
            range=[-2, 2]
        )
    )
    
    save_and_compress(fig, 'static/images/plots/raychaudhuri_focusing.png')
    return fig


def plot_expansion_scalar():
    """
    图4: 膨胀标量 θ 随时间的演化
    展示 Raychaudhuri 方程的解
    """
    fig = go.Figure()
    
    tau = np.linspace(0, 2, 200)
    
    # 不同的初始膨胀值
    theta_0_values = [-1, -0.5, -0.2, 0, 0.3]
    colors = ['#FF3B30', '#FF9500', '#FFCC00', '#34C759', '#007AFF']
    
    for theta_0, color in zip(theta_0_values, colors):
        # Raychaudhuri 方程的近似解
        # dθ/dτ = -R_{μν}u^μu^ν - (1/3)θ² - σ² + ω²
        # 假设强能量条件，Ricci 投影为正，σ=0，ω=0
        # dθ/dτ = -R - (1/3)θ²
        
        # 简化模型：θ(τ) = θ_0 / (1 + (θ_0/3)*τ) 当 θ_0 < 0 时发散
        if theta_0 < 0:
            # 发散解
            tau_max = -3/theta_0 * 0.95
            tau_plot = tau[tau < tau_max]
            theta = theta_0 / (1 + (theta_0/3) * tau_plot)
        else:
            tau_plot = tau
            theta = theta_0 / (1 + (theta_0/3) * tau) if theta_0 != 0 else -0.1 * tau
        
        label = f'θ₀ = {theta_0}'
        fig.add_trace(go.Scatter(
            x=tau_plot, y=theta,
            mode='lines',
            line=dict(color=color, width=2.5),
            name=label
        ))
        
        # 标记奇点位置（当 θ → -∞）
        if theta_0 < 0:
            singularity_tau = -3/theta_0
            fig.add_trace(go.Scatter(
                x=[singularity_tau], y=[-5],
                mode='markers',
                marker=dict(size=10, color=color, symbol='x'),
                showlegend=False
            ))
    
    # 添加水平参考线 θ = 0
    fig.add_hline(y=0, line=dict(color='#999999', width=1, dash='dash'))
    
    # 添加说明
    fig.add_annotation(
        x=1.5, y=-3,
        text='<b>θ → -∞ 表示奇点</b><br>膨胀标量在有限时间内发散',
        showarrow=False,
        font=dict(size=11, color='#FF3B30'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#FF3B30',
        borderwidth=1
    )
    
    fig.update_layout(
        title=dict(text='Raychaudhuri方程：膨胀标量演化', font=dict(size=16)),
        xaxis_title='固有时 τ',
        yaxis_title='膨胀标量 θ',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        width=800,
        height=500,
        xaxis=dict(range=[0, 2]),
        yaxis=dict(range=[-5, 1])
    )
    
    save_and_compress(fig, 'static/images/plots/expansion_scalar.png')
    return fig


def plot_black_hole_formation():
    """
    图5: 黑洞形成过程示意图
    展示恒星坍缩形成黑洞和事件视界
    """
    fig = go.Figure()
    
    # 时间轴
    t = np.linspace(0, 3, 100)
    
    # 恒星表面半径随时间坍缩
    # 使用简化模型：R(t) = R_0 * sqrt(1 - t/t_collapse) for t < t_collapse
    R_0 = 2.0  # 初始半径
    t_collapse = 2.5  # 坍缩时间
    
    # 恒星半径（外部）
    R_star = R_0 * np.sqrt(np.maximum(0, 1 - t/t_collapse))
    
    #  Schwarzschild 半径 (事件视界)
    R_s = 0.8  # Schwarzschild 半径
    
    # 绘制恒星表面
    fig.add_trace(go.Scatter(
        x=t, y=R_star,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='恒星表面'
    ))
    
    # 绘制 Schwarzschild 半径（事件视界）
    fig.add_hline(y=R_s, line=dict(color='#FF3B30', width=2, dash='dash'),
                  annotation_text='事件视界 (r = 2M)',
                  annotation_position='right')
    
    # 填充区域：外部空间
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([R_star, np.full_like(t, 3)]),
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.1)',
        line=dict(width=0),
        name='外部空间',
        showlegend=True
    ))
    
    # 填充区域：黑洞内部
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([R_star, np.zeros_like(t)]),
        fill='toself',
        fillcolor='rgba(0, 0, 0, 0.15)',
        line=dict(width=0),
        name='黑洞内部',
        showlegend=True
    ))
    
    # 标记 trapped surface 形成点
    trapped_idx = np.where(R_star <= R_s)[0]
    if len(trapped_idx) > 0:
        t_trapped = t[trapped_idx[0]]
        fig.add_trace(go.Scatter(
            x=[t_trapped], y=[R_s],
            mode='markers+text',
            marker=dict(size=15, color='#FF9500'),
            text=['Trapped\nSurface形成'],
            textposition='top left',
            textfont=dict(size=10, color='#FF9500'),
            name='Trapped Surface形成',
            showlegend=False
        ))
        
        # 垂直线标记
        fig.add_vline(x=t_trapped, line=dict(color='#FF9500', width=1, dash='dot'))
    
    # 标记奇点形成
    fig.add_trace(go.Scatter(
        x=[t_collapse], y=[0],
        mode='markers+text',
        marker=dict(size=15, color='black', symbol='x'),
        text=['奇点形成'],
        textposition='bottom right',
        textfont=dict(size=11, color='black'),
        showlegend=False
    ))
    
    # 添加光锥方向示意
    for t_pos in [0.5, 1.2, 1.8]:
        r_pos = R_0 * np.sqrt(max(0, 1 - t_pos/t_collapse))
        if r_pos > R_s:
            # 光锥向外
            fig.add_annotation(
                x=t_pos, y=r_pos,
                ax=t_pos+0.2, ay=r_pos+0.3,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#007AFF'
            )
    
    fig.update_layout(
        title=dict(text='引力坍缩与黑洞形成', font=dict(size=16)),
        xaxis_title='时间 t',
        yaxis_title='半径 r',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        width=800,
        height=500,
        xaxis=dict(range=[0, 3]),
        yaxis=dict(range=[0, 3])
    )
    
    save_and_compress(fig, 'static/images/plots/black_hole_formation.png')
    return fig


def plot_singularity_types():
    """
    图6: 不同类型的奇点示意图
    展示类空奇点、类时奇点
    """
    fig = go.Figure()
    
    # 创建子图布局
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('类空奇点 (Schwarzschild)', '类时奇点 (Reissner-Nordström)'),
        horizontal_spacing=0.1
    )
    
    # === 左图: Schwarzschild 奇点 (类空) ===
    t = np.linspace(-2, 2, 50)
    r = np.linspace(0.1, 3, 50)
    T, R = np.meshgrid(t, r)
    
    # 简化的 Penrose 图： Schwarzschild
    # 奇点是一条水平线 (r=0, 类空)
    
    # 绘制奇点
    fig.add_trace(go.Scatter(
        x=[-2, 2], y=[0, 0],
        mode='lines',
        line=dict(color='#FF3B30', width=4),
        name='奇点 (r=0)',
        showlegend=True
    ), row=1, col=1)
    
    # 事件视界
    fig.add_vline(x=0, line=dict(color='#007AFF', width=2), row=1, col=1)
    
    # 标记区域
    fig.add_annotation(x=-1, y=1.5, text='区域 I<br>(渐近平直)', showarrow=False,
                       font=dict(size=10, color='#333333'), row=1, col=1)
    fig.add_annotation(x=1, y=1.5, text='区域 II<br>(黑洞内部)', showarrow=False,
                       font=dict(size=10, color='#333333'), row=1, col=1)
    fig.add_annotation(x=1, y=-1.5, text='奇点', showarrow=False,
                       font=dict(size=10, color='#FF3B30'), row=1, col=1)
    
    # === 右图: Reissner-Nordström 奇点 (类时) ===
    
    # 奇点是一条垂直线 (类时，可避开)
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-2, 2],
        mode='lines',
        line=dict(color='#34C759', width=4),
        name='奇点 (可避开)',
        showlegend=True
    ), row=1, col=2)
    
    # 内外视界
    fig.add_vline(x=-0.5, line=dict(color='#007AFF', width=2, dash='dash'), row=1, col=2)
    fig.add_vline(x=0.5, line=dict(color='#007AFF', width=2, dash='dash'), row=1, col=2)
    
    # 标记
    fig.add_annotation(x=-1.2, y=1.5, text='外部宇宙', showarrow=False,
                       font=dict(size=10, color='#333333'), row=1, col=2)
    fig.add_annotation(x=0, y=1.5, text='内部区域', showarrow=False,
                       font=dict(size=10, color='#333333'), row=1, col=2)
    fig.add_annotation(x=0, y=-1.5, text='类时奇点<br>(可绕行)', showarrow=False,
                       font=dict(size=10, color='#34C759'), row=1, col=2)
    
    # 更新布局
    fig.update_layout(
        title=dict(text='奇点的类型', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        showlegend=True,
        legend=dict(x=0.4, y=-0.15, orientation='h'),
        width=900,
        height=400
    )
    
    for i in range(1, 3):
        fig.update_xaxes(title_text='时间', row=1, col=i, range=[-2, 2])
        fig.update_yaxes(title_text='空间', row=1, col=i, range=[-2, 2])
    
    save_and_compress(fig, 'static/images/plots/singularity_types.png', width=900, height=400)
    return fig


def plot_causal_structure():
    """
    图7: 全局因果结构示意图
    展示 Cauchy 面、全局双曲性
    """
    fig = go.Figure()
    
    # Cauchy 面 (类空超曲面)
    x_cauchy = np.linspace(-2, 2, 100)
    y_cauchy = np.zeros_like(x_cauchy)
    
    fig.add_trace(go.Scatter(
        x=x_cauchy, y=y_cauchy,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        name='Cauchy面 Σ'
    ))
    
    # 从 Cauchy 面发出的未来零测地线
    x_starts = np.linspace(-1.5, 1.5, 7)
    for x0 in x_starts:
        # 未来光锥
        t = np.linspace(0, 1.5, 30)
        x_right = x0 + t
        y_right = t
        x_left = x0 - t
        y_left = t
        
        fig.add_trace(go.Scatter(
            x=x_right, y=y_right,
            mode='lines',
            line=dict(color='#007AFF', width=1.5),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x_left, y=y_left,
            mode='lines',
            line=dict(color='#007AFF', width=1.5),
            showlegend=False
        ))
    
    # 过去光锥
    for x0 in x_starts:
        t = np.linspace(0, 1.5, 30)
        x_right = x0 + t
        y_right = -t
        x_left = x0 - t
        y_left = -t
        
        fig.add_trace(go.Scatter(
            x=x_right, y=y_right,
            mode='lines',
            line=dict(color='#34C759', width=1.5),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x_left, y=y_left,
            mode='lines',
            line=dict(color='#34C759', width=1.5),
            showlegend=False
        ))
    
    # D+(Σ) 和 D-(Σ) 区域
    # 未来依赖域
    fig.add_trace(go.Scatter(
        x=[-1.5, 1.5, 0, -1.5],
        y=[0, 0, 1.5, 0],
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.1)',
        line=dict(width=0),
        name='D⁺(Σ) 未来依赖域'
    ))
    
    # 过去依赖域
    fig.add_trace(go.Scatter(
        x=[-1.5, 1.5, 0, -1.5],
        y=[0, 0, -1.5, 0],
        fill='toself',
        fillcolor='rgba(52, 199, 89, 0.1)',
        line=dict(width=0),
        name='D⁻(Σ) 过去依赖域'
    ))
    
    # 添加标注
    fig.add_annotation(x=0, y=0.7, text='D⁺(Σ)', showarrow=False,
                       font=dict(size=14, color='#007AFF'))
    fig.add_annotation(x=0, y=-0.7, text='D⁻(Σ)', showarrow=False,
                       font=dict(size=14, color='#34C759'))
    fig.add_annotation(x=0, y=0.1, text='Σ (Cauchy面)', showarrow=False,
                       font=dict(size=12, color='white'))
    
    # 说明文字
    fig.add_annotation(
        x=2.3, y=1.2,
        text='<b>全局双曲时空</b><br>• 存在 Cauchy 面<br>• 决定论适用<br>• 初值问题良定义',
        showarrow=False,
        font=dict(size=10, color='#333333'),
        align='left',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#007AFF',
        borderwidth=1
    )
    
    fig.update_layout(
        title=dict(text='全局双曲时空的因果结构', font=dict(size=16)),
        xaxis_title='空间',
        yaxis_title='时间',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        width=800,
        height=600,
        xaxis=dict(range=[-2.5, 3], zeroline=False),
        yaxis=dict(range=[-2, 2], zeroline=False)
    )
    
    save_and_compress(fig, 'static/images/plots/causal_structure.png')
    return fig


if __name__ == '__main__':
    print("生成彭罗斯-霍金奇点定理相关图形...")
    
    print("\n1. 生成光锥结构图...")
    plot_light_cone_structure()
    
    print("\n2. 生成 Trapped Surface 示意图...")
    plot_trapped_surface()
    
    print("\n3. 生成测地线聚焦效应图...")
    plot_raychaudhuri_focusing()
    
    print("\n4. 生成膨胀标量演化图...")
    plot_expansion_scalar()
    
    print("\n5. 生成黑洞形成过程图...")
    plot_black_hole_formation()
    
    print("\n6. 生成奇点类型对比图...")
    plot_singularity_types()
    
    print("\n7. 生成因果结构图...")
    plot_causal_structure()
    
    print("\n✅ 所有图形生成完成！")
    
    # 验证文件
    print("\n生成的文件:")
    for f in os.listdir('static/images/plots'):
        if 'singularity' in f or 'light_cone' in f or 'trapped' in f or 'raychaudhuri' in f or 'expansion' in f or 'black_hole' in f or 'causal' in f:
            filepath = os.path.join('static/images/plots', f)
            size = os.path.getsize(filepath)
            print(f"  - {f}: {size/1024:.1f} KB")
