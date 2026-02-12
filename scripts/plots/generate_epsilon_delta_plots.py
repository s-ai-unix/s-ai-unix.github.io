"""
生成 epsilon-delta 语言相关的 Plotly 图形
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os

def save_and_compress(fig, filepath):
    """保存并压缩图片"""
    fig.write_image(filepath, scale=2)
    
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")


def plot_epsilon_delta_illustration():
    """
    可视化 epsilon-delta 定义：
    展示函数 f(x) = x^2 在 x = 2 处的连续性
    """
    x = np.linspace(0, 4, 500)
    y = x**2
    
    # 关键点
    a = 2
    L = a**2  # 4
    
    # epsilon 和 delta 值
    epsilon = 1.5
    delta = 0.3  # 手动调整以获得更好的可视化效果
    
    fig = go.Figure()
    
    # 绘制函数曲线
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='$f(x) = x^2$',
        line=dict(color='#007AFF', width=3)
    ))
    
    # 绘制 (a, L) 点
    fig.add_trace(go.Scatter(
        x=[a], y=[L],
        mode='markers',
        name=f'点 $({a}, {L})$',
        marker=dict(color='#FF3B30', size=12, symbol='circle')
    ))
    
    # 绘制 epsilon 区间 (水平带状区域)
    x_fill = np.linspace(a - delta, a + delta, 100)
    y_upper = np.full_like(x_fill, L + epsilon)
    y_lower = np.full_like(x_fill, L - epsilon)
    
    # epsilon 带状区域
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 149, 0, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'$\\epsilon = {epsilon}$ 区间',
        showlegend=True
    ))
    
    # delta 区间 (垂直带状区域)
    y_fill = np.linspace(0, 16, 100)
    fig.add_trace(go.Scatter(
        x=np.concatenate([np.full_like(y_fill, a - delta), np.full_like(y_fill, a + delta)[::-1]]),
        y=np.concatenate([y_fill, y_fill[::-1]]),
        fill='toself',
        fillcolor='rgba(52, 199, 89, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'$\\delta = {delta}$ 区间',
        showlegend=True
    ))
    
    # 标注线
    # epsilon 线
    fig.add_hline(y=L, line=dict(color='#8E8E93', width=1, dash='dash'), 
                  annotation_text='$L = 4$', annotation_position='right')
    fig.add_hline(y=L + epsilon, line=dict(color='#FF9500', width=1, dash='dot'),
                  annotation_text='$L + \\epsilon$', annotation_position='right')
    fig.add_hline(y=L - epsilon, line=dict(color='#FF9500', width=1, dash='dot'),
                  annotation_text='$L - \\epsilon$', annotation_position='right')
    
    # delta 线
    fig.add_vline(x=a, line=dict(color='#8E8E93', width=1, dash='dash'),
                  annotation_text='$a = 2$', annotation_position='top')
    fig.add_vline(x=a + delta, line=dict(color='#34C759', width=1, dash='dot'),
                  annotation_text='$a + \\delta$', annotation_position='top')
    fig.add_vline(x=a - delta, line=dict(color='#34C759', width=1, dash='dot'),
                  annotation_text='$a - \\delta$', annotation_position='top')
    
    # 箭头说明
    fig.add_annotation(x=2.8, y=L + epsilon/2, text='误差范围 $\\epsilon$', 
                       showarrow=False, font=dict(size=12, color='#FF9500'))
    fig.add_annotation(x=a + delta/2, y=1, text='邻域半径 $\\delta$', 
                       showarrow=False, font=dict(size=12, color='#34C759'))
    
    fig.update_layout(
        title=dict(text='Epsilon-Delta 定义可视化：$f(x) = x^2$ 在 $x = 2$ 处的连续性', 
                   font=dict(size=16)),
        xaxis_title='$x$',
        yaxis_title='$f(x)$',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        width=900,
        height=600
    )
    
    fig.update_xaxes(range=[0, 4])
    fig.update_yaxes(range=[0, 16])
    
    save_and_compress(fig, 'static/images/plots/epsilon_delta_illustration.png')
    return fig


def plot_limit_concept_evolution():
    """
    极限概念的演变：从直观到严格
    展示三种不同定义方式对同一函数的理解
    """
    x = np.linspace(-3, 3, 500)
    
    # 函数: sin(x)/x，在 x=0 处 removable discontinuity
    y = np.sinc(x / np.pi)  # sinc(x) = sin(pi*x)/(pi*x)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            '直观理解："趋近于"',
            '无穷小方法："无限接近"', 
            'Epsilon-Delta："任意接近"'
        ),
        horizontal_spacing=0.12
    )
    
    # 子图1: 直观理解
    fig.add_trace(go.Scatter(
        x=x[x != 0], y=y[x != 0],
        mode='lines',
        line=dict(color='#007AFF', width=2.5),
        showlegend=False
    ), row=1, col=1)
    
    # 添加 "趋近" 箭头 - 从左侧指向原点
    fig.add_annotation(
        x=-0.8, y=0.85, 
        text='趋近',
        showarrow=True,
        arrowhead=2, 
        arrowcolor='#FF3B30',
        arrowsize=1.5,
        arrowwidth=2,
        ax=40,  # 箭头指向右侧
        ay=0,
        font=dict(size=12, color='#FF3B30'),
        row=1, col=1
    )
    
    # 目标值标注
    fig.add_trace(go.Scatter(
        x=[0], y=[1],
        mode='markers+text',
        marker=dict(color='#34C759', size=10, symbol='diamond'),
        text=['极限 = 1'],
        textposition='top center',
        textfont=dict(size=11, color='#34C759'),
        showlegend=False
    ), row=1, col=1)
    
    # 子图2: 无穷小方法
    fig.add_trace(go.Scatter(
        x=x[x != 0], y=y[x != 0],
        mode='lines',
        line=dict(color='#007AFF', width=2.5),
        showlegend=False
    ), row=1, col=2)
    
    # 无穷小标记 - 在原点附近添加 dx 标记
    dx_pos = 0.3
    dy_val = np.sinc(dx_pos / np.pi)
    
    # 绘制 dx 区间
    fig.add_vrect(
        x0=-dx_pos, x1=dx_pos,
        fillcolor='rgba(175, 82, 222, 0.15)',
        line_width=0,
        row=1, col=2
    )
    
    # 无穷小标注
    fig.add_annotation(
        x=0, y=0.7,
        text='$dx$ 为无穷小量',
        showarrow=False,
        font=dict(size=12, color='#AF52DE'),
        row=1, col=2
    )
    fig.add_annotation(
        x=0, y=0.62,
        text='$\\frac{\\sin(dx)}{dx} \\approx 1$',
        showarrow=False,
        font=dict(size=12, color='#AF52DE'),
        row=1, col=2
    )
    
    # 在 dx 位置画标记点
    fig.add_trace(go.Scatter(
        x=[dx_pos], y=[dy_val],
        mode='markers',
        marker=dict(color='#AF52DE', size=8, symbol='circle'),
        showlegend=False
    ), row=1, col=2)
    fig.add_annotation(
        x=dx_pos + 0.15, y=dy_val + 0.02,
        text='$dx$',
        showarrow=False,
        font=dict(size=10, color='#AF52DE'),
        row=1, col=2
    )
    
    # 子图3: Epsilon-Delta
    fig.add_trace(go.Scatter(
        x=x[x != 0], y=y[x != 0],
        mode='lines',
        line=dict(color='#007AFF', width=2.5),
        showlegend=False
    ), row=1, col=3)
    
    # 添加 epsilon 带状区域
    epsilon = 0.08
    delta = 0.5
    x_band = np.linspace(-delta, delta, 50)
    y_upper = np.full_like(x_band, 1 + epsilon)
    y_lower = np.full_like(x_band, 1 - epsilon)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_band, x_band[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 149, 0, 0.25)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ), row=1, col=3)
    
    # epsilon 水平线
    fig.add_hline(y=1 + epsilon, line=dict(color='#FF9500', width=1.5, dash='dot'), row=1, col=3)
    fig.add_hline(y=1 - epsilon, line=dict(color='#FF9500', width=1.5, dash='dot'), row=1, col=3)
    fig.add_hline(y=1, line=dict(color='#8E8E93', width=1, dash='dash'), row=1, col=3)
    
    # delta 垂直线
    fig.add_vline(x=delta, line=dict(color='#34C759', width=1.5, dash='dot'), row=1, col=3)
    fig.add_vline(x=-delta, line=dict(color='#34C759', width=1.5, dash='dot'), row=1, col=3)
    
    # 标注
    fig.add_annotation(
        x=1.8, y=1 + epsilon/2,
        text='$\\epsilon$ 误差带',
        showarrow=False,
        font=dict(size=11, color='#FF9500'),
        row=1, col=3
    )
    fig.add_annotation(
        x=delta + 0.3, y=0.65,
        text='$\\delta$ 邻域',
        showarrow=False,
        font=dict(size=11, color='#34C759'),
        row=1, col=3
    )
    
    fig.update_layout(
        title=dict(text='极限概念的三种理解方式', font=dict(size=18)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1200,
        height=500,
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    for i in range(1, 4):
        fig.update_xaxes(range=[-3, 3], row=1, col=i, title_text='$x$')
        fig.update_yaxes(range=[0.5, 1.25], row=1, col=i, title_text='$y$')
    
    save_and_compress(fig, 'static/images/plots/limit_concept_evolution.png')
    return fig


def plot_continuity_types():
    """
    不同类型的连续性示例
    """
    x = np.linspace(-2, 2, 500)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '连续函数：$f(x) = x^2$',
            '可去间断点：$f(x) = \\frac{\\sin x}{x}$ (补充定义)',
            '跳跃间断点：符号函数',
            '本质间断点：$f(x) = \\sin(\\frac{1}{x})$'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 子图1: 连续函数 x^2
    fig.add_trace(go.Scatter(
        x=x, y=x**2,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(color='#FF3B30', size=10),
        showlegend=False
    ), row=1, col=1)
    
    # 子图2: 可去间断点
    x2 = x[x != 0]
    y2 = np.sin(x2) / x2
    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[0], y=[1],
        mode='markers',
        marker=dict(color='#34C759', size=10, symbol='circle-open'),
        showlegend=False
    ), row=1, col=2)
    
    # 子图3: 跳跃间断点
    x3_left = x[x < 0]
    x3_right = x[x >= 0]
    fig.add_trace(go.Scatter(
        x=x3_left, y=np.ones_like(x3_left) * (-1),
        mode='lines',
        line=dict(color='#007AFF', width=2),
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x3_right, y=np.ones_like(x3_right),
        mode='lines',
        line=dict(color='#007AFF', width=2),
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[1],
        mode='markers',
        marker=dict(color='#FF3B30', size=10),
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[-1],
        mode='markers',
        marker=dict(color='#007AFF', size=10, symbol='circle-open'),
        showlegend=False
    ), row=2, col=1)
    
    # 子图4: 本质间断点
    x4 = np.linspace(-0.5, 0.5, 2000)
    x4 = x4[x4 != 0]
    y4 = np.sin(1 / x4)
    fig.add_trace(go.Scatter(
        x=x4, y=y4,
        mode='lines',
        line=dict(color='#007AFF', width=1),
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        title=dict(text='连续性的分类与间断点类型', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=900,
        height=700
    )
    
    # 更新各子图的坐标轴
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(range=[-2, 2], row=i, col=j)
    
    fig.update_yaxes(range=[-0.5, 4.5], row=1, col=1)
    fig.update_yaxes(range=[0, 1.5], row=1, col=2)
    fig.update_yaxes(range=[-1.5, 1.5], row=2, col=1)
    fig.update_yaxes(range=[-1.5, 1.5], row=2, col=2)
    
    save_and_compress(fig, 'static/images/plots/continuity_types.png')
    return fig


def plot_weierstrass_function():
    """
    魏尔斯特拉斯函数：处处连续但处处不可导
    """
    def weierstrass(x, a=0.5, b=3, n_terms=50):
        """魏尔斯特拉斯函数近似"""
        result = np.zeros_like(x)
        for n in range(n_terms):
            result += a**n * np.cos(b**n * np.pi * x)
        return result
    
    x = np.linspace(-2, 2, 5000)
    y = weierstrass(x, a=0.5, b=3, n_terms=30)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#007AFF', width=1),
        name='魏尔斯特拉斯函数'
    ))
    
    # 添加注释
    fig.add_annotation(x=0, y=1.5, text='处处连续', showarrow=False,
                       font=dict(size=14, color='#34C759'))
    fig.add_annotation(x=0, y=1.2, text='处处不可导', showarrow=False,
                       font=dict(size=14, color='#FF3B30'))
    
    fig.update_layout(
        title=dict(text='魏尔斯特拉斯函数：连续性与可导性的分离', font=dict(size=16)),
        xaxis_title='$x$',
        yaxis_title='$W(x)$',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        width=900,
        height=500
    )
    
    save_and_compress(fig, 'static/images/plots/weierstrass_function.png')
    return fig


def plot_uniform_continuity():
    """
    一致连续性的可视化
    """
    x = np.linspace(0.01, 3, 500)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '在 $(0, 1]$ 上不一致连续：$f(x) = \\frac{1}{x}$',
            '在 $[1, \\infty)$ 上一致连续：$f(x) = \\frac{1}{x}$'
        ),
        horizontal_spacing=0.12
    )
    
    # 子图1: 在 (0, 1] 上
    y1 = 1 / x
    fig.add_trace(go.Scatter(
        x=x, y=y1,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        showlegend=False
    ), row=1, col=1)
    
    # 标注不同位置的 delta 变化
    fig.add_annotation(x=0.2, y=5, text='相同的 $\\epsilon$<br>需要更小的 $\\delta$', 
                       showarrow=True, arrowhead=2, ax=-40, ay=-40,
                       font=dict(size=11, color='#FF9500'), row=1, col=1)
    fig.add_annotation(x=2, y=0.5, text='更大的 $\\delta$ 即可', 
                       showarrow=True, arrowhead=2, ax=40, ay=-30,
                       font=dict(size=11, color='#34C759'), row=1, col=1)
    
    # 子图2: 在 [1, inf) 上
    x2 = np.linspace(1, 5, 500)
    y2 = 1 / x2
    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_annotation(x=3, y=0.5, text='相同的 $\\delta$ 适用于所有点', 
                       showarrow=False, font=dict(size=12, color='#34C759'), row=1, col=2)
    
    fig.update_layout(
        title=dict(text='一致连续性：$\\delta$ 是否与位置无关', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1100,
        height=500
    )
    
    fig.update_xaxes(range=[0, 3], row=1, col=1)
    fig.update_yaxes(range=[0, 6], row=1, col=1)
    fig.update_xaxes(range=[1, 5], row=1, col=2)
    fig.update_yaxes(range=[0, 1.2], row=1, col=2)
    
    save_and_compress(fig, 'static/images/plots/uniform_continuity.png')
    return fig


if __name__ == '__main__':
    print("开始生成 epsilon-delta 相关图形...")
    
    plot_epsilon_delta_illustration()
    print("✅ 已生成 epsilon_delta_illustration.png")
    
    plot_limit_concept_evolution()
    print("✅ 已生成 limit_concept_evolution.png")
    
    plot_continuity_types()
    print("✅ 已生成 continuity_types.png")
    
    plot_weierstrass_function()
    print("✅ 已生成 weierstrass_function.png")
    
    plot_uniform_continuity()
    print("✅ 已生成 uniform_continuity.png")
    
    print("\n所有图形生成完成！")
