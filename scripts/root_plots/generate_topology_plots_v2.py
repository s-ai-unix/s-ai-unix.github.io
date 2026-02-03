"""
生成拓扑学相关的配图（简化版本）
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os

def save_and_compress(fig, filepath, width=800, height=500, scale=2):
    """保存并压缩图片"""
    fig.write_image(filepath, width=width, height=height, scale=scale)
    
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存: {filepath}")


def plot_mobius_strip_2d():
    """绘制莫比乌斯带的2D投影示意 - 正确显示扭转结构"""
    fig = go.Figure()
    
    # 莫比乌斯带参数方程（3D投影到2D）
    # u: 沿带长度方向 [0, 2π]
    # v: 沿带宽方向 [-w, w]
    u = np.linspace(0, 2*np.pi, 200)
    w = 0.3  # 半宽度
    
    # 计算莫比乌斯带的两条边缘（上边缘和下边缘）
    # 参数方程: 
    # x = (1 + v*cos(u/2)) * cos(u)
    # y = (1 + v*cos(u/2)) * sin(u)
    # z = v * sin(u/2)
    
    # 上边缘 (v = w)
    x_top = (1 + w * np.cos(u/2)) * np.cos(u)
    y_top = (1 + w * np.cos(u/2)) * np.sin(u)
    z_top = w * np.sin(u/2)
    
    # 下边缘 (v = -w)
    x_bot = (1 - w * np.cos(u/2)) * np.cos(u)
    y_bot = (1 - w * np.cos(u/2)) * np.sin(u)
    z_bot = -w * np.sin(u/2)
    
    # 中心线 (v = 0)
    x_center = np.cos(u)
    y_center = np.sin(u)
    z_center = np.zeros_like(u)
    
    # 使用3D投影到2D: 使用x和y坐标，用颜色表示z深度（扭转效果）
    # 上边缘
    fig.add_trace(go.Scatter(
        x=x_top, y=y_top,
        mode='lines',
        line=dict(color='#007AFF', width=4),
        name='上边缘',
        showlegend=False
    ))
    
    # 下边缘
    fig.add_trace(go.Scatter(
        x=x_bot, y=y_bot,
        mode='lines',
        line=dict(color='#34C759', width=4),
        name='下边缘',
        showlegend=False
    ))
    
    # 用填充区域表示带子
    # 合并上下边缘点（上边缘正向，下边缘反向）
    x_fill = np.concatenate([x_top, x_bot[::-1]])
    y_fill = np.concatenate([y_top, y_bot[::-1]])
    
    fig.add_trace(go.Scatter(
        x=x_fill, y=y_fill,
        mode='lines',
        line=dict(color='#007AFF', width=0),
        fill='toself', fillcolor='rgba(0, 122, 255, 0.25)',
        name='莫比乌斯带',
        showlegend=False
    ))
    
    # 重新绘制边缘线（在填充层之上）
    fig.add_trace(go.Scatter(
        x=x_top, y=y_top,
        mode='lines',
        line=dict(color='#007AFF', width=3),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=x_bot, y=y_bot,
        mode='lines',
        line=dict(color='#34C759', width=3),
        showlegend=False
    ))
    
    # 中心线（虚线）
    fig.add_trace(go.Scatter(
        x=x_center, y=y_center,
        mode='lines',
        line=dict(color='#FF3B30', width=2, dash='dash'),
        name='中心线',
        showlegend=True
    ))
    
    # 添加关键点标记：起点和终点（在中心线上是同一个点，但在带子上是"对面"）
    # 起点 (u=0)
    fig.add_trace(go.Scatter(
        x=[1], y=[0],
        mode='markers+text',
        marker=dict(size=12, color='#FF9500', symbol='circle'),
        text=['起点 A'],
        textposition='top right',
        textfont=dict(size=11, color='#333333'),
        showlegend=False
    ))
    
    # 沿中心线走一圈后的位置（在带子上是"背面"）
    # 用箭头表示路径
    fig.add_annotation(x=-0.7, y=0.7, ax=-0.3, ay=0.3,
                       showarrow=True, arrowhead=2, arrowcolor='#FF3B30',
                       arrowwidth=2, arrowsize=1.5)
    
    # 添加说明文字
    fig.add_annotation(x=-0.9, y=0.9, text='绕一圈后<br>回到"背面"',
                       showarrow=False, font=dict(size=10, color='#666666'),
                       align='left')
    
    # 添加粘合点标记（展示扭转粘合）
    fig.add_trace(go.Scatter(
        x=[1.3], y=[0],
        mode='markers+text',
        marker=dict(size=8, color='#AF52DE', symbol='diamond'),
        text=['粘合处'],
        textposition='bottom right',
        textfont=dict(size=9, color='#666666'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(text='莫比乌斯带：单侧不可定向曲面', font=dict(size=16)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, 
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.02, y=0.02, bgcolor='rgba(255,255,255,0.8)'),
        width=700, height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def plot_topology_concepts():
    """绘制拓扑空间核心概念对比"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('开集 U 中的点', '连续映射 f', '同胚映射', '紧致性'),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # 左上：开集与邻域
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 开集 U（蓝色）
    fig.add_trace(go.Scatter(
        x=1.5*np.cos(theta), y=1.5*np.sin(theta),
        mode='lines', line=dict(color='#007AFF', width=2),
        fill='toself', fillcolor='rgba(0, 122, 255, 0.3)',
        showlegend=False
    ), row=1, col=1)
    
    # 邻域 V（绿色）
    fig.add_trace(go.Scatter(
        x=0.6*np.cos(theta), y=0.6*np.sin(theta),
        mode='lines', line=dict(color='#34C759', width=2),
        fill='toself', fillcolor='rgba(52, 199, 89, 0.3)',
        showlegend=False
    ), row=1, col=1)
    
    # 点 p
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=10, color='#FF3B30'),
        text=['$p$'], textposition='top right',
        showlegend=False
    ), row=1, col=1)
    
    # 右上：连续映射
    x = np.linspace(-2, 2, 100)
    y = np.sin(x) + 0.2*x
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        showlegend=False
    ), row=1, col=2)
    
    # epsilon 带
    x0, y0 = 0.5, np.sin(0.5) + 0.2*0.5
    fig.add_hrect(y0=y0-0.3, y1=y0+0.3, row=1, col=2,
                  fillcolor='rgba(255, 149, 0, 0.2)', line_width=0)
    fig.add_vrect(x0=0.3, x1=0.7, row=1, col=2,
                  fillcolor='rgba(0, 122, 255, 0.15)', line_width=0)
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode='markers',
                             marker=dict(size=8, color='#FF3B30'), showlegend=False), row=1, col=2)
    
    # 左下：同胚映射示意
    # 空间 X（圆）
    fig.add_trace(go.Scatter(
        x=2*np.cos(theta)-3, y=2*np.sin(theta),
        mode='lines', line=dict(color='#007AFF', width=2),
        fill='toself', fillcolor='rgba(0, 122, 255, 0.2)',
        showlegend=False
    ), row=2, col=1)
    
    # 空间 Y（正方形）
    square_x = [-1, 3, 3, -1, -1]
    square_y = [-2, -2, 2, 2, -2]
    fig.add_trace(go.Scatter(
        x=square_x, y=square_y,
        mode='lines', line=dict(color='#FF9500', width=2),
        fill='toself', fillcolor='rgba(255, 149, 0, 0.2)',
        showlegend=False
    ), row=2, col=1)
    
    # 映射箭头
    fig.add_annotation(x=-1, y=0, ax=-2.8, ay=0,
                       showarrow=True, arrowhead=2, arrowcolor='#34C759',
                       arrowwidth=2, row=2, col=1)
    fig.add_annotation(x=1, y=0, text='$f$', showarrow=False, row=2, col=1)
    
    # 右下：紧致性
    x_closed = np.linspace(0, 4, 50)
    fig.add_trace(go.Scatter(
        x=x_closed, y=np.zeros_like(x_closed),
        mode='lines', line=dict(color='#007AFF', width=3),
        showlegend=False
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(x=[0, 4], y=[0, 0], mode='markers',
                             marker=dict(size=10, color='#007AFF'), showlegend=False), row=2, col=2)
    
    # 覆盖
    cover_intervals = [(0, 1.5), (1, 3), (2.5, 4)]
    colors = ['#FF9500', '#34C759', '#AF52DE']
    for (a, b), c in zip(cover_intervals, colors):
        fig.add_trace(go.Scatter(x=[a, b], y=[0.1, 0.1], mode='lines',
                                 line=dict(color=c, width=6), opacity=0.5, showlegend=False), row=2, col=2)
    
    fig.add_annotation(x=2, y=0.35, text='有限子覆盖', showarrow=False, row=2, col=2)
    
    # 更新所有子图的坐标轴
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
    
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        width=900, height=700
    )
    
    return fig


def plot_connected_compact():
    """绘制连通性与紧致性对比"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('连通但不紧致：实直线 $\mathbb{R}$', '紧致且连通：闭区间 [a,b]'),
        horizontal_spacing=0.1
    )
    
    # 左图：实直线（连通但不紧致）
    x_line = np.linspace(-3, 3, 200)
    fig.add_trace(go.Scatter(
        x=x_line, y=np.zeros_like(x_line),
        mode='lines',
        line=dict(color='#007AFF', width=3),
        showlegend=False
    ), row=1, col=1)
    
    # 无限覆盖
    for i in range(-3, 4):
        fig.add_trace(go.Scatter(
            x=[i-0.4, i+0.4], y=[0.1, 0.1],
            mode='lines',
            line=dict(color='#FF9500', width=4),
            opacity=0.4,
            showlegend=False
        ), row=1, col=1)
    
    fig.add_annotation(x=0, y=0.35, text='无限覆盖，无有限子覆盖', showarrow=False, row=1, col=1)
    
    # 右图：闭区间（紧致且连通）
    x_closed = np.linspace(-2, 2, 100)
    fig.add_trace(go.Scatter(
        x=x_closed, y=np.zeros_like(x_closed),
        mode='lines',
        line=dict(color='#34C759', width=3),
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[-2, 2], y=[0, 0],
        mode='markers',
        marker=dict(size=10, color='#34C759'),
        showlegend=False
    ), row=1, col=2)
    
    # 有限覆盖
    intervals = [(-2, -0.5), (-1, 1), (0.5, 2)]
    colors = ['#FF9500', '#34C759', '#AF52DE']
    for (a, b), c in zip(intervals, colors):
        fig.add_trace(go.Scatter(
            x=[a, b], y=[0.1, 0.1],
            mode='lines',
            line=dict(color=c, width=4),
            opacity=0.5,
            showlegend=False
        ), row=1, col=2)
    
    fig.add_annotation(x=0, y=0.35, text='存在有限子覆盖', showarrow=False, row=1, col=2)
    
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 0.5], visible=False)
    
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        width=900, height=350,
        title=dict(text='连通性与紧致性的关系', font=dict(size=16))
    )
    
    return fig


def plot_manifold_concept():
    """绘制流形概念的示意图"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('流形 M', '坐标卡 $(U, \phi)$', '欧氏空间中的像'),
        horizontal_spacing=0.08
    )
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 左图：流形（圆）
    x_circle = 2*np.cos(theta)
    y_circle = 2*np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines', line=dict(color='#007AFF', width=2),
        fill='toself', fillcolor='rgba(0, 122, 255, 0.15)',
        showlegend=False
    ), row=1, col=1)
    
    # 高亮局部区域
    theta_local = np.linspace(np.pi/4, 3*np.pi/4, 50)
    x_local = 2*np.cos(theta_local)
    y_local = 2*np.sin(theta_local)
    fig.add_trace(go.Scatter(
        x=x_local, y=y_local,
        mode='lines', line=dict(color='#FF9500', width=4),
        showlegend=False
    ), row=1, col=1)
    fig.add_annotation(x=0, y=2.5, text='$U$', showarrow=False, row=1, col=1)
    
    # 中图：同胚映射
    fig.add_trace(go.Scatter(
        x=x_local, y=y_local,
        mode='lines', line=dict(color='#FF9500', width=3),
        showlegend=False
    ), row=1, col=2)
    fig.add_annotation(x=1.5, y=1.5, text='$\phi: U \\to \\mathbb{R}^n$', showarrow=False, row=1, col=2)
    
    # 右图：欧氏空间中的开集
    x_open = np.linspace(-1.5, 1.5, 100)
    y_open = np.linspace(-0.5, 0.5, 100)
    X, Y = np.meshgrid(x_open, y_open)
    
    # 只画一个矩形区域
    rect_x = [-1.5, 1.5, 1.5, -1.5, -1.5]
    rect_y = [-0.5, -0.5, 0.5, 0.5, -0.5]
    fig.add_trace(go.Scatter(
        x=rect_x, y=rect_y,
        mode='lines', line=dict(color='#34C759', width=2),
        fill='toself', fillcolor='rgba(52, 199, 89, 0.3)',
        showlegend=False
    ), row=1, col=3)
    fig.add_annotation(x=0, y=0.8, text='$\\phi(U) \\subset \\mathbb{R}^n$', showarrow=False, row=1, col=3)
    
    for i in range(1, 4):
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=i)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=i)
    
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        width=1000, height=350,
        title=dict(text='流形的局部坐标卡结构', font=dict(size=16))
    )
    
    return fig


def main():
    """生成所有配图"""
    output_dir = '../static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成拓扑学配图...")
    
    # 1. 莫比乌斯带
    print("1. 生成莫比乌斯带图...")
    fig1 = plot_mobius_strip_2d()
    save_and_compress(fig1, f'{output_dir}/mobius-strip.png', width=700, height=500)
    
    # 2. 拓扑空间核心概念
    print("2. 生成拓扑空间概念图...")
    fig2 = plot_topology_concepts()
    save_and_compress(fig2, f'{output_dir}/topology-concepts.png', width=900, height=700)
    
    # 3. 连通性与紧致性
    print("3. 生成连通紧致对比图...")
    fig3 = plot_connected_compact()
    save_and_compress(fig3, f'{output_dir}/connected-compact.png', width=900, height=350)
    
    # 4. 流形概念
    print("4. 生成流形概念图...")
    fig4 = plot_manifold_concept()
    save_and_compress(fig4, f'{output_dir}/manifold-concept.png', width=1000, height=350)
    
    print("\n✅ 所有配图生成完成！")


if __name__ == '__main__':
    main()
