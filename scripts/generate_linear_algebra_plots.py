#!/usr/bin/env python3
"""
生成线性代数文章的静态 Plotly 图形
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path

# 输出目录
PLOT_DIR = Path("static/images/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 苹果风格配色
APPLE_BLUE = "#007AFF"
APPLE_GREEN = "#34C759"
APPLE_ORANGE = "#FF9500"
APPLE_RED = "#FF3B30"
APPLE_GRAY = "#8E8E93"
APPLE_PURPLE = "#AF52DE"

def plot_vector_transformation():
    """图1：线性变换示例"""
    print("生成 vector-transformation 图形...")

    fig = go.Figure()

    # 原始基向量
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0],
        mode='lines+markers',
        name='e₁',
        line=dict(color=APPLE_BLUE, width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 1],
        mode='lines+markers',
        name='e₂',
        line=dict(color=APPLE_GREEN, width=3),
        marker=dict(size=8)
    ))

    # 变换后的基向量 (旋转45度并缩放)
    theta = np.pi / 4
    scale = 1.5

    fig.add_trace(go.Scatter(
        x=[0, scale * np.cos(theta)],
        y=[0, scale * np.sin(theta)],
        mode='lines+markers',
        name='T(e₁)',
        line=dict(color=APPLE_BLUE, width=3, dash='dash'),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=[0, -scale * np.sin(theta)],
        y=[0, scale * np.cos(theta)],
        mode='lines+markers',
        name='T(e₂)',
        line=dict(color=APPLE_GREEN, width=3, dash='dash'),
        marker=dict(size=8)
    ))

    # 变换后的网格点
    x_grid = np.linspace(-2, 2, 11)
    y_grid = np.linspace(-2, 2, 11)
    X, Y = np.meshgrid(x_grid, y_grid)

    # 应用变换
    X_transformed = scale * (X * np.cos(theta) - Y * np.sin(theta))
    Y_transformed = scale * (X * np.sin(theta) + Y * np.cos(theta))

    fig.add_trace(go.Scatter(
        x=X_transformed.flatten(),
        y=Y_transformed.flatten(),
        mode='markers',
        name='变换后的网格',
        marker=dict(color=APPLE_GRAY, size=2, opacity=0.5),
        showlegend=False
    ))

    fig.update_layout(
        title='线性变换示例',
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        width=800,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 保存为静态图片
    fig.write_image(PLOT_DIR / "vector-transformation.png", scale=2)
    print(f"  ✓ 保存到 {PLOT_DIR / 'vector-transformation.png'}")
    return fig

def plot_eigenvectors():
    """图2：对称矩阵的特征向量"""
    print("生成 eigenvectors 图形...")

    fig = go.Figure()

    # 生成单位圆
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        name='单位圆',
        line=dict(color=APPLE_GRAY, width=2),
        showlegend=False
    ))

    # 特征向量
    lambda1, lambda2 = 2, 0.5

    # 第一个特征向量 (绿色)
    fig.add_trace(go.Scatter(
        x=[0, lambda1 * np.cos(0)],
        y=[0, lambda1 * np.sin(0)],
        mode='lines+markers',
        name=f'特征向量 1 (λ={lambda1})',
        line=dict(color=APPLE_GREEN, width=4),
        marker=dict(size=12)
    ))

    # 第二个特征向量 (橙色)
    fig.add_trace(go.Scatter(
        x=[0, lambda2 * np.cos(np.pi/2)],
        y=[0, lambda2 * np.sin(np.pi/2)],
        mode='lines+markers',
        name=f'特征向量 2 (λ={lambda2})',
        line=dict(color=APPLE_ORANGE, width=4),
        marker=dict(size=12)
    ))

    # 变换后的椭圆
    x_ellipse = lambda1 * np.cos(theta)
    y_ellipse = lambda2 * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=x_ellipse, y=y_ellipse,
        mode='lines',
        name='变换后的椭圆',
        line=dict(color=APPLE_BLUE, width=3),
        fill='toself',
        fillcolor='rgba(0,122,255,0.1)'
    ))

    fig.update_layout(
        title='对称矩阵的特征向量',
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        width=800,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.write_image(PLOT_DIR / "eigenvectors.png", scale=2)
    print(f"  ✓ 保存到 {PLOT_DIR / 'eigenvectors.png'}")
    return fig

def plot_svd_visualization():
    """图3：SVD 分解可视化"""
    print("生成 svd-visualization 图形...")

    from scipy.linalg import svd
    np.random.seed(42)

    # 生成数据
    n_points = 100
    x = np.random.randn(n_points)
    y = 0.5 * x + np.random.randn(n_points) * 0.3

    # 中心化
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    # SVD
    X = np.column_stack([x_centered, y_centered])
    U, s, Vt = svd(X)

    fig = go.Figure()

    # 原始数据
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='原始数据',
        marker=dict(color=APPLE_GRAY, size=6, opacity=0.6)
    ))

    # 第一主成分方向
    pc1_x = [np.mean(x), np.mean(x) + Vt[0, 0] * s[0] * 0.5]
    pc1_y = [np.mean(y), np.mean(y) + Vt[0, 1] * s[0] * 0.5]

    fig.add_trace(go.Scatter(
        x=pc1_x, y=pc1_y,
        mode='lines+markers',
        name=f'第一主成分 (σ={s[0]:.2f})',
        line=dict(color=APPLE_BLUE, width=4),
        marker=dict(size=10)
    ))

    # 第二主成分方向
    pc2_x = [np.mean(x), np.mean(x) + Vt[1, 0] * s[1] * 0.5]
    pc2_y = [np.mean(y), np.mean(y) + Vt[1, 1] * s[1] * 0.5]

    fig.add_trace(go.Scatter(
        x=pc2_x, y=pc2_y,
        mode='lines+markers',
        name=f'第二主成分 (σ={s[1]:.2f})',
        line=dict(color=APPLE_GREEN, width=4),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title='SVD 分解可视化',
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        width=800,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.write_image(PLOT_DIR / "svd-visualization.png", scale=2)
    print(f"  ✓ 保存到 {PLOT_DIR / 'svd-visualization.png'}")
    return fig

def plot_pca_example():
    """图4：PCA 降维示例"""
    print("生成 pca-example 图形...")

    np.random.seed(42)

    # 生成两个聚类的数据
    n_samples = 50

    # 第一个聚类
    x1 = np.random.randn(n_samples) * 0.3 + 1
    y1 = x1 * 0.8 + np.random.randn(n_samples) * 0.2

    # 第二个聚类
    x2 = np.random.randn(n_samples) * 0.3 - 1
    y2 = x2 * 0.8 + np.random.randn(n_samples) * 0.2

    # 计算第一主成分
    all_x = np.concatenate([x1, x2])
    all_y = np.concatenate([y1, y2])

    X = np.column_stack([all_x - np.mean(all_x), all_y - np.mean(all_y)])
    from scipy.linalg import svd
    U, s, Vt = svd(X)

    # 创建子图
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=('原始二维数据', '投影到第一主成分'),
        horizontal_spacing=0.1
    )

    # 左图：原始数据
    fig.add_trace(go.Scatter(
        x=x1, y=y1,
        mode='markers',
        name='聚类 1',
        marker=dict(color=APPLE_RED, size=8, opacity=0.7),
        legendgroup='cluster1'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        mode='markers',
        name='聚类 2',
        marker=dict(color=APPLE_BLUE, size=8, opacity=0.7),
        legendgroup='cluster2'
    ), row=1, col=1)

    # 第一主成分方向线
    pc_x = [np.mean(all_x) - Vt[0, 0] * 2, np.mean(all_x) + Vt[0, 0] * 2]
    pc_y = [np.mean(all_y) - Vt[0, 1] * 2, np.mean(all_y) + Vt[0, 1] * 2]

    fig.add_trace(go.Scatter(
        x=pc_x, y=pc_y,
        mode='lines',
        name='第一主成分',
        line=dict(color=APPLE_GREEN, width=3),
        showlegend=False
    ), row=1, col=1)

    # 投影到第一主成分
    proj1 = (x1 - np.mean(all_x)) * Vt[0, 0] + (y1 - np.mean(all_y)) * Vt[0, 1]
    proj2 = (x2 - np.mean(all_x)) * Vt[0, 0] + (y2 - np.mean(all_y)) * Vt[0, 1]

    # 右图：投影后的数据
    fig.add_trace(go.Scatter(
        x=proj1, y=[0] * len(proj1),
        mode='markers',
        name='聚类 1 (投影)',
        marker=dict(color=APPLE_RED, size=8, opacity=0.7),
        showlegend=False,
        legendgroup='cluster1'
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=proj2, y=[0] * len(proj2),
        mode='markers',
        name='聚类 2 (投影)',
        marker=dict(color=APPLE_BLUE, size=8, opacity=0.7),
        showlegend=False,
        legendgroup='cluster2'
    ), row=1, col=2)

    fig.update_xaxes(title_text='x', row=1, col=1)
    fig.update_yaxes(title_text='y', row=1, col=1)
    fig.update_xaxes(title_text='第一主成分', row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        width=1000,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.write_image(PLOT_DIR / "pca-example.png", scale=2)
    print(f"  ✓ 保存到 {PLOT_DIR / 'pca-example.png'}")
    return fig

def plot_neural_network_matrices():
    """图5：神经网络中的矩阵运算"""
    print("生成 neural-network-matrices 图形...")

    fig = go.Figure()

    # 简化的神经网络架构
    layers = [2, 3, 2]  # 输入层2个神经元，隐藏层3个，输出层2个
    layer_names = ['输入层', '隐藏层', '输出层']

    # 绘制神经元
    node_positions = []
    for i, (num_nodes, layer_name) in enumerate(zip(layers, layer_names)):
        layer_x = i
        for j in range(num_nodes):
            layer_y = (j - (num_nodes - 1) / 2) / max(layers) * 2
            node_positions.append((layer_x, layer_y, i, j))

            fig.add_trace(go.Scatter(
                x=[layer_x], y=[layer_y],
                mode='markers',
                name=layer_name if j == 0 else '',
                marker=dict(
                    color=APPLE_BLUE if i == 0 else APPLE_GREEN if i == 1 else APPLE_ORANGE,
                    size=30,
                    line=dict(color='white', width=2)
                ),
                showlegend=False,
                hovertext=f'{layer_name}<br>神经元 {j+1}'
            ))

    # 绘制连接（仅显示部分连接以避免混乱）
    for i in range(len(node_positions)):
        for j in range(len(node_positions)):
            x1, y1, l1, n1 = node_positions[i]
            x2, y2, l2, n2 = node_positions[j]

            # 只连接相邻层
            if l2 == l1 + 1:
                fig.add_trace(go.Scatter(
                    x=[x1, x2, None],
                    y=[y1, y2, None],
                    mode='lines',
                    line=dict(color=APPLE_GRAY, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # 添加矩阵运算标注
    annotations = [
        dict(x=0.5, y=1.3, text='X<br>(输入向量)', showarrow=False, font=dict(size=12)),
        dict(x=1.5, y=1.3, text='σ(W₁X + b₁)<br>(隐藏层)', showarrow=False, font=dict(size=12)),
        dict(x=2.5, y=1.3, text='σ(W₂H + b₂)<br>(输出)', showarrow=False, font=dict(size=12)),
    ]

    fig.update_layout(
        title='神经网络中的矩阵运算',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=annotations,
        width=800,
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fig.write_image(PLOT_DIR / "neural-network-matrices.png", scale=2)
    print(f"  ✓ 保存到 {PLOT_DIR / 'neural-network-matrices.png'}")
    return fig

def plot_attention_mechanism():
    """图6：自注意力机制矩阵计算"""
    print("生成 attention-mechanism 图形...")

    np.random.seed(42)

    # 生成注意力分数矩阵
    n_tokens = 5
    attention_scores = np.random.randn(n_tokens, n_tokens)
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)

    token_names = [f'Token {i+1}' for i in range(n_tokens)]

    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=('注意力分数 (QKᵀ)', '注意力权重 (Softmax)'),
        horizontal_spacing=0.15
    )

    # 左图：注意力分数
    fig.add_trace(go.Heatmap(
        z=attention_scores,
        x=token_names,
        y=token_names,
        colorscale='RdBu',
        showscale=True,
        colorbar=dict(title='分数', x=0.45)
    ), row=1, col=1)

    # 右图：注意力权重
    fig.add_trace(go.Heatmap(
        z=attention_weights,
        x=token_names,
        y=token_names,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title='权重', x=1.0)
    ), row=1, col=2)

    fig.update_xaxes(title_text='Key', row=1, col=1)
    fig.update_yaxes(title_text='Query', row=1, col=1)
    fig.update_xaxes(title_text='Key', row=1, col=2)
    fig.update_yaxes(title_text='', row=1, col=2)

    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1000,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.write_image(PLOT_DIR / "attention-mechanism.png", scale=2)
    print(f"  ✓ 保存到 {PLOT_DIR / 'attention-mechanism.png'}")
    return fig

def main():
    """生成所有图形"""
    print("开始生成线性代数文章的静态图形...")

    plot_vector_transformation()
    plot_eigenvectors()
    plot_svd_visualization()
    plot_pca_example()
    plot_neural_network_matrices()
    plot_attention_mechanism()

    print("\n✅ 所有图形生成完成！")

if __name__ == "__main__":
    import plotly.subplots as sp
    from scipy.linalg import svd
    main()
