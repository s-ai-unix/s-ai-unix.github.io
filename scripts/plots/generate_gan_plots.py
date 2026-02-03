import plotly.graph_objects as go
import numpy as np
import os

# 创建输出目录
os.makedirs('static/images/math', exist_ok=True)

# ============================================
# 图1: GAN 架构示意图
# ============================================
fig1 = go.Figure()

# 生成器节点
z_node = go.Scatter(
    x=[1], y=[3], mode='markers+text',
    marker=dict(size=30, color='#FF9500', line=dict(width=3, color='#FF9500')),
    text=['z'], textposition='middle center',
    textfont=dict(size=14, color='white'),
    name='噪声 z'
)

g_layers = [
    (3, 3, '全连接'),
    (5, 3, '上采样1'),
    (7, 3, '上采样2'),
    (9, 3, '上采样3'),
    (11, 3, '输出层')
]

for x, y, label in g_layers:
    fig1.add_trace(go.Scatter(
        x=[x], y=[y], mode='markers+text',
        marker=dict(size=25, color='#5AC8FA', line=dict(width=2, color='#5AC8FA')),
        text=[label], textposition='middle center',
        textfont=dict(size=10, color='white'),
        showlegend=False
    ))

# 判别器节点
x_real = go.Scatter(
    x=[11], y=[1], mode='markers+text',
    marker=dict(size=30, color='#34C759', line=dict(width=3, color='#34C759')),
    text=['x'], textposition='middle center',
    textfont=dict(size=14, color='white'),
    name='真实 x'
)

x_fake = go.Scatter(
    x=[13], y=[3], mode='markers+text',
    marker=dict(size=30, color='#FF9500', line=dict(width=2, color='#FF9500')),
    text=['x̃'], textposition='middle center',
    textfont=dict(size=14, color='white'),
    name='生成 x̃'
)

d_layers = [
    (15, 2, '卷积1'),
    (17, 2, '卷积2'),
    (19, 2, '卷积3'),
    (21, 2, '输出D')
]

for x, y, label in d_layers:
    fig1.add_trace(go.Scatter(
        x=[x], y=[y], mode='markers+text',
        marker=dict(size=25, color='#AF52DE', line=dict(width=2, color='#AF52DE')),
        text=[label], textposition='middle center',
        textfont=dict(size=10, color='white'),
        showlegend=False
    ))

# 连接线
# 生成器内部连接
for i in range(len(g_layers)-1):
    fig1.add_trace(go.Scatter(
        x=[g_layers[i][0], g_layers[i+1][0]],
        y=[g_layers[i][1], g_layers[i+1][1]],
        mode='lines', line=dict(color='#FF9500', width=2), showlegend=False
    ))

# z -> G[0]
fig1.add_trace(go.Scatter(
    x=[1, g_layers[0][0]], y=[3, g_layers[0][1]],
    mode='lines', line=dict(color='#FF9500', width=2), showlegend=False
))

# G[-1] -> x_fake
fig1.add_trace(go.Scatter(
    x=[g_layers[-1][0], x_fake.x[0]], y=[g_layers[-1][1], x_fake.y[0]],
    mode='lines', line=dict(color='#FF9500', width=2), showlegend=False
))

# x_real, x_fake -> 判别器
fig1.add_trace(go.Scatter(
    x=[x_real.x[0], x_fake.x[0]], y=[x_real.y[0], x_fake.y[0]],
    mode='lines', line=dict(color='#FF9500', width=2), showlegend=False
))

# 判别器内部连接
for i in range(len(d_layers)-1):
    fig1.add_trace(go.Scatter(
        x=[d_layers[i][0], d_layers[i+1][0]],
        y=[d_layers[i][1], d_layers[i+1][1]],
        mode='lines', line=dict(color='#AF52DE', width=2), showlegend=False
    ))

fig1.update_layout(
    title='GAN 架构示意图',
    xaxis=dict(showgrid=False, showticklabels=False, range=[0, 23]),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, 4]),
    width=1000,
    height=400,
    font=dict(size=12),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig1.write_image('static/images/math/gan-architecture.png', scale=2)

# ============================================
# 图2: 训练损失函数收敛曲线
# ============================================
epochs = np.arange(0, 500)
# 模拟损失曲线
d_loss = 0.69 - 0.15 * np.exp(-epochs/100) + 0.1 * np.sin(epochs/30) * np.exp(-epochs/200)
g_loss = 0.8 - 0.2 * np.exp(-epochs/80) + 0.15 * np.cos(epochs/25) * np.exp(-epochs/180)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=epochs, y=d_loss,
    mode='lines',
    line=dict(color='#FF3B30', width=2),
    name='判别器损失'
))
fig2.add_trace(go.Scatter(
    x=epochs, y=g_loss,
    mode='lines',
    line=dict(color='#007AFF', width=2),
    name='生成器损失'
))

fig2.update_layout(
    title='GAN 训练损失曲线',
    xaxis_title='训练轮数 (Epochs)',
    yaxis_title='损失值',
    template='plotly_white',
    width=800,
    height=500,
    font=dict(size=14),
    plot_bgcolor='white',
    xaxis=dict(gridcolor='#e0e0e0', zerolinecolor='#333'),
    yaxis=dict(gridcolor='#e0e0e0', zerolinecolor='#333'),
    legend=dict(x=0.7, y=0.95)
)

fig2.write_image('static/images/math/gan-training-loss.png', scale=2)

# ============================================
# 图3: Wasserstein 距离直观理解
# ============================================
# 创建两个分布
x = np.linspace(-4, 4, 500)
p_data = np.exp(-x**2 / 2) / np.sqrt(2*np.pi)  # 标准正态分布
p_gen = np.exp(-(x-1)**2 / 2) / np.sqrt(2*np.pi)  # 平移的正态分布

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=x, y=p_data,
    mode='lines',
    line=dict(color='#34C759', width=3),
    fill='tonexty',
    fillcolor='rgba(52, 199, 89, 0.2)',
    name='真实分布 p_data'
))
fig3.add_trace(go.Scatter(
    x=x, y=p_gen,
    mode='lines',
    line=dict(color='#FF9500', width=3),
    fill='tonexty',
    fillcolor='rgba(255, 149, 0, 0.2)',
    name='生成分布 p_gen'
))

# 添加移动质量的箭头
for i in range(5):
    x_pos = -2 + i * 0.8
    fig3.add_annotation(
        x=x_pos, y=p_data[np.argmin(np.abs(x-x_pos))] + 0.05,
        ax=x_pos+1, ay=p_gen[np.argmin(np.abs(x-(x_pos+1)))] + 0.05,
        arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#AF52DE'
    )

fig3.update_layout(
    title='Wasserstein 距离的直观理解',
    xaxis_title='x',
    yaxis_title='概率密度',
    template='plotly_white',
    width=800,
    height=500,
    font=dict(size=14),
    plot_bgcolor='white',
    xaxis=dict(gridcolor='#e0e0e0', zerolinecolor='#333'),
    yaxis=dict(gridcolor='#e0e0e0', zerolinecolor='#333'),
    legend=dict(x=0.65, y=0.95)
)

fig3.write_image('static/images/math/wasserstein-distance.png', scale=2)

# ============================================
# 图4: JS 散度 vs Wasserstein 距离
# ============================================
x_vals = np.linspace(-3, 3, 100)
# 两个高斯分布,均值逐渐远离
js_div = []
wasserstein_dist = []

for mean_diff in x_vals:
    p1 = lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi)
    p2 = lambda x: np.exp(-(x-mean_diff)**2/2) / np.sqrt(2*np.pi)

    # 近似 JS 散度
    x_grid = np.linspace(-5, 5, 1000)
    m = (p1(x_grid) + p2(x_grid)) / 2
    kl1 = np.sum(p1(x_grid) * np.log(p1(x_grid) / (m + 1e-10) + 1e-10)) * (10/1000)
    kl2 = np.sum(p2(x_grid) * np.log(p2(x_grid) / (m + 1e-10) + 1e-10)) * (10/1000)
    js_div.append((kl1 + kl2) / 2)

    # Wasserstein 距离(对于1D高斯就是均值差的绝对值)
    wasserstein_dist.append(abs(mean_diff))

fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=x_vals, y=js_div,
    mode='lines',
    line=dict(color='#FF3B30', width=3),
    name='JS 散度'
))
fig4.add_trace(go.Scatter(
    x=x_vals, y=wasserstein_dist,
    mode='lines',
    line=dict(color='#007AFF', width=3),
    name='Wasserstein 距离'
))

fig4.update_layout(
    title='JS 散度 vs Wasserstein 距离',
    xaxis_title='分布均值差',
    yaxis_title='距离值',
    template='plotly_white',
    width=800,
    height=500,
    font=dict(size=14),
    plot_bgcolor='white',
    xaxis=dict(gridcolor='#e0e0e0', zerolinecolor='#333'),
    yaxis=dict(gridcolor='#e0e0e0', zerolinecolor='#333'),
    legend=dict(x=0.65, y=0.95)
)

fig4.write_image('static/images/math/js-vs-wasserstein.png', scale=2)

# ============================================
# 图5: 潜空间插值可视化
# ============================================
from plotly.subplots import make_subplots

np.random.seed(42)
z1 = np.random.randn(10)
z2 = np.random.randn(10)

# 插值
interpolation_ratios = np.linspace(0, 1, 6)
interpolated_z = [(1-t)*z1 + t*z2 for t in interpolation_ratios]

fig5 = make_subplots(rows=2, cols=3, subplot_titles=[f'α={t:.1f}' for t in interpolation_ratios])

for i, z in enumerate(interpolated_z):
    # 模拟一个5x5的图像
    img = z.reshape(5, 2)
    img = np.tile(img[:, 0].reshape(5, 1), (1, 5))

    row = i // 3 + 1
    col = i % 3 + 1

    fig5.add_trace(
        go.Heatmap(z=img, colorscale='Viridis', showscale=(i==5)),
        row=row, col=col
    )

fig5.update_layout(
    title='潜空间插值',
    width=800,
    height=600,
    font=dict(size=14)
)

fig5.write_image('static/images/math/latent-interpolation.png', scale=2)

print("所有图片生成完成!")
print("- gan-architecture.png")
print("- gan-training-loss.png")
print("- wasserstein-distance.png")
print("- js-vs-wasserstein.png")
print("- latent-interpolation.png")
