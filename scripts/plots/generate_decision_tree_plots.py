#!/usr/bin/env python3
"""
决策树算法配图生成
包含：决策树结构、信息增益、基尼指数、集成学习等
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

os.makedirs('static/images/plots', exist_ok=True)

template = 'plotly_white'
primary_color = '#007AFF'
secondary_color = '#34C759'
accent_color = '#FF9500'

def save_fig(fig, filename):
    fig.write_image(f'static/images/plots/{filename}', scale=2)
    print(f'✅ {filename} 已保存')

print("开始生成决策树算法配图...")

# ========== 图1: 决策树结构示意 ==========
print("生成图1: 决策树结构...")

fig1 = go.Figure()

# 节点位置
nodes = {
    'root': (0.5, 1),
    'left': (0.25, 0.7),
    'right': (0.75, 0.7),
    'll': (0.15, 0.4),
    'lr': (0.35, 0.4),
    'rl': (0.65, 0.4),
    'rr': (0.85, 0.4),
}

# 绘制边
edges = [('root', 'left'), ('root', 'right'), 
         ('left', 'll'), ('left', 'lr'),
         ('right', 'rl'), ('right', 'rr')]

for start, end in edges:
    x0, y0 = nodes[start]
    x1, y1 = nodes[end]
    fig1.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode='lines',
        line=dict(color='gray', width=2),
        showlegend=False
    ))

# 绘制节点
node_labels = {
    'root': ('根节点', primary_color),
    'left': ('A<阈值', secondary_color),
    'right': ('A≥阈值', secondary_color),
    'll': ('类别1', accent_color),
    'lr': ('类别2', accent_color),
    'rl': ('类别3', accent_color),
    'rr': ('类别4', accent_color)
}

for name, (x, y) in nodes.items():
    label, color = node_labels[name]
    # 绘制圆圈
    fig1.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers',
        marker=dict(size=55, color=color),
        showlegend=False
    ))
    # 绘制文字（在圆圈上方）
    fig1.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='text',
        text=[label],
        textposition='middle center',
        textfont=dict(size=11, color='white', family='Arial, sans-serif'),
        showlegend=False
    ))

fig1.update_layout(
    title='决策树结构：从根节点到叶节点的决策路径',
    xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.2]),
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=800,
    height=600
)

save_fig(fig1, 'decision_tree_structure.png')

# ========== 图2: 信息熵与信息增益 ==========
print("生成图2: 信息熵...")

# 熵函数 H(p) = -p*log(p) - (1-p)*log(1-p)
p = np.linspace(0.01, 0.99, 100)
entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=p, y=entropy,
    mode='lines',
    line=dict(color=primary_color, width=3),
    name='熵 H(p)'
))

# 标注最大熵点
fig2.add_annotation(
    x=0.5, y=1,
    text='最大熵 = 1<br>(最不确定)',
    showarrow=True,
    arrowhead=2,
    ax=40, ay=-40
)

fig2.add_annotation(
    x=0.05, y=0.2,
    text='熵 = 0<br>(纯节点)',
    showarrow=True,
    arrowhead=2,
    ax=60, ay=0
)

fig2.update_layout(
    title='信息熵：度量节点的不纯度',
    xaxis_title='正类比例 p',
    yaxis_title='熵 H(p)',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=700,
    height=450
)

save_fig(fig2, 'information_entropy.png')

# ========== 图3: 基尼指数 ==========
print("生成图3: 基尼指数...")

# 基尼指数 G(p) = 2*p*(1-p)
gini = 2 * p * (1-p)

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=p, y=gini,
    mode='lines',
    line=dict(color=accent_color, width=3),
    name='基尼指数 G(p)'
))

fig3.add_trace(go.Scatter(
    x=p, y=entropy,
    mode='lines',
    line=dict(color=primary_color, width=2, dash='dash'),
    name='熵 H(p) (对比)'
))

fig3.update_layout(
    title='基尼指数 vs 熵：两种不纯度度量',
    xaxis_title='正类比例 p',
    yaxis_title='不纯度',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=700,
    height=450,
    legend=dict(x=0.02, y=0.98)
)

save_fig(fig3, 'gini_index.png')

# ========== 图4: 特征空间划分 ==========
print("生成图4: 特征空间划分...")

# 生成二维分类数据
np.random.seed(42)
n_samples = 200

# 类别0
X0 = np.random.randn(n_samples//2, 2) + np.array([2, 2])
# 类别1
X1 = np.random.randn(n_samples//2, 2) + np.array([-2, -2])

X = np.vstack([X0, X1])
y = np.array([0]* (n_samples//2) + [1]* (n_samples//2))

fig4 = go.Figure()

# 绘制数据点
fig4.add_trace(go.Scatter(
    x=X0[:, 0], y=X0[:, 1],
    mode='markers',
    marker=dict(size=8, color=primary_color, symbol='circle'),
    name='类别0'
))

fig4.add_trace(go.Scatter(
    x=X1[:, 0], y=X1[:, 1],
    mode='markers',
    marker=dict(size=8, color=accent_color, symbol='square'),
    name='类别1'
))

# 绘制决策边界
fig4.add_vline(x=0, line=dict(color='red', width=2, dash='dash'))
fig4.add_hline(y=0, line=dict(color='green', width=2, dash='dash'))

# 添加区域标注
fig4.add_annotation(x=3, y=3, text='区域A<br>(类别0)', showarrow=False, font=dict(size=12))
fig4.add_annotation(x=-3, y=-3, text='区域B<br>(类别1)', showarrow=False, font=dict(size=12))
fig4.add_annotation(x=3, y=-3, text='区域C<br>(混合)', showarrow=False, font=dict(size=12))
fig4.add_annotation(x=-3, y=3, text='区域D<br>(混合)', showarrow=False, font=dict(size=12))

fig4.update_layout(
    title='决策树对特征空间的轴平行划分',
    xaxis_title='特征1',
    yaxis_title='特征2',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=700,
    height=600
)

save_fig(fig4, 'feature_space_partition.png')

# ========== 图5: 随机森林投票机制 ==========
print("生成图5: 随机森林...")

fig5 = make_subplots(
    rows=2, cols=3,
    subplot_titles=('树1', '树2', '树3', '树4', '树5', '投票结果'),
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
           [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
)

# 生成5棵不同的决策树结果
np.random.seed(42)
x_range = np.linspace(-3, 3, 100)

for i in range(5):
    row = i // 3 + 1
    col = i % 3 + 1
    
    # 模拟每棵树的决策边界略有不同
    offset = np.random.randn() * 0.3
    boundary = offset
    
    # 绘制数据点
    fig5.add_trace(go.Scatter(
        x=X0[:, 0], y=X0[:, 1],
        mode='markers',
        marker=dict(size=5, color=primary_color),
        showlegend=False
    ), row=row, col=col)
    
    fig5.add_trace(go.Scatter(
        x=X1[:, 0], y=X1[:, 1],
        mode='markers',
        marker=dict(size=5, color=accent_color),
        showlegend=False
    ), row=row, col=col)
    
    # 决策边界
    fig5.add_vline(x=boundary, line=dict(color='red', width=2), row=row, col=col)

# 最后一格显示投票结果（集成）
fig5.add_trace(go.Scatter(
    x=X0[:, 0], y=X0[:, 1],
    mode='markers',
    marker=dict(size=5, color=primary_color),
    showlegend=False
), row=2, col=3)

fig5.add_trace(go.Scatter(
    x=X1[:, 0], y=X1[:, 1],
    mode='markers',
    marker=dict(size=5, color=accent_color),
    showlegend=False
), row=2, col=3)

fig5.add_vline(x=0, line=dict(color='green', width=3), row=2, col=3)

fig5.update_layout(
    title='随机森林：多棵决策树的Bagging集成',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=900,
    height=600
)

save_fig(fig5, 'random_forest.png')

# ========== 图6: 梯度提升过程 ==========
print("生成图6: 梯度提升...")

# 一维回归问题
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_true = np.sin(x) + 0.1 * x

# 模拟梯度提升过程
fig6 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('第1轮', '第2轮', '第5轮', '第20轮（最终）'),
    vertical_spacing=0.15
)

iterations = [1, 2, 5, 20]
for idx, n_iter in enumerate(iterations):
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    # 模拟预测结果（逐步逼近真实值）
    learning_rate = 0.1
    y_pred = np.zeros_like(x)
    for i in range(n_iter):
        residual = y_true - y_pred
        # 简单模拟树拟合残差
        y_pred += learning_rate * residual * 0.3
    
    fig6.add_trace(go.Scatter(
        x=x, y=y_true,
        mode='lines',
        line=dict(color='gray', width=2),
        name='真实值',
        showlegend=(idx==0)
    ), row=row, col=col)
    
    fig6.add_trace(go.Scatter(
        x=x, y=y_pred,
        mode='lines',
        line=dict(color=accent_color, width=2),
        name='预测值',
        showlegend=(idx==0)
    ), row=row, col=col)
    
    # 残差
    fig6.add_trace(go.Scatter(
        x=x, y=y_true - y_pred,
        mode='lines',
        line=dict(color='red', width=1, dash='dash'),
        name='残差',
        showlegend=(idx==0)
    ), row=row, col=col)

fig6.update_layout(
    title='梯度提升：串行训练，逐步减小残差',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=800,
    height=600
)

save_fig(fig6, 'gradient_boosting.png')

# ========== 图7: 算法对比 ==========
print("生成图7: 算法性能对比...")

algorithms = ['决策树', '随机森林', 'XGBoost', 'LightGBM', 'CatBoost']
accuracy = [0.82, 0.91, 0.94, 0.93, 0.945]
training_time = [1, 5, 8, 3, 12]  # 相对时间

fig7 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('准确率对比', '训练时间对比（相对值）'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}]]
)

# 准确率
fig7.add_trace(go.Bar(
    x=algorithms,
    y=accuracy,
    marker_color=[accent_color, primary_color, secondary_color, '#FF6B6B', '#4ECDC4'],
    text=[f'{a:.1%}' for a in accuracy],
    textposition='outside',
    showlegend=False
), row=1, col=1)

# 训练时间
fig7.add_trace(go.Bar(
    x=algorithms,
    y=training_time,
    marker_color=[accent_color, primary_color, secondary_color, '#FF6B6B', '#4ECDC4'],
    text=[f'{t}x' for t in training_time],
    textposition='outside',
    showlegend=False
), row=1, col=2)

fig7.update_layout(
    title='决策树及其衍生算法性能对比',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=900,
    height=450
)

save_fig(fig7, 'algorithm_comparison.png')

# ========== 图8: 过拟合与剪枝 ==========
print("生成图8: 过拟合与剪枝...")

depth = np.arange(1, 21)
# 模拟训练误差和验证误差
train_error = 0.5 * np.exp(-depth / 3) + 0.05
val_error = 0.5 * np.exp(-depth / 5) + 0.05 + 0.002 * (depth - 8)**2

fig8 = go.Figure()

fig8.add_trace(go.Scatter(
    x=depth, y=train_error,
    mode='lines+markers',
    line=dict(color=primary_color, width=2),
    name='训练误差'
))

fig8.add_trace(go.Scatter(
    x=depth, y=val_error,
    mode='lines+markers',
    line=dict(color=accent_color, width=2),
    name='验证误差'
))

# 标注最优深度
optimal_idx = np.argmin(val_error)
fig8.add_vline(x=depth[optimal_idx], line=dict(color='red', dash='dash', width=2))
fig8.add_annotation(
    x=depth[optimal_idx], y=val_error[optimal_idx],
    text='最优深度<br>(剪枝点)',
    showarrow=True,
    arrowhead=2,
    ax=40, ay=-40
)

# 标注过拟合区域
fig8.add_annotation(
    x=15, y=0.15,
    text='过拟合区域',
    showarrow=False,
    font=dict(color='red')
)

fig8.update_layout(
    title='过拟合与剪枝：树深度对泛化性能的影响',
    xaxis_title='树深度',
    yaxis_title='误差',
    template=template,
    font=dict(family='Arial, sans-serif', size=12),
    width=700,
    height=450,
    legend=dict(x=0.02, y=0.98)
)

save_fig(fig8, 'overfitting_pruning.png')

print("\n✅ 所有决策树配图生成完成！")
