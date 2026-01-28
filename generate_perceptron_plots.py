import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

os.makedirs('static/images/plots', exist_ok=True)

def apply_apple_style(fig, title, width=800, height=500):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family='Arial, sans-serif')),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=width,
        height=height,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# ============ 图1: 线性可分 vs 线性不可分对比 ============
print("生成图1: 线性可分 vs 线性不可分...")

fig1 = make_subplots(rows=1, cols=2,
                     subplot_titles=('AND问题 - 线性可分', 'XOR问题 - 线性不可分'),
                     specs=[[{'type': 'xy'}, {'type': 'xy'}]])

# AND 问题数据
and_x0, and_y0 = [0, 0, 1], [0, 1, 0]  # 输出0
and_x1, and_y1 = [1], [1]  # 输出1

fig1.add_trace(go.Scatter(
    x=and_x0, y=and_y0, mode='markers',
    marker=dict(size=20, color='#FF3B30', symbol='circle'),
    name='输出 0', showlegend=True
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=and_x1, y=and_y1, mode='markers',
    marker=dict(size=20, color='#007AFF', symbol='circle'),
    name='输出 1', showlegend=True
), row=1, col=1)

# 添加决策边界线
x_line = np.linspace(-0.5, 1.5, 100)
y_line = 1.5 - x_line  # 决策边界
fig1.add_trace(go.Scatter(
    x=x_line, y=y_line, mode='lines',
    line=dict(color='#34C759', width=3, dash='solid'),
    name='决策边界', showlegend=True
), row=1, col=1)

# 添加阴影区域表示分类区域
fig1.add_trace(go.Scatter(
    x=[-0.5, 1.5, 1.5, -0.5], y=[2, 2, 1.5-1.5, 1.5+0.5],
    fill='toself', fillcolor='rgba(0,122,255,0.1)',
    line=dict(width=0), showlegend=False
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=[-0.5, 1.5, 1.5, -0.5], y=[1.5-1.5, 1.5+0.5, -0.5, -0.5],
    fill='toself', fillcolor='rgba(255,59,48,0.1)',
    line=dict(width=0), showlegend=False
), row=1, col=1)

# XOR 问题数据
xor_x0, xor_y0 = [0, 1], [0, 1]  # 输出0
xor_x1, xor_y1 = [0, 1], [1, 0]  # 输出1

fig1.add_trace(go.Scatter(
    x=xor_x0, y=xor_y0, mode='markers',
    marker=dict(size=20, color='#FF3B30', symbol='circle'),
    name='输出 0', showlegend=False
), row=1, col=2)

fig1.add_trace(go.Scatter(
    x=xor_x1, y=xor_y1, mode='markers',
    marker=dict(size=20, color='#007AFF', symbol='circle'),
    name='输出 1', showlegend=False
), row=1, col=2)

# 尝试添加一条直线（说明无法分类）
fig1.add_trace(go.Scatter(
    x=[-0.5, 1.5], y=[1.5, -0.5], mode='lines',
    line=dict(color='#FF9500', width=2, dash='dash'),
    showlegend=False
), row=1, col=2)

# 添加"无法分类"文字
fig1.add_annotation(x=0.5, y=0.5, text='任何直线<br>都无法<br>正确分类',
                   showarrow=False, font=dict(size=14, color='#FF3B30'),
                   row=1, col=2)

# 添加红叉表示分类错误
fig1.add_trace(go.Scatter(
    x=[0.25, 0.75], y=[0.25, 0.75], mode='markers',
    marker=dict(size=15, color='#FF3B30', symbol='x'),
    showlegend=False
), row=1, col=2)

fig1.update_xaxes(range=[-0.5, 1.5], dtick=1, title='x₁')
fig1.update_yaxes(range=[-0.5, 1.5], dtick=1, title='x₂')

apply_apple_style(fig1, '', 900, 450)
fig1.write_image('static/images/plots/perceptron-linear-separability.png', scale=2)

# ============ 图2: 多层感知机解决XOR问题的决策边界 ============
print("生成图2: 多层感知机解决XOR...")

fig2 = go.Figure()

# XOR 数据点
fig2.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode='markers+text',
    marker=dict(size=25, color='#FF3B30', symbol='circle'),
    text=['0', '0'], textposition='middle center',
    textfont=dict(size=14, color='white'),
    name='类别 0'
))

fig2.add_trace(go.Scatter(
    x=[0, 1], y=[1, 0], mode='markers+text',
    marker=dict(size=25, color='#007AFF', symbol='circle'),
    text=['1', '1'], textposition='middle center',
    textfont=dict(size=14, color='white'),
    name='类别 1'
))

# 绘制两条隐藏层的决策边界
x_range = np.linspace(-0.5, 1.5, 100)

# 第一条边界: x1 + x2 = 0.5
y1 = 0.5 - x_range
fig2.add_trace(go.Scatter(
    x=x_range, y=y1, mode='lines',
    line=dict(color='#FF9500', width=3),
    name='隐藏层1边界'
))

# 第二条边界: x1 + x2 = 1.5
y2 = 1.5 - x_range
fig2.add_trace(go.Scatter(
    x=x_range, y=y2, mode='lines',
    line=dict(color='#34C759', width=3),
    name='隐藏层2边界'
))

# 添加区域标注
fig2.add_annotation(x=0.5, y=0.1, text='区域A', showarrow=False, font=dict(size=12))
fig2.add_annotation(x=0.5, y=0.9, text='区域B', showarrow=False, font=dict(size=12))
fig2.add_annotation(x=0.1, y=0.5, text='区域C', showarrow=False, font=dict(size=12))
fig2.add_annotation(x=0.9, y=0.5, text='区域D', showarrow=False, font=dict(size=12))

# 添加说明
fig2.add_annotation(x=-0.3, y=1.3, 
                   text='两条直线将空间<br>划分为4个区域，<br>实现非线性分类',
                   showarrow=False, font=dict(size=11),
                   bgcolor='rgba(255,255,255,0.9)')

fig2.update_xaxes(range=[-0.5, 1.5], title='x₁', dtick=1)
fig2.update_yaxes(range=[-0.5, 1.5], title='x₂', dtick=1)
fig2.update_layout(
    xaxis_title='x₁',
    yaxis_title='x₂',
    legend=dict(x=0.02, y=0.98)
)

apply_apple_style(fig2, '多层感知机：两条直线实现XOR分类', 700, 550)
fig2.write_image('static/images/plots/perceptron-mlp-xor.png', scale=2)

# ============ 图3: 感知机学习过程的权重更新 ============
print("生成图3: 感知机学习过程...")

fig3 = go.Figure()

# 模拟AND问题的学习过程
np.random.seed(42)
epochs = 20
weights_history = []
bias_history = []

# 初始权重
w1, w2, b = 0.1, -0.1, 0.0
lr = 0.1

# AND数据
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

w1_list, w2_list, b_list = [w1], [w2], [b]

for epoch in range(epochs):
    for i in range(len(X)):
        pred = 1 if w1 * X[i][0] + w2 * X[i][1] + b >= 0 else 0
        error = y[i] - pred
        w1 += lr * error * X[i][0]
        w2 += lr * error * X[i][1]
        b += lr * error
    w1_list.append(w1)
    w2_list.append(w2)
    b_list.append(b)

epoch_range = list(range(len(w1_list)))

fig3.add_trace(go.Scatter(
    x=epoch_range, y=w1_list, mode='lines+markers',
    name='w₁ (权重1)', line=dict(width=3, color='#007AFF'),
    marker=dict(size=8)
))

fig3.add_trace(go.Scatter(
    x=epoch_range, y=w2_list, mode='lines+markers',
    name='w₂ (权重2)', line=dict(width=3, color='#34C759'),
    marker=dict(size=8)
))

fig3.add_trace(go.Scatter(
    x=epoch_range, y=b_list, mode='lines+markers',
    name='b (偏置)', line=dict(width=3, color='#FF9500'),
    marker=dict(size=8)
))

# 添加收敛区域标注
fig3.add_hrect(y0=0.45, y1=0.55, line_width=0, 
               fillcolor="rgba(52,199,89,0.2)",
               annotation_text="收敛区域", annotation_position="top")

fig3.update_layout(
    xaxis_title='迭代次数',
    yaxis_title='参数值',
    legend=dict(x=0.02, y=0.98)
)

apply_apple_style(fig3, '感知机学习过程：权重收敛动态', 800, 500)
fig3.write_image('static/images/plots/perceptron-learning-process.png', scale=2)

# ============ 图4: 激活函数对比 ============
print("生成图4: 激活函数对比...")

fig4 = make_subplots(rows=2, cols=2,
                     subplot_titles=('阶跃函数 (Step)', 'Sigmoid函数', 'ReLU函数', 'Tanh函数'),
                     specs=[[{'type': 'xy'}, {'type': 'xy'}],
                            [{'type': 'xy'}, {'type': 'xy'}]])

x = np.linspace(-5, 5, 500)

# Step函数
y_step = np.where(x >= 0, 1, 0)
fig4.add_trace(go.Scatter(
    x=x, y=y_step, mode='lines',
    line=dict(width=3, color='#007AFF'),
    showlegend=False
), row=1, col=1)

# Sigmoid函数
y_sigmoid = 1 / (1 + np.exp(-x))
fig4.add_trace(go.Scatter(
    x=x, y=y_sigmoid, mode='lines',
    line=dict(width=3, color='#34C759'),
    showlegend=False
), row=1, col=2)

# ReLU函数
y_relu = np.maximum(0, x)
fig4.add_trace(go.Scatter(
    x=x, y=y_relu, mode='lines',
    line=dict(width=3, color='#FF9500'),
    showlegend=False
), row=2, col=1)

# Tanh函数
y_tanh = np.tanh(x)
fig4.add_trace(go.Scatter(
    x=x, y=y_tanh, mode='lines',
    line=dict(width=3, color='#FF3B30'),
    showlegend=False
), row=2, col=2)

# 更新坐标轴
for i in range(1, 3):
    for j in range(1, 3):
        fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3,
                      row=i, col=j)
        fig4.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3,
                      row=i, col=j)
        fig4.update_xaxes(range=[-5, 5], row=i, col=j)

fig4.update_yaxes(range=[-0.2, 1.2], row=1, col=1)
fig4.update_yaxes(range=[-0.2, 1.2], row=1, col=2)
fig4.update_yaxes(range=[-0.5, 5], row=2, col=1)
fig4.update_yaxes(range=[-1.2, 1.2], row=2, col=2)

apply_apple_style(fig4, '', 900, 700)
fig4.write_image('static/images/plots/perceptron-activation-functions.png', scale=2)

# ============ 图5: 感知机发展历程时间线 ============
print("生成图5: 发展历程时间线...")

fig5 = go.Figure()

# 定义里程碑事件
years = [1943, 1957, 1958, 1969, 1986, 1989, 2006, 2012]
events = [
    '麦卡洛克-皮茨<br>神经元模型',
    '罗森布拉特<br>提出感知机',
    '感知机<br>硬件实现',
    '明斯基&帕普特<br>《感知机》',
    '反向传播<br>算法普及',
    '万能逼近<br>定理证明',
    '深度学习<br>复兴',
    'AlexNet<br>ImageNet胜利'
]
importance = [3, 5, 4, 4, 5, 4, 4, 5]  # 重要程度，用于标记大小
colors = ['#007AFF', '#34C759', '#34C759', '#FF3B30', '#34C759', '#007AFF', '#FF9500', '#FF9500']

fig5.add_trace(go.Scatter(
    x=years, y=[0]*len(years), mode='markers+text',
    marker=dict(size=[i*10 for i in importance], color=colors, line=dict(width=2, color='white')),
    text=events, textposition='top center',
    textfont=dict(size=10),
    hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
))

# 添加时间轴线
fig5.add_trace(go.Scatter(
    x=[1940, 2015], y=[0, 0], mode='lines',
    line=dict(width=3, color='#8E8E93'),
    showlegend=False
))

# 添加时期标注
fig5.add_annotation(x=1950, y=-0.3, text='诞生期', showarrow=False, font=dict(size=12, color='#007AFF'))
fig5.add_annotation(x=1970, y=-0.3, text='寒冬期', showarrow=False, font=dict(size=12, color='#FF3B30'))
fig5.add_annotation(x=1995, y=-0.3, text='复兴期', showarrow=False, font=dict(size=12, color='#34C759'))
fig5.add_annotation(x=2010, y=-0.3, text='爆发期', showarrow=False, font=dict(size=12, color='#FF9500'))

fig5.update_xaxes(range=[1940, 2015], title='年份')
fig5.update_yaxes(range=[-0.5, 0.8], visible=False)

apply_apple_style(fig5, '感知机与神经网络发展历程', 1000, 400)
fig5.write_image('static/images/plots/perceptron-timeline.png', scale=2)

print("所有感知机图形生成完成!")
