import plotly.graph_objects as go
import numpy as np

fig = go.Figure()

# 节点位置
nodes_x = [1, 2.5, 4, 4, 5.5, 7, 8.5]
nodes_y = [0, 0, 0.6, -0.6, 0, 0, 0]
nodes_text = ['输入 x', '编码器', 'μ(x)', 'σ(x)', '采样 z', '解码器', '重构 x̂']
nodes_color = ['#007AFF', '#34C759', '#FF9500', '#FF9500', '#AF52DE', '#34C759', '#007AFF']
sizes = [60, 60, 55, 55, 55, 60, 60]

for i, (x, y, text, color, size) in enumerate(zip(nodes_x, nodes_y, nodes_text, nodes_color, sizes)):
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text',
                             marker=dict(size=size, color=color),
                             text=[text], textposition='middle center',
                             textfont=dict(size=10 if i in [2,3,4] else 11, color='white'),
                             showlegend=False))

# 箭头（使用线段代替）
arrows_x = [[1.7, 2.2], [3.2, 3.6], [3.2, 3.6], [4.6, 5.0], [4.6, 5.0], [6.2, 6.7], [7.7, 8.2]]
arrows_y = [[0, 0], [0.2, 0.45], [-0.2, -0.45], [0.4, 0.1], [-0.4, -0.1], [0, 0], [0, 0]]

for ax, ay in zip(arrows_x, arrows_y):
    fig.add_trace(go.Scatter(x=ax, y=ay, mode='lines', line=dict(color='#8E8E93', width=2), showlegend=False))

fig.add_annotation(x=4, y=1.4, text='q(z|x) = N(μ(x), σ²(x))', showarrow=False, font=dict(size=10))
fig.add_annotation(x=8.5, y=-1.4, text='p(x|z) = N(解码器(z), σ²)', showarrow=False, font=dict(size=10))

fig.update_layout(title='变分自编码器（VAE）中的条件期望', template='plotly_white', font=dict(family='Arial', size=12),
                  xaxis=dict(range=[0, 9.5], showgrid=False, showticklabels=False, zeroline=False),
                  yaxis=dict(range=[-2, 2], showgrid=False, showticklabels=False, zeroline=False),
                  margin=dict(l=30, r=30, t=80, b=30), height=450)

fig.write_image('static/images/plots/vae-conditional-expectation.png', width=950, height=450, scale=2)
print("图6完成")
