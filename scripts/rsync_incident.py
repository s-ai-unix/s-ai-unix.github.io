#!/usr/bin/env python3
"""生成事故状态示意图"""
import plotly.graph_objects as go

fig = go.Figure()

# 节点定义
nodes = [
    # 事故前
    dict(x=0.2, y=1.0, label='事故前\n~/.claude/skills/', color='#34C759', size=55),
    dict(x=0.2, y=0.6, label='100+ Skills\n完整', color='#34C759', size=40),
    # 错误命令
    dict(x=0.5, y=1.0, label='rsync --delete\nSOURCE=空目录\nDEST=~/.claude/skills/', color='#FF9500', size=50),
    # 事故后
    dict(x=0.8, y=1.0, label='事故后', color='#FF3B30', size=40),
    dict(x=0.8, y=0.6, label='全部 Skills\n被删除', color='#FF3B30', size=40),
    # 恢复
    dict(x=1.1, y=1.0, label='git push\n恢复', color='#007AFF', size=45),
    dict(x=1.1, y=0.6, label='.git 目录\n完好', color='#34C759', size=40),
]

# 绘制节点
for n in nodes:
    fig.add_trace(go.Scatter(
        x=[n['x']], y=[n['y']],
        mode='markers+text',
        marker=dict(size=n['size'], color=n['color'], opacity=0.8,
                    line=dict(color='white', width=2)),
        text=[n['label']],
        textposition='middle center',
        textfont=dict(size=9, color='white'),
        showlegend=False
    ))

# 箭头
arrows = [
    (0.32, 1.0, 0.42, 1.0, '#FF3B30', '执行'),
    (0.58, 1.0, 0.72, 1.0, '#FF3B30', '删除!'),
    (0.88, 1.0, 1.0, 1.0, '#34C759', '恢复'),
]

for x0, y0, x1, y1, color, label in arrows:
    fig.add_annotation(
        ax=x0, ay=y0, axref='x', ayref='y',
        x=x1, y=y1, xref='x', yref='y',
        arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=color,
        showarrow=True
    )

fig.update_layout(
    width=1000, height=400,
    plot_bgcolor='white',
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.1, 1.3]),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0.2, 1.3]),
    margin=dict(t=20, b=20, l=20, r=20)
)

fig.write_image('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/rsync-incident-state.png', scale=2)
print('done')
