import plotly.graph_objects as go

# Define colors for Apple style
COLORS = {
    'primary': '#007AFF',    # Blue
    'secondary': '#5856D6',  # Purple
    'success': '#34C759',    # Green
    'warning': '#FF9500',    # Orange
    'text': '#1D1D1F',
    'bg': '#F5F5F7'
}

fig = go.Figure()

# Background boxes for Sandbox Groups
fig.add_shape(type="rect", x0=0, y0=4, x1=4, y1=9, line=dict(color=COLORS['primary'], width=2), fillcolor='rgba(0,122,255,0.1)')
fig.add_shape(type="rect", x0=6, y0=4, x1=10, y1=9, line=dict(color=COLORS['secondary'], width=2), fillcolor='rgba(88,86,214,0.1)')

# Global DB Area
fig.add_shape(type="rect", x0=3, y0=0, x1=7, y1=2, line=dict(color=COLORS['success'], width=2, dash="dash"), fillcolor='rgba(52,199,89,0.1)')

# Sandbox Titles
fig.add_annotation(x=2, y=8.5, text="工作群组 A (沙盒)", showarrow=False, font=dict(size=14, color=COLORS['primary']))
fig.add_annotation(x=8, y=8.5, text="工作群组 B (沙盒)", showarrow=False, font=dict(size=14, color=COLORS['secondary']))
fig.add_annotation(x=5, y=1, text="全局语义 SQLite 库", showarrow=False, font=dict(size=14, color=COLORS['success']))

# Elements in Sandbox A
fig.add_trace(go.Scatter(
    x=[2], y=[7], mode='markers+text',
    marker=dict(size=50, color=COLORS['primary'], symbol='circle'),
    text=['提炼方法论'], textposition='middle center',
    textfont=dict(color='white', size=12), showlegend=False
))
fig.add_trace(go.Scatter(
    x=[2], y=[5.5], mode='markers+text',
    marker=dict(size=40, color=COLORS['warning'], symbol='hexagon'),
    text=['高价值脱敏规律'], textposition='middle center',
    textfont=dict(color='white', size=10), showlegend=False
))
# Lines inside Sandbox A
fig.add_trace(go.Scatter(x=[2, 2], y=[7, 5.5], mode='lines', line=dict(color=COLORS['primary'], width=2), showlegend=False))

# IPC Bus (horizontal line)
fig.add_shape(type="line", x0=-1, y0=3, x1=11, y1=3, line=dict(color=COLORS['text'], width=3, dash="dot"))
fig.add_annotation(x=5, y=3.2, text="IPC 跨进程层 (受限只读管道)", showarrow=False, font=dict(size=12))

# Broadcast arrow Sandbox A -> IPC
fig.add_annotation(x=2, y=3, ax=2, ay=5.5, text="ToolCall: BroadcastPattern", showarrow=True, arrowhead=2, axis=True, font=dict(color=COLORS['warning']))

# IPC -> Global DB
fig.add_annotation(x=5, y=2, ax=5, ay=3, text="向量化并持久化", showarrow=True, arrowhead=2, arrowcolor=COLORS['success'], axis=True)

# Global DB -> Sandbox B via IPC
fig.add_annotation(x=8, y=3, ax=5, ay=2, text="跨边界语义检索", showarrow=True, arrowhead=2, arrowcolor=COLORS['secondary'], font=dict(color=COLORS['secondary']))

# Element in Sandbox B
fig.add_trace(go.Scatter(
    x=[8], y=[6], mode='markers+text',
    marker=dict(size=60, color=COLORS['secondary'], symbol='circle'),
    text=['应用通用架构经验'], textposition='middle center',
    textfont=dict(color='white', size=12), showlegend=False
))

fig.add_annotation(x=8, y=6, ax=8, ay=3, text="自动注入 L1 上下文", showarrow=True, arrowhead=2, arrowcolor=COLORS['secondary'])

fig.update_layout(
    title='基于 IPC 与 SQLite 的跨沙盒跨周期记忆流',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 11]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 10]),
    plot_bgcolor='white',
    width=900, height=600
)

import os
os.makedirs('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots', exist_ok=True)
fig.write_image('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/ipc-swarm-memory.png', scale=2)
print("ipc memory image saved")
