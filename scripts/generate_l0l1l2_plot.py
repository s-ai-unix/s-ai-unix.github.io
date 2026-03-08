import plotly.graph_objects as go

# Define colors for Apple style
COLORS = {
    'primary': '#007AFF',    # L0 Light Blue
    'secondary': '#5856D6',  # L1 Purple
    'success': '#34C759',    # L2 Green
    'warning': '#FF9500',    # Orange
    'text': '#1D1D1F'
}

fig = go.Figure()

# --- Diagram 1: L0/L1/L2 Hierarchy ---
# User Query Box
fig.add_trace(go.Scatter(
    x=[0], y=[8], 
    mode='markers+text',
    marker=dict(size=60, color=COLORS['text'], symbol='square'),
    text=['用户提问'], textposition='middle center',
    textfont=dict(color='white', size=14),
    showlegend=False
))

# L0 Level
fig.add_trace(go.Scatter(
    x=[-2, 0, 2], y=[6, 6, 6],
    mode='markers+text',
    marker=dict(size=50, color=COLORS['primary'], symbol='circle'),
    text=['极简', 'L0', '摘要'], textposition='middle center',
    textfont=dict(color='white', size=14),
    name='L0 层'
))

# L1 Level
fig.add_trace(go.Scatter(
    x=[-3, -1, 1, 3], y=[3, 3, 3, 3],
    mode='markers+text',
    marker=dict(size=40, color=COLORS['secondary'], symbol='circle'),
    text=['', 'L1', '结构', ''], textposition='middle center',
    textfont=dict(color='white', size=12),
    name='L1 层'
))

# L2 Level
fig.add_trace(go.Scatter(
    x=[-4, -2, 0, 2, 4], y=[0, 0, 0, 0, 0],
    mode='markers+text',
    marker=dict(size=30, color=COLORS['success'], symbol='circle'),
    text=['', '', 'L2 完整记录', '', ''], textposition='middle center',
    textfont=dict(color='white', size=10),
    name='L2 层'
))

# Add edges (Lines between L0 -> L1 -> L2)
edges_x = []
edges_y = []
# Query -> L0
for i in [-2, 0, 2]:
    edges_x.extend([0, i, None])
    edges_y.extend([8, 6, None])
# L0 -> L1
edges_x.extend([-2, -3, None, -2, -1, None, 0, -1, None, 0, 1, None, 2, 1, None, 2, 3, None])
edges_y.extend([6, 3, None, 6, 3, None, 6, 3, None, 6, 3, None, 6, 3, None, 6, 3, None])
# L1 -> L2
edges_x.extend([-3, -4, None, -3, -2, None, -1, -2, None, -1, 0, None, 1, 0, None, 1, 2, None, 3, 2, None, 3, 4, None])
edges_y.extend([3, 0, None, 3, 0, None, 3, 0, None, 3, 0, None, 3, 0, None, 3, 0, None, 3, 0, None, 3, 0, None])

fig.add_trace(go.Scatter(
    x=edges_x, y=edges_y,
    mode='lines',
    line=dict(color='#C7C7CC', width=1),
    hoverinfo='none',
    showlegend=False
))

# Annotations for retrieval depth
fig.add_annotation(x=3, y=7, text="高频检索，成本极低", showarrow=False, font=dict(color=COLORS['primary']))
fig.add_annotation(x=4, y=4, text="预处理上下文垫底", showarrow=False, font=dict(color=COLORS['secondary']))
fig.add_annotation(x=5, y=1, text="遇到瓶颈下钻读取原链", showarrow=False, font=dict(color=COLORS['success']))

fig.update_layout(
    title='L0-L2 抽象加载层级架构',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white',
    width=800, height=500
)

import os
os.makedirs('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots', exist_ok=True)
fig.write_image('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/nanoclaw-memory-hierarchy.png', scale=2)
print("hierarchy image saved")
