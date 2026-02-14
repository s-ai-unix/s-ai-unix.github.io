import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 设置苹果风格配色
APPLE_COLORS = {
    'primary': '#007AFF',
    'success': '#34C759',
    'warning': '#FF9500',
    'danger': '#FF3B30',
    'purple': '#AF52DE',
    'gray': '#8E8E93'
}

# 数据：不同语言的Token效率
text_lengths = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

# 中文：约1.5个字符/Token
chinese_tokens = text_lengths / 1.5

# 英文：约4个字符/Token
english_tokens = text_lengths / 4

# 代码：约3.5个字符/Token
code_tokens = text_lengths / 3.5

# 创建图表
fig = go.Figure()

# 添加中文曲线
fig.add_trace(go.Scatter(
    x=text_lengths,
    y=chinese_tokens,
    mode='lines+markers',
    name='中文',
    line=dict(color=APPLE_COLORS['primary'], width=3),
    marker=dict(size=8)
))

# 添加英文曲线
fig.add_trace(go.Scatter(
    x=text_lengths,
    y=english_tokens,
    mode='lines+markers',
    name='英文',
    line=dict(color=APPLE_COLORS['success'], width=3),
    marker=dict(size=8)
))

# 添加代码曲线
fig.add_trace(go.Scatter(
    x=text_lengths,
    y=code_tokens,
    mode='lines+markers',
    name='代码',
    line=dict(color=APPLE_COLORS['warning'], width=3),
    marker=dict(size=8)
))

# 添加参考线：y=x（1:1关系）
fig.add_trace(go.Scatter(
    x=text_lengths,
    y=text_lengths,
    mode='lines',
    name='字符数（参考）',
    line=dict(color=APPLE_COLORS['gray'], width=2, dash='dash')
))

# 更新布局
fig.update_layout(
    title=dict(
        text='不同文本类型的Token数量对比',
        font=dict(size=18, family='-apple-system, BlinkMacSystemFont, "SF Pro Text", Segoe UI, Roboto, sans-serif'),
        x=0.5
    ),
    xaxis=dict(
        title=dict(text='字符数', font=dict(size=14)),
        gridcolor='#E5E5EA',
        linecolor='#C7C7CC'
    ),
    yaxis=dict(
        title=dict(text='Token数', font=dict(size=14)),
        gridcolor='#E5E5EA',
        linecolor='#C7C7CC'
    ),
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#E5E5EA',
        borderwidth=1
    ),
    plot_bgcolor='#F5F5F7',
    paper_bgcolor='white',
    font=dict(family='-apple-system, BlinkMacSystemFont, "SF Pro Text", Segoe UI, Roboto, sans-serif', size=12),
    margin=dict(l=60, r=40, t=80, b=60)
)

# 保存图片
output_path = 'static/images/plots/token-char-relation.png'
fig.write_image(output_path, scale=2, width=900, height=500)

# 压缩图片
try:
    import subprocess
    subprocess.run([
        'pngquant', '--quality=70-85', '--force',
        '--output', output_path, output_path
    ], check=False)
    print(f"✅ 已保存并压缩: {output_path}")
except:
    print(f"✅ 已保存: {output_path}")

# 显示图表
fig.show()
