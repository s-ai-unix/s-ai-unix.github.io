import plotly.graph_objects as go

# 苹果风格配色
COLORS = {
    'primary': '#007AFF',    # 蓝色
    'secondary': '#5856D6',  # 紫色
    'success': '#34C759',    # 绿色
    'danger': '#FF3B30',     # 红色
    'warning': '#FF9500'     # 橙色
}

fig = go.Figure()

# 画象限底色
fig.add_shape(type="rect", x0=0, y0=5, x1=5, y1=10, fillcolor="rgba(255, 59, 48, 0.1)", line=dict(width=0))
fig.add_shape(type="rect", x0=5, y0=0, x1=10, y1=5, fillcolor="rgba(52, 199, 89, 0.1)", line=dict(width=0))
fig.add_shape(type="rect", x0=0, y0=0, x1=5, y1=5, fillcolor="rgba(255, 149, 0, 0.1)", line=dict(width=0))
fig.add_shape(type="rect", x0=5, y0=5, x1=10, y1=10, fillcolor="rgba(88, 86, 214, 0.1)", line=dict(width=0))

# 绘制十字轴
fig.add_hline(y=5, line_width=2, line_dash="dash", line_color="gray")
fig.add_vline(x=5, line_width=2, line_dash="dash", line_color="gray")

# 象限标注文字
fig.add_annotation(x=2.5, y=9.5, text="<b>高危区：极易被取代</b>", showarrow=False, font=dict(size=16, color=COLORS['danger']))
fig.add_annotation(x=7.5, y=0.5, text="<b>安全区：极难被取代</b>", showarrow=False, font=dict(size=16, color=COLORS['success']))

# 绘制散点
jobs = [
    # 结构化/规则化高，模糊性/责任低 (高危区)
    {"name": "初级程序员<br>(代码执行)", "x": 1.5, "y": 9, "color": COLORS['danger']},
    {"name": "基础数据提取<br>(写SQL/报表)", "x": 2, "y": 8.5, "color": COLORS['danger']},
    {"name": "文字校对与翻译", "x": 1, "y": 8, "color": COLORS['danger']},
    
    # 结构化中等，责任中等 (过渡区)
    {"name": "产品经理<br>(梳理需求/画原型)", "x": 3.5, "y": 6.5, "color": COLORS['warning']},
    {"name": "高级数据挖掘", "x": 4.5, "y": 5.5, "color": COLORS['warning']},
    
    # 结构化低，模糊性/责任高 (安全区)
    {"name": "AI合规审查与认证", "x": 8.5, "y": 2, "color": COLORS['success']},
    {"name": "系统架构师", "x": 7, "y": 3, "color": COLORS['success']},
    {"name": "战略咨询顾问", "x": 9, "y": 1, "color": COLORS['success']},
    
    # 结构化低，但模糊性责任也不极高 (物理交互等)
    {"name": "心理咨询/物理劳动", "x": 8, "y": 4.5, "color": COLORS['secondary']},
]

for job in jobs:
    # 绘制点
    fig.add_trace(go.Scatter(
        x=[job['x']], y=[job['y']],
        mode='markers+text',
        marker=dict(size=30, color=job['color'], line=dict(width=2, color='white')),
        text=[job['name']],
        textposition='top center',
        textfont=dict(size=14, color='#1D1D1F'),
        showlegend=False
    ))

# 布局设置
fig.update_layout(
    title='<b>职业被替代风险矩阵图</b>',
    title_font=dict(size=24),
    xaxis=dict(
        title=dict(text='<b>模糊性处理与责任承担能力 (Accountability & Ambiguity)</b>', font=dict(size=16)),
        range=[0, 10],
        showticklabels=False,
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title=dict(text='<b>结构化/规则化程度 (Structured & Routine)</b>', font=dict(size=16)),
        range=[0, 10],
        showticklabels=False,
        showgrid=False,
        zeroline=False
    ),
    width=900,
    height=650,
    plot_bgcolor='white',
    margin=dict(l=60, r=40, t=80, b=60)
)

fig.write_image('static/images/plots/2026-04-26-ai-jobs-quadrant.png', scale=2)
print("Quadrant chart generated successfully.")
