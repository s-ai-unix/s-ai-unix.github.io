#!/usr/bin/env python3
"""
生成 Scala 语言演进时间线图
"""
import plotly.graph_objects as go
import subprocess

# Scala 发展历程数据
scala_events = [
    # (年份, 事件, 类型, 描述)
    (2001, "项目启动", "major", "Martin Odersky 在 EPFL\n开始设计 Scala"),
    (2003, "Scala 1.0", "major", "内部版本发布\n基于 JVM 的新语言"),
    (2004, "公开发布", "major", "正式面向公众发布\n兼容 Java 字节码"),
    (2006, "Scala 2.0", "major", "全新编译器架构\n更稳定的类型系统"),
    (2007, "Android 支持", "community", "开始支持 Android 开发\n移动平台新选择"),
    (2009, "Twitter 采用", "community", "Twitter 后端从 Ruby\n迁移至 Scala"),
    (2010, "Scala 2.8", "major", "全新集合框架\n命名参数与默认参数"),
    (2011, "Typesafe 成立", "community", "Typesafe 公司成立\n提供商业支持"),
    (2011, "Scala 2.9", "version", "并行集合加入\nJava 互操作性增强"),
    (2013, "Scala 2.10", "major", "反射 API、隐式类\n字符串插值、Future"),
    (2014, "Scala 2.11", "version", "模块化设计\n二进制兼容性改进"),
    (2016, "Scala 2.12", "major", "Java 8 专用版本\nLambda 表达式优化"),
    (2017, "Scala Native", "major", "Scala Native 发布\n编译为原生代码"),
    (2018, "Scala 2.13", "major", "新集合库、改进的\n隐式解析、性能提升"),
    (2020, "Scala.js 1.0", "major", "Scala.js 正式发布\n编译至 JavaScript"),
    (2021, "Scala 3.0", "major", "Dotty 编译器稳定\n全新语法与类型系统"),
    (2022, "Scala 3.1-3.2", "version", "稳定版本迭代\n逐步完善新特性"),
    (2023, "Scala 3.3 LTS", "major", "首个长期支持版本\nTypeScript 式类型"),
    (2024, "Scala 3.4", "version", "改进编辑器支持\n更好的报错信息"),
]

# 创建时间线图
fig = go.Figure()

# 颜色映射
colors = {
    "major": "#007AFF",   # 蓝色 - 主要版本
    "version": "#34C759", # 绿色 - 版本更新
    "community": "#FF9500" # 橙色 - 社区事件
}

# 分离 y 坐标以避免重叠
y_positions = []
current_y = 3.5
last_year = None
for year, event, etype, desc in scala_events:
    if last_year is not None and year - last_year < 2:
        current_y = -current_y + 0.4 if current_y > 0 else -current_y - 0.4
    else:
        current_y = 3.5 if current_y < 0 else -3.5
    y_positions.append(current_y)
    last_year = year

# 添加事件点和标签
for i, (year, event, etype, desc) in enumerate(scala_events):
    y = y_positions[i]
    color = colors[etype]
    
    # 添加节点
    fig.add_trace(go.Scatter(
        x=[year],
        y=[0],
        mode='markers',
        marker=dict(size=12, color=color, line=dict(width=2, color='white')),
        name=event,
        showlegend=False,
        hovertemplate=f"<b>{event}</b><br>{year}年<br>{desc}<extra></extra>"
    ))
    
    # 添加连接线
    fig.add_trace(go.Scatter(
        x=[year, year],
        y=[0, y * 0.7],
        mode='lines',
        line=dict(color=color, width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 添加文字标签
    fig.add_annotation(
        x=year,
        y=y,
        text=f"<b>{event}</b><br><span style='font-size:9px'>{desc[:15]}...</span>",
        showarrow=False,
        font=dict(size=9, color='#333'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=color,
        borderwidth=1,
        borderpad=3,
        align='center'
    )

# 添加时间轴线
fig.add_trace(go.Scatter(
    x=[2000, 2025],
    y=[0, 0],
    mode='lines',
    line=dict(color='#666', width=2),
    showlegend=False,
    hoverinfo='skip'
))

# 添加图例
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=10, color=colors['major']),
    name='主要版本/里程碑'
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=10, color=colors['version']),
    name='版本更新'
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=10, color=colors['community']),
    name='社区事件'
))

# 更新布局
fig.update_layout(
    title=dict(
        text='Scala 语言演进时间线 (2001-2024)',
        font=dict(size=18, color='#333'),
        x=0.5
    ),
    xaxis=dict(
        title='年份',
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)',
        tickmode='linear',
        dtick=3,
        range=[2000, 2026]
    ),
    yaxis=dict(
        visible=False,
        range=[-6, 6]
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial, sans-serif', size=12),
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.15,
        xanchor='center',
        x=0.5
    ),
    margin=dict(l=50, r=50, t=80, b=80),
    height=500
)

# 保存
output_path = 'static/images/plots/scala-timeline.png'
fig.write_image(output_path, scale=2)

# 压缩
subprocess.run([
    'pngquant', '--quality=70-85', '--force',
    '--output', output_path, output_path
], check=False)

print(f"✅ 已生成并压缩: {output_path}")
