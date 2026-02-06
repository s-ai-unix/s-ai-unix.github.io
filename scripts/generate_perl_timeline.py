#!/usr/bin/env python3
"""
生成 Perl 语言演进时间线图
"""
import plotly.graph_objects as go
import subprocess
import os

# Perl 发展历程数据
perl_events = [
    # (年份, 事件, 类型, 描述)
    (1987, "Perl 1.0", "major", "Larry Wall 发布 Perl 1.0\n文本处理语言诞生"),
    (1988, "Perl 2.0", "version", "新增正则表达式支持\n增强报告生成功能"),
    (1989, "Perl 3.0", "major", "新增二进制数据处理\n支持面向对象雏形"),
    (1991, "Perl 4.0", "version", "发布 perlmod 模块系统\nCPAN 前身出现"),
    (1994, "Perl 5.0", "major", "完全面向对象编程\n模块系统、引用、正则增强"),
    (1995, "CPAN 成立", "community", "综合 Perl 归档网成立\n模块生态爆发式增长"),
    (2000, "Perl 5.6", "version", "支持 64 位系统\n引入 our 关键字"),
    (2002, "Perl 5.8", "version", "Unicode 全面支持\n线程模型改进"),
    (2007, "Perl 5.10", "version", "state 关键字\n智能匹配 =~"),
    (2010, "Perl 5.12", "version", "Yada Yada 操作符\n改进包声明"),
    (2011, "Perl 5.14", "version", "非破坏性替换 /r\n正则性能提升"),
    (2012, "Perl 5.16", "version", "__SUB__ 当前子例程引用"),
    (2013, "Perl 5.18", "version", "更好的 Unicode 支持\n哈希随机化"),
    (2014, "Perl 5.20", "version", "子例程签名实验性支持\n引入 %hash{...}"),
    (2015, "Perl 5.22", "version", "bitwise 操作符改进\n正则锚点优化"),
    (2017, "Perl 5.26", "version", "移除当前目录 . 从 @INC\n缩进的 Here-doc"),
    (2018, "Perl 5.28", "version", "更快的 UTF-8 处理\n删除 $* 变量"),
    (2019, "Perl 5.30", "version", "更安全的 eval\n性能优化"),
    (2020, "Perl 7 宣布", "major", "Perl 7 路线图发布\n向后兼容的现代化"),
    (2021, "Perl 5.34", "version", "try/catch 实验性支持\n迭代多值返回"),
    (2022, "Perl 5.36", "version", "标准启用 strict\n内置函数签名稳定"),
    (2023, "Perl 5.38", "version", "class 关键字实验性\n更好的 OO 支持"),
    (2024, "Perl 5.40", "version", "稳定版 class 特性\ntry/catch/finally"),
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
for year, event, etype, desc in perl_events:
    if last_year is not None and year - last_year < 3:
        current_y = -current_y + 0.3 if current_y > 0 else -current_y - 0.3
    else:
        current_y = 3.5 if current_y < 0 else -3.5
    y_positions.append(current_y)
    last_year = year

# 添加事件点和标签
for i, (year, event, etype, desc) in enumerate(perl_events):
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
    
    # 添加文字标签（外部）
    fig.add_annotation(
        x=year,
        y=y,
        text=f"<b>{event}</b><br><span style='font-size:10px'>{desc[:20]}...</span>",
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
    x=[1986, 2025],
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
    name='主要版本'
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
        text='Perl 语言演进时间线 (1987-2024)',
        font=dict(size=18, color='#333'),
        x=0.5
    ),
    xaxis=dict(
        title='年份',
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)',
        tickmode='linear',
        dtick=5,
        range=[1985, 2026]
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
output_path = 'static/images/plots/perl-timeline.png'
fig.write_image(output_path, scale=2)

# 压缩
subprocess.run([
    'pngquant', '--quality=70-85', '--force',
    '--output', output_path, output_path
], check=False)

print(f"✅ 已生成并压缩: {output_path}")
