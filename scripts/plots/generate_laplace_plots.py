#!/usr/bin/env python3
"""
为拉普拉斯变换历史文章生成静态 Plotly 图形
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 创建输出目录
os.makedirs('static/images/plots', exist_ok=True)

# 苹果风格配色
APPLE_BLUE = '#007AFF'
APPLE_GREEN = '#34C759'
APPLE_ORANGE = '#FF9500'
APPLE_RED = '#FF3B30'
APPLE_GRAY = '#8E8E93'

def apply_apple_style(fig, title, width=900, height=500):
    """应用苹果风格样式"""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family='Arial, sans-serif', color='#1d1d1f')),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12, color='#1d1d1f'),
        width=width,
        height=height,
        margin=dict(l=60, r=50, t=100, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

# ============ 图1: 衰减因子可视化 ============
print("生成图1: 衰减因子可视化...")

fig1 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '原函数 f(t) = 1（无衰减）',
        '衰减因子 e^(-σt), σ=0.5',
        '原函数 × 衰减因子',
        '积分收敛区域示意'
    ),
    specs=[[{'type': 'xy'}, {'type': 'xy'}],
           [{'type': 'xy'}, {'type': 'xy'}]],
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

t = np.linspace(0, 10, 500)
sigma = 0.5

# 子图1: 原函数 f(t) = 1
fig1.add_trace(go.Scatter(
    x=t, y=np.ones_like(t),
    mode='lines',
    name='f(t) = 1',
    line=dict(width=3, color=APPLE_BLUE),
    fill='tozeroy',
    fillcolor='rgba(0,122,255,0.1)',
    showlegend=False
), row=1, col=1)

# 添加积分不收敛的标注
fig1.add_annotation(
    x=5, y=0.5,
    text='积分不收敛',
    showarrow=False,
    font=dict(size=14, color=APPLE_RED),
    row=1, col=1
)

# 子图2: 衰减因子
fig1.add_trace(go.Scatter(
    x=t, y=np.exp(-sigma * t),
    mode='lines',
    name=f'e^(-{sigma}t)',
    line=dict(width=3, color=APPLE_ORANGE),
    showlegend=False
), row=1, col=2)

# 子图3: 原函数 × 衰减因子
fig1.add_trace(go.Scatter(
    x=t, y=np.exp(-sigma * t),
    mode='lines',
    name='f(t)·e^(-σt)',
    line=dict(width=3, color=APPLE_GREEN),
    fill='tozeroy',
    fillcolor='rgba(52,199,89,0.2)',
    showlegend=False
), row=2, col=1)

# 添加积分收敛的标注
fig1.add_annotation(
    x=5, y=0.1,
    text='积分收敛',
    showarrow=False,
    font=dict(size=14, color=APPLE_GREEN),
    row=2, col=1
)

# 子图4: 复平面收敛区域示意
s_real = np.linspace(-1, 3, 100)
s_imag = np.linspace(-3, 3, 100)
S_real, S_imag = np.meshgrid(s_real, s_imag)

# 收敛区域：Re(s) > 0
convergence = (S_real > 0).astype(float)

fig1.add_trace(go.Contour(
    x=s_real,
    y=s_imag,
    z=convergence,
    colorscale=[[0, 'white'], [1, 'rgba(0,122,255,0.3)']],
    showscale=False,
    contours=dict(coloring='heatmap'),
    line=dict(width=0),
    name='收敛区域'
), row=2, col=2)

# 添加虚轴标记
fig1.add_vline(x=0, line_dash="solid", line_color=APPLE_RED, line_width=2, row=2, col=2)

# 添加标注
fig1.add_annotation(x=1.5, y=2, text='收敛区域<br>(Re(s) > 0)', showarrow=False,
                   font=dict(size=12, color=APPLE_BLUE), row=2, col=2)
fig1.add_annotation(x=-0.5, y=2, text='发散区域', showarrow=False,
                   font=dict(size=12, color=APPLE_GRAY), row=2, col=2)

# 更新所有子图的轴标签
fig1.update_xaxes(title_text='t', row=1, col=1)
fig1.update_yaxes(title_text='f(t)', row=1, col=1)
fig1.update_xaxes(title_text='t', row=1, col=2)
fig1.update_yaxes(title_text='e^(-σt)', row=1, col=2)
fig1.update_xaxes(title_text='t', row=2, col=1)
fig1.update_yaxes(title_text='f(t)·e^(-σt)', row=2, col=1)
fig1.update_xaxes(title_text='Re(s)', row=2, col=2)
fig1.update_yaxes(title_text='Im(s)', row=2, col=2)

apply_apple_style(fig1, '', 1000, 700)
fig1.update_layout(title=dict(text='拉普拉斯变换的衰减因子原理', font=dict(size=20)))
fig1.write_image('static/images/plots/laplace-convergence.png', scale=2)

# ============ 图2: RL 电路响应 ============
print("生成图2: RL 电路响应...")

fig2 = make_subplots(
    rows=2, cols=1,
    subplot_titles=('输入电压 V(t)', '电流响应 i(t)'),
    vertical_spacing=0.15
)

t = np.linspace(0, 5, 500)
L = 1.0  # 电感
R = 2.0  # 电阻
tau = L / R  # 时间常数
V0 = 10  # 电压幅值

# 输入电压：阶跃函数
V_input = np.where(t >= 0.5, V0, 0)

# 电流响应：i(t) = (V0/R) * (1 - exp(-t/τ)) 对于 t >= 0.5
i_response = np.where(t >= 0.5, 
                      (V0/R) * (1 - np.exp(-(t - 0.5)/tau)), 
                      0)

# 子图1: 输入电压
fig2.add_trace(go.Scatter(
    x=t, y=V_input,
    mode='lines',
    name='输入电压 V(t)',
    line=dict(width=3, color=APPLE_BLUE),
    fill='tozeroy',
    fillcolor='rgba(0,122,255,0.1)'
), row=1, col=1)

# 添加时间常数标注
fig2.add_vline(x=0.5 + tau, line_dash="dash", line_color=APPLE_ORANGE, 
               line_width=2, row=1, col=1)
fig2.add_annotation(x=0.5 + tau, y=8, text=f'τ = L/R = {tau}s', 
                   showarrow=True, arrowhead=2, row=1, col=1,
                   font=dict(size=12, color=APPLE_ORANGE))

# 子图2: 电流响应
fig2.add_trace(go.Scatter(
    x=t, y=i_response,
    mode='lines',
    name='电流 i(t)',
    line=dict(width=3, color=APPLE_GREEN),
    fill='tozeroy',
    fillcolor='rgba(52,199,89,0.2)'
), row=2, col=1)

# 添加稳态值标记
fig2.add_hline(y=V0/R, line_dash="dash", line_color=APPLE_RED, 
               line_width=2, row=2, col=1)
fig2.add_annotation(x=4, y=V0/R + 0.3, text=f'稳态值 V₀/R = {V0/R}A', 
                   showarrow=False, row=2, col=1,
                   font=dict(size=12, color=APPLE_RED))

# 更新轴标签
fig2.update_xaxes(title_text='时间 t (s)', row=1, col=1)
fig2.update_yaxes(title_text='电压 V (V)', row=1, col=1)
fig2.update_xaxes(title_text='时间 t (s)', row=2, col=1)
fig2.update_yaxes(title_text='电流 i (A)', row=2, col=1)

apply_apple_style(fig2, '', 900, 700)
fig2.update_layout(title=dict(text='RL 电路的阶跃响应 (L=1H, R=2Ω)', font=dict(size=20)))
fig2.write_image('static/images/plots/laplace-rl-circuit.png', scale=2)

# ============ 图3: 拉普拉斯变换对示例 ============
print("生成图3: 拉普拉斯变换对...")

fig3 = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        '单位阶跃 u(t)',
        '指数函数 e^(-at)',
        '正弦函数 sin(ωt)',
        '1/s',
        '1/(s+a)',
        'ω/(s²+ω²)'
    ),
    specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
           [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]],
    vertical_spacing=0.2,
    horizontal_spacing=0.1
)

t = np.linspace(0, 5, 500)
s = np.linspace(0.1, 5, 500)
a = 1.0
omega = 2.0

# 第1行: 原函数（时域）
# 单位阶跃
fig3.add_trace(go.Scatter(
    x=t, y=np.where(t >= 0, 1, 0),
    mode='lines',
    name='u(t)',
    line=dict(width=3, color=APPLE_BLUE),
    showlegend=False
), row=1, col=1)

# 指数函数
fig3.add_trace(go.Scatter(
    x=t, y=np.exp(-a * t),
    mode='lines',
    name='e^(-at)',
    line=dict(width=3, color=APPLE_GREEN),
    showlegend=False
), row=1, col=2)

# 正弦函数
fig3.add_trace(go.Scatter(
    x=t, y=np.sin(omega * t),
    mode='lines',
    name='sin(ωt)',
    line=dict(width=3, color=APPLE_ORANGE),
    showlegend=False
), row=1, col=3)

# 第2行: 像函数（s域）
# 1/s
fig3.add_trace(go.Scatter(
    x=s, y=1/s,
    mode='lines',
    name='1/s',
    line=dict(width=3, color=APPLE_BLUE),
    showlegend=False
), row=2, col=1)

# 1/(s+a)
fig3.add_trace(go.Scatter(
    x=s, y=1/(s + a),
    mode='lines',
    name='1/(s+a)',
    line=dict(width=3, color=APPLE_GREEN),
    showlegend=False
), row=2, col=2)

# ω/(s²+ω²)
fig3.add_trace(go.Scatter(
    x=s, y=omega / (s**2 + omega**2),
    mode='lines',
    name='ω/(s²+ω²)',
    line=dict(width=3, color=APPLE_ORANGE),
    showlegend=False
), row=2, col=3)

# 更新所有子图的轴标签
fig3.update_xaxes(title_text='t', row=1, col=1)
fig3.update_yaxes(title_text='f(t)', row=1, col=1)
fig3.update_xaxes(title_text='t', row=1, col=2)
fig3.update_yaxes(title_text='f(t)', row=1, col=2)
fig3.update_xaxes(title_text='t', row=1, col=3)
fig3.update_yaxes(title_text='f(t)', row=1, col=3)

fig3.update_xaxes(title_text='s', row=2, col=1)
fig3.update_yaxes(title_text='F(s)', row=2, col=1)
fig3.update_xaxes(title_text='s', row=2, col=2)
fig3.update_yaxes(title_text='F(s)', row=2, col=2)
fig3.update_xaxes(title_text='s', row=2, col=3)
fig3.update_yaxes(title_text='F(s)', row=2, col=3)

# 添加变换箭头标注
fig3.add_annotation(x=0.5, y=-0.15, text='ℒ', showarrow=False,
                   font=dict(size=20, color=APPLE_GRAY),
                   xref='paper', yref='paper')

apply_apple_style(fig3, '', 1000, 600)
fig3.update_layout(title=dict(text='常见拉普拉斯变换对', font=dict(size=20)))
fig3.write_image('static/images/plots/laplace-transform-pairs.png', scale=2)

# ============ 图4: 微分定理可视化 ============
print("生成图4: 微分定理可视化...")

fig4 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('时域：微分运算', 's域：代数乘法'),
    horizontal_spacing=0.15
)

t = np.linspace(0, 4, 500)
s = np.linspace(0.5, 5, 500)

# 原函数: f(t) = t^2
f = t**2
# 导数: f'(t) = 2t
f_prime = 2*t
# 二阶导: f''(t) = 2
f_double = 2*np.ones_like(t)

# 子图1: 时域微分
fig4.add_trace(go.Scatter(
    x=t, y=f,
    mode='lines',
    name='f(t) = t²',
    line=dict(width=3, color=APPLE_BLUE)
), row=1, col=1)

fig4.add_trace(go.Scatter(
    x=t, y=f_prime,
    mode='lines',
    name="f'(t) = 2t",
    line=dict(width=3, color=APPLE_GREEN)
), row=1, col=1)

fig4.add_trace(go.Scatter(
    x=t, y=f_double,
    mode='lines',
    name="f''(t) = 2",
    line=dict(width=3, color=APPLE_ORANGE, dash='dash')
), row=1, col=1)

# 子图2: s域代数运算
# F(s) = 2/s^3
F = 2 / s**3
# sF(s) - f(0) = 2/s^2 (对应 f'(t))
sF = 2 / s**2
# s²F(s) - sf(0) - f'(0) = 2/s (对应 f''(t))
s2F = 2 / s

fig4.add_trace(go.Scatter(
    x=s, y=F,
    mode='lines',
    name='F(s) = 2/s³',
    line=dict(width=3, color=APPLE_BLUE),
    showlegend=False
), row=1, col=2)

fig4.add_trace(go.Scatter(
    x=s, y=sF,
    mode='lines',
    name='sF(s) = 2/s²',
    line=dict(width=3, color=APPLE_GREEN),
    showlegend=False
), row=1, col=2)

fig4.add_trace(go.Scatter(
    x=s, y=s2F,
    mode='lines',
    name='s²F(s) = 2/s',
    line=dict(width=3, color=APPLE_ORANGE, dash='dash'),
    showlegend=False
), row=1, col=2)

# 添加公式标注
fig4.add_annotation(x=0.25, y=1.05, text="微分运算", showarrow=False,
                   font=dict(size=14, color=APPLE_GRAY),
                   xref='paper', yref='paper')
fig4.add_annotation(x=0.75, y=1.05, text="代数乘法", showarrow=False,
                   font=dict(size=14, color=APPLE_GRAY),
                   xref='paper', yref='paper')

fig4.update_xaxes(title_text='t', row=1, col=1)
fig4.update_yaxes(title_text='函数值', row=1, col=1)
fig4.update_xaxes(title_text='s', row=1, col=2)
fig4.update_yaxes(title_text='像函数值', row=1, col=2)

apply_apple_style(fig4, '', 1000, 450)
fig4.update_layout(title=dict(text='微分定理：微分方程 ↔ 代数方程', font=dict(size=20)))
fig4.write_image('static/images/plots/laplace-differential-theorem.png', scale=2)

print("\n所有图形生成完成!")
print("生成的文件:")
print("  - static/images/plots/laplace-convergence.png")
print("  - static/images/plots/laplace-rl-circuit.png")
print("  - static/images/plots/laplace-transform-pairs.png")
print("  - static/images/plots/laplace-differential-theorem.png")
