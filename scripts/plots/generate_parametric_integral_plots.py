#!/usr/bin/env python3
"""
生成含参变量积分相关的 Plotly 图形
导出为 PNG 格式并压缩
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os

# 设置默认模板和颜色
template = 'plotly_white'
primary_color = '#007AFF'
secondary_color = '#34C759'
tertiary_color = '#FF9500'
accent_color = '#AF52DE'

output_dir = 'static/images/plots'


def save_and_compress(fig, filepath, width=900, height=600, scale=2):
    """保存并压缩图片"""
    full_path = os.path.join(output_dir, filepath)
    fig.write_image(full_path, width=width, height=height, scale=scale)
    
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', full_path, full_path
        ], check=False)
    
    print(f"✅ 已保存并压缩: {full_path}")


def plot_parametric_integral_family():
    """
    图1: 含参积分函数族示例
    展示 F(t) = ∫_0^1 e^(-tx) dx 对于不同 t 值的被积函数
    """
    x = np.linspace(0, 1, 200)
    t_values = [0.5, 1, 2, 3, 5]
    colors = ['#007AFF', '#34C759', '#FF9500', '#AF52DE', '#FF3B30']
    
    fig = go.Figure()
    
    for t, color in zip(t_values, colors):
        y = np.exp(-t * x)
        # 计算积分值
        integral_val = (1 - np.exp(-t)) / t if t != 0 else 1
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f't={t}, F(t)={integral_val:.3f}',
            line=dict(color=color, width=2.5)
        ))
    
    fig.update_layout(
        title=dict(
            text='含参积分示例：不同参数 t 下的被积函数 e^(-tx)',
            font=dict(size=16)
        ),
        xaxis_title='x',
        yaxis_title='f(x,t) = e^(-tx)',
        template=template,
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(t=100)
    )
    
    save_and_compress(fig, 'parametric-integral-family.png', width=900, height=550)


def plot_gamma_function():
    """
    图2: Gamma 函数图像
    Gamma(t) = ∫_0^∞ x^(t-1) e^(-x) dx
    """
    # 使用递推关系近似计算 Gamma 函数
    def gamma_approx(t):
        """使用 Lanczos 近似或直接使用 scipy 的近似"""
        # 对于正实数，使用简单的近似
        # Gamma(n) = (n-1)! 对于正整数
        # 这里我们用数值积分近似
        if t <= 0:
            return np.nan
        x = np.linspace(0, 20, 1000)
        dx = x[1] - x[0]
        integrand = x**(t-1) * np.exp(-x)
        return np.trapz(integrand, x)
    
    t = np.linspace(0.1, 5, 300)
    gamma_values = [gamma_approx(ti) for ti in t]
    
    # 标记整数点
    t_int = np.array([1, 2, 3, 4, 5])
    gamma_int = [gamma_approx(ti) for ti in t_int]
    
    fig = go.Figure()
    
    # 主曲线
    fig.add_trace(go.Scatter(
        x=t, y=gamma_values,
        mode='lines',
        name='Γ(t)',
        line=dict(color=primary_color, width=2.5)
    ))
    
    # 整数点标记
    fig.add_trace(go.Scatter(
        x=t_int, y=gamma_int,
        mode='markers+text',
        name='整数值 Γ(n)=(n-1)!',
        marker=dict(size=12, color=tertiary_color, symbol='diamond'),
        text=[f'Γ({int(ti)})={(gi)}' for ti, gi in zip(t_int, gamma_int)],
        textposition='top center',
        textfont=dict(size=10)
    ))
    
    # 添加 y=1 参考线
    fig.add_hline(y=1, line=dict(dash='dot', color='gray', width=1))
    
    fig.update_layout(
        title=dict(
            text='Gamma 函数：Γ(t) = ∫₀^∞ x^(t-1) e^(-x) dx',
            font=dict(size=16)
        ),
        xaxis_title='t',
        yaxis_title='Γ(t)',
        template=template,
        font=dict(family='Arial, sans-serif', size=12),
        yaxis=dict(range=[0, 8]),
        showlegend=True,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )
    
    save_and_compress(fig, 'gamma-function.png', width=900, height=550)


def plot_leibnitz_rule_geometry():
    """
    图3: 莱布尼茨积分法则的几何解释
    展示积分区域和参数变化的影响
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('固定积分限', '变积分限'),
        horizontal_spacing=0.15
    )
    
    # 左图：固定积分限 [0,1]
    x1 = np.linspace(0, 1, 100)
    y1_base = np.zeros_like(x1)
    y1_top = np.exp(-x1)
    
    # 填充区域
    fig.add_trace(go.Scatter(
        x=list(x1) + list(x1[::-1]),
        y=list(y1_top) + list(y1_base[::-1]),
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        line=dict(width=0),
        name='积分区域',
        showlegend=True
    ), row=1, col=1)
    
    # 边界线
    fig.add_trace(go.Scatter(
        x=x1, y=y1_top,
        mode='lines',
        line=dict(color=primary_color, width=2.5),
        name='f(x) = e^(-x)',
        showlegend=False
    ), row=1, col=1)
    
    # 标记积分限
    fig.add_vline(x=0, line=dict(dash='dash', color='gray'), row=1, col=1)
    fig.add_vline(x=1, line=dict(dash='dash', color='gray'), row=1, col=1)
    
    # 右图：变积分限 a(t) 到 b(t)
    t_val = 0.5  # 固定一个 t 值
    a_t = 0.2 + 0.1 * t_val
    b_t = 0.8 + 0.15 * t_val
    
    x2 = np.linspace(a_t, b_t, 100)
    y2 = np.exp(-2 * x2)  # f(x,t) = e^(-2x)
    y2_base = np.zeros_like(x2)
    
    # 填充区域（带渐变效果）
    fig.add_trace(go.Scatter(
        x=list(x2) + list(x2[::-1]),
        y=list(y2) + list(y2_base[::-1]),
        fill='toself',
        fillcolor='rgba(52, 199, 89, 0.25)',
        line=dict(width=0),
        name='变限积分区域',
        showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        mode='lines',
        line=dict(color=secondary_color, width=2.5),
        name='f(x,t) = e^(-2x)',
        showlegend=False
    ), row=1, col=2)
    
    # 标记变限
    fig.add_vline(x=a_t, line=dict(dash='dash', color=tertiary_color, width=2), row=1, col=2)
    fig.add_vline(x=b_t, line=dict(dash='dash', color=tertiary_color, width=2), row=1, col=2)
    
    # 添加注释
    fig.add_annotation(
        x=a_t, y=-0.05,
        text=f'a(t)',
        showarrow=False,
        font=dict(size=11, color=tertiary_color),
        row=1, col=2
    )
    fig.add_annotation(
        x=b_t, y=-0.05,
        text=f'b(t)',
        showarrow=False,
        font=dict(size=11, color=tertiary_color),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(
            text='莱布尼茨积分法则：积分区域的变化',
            font=dict(size=16)
        ),
        template=template,
        font=dict(family='Arial, sans-serif', size=12),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='center',
            x=0.5
        )
    )
    
    fig.update_xaxes(title_text='x', row=1, col=1)
    fig.update_xaxes(title_text='x', range=[0, 1.2], row=1, col=2)
    fig.update_yaxes(title_text='f(x)', row=1, col=1)
    fig.update_yaxes(title_text='f(x,t)', row=1, col=2)
    
    save_and_compress(fig, 'leibnitz-rule-geometry.png', width=950, height=450)


def plot_continuity_demo():
    """
    图4: 含参积分连续性演示
    展示 F(t) 随参数 t 的连续变化
    """
    # 计算 F(t) = ∫_0^1 e^(-tx) dx = (1-e^(-t))/t
    t = np.linspace(0.1, 5, 200)
    F_t = (1 - np.exp(-t)) / t
    
    # 添加 t→0 的极限值 F(0) = 1
    t_full = np.concatenate([[0], t])
    F_full = np.concatenate([[1], F_t])
    
    fig = go.Figure()
    
    # 主曲线
    fig.add_trace(go.Scatter(
        x=t_full, y=F_full,
        mode='lines',
        name='F(t)',
        line=dict(color=primary_color, width=3)
    ))
    
    # 标记几个关键点
    key_points = [0.5, 1, 2, 3]
    for tp in key_points:
        Fp = (1 - np.exp(-tp)) / tp
        fig.add_trace(go.Scatter(
            x=[tp], y=[Fp],
            mode='markers',
            marker=dict(size=10, color=tertiary_color, symbol='circle'),
            showlegend=False
        ))
    
    # 添加极限标注
    fig.add_annotation(
        x=0, y=1,
        text='F(0)=1',
        showarrow=True,
        arrowhead=2,
        ax=30, ay=-40,
        font=dict(size=11)
    )
    
    # 渐近线
    fig.add_hline(y=0, line=dict(dash='dot', color='gray', width=1))
    
    fig.update_layout(
        title=dict(
            text='含参积分的连续性：F(t) = ∫₀¹ e^(-tx) dx',
            font=dict(size=16)
        ),
        xaxis_title='参数 t',
        yaxis_title='F(t)',
        template=template,
        font=dict(family='Arial, sans-serif', size=12),
        yaxis=dict(range=[0, 1.1])
    )
    
    save_and_compress(fig, 'parametric-integral-continuity.png', width=900, height=500)


def plot_error_function():
    """
    图5: 误差函数 erf(x) 图像
    展示含参积分在概率论中的应用
    """
    def erf_approx(x):
        """误差函数近似"""
        # 使用数值积分计算
        t = np.linspace(0, abs(x), 500)
        if len(t) < 2:
            return 0
        dt = t[1] - t[0]
        integrand = np.exp(-t**2)
        result = (2/np.sqrt(np.pi)) * np.trapz(integrand, t)
        return result if x >= 0 else -result
    
    x = np.linspace(-3, 3, 400)
    y = [erf_approx(xi) for xi in x]
    
    fig = go.Figure()
    
    # 主曲线
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='erf(x)',
        line=dict(color=primary_color, width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 122, 255, 0.15)'
    ))
    
    # 渐近线 y=±1
    fig.add_hline(y=1, line=dict(dash='dash', color=tertiary_color, width=2),
                  annotation_text='y=1', annotation_position='right')
    fig.add_hline(y=-1, line=dict(dash='dash', color=tertiary_color, width=2),
                  annotation_text='y=-1', annotation_position='right')
    
    # 标记关键点
    key_x = [0, 1, -1]
    for kx in key_x:
        ky = erf_approx(kx)
        fig.add_trace(go.Scatter(
            x=[kx], y=[ky],
            mode='markers',
            marker=dict(size=12, color=secondary_color, symbol='diamond'),
            showlegend=False
        ))
        fig.add_annotation(
            x=kx, y=ky,
            text=f'erf({kx})={ky:.3f}',
            showarrow=True,
            arrowhead=2,
            ax=40 if kx >= 0 else -40,
            ay=-30,
            font=dict(size=10)
        )
    
    fig.update_layout(
        title=dict(
            text='误差函数：erf(x) = (2/√π) ∫₀^x e^(-t²) dt',
            font=dict(size=16)
        ),
        xaxis_title='x',
        yaxis_title='erf(x)',
        template=template,
        font=dict(family='Arial, sans-serif', size=12),
        yaxis=dict(range=[-1.2, 1.2]),
        showlegend=False
    )
    
    save_and_compress(fig, 'error-function.png', width=900, height=550)


def plot_feynman_trick_demo():
    """
    图6: 费曼技巧演示
    展示如何通过引入参数简化积分计算
    """
    # 演示 I(a) = ∫_0^(π/2) ln(a²cos²x + sin²x) dx
    def I_integral(a):
        x = np.linspace(0, np.pi/2, 500)
        integrand = np.log(a**2 * np.cos(x)**2 + np.sin(x)**2)
        return np.trapz(integrand, x)
    
    a = np.linspace(0.1, 3, 100)
    I_values = [I_integral(ai) for ai in a]
    
    # 解析解：I(a) = π ln((a+1)/2)
    I_analytic = np.pi * np.log((a + 1) / 2)
    
    fig = go.Figure()
    
    # 数值积分结果
    fig.add_trace(go.Scatter(
        x=a, y=I_values,
        mode='lines',
        name='数值积分',
        line=dict(color=primary_color, width=3)
    ))
    
    # 解析解
    fig.add_trace(go.Scatter(
        x=a, y=I_analytic,
        mode='lines',
        name='解析解 I(a) = π ln((a+1)/2)',
        line=dict(color=tertiary_color, width=2.5, dash='dash')
    ))
    
    # 标记 a=1 点
    fig.add_trace(go.Scatter(
        x=[1], y=[0],
        mode='markers',
        marker=dict(size=12, color=secondary_color, symbol='star'),
        name='I(1) = 0',
        showlegend=True
    ))
    
    fig.add_annotation(
        x=1, y=0,
        text='I(1)=0',
        showarrow=True,
        arrowhead=2,
        ax=-40, ay=-40,
        font=dict(size=11)
    )
    
    fig.update_layout(
        title=dict(
            text='费曼技巧示例：I(a) = ∫₀^(π/2) ln(a²cos²x + sin²x) dx',
            font=dict(size=16)
        ),
        xaxis_title='参数 a',
        yaxis_title='I(a)',
        template=template,
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(t=100)
    )
    
    save_and_compress(fig, 'feynman-trick-demo.png', width=900, height=550)


def plot_historical_timeline():
    """
    图7: 含参变量积分发展历史时间线
    """
    # 历史事件
    events = [
        ('1666', '牛顿', '流数法\n微积分创立'),
        ('1675', '莱布尼茨', '积分符号\n∫ 引入'),
        ('1694', '约翰·伯努利', '变分法\n奠基'),
        ('1730', '欧拉', 'Γ函数\n系统研究'),
        ('1810', '拉普拉斯', '拉普拉斯变换\n含参积分应用'),
        ('1828', '格林', '格林公式\n积分理论'),
        ('1857', '黎曼', '黎曼积分\n严格化'),
        ('1949', '费曼', '费曼技巧\n路径积分')
    ]
    
    years = [int(e[0]) for e in events]
    y_pos = [0] * len(events)
    
    fig = go.Figure()
    
    # 时间线主线
    fig.add_trace(go.Scatter(
        x=[1640, 2000],
        y=[0, 0],
        mode='lines',
        line=dict(color='gray', width=2),
        showlegend=False
    ))
    
    # 事件点
    colors = [primary_color, secondary_color, tertiary_color, accent_color] * 2
    for i, (year, person, desc) in enumerate(events):
        color = colors[i]
        y_offset = 0.3 if i % 2 == 0 else -0.3
        
        # 连接线
        fig.add_trace(go.Scatter(
            x=[int(year), int(year)],
            y=[0, y_offset * 0.7],
            mode='lines',
            line=dict(color=color, width=1.5),
            showlegend=False
        ))
        
        # 事件点
        fig.add_trace(go.Scatter(
            x=[int(year)],
            y=[0],
            mode='markers',
            marker=dict(size=14, color=color, symbol='circle'),
            showlegend=False
        ))
        
        # 文字标签
        fig.add_annotation(
            x=int(year),
            y=y_offset,
            text=f'<b>{year}</b><br>{person}<br>{desc}',
            showarrow=False,
            font=dict(size=10),
            align='center',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=color,
            borderwidth=1,
            borderpad=4
        )
    
    fig.update_layout(
        title=dict(
            text='含参变量积分发展简史',
            font=dict(size=16)
        ),
        template=template,
        font=dict(family='Arial, sans-serif', size=11),
        xaxis=dict(
            range=[1640, 2000],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-0.6, 0.6],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        plot_bgcolor='white',
        height=400,
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    save_and_compress(fig, 'parametric-integral-history.png', width=1100, height=400)


def main():
    """生成所有图形"""
    print("开始生成含参变量积分配图...")
    
    plot_parametric_integral_family()
    plot_gamma_function()
    plot_leibnitz_rule_geometry()
    plot_continuity_demo()
    plot_error_function()
    plot_feynman_trick_demo()
    plot_historical_timeline()
    
    print("\n所有图形生成完成！")
    
    # 检查文件大小
    print("\n文件大小统计：")
    for fname in os.listdir(output_dir):
        if fname.startswith('parametric-integral') or fname.startswith('gamma') or \
           fname.startswith('leibnitz') or fname.startswith('error') or \
           fname.startswith('feynman'):
            fpath = os.path.join(output_dir, fname)
            size = os.path.getsize(fpath)
            print(f"  {fname}: {size/1024:.1f} KB")


if __name__ == '__main__':
    main()
