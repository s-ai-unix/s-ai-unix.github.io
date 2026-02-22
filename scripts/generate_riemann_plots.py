import plotly.graph_objects as go
import numpy as np
import subprocess
from pathlib import Path

def save_and_compress(fig, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(filepath, scale=2)
    if filepath.endswith('.png'):
        subprocess.run(['pngquant', '--quality=70-85', '--force', '--output', filepath, filepath], check=False)
    print(f"✅ 已保存并压缩: {filepath}")

def generate_analytic_continuation():
    fig = go.Figure()
    
    # 发散区 Re(s) <= 1
    fig.add_shape(type="rect", x0=-4, y0=-4, x1=1, y1=4,
                  fillcolor="rgba(255, 59, 48, 0.15)", line=dict(width=0))
                  
    # 收敛区 Re(s) > 1
    fig.add_shape(type="rect", x0=1, y0=-4, x1=4, y1=4,
                  fillcolor="rgba(52, 199, 89, 0.15)", line=dict(width=0))
                  
    # 边界 Re(s) = 1
    fig.add_shape(type="line", x0=1, y0=-4, x1=1, y1=4,
                  line=dict(color="#1D1D1F", dash="dash", width=2))
                  
    # 标注
    fig.add_annotation(x=2.5, y=0, text="<b>原始无穷级数收敛区</b><br>$\text{Re}(s) > 1$", 
                       showarrow=False, font=dict(size=14, color="#1D1D1F"))
                       
    fig.add_annotation(x=-1.5, y=0, text="<b>解析延拓扩展区</b><br>$\text{Re}(s) \le 1$", 
                       showarrow=False, font=dict(size=14, color="#1D1D1F"))
                       
    fig.add_annotation(x=1, y=3.5, text="<b>极大极点边界</b> $s=1$", 
                       showarrow=True, arrowhead=2, ax=-60, ay=0, font=dict(size=12))

    fig.update_layout(
        title="黎曼 $\zeta$ 函数的解析延拓示意",
        xaxis=dict(title="实部 $\sigma$", range=[-4, 4], zeroline=True, zerolinecolor="#86868B"),
        yaxis=dict(title="虚部 $t$", range=[-4, 4], zeroline=True, zerolinecolor="#86868B"),
        template='plotly_white',
        width=800, height=600,
        font=dict(family='-apple-system, BlinkMacSystemFont, "SF Pro Text", Segoe UI, Roboto, sans-serif', size=14)
    )
    
    save_and_compress(fig, 'static/images/plots/riemann-analytic-continuation.png')

def generate_critical_line():
    fig = go.Figure()
    
    # 临界带 0 < Re(s) < 1
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=30,
                  fillcolor="rgba(88, 86, 214, 0.15)", line=dict(width=0))
                  
    # 临界线 Re(s) = 0.5
    fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=30,
                  line=dict(color="#FF3B30", dash="dash", width=2))
                  
    # 前几个非平凡零点
    zeros_y = [14.1347, 21.0220, 25.0108, 30.4248]
    zeros_x = [0.5] * len(zeros_y)
    
    fig.add_trace(go.Scatter(
        x=zeros_x, y=zeros_y, mode='markers',
        marker=dict(symbol='diamond', size=12, color='#34C759', line=dict(width=1, color='white')),
        name='非平凡零点'
    ))
    
    for i, y in enumerate(zeros_y):
        fig.add_annotation(
            x=0.5, y=y, text=f"$\gamma_{i+1} \\approx {y:.1f}$",
            showarrow=True, arrowhead=2, ax=40, ay=0,
            font=dict(size=12)
        )
        
    fig.add_annotation(x=0.5, y=5, text="<b>临界线</b> $\text{Re}(s)=1/2$",
                       showarrow=True, arrowhead=2, ax=-60, ay=0,
                       font=dict(size=14, color="#FF3B30"))

    fig.add_annotation(x=0.5, y=28, text="<b>临界带</b> $0 < \text{Re}(s) < 1$",
                       showarrow=False, font=dict(size=14, color="#5856D6"))

    fig.update_layout(
        title="复平面上的非平凡零点与临界线",
        xaxis=dict(title="实部 $\sigma$", range=[-0.5, 1.5], zeroline=True, zerolinecolor="#86868B"),
        yaxis=dict(title="虚部 $t$", range=[0, 32], zeroline=False),
        template='plotly_white',
        width=800, height=600,
        showlegend=False,
        font=dict(family='-apple-system, BlinkMacSystemFont, "SF Pro Text", Segoe UI, Roboto, sans-serif', size=14)
    )
    
    save_and_compress(fig, 'static/images/plots/riemann-critical-line.png')

if __name__ == '__main__':
    generate_analytic_continuation()
    generate_critical_line()
