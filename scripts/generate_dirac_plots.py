#!/usr/bin/env python3
"""
Generate Plotly visualizations for Dirac Equation article.
Images will be saved as PNG files in static/images/math/
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# Create output directory
output_dir = "static/images/math"
os.makedirs(output_dir, exist_ok=True)

# Set Plotly style
template = "plotly_white"
colors = {
    'blue': '#007AFF',
    'green': '#34C759',
    'orange': '#FF9500',
    'red': '#FF3B30',
    'purple': '#AF52DE',
    'cyan': '#00D4AA'
}

print("=" * 60)
print("Generating Plotly visualizations for Dirac Equation")
print("=" * 60)

# 1. Energy-Momentum Relation for Relativistic Particles
print("\n[1/5] Energy-Momentum Relation...")
p = np.linspace(-3, 3, 500)
m = 1  # Normalized mass
c = 1  # Normalized speed of light

# Positive and negative energy solutions
E_positive = np.sqrt(p**2 * c**2 + m**2 * c**4)
E_negative = -np.sqrt(p**2 * c**2 + m**2 * c**4)

fig1 = go.Figure()

# Add positive energy branch
fig1.add_trace(go.Scatter(
    x=p,
    y=E_positive,
    mode='lines',
    name='正能量',
    line=dict(color=colors['blue'], width=3),
    legendgroup='energy'
))

# Add negative energy branch
fig1.add_trace(go.Scatter(
    x=p,
    y=E_negative,
    mode='lines',
    name='负能量',
    line=dict(color=colors['red'], width=3, dash='dash'),
    legendgroup='energy'
))

# Add non-relativistic approximation (dotted)
E_classical = m * c**2 + p**2 / (2 * m)
fig1.add_trace(go.Scatter(
    x=p,
    y=E_classical,
    mode='lines',
    name='经典近似 (p²/2m)',
    line=dict(color=colors['green'], width=2, dash='dot'),
    legendgroup='approx'
))

fig1.update_layout(
    title='相对论能量-动量关系 E² = p²c² + m²c⁴',
    xaxis_title='动量 p (归一化单位)',
    yaxis_title='能量 E (归一化单位)',
    template=template,
    font=dict(family='Arial, sans-serif', size=14),
    hovermode='x unified',
    height=500,
    width=800
)

fig1.write_image(f'{output_dir}/dirac-energy-momentum.png', width=800, height=500, scale=2)
print(f"  ✓ Saved: dirac-energy-momentum.png")

# 2. Dirac Sea Visualization
print("\n[2/5] Dirac Sea Visualization...")
E_levels = np.linspace(-5, -0.1, 50)

fig2 = go.Figure()

# Create Dirac sea levels
for i, E in enumerate(E_levels):
    fig2.add_trace(go.Scatter(
        x=[0, 1],
        y=[E, E],
        mode='lines',
        line=dict(color=colors['blue'], width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

# Add positive energy levels
E_pos_levels = np.linspace(0.1, 5, 50)
for i, E in enumerate(E_pos_levels):
    fig2.add_trace(go.Scatter(
        x=[0, 1],
        y=[E, E],
        mode='lines',
        line=dict(color=colors['green'], width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

# Add hole
fig2.add_trace(go.Scatter(
    x=[0.5, 0.5],
    y=[-2, -2],
    mode='markers',
    name='空穴 (正电子)',
    marker=dict(
        color=colors['orange'],
        size=20,
        symbol='circle-open',
        line=dict(color='white', width=2)
    ),
    hovertemplate='空穴: %{y} eV<extra></extra>'
))

# Add electron
fig2.add_trace(go.Scatter(
    x=[0.5, 0.5],
    y=[1, 1],
    mode='markers',
    name='电子',
    marker=dict(
        color=colors['cyan'],
        size=15,
        symbol='circle',
        line=dict(color='white', width=2)
    ),
    hovertemplate='电子: %{y} eV<extra></extra>'
))

fig2.update_layout(
    title='狄拉克海模型',
    xaxis=dict(showticklabels=False, showgrid=False, range=[-0.1, 1.1]),
    yaxis_title='能量 E (eV)',
    template=template,
    font=dict(family='Arial, sans-serif', size=14),
    hovermode='closest',
    height=500,
    width=600,
    annotations=[
        dict(
            x=0.5, y=-5.5,
            text="负能量海 (被电子填满)",
            showarrow=False,
            font=dict(size=12, color=colors['blue'])
        ),
        dict(
            x=0.5, y=5.5,
            text="正能量态 (空穴=正电子)",
            showarrow=False,
            font=dict(size=12, color=colors['green'])
        )
    ]
)

fig2.write_image(f'{output_dir}/dirac-sea.png', width=600, height=500, scale=2)
print(f"  ✓ Saved: dirac-sea.png")

# 3. Spin States Visualization
print("\n[3/5] Spin States...")
theta = np.linspace(0, 2*np.pi, 200)

# Spin up state
psi_up_real = np.cos(theta/2)
psi_up_imag = np.sin(theta/2)
mag_up = np.abs(np.sqrt(psi_up_real**2 + psi_up_imag**2))**2

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=theta,
    y=mag_up,
    mode='lines',
    name='自旋向上 |↑⟩',
    line=dict(color=colors['blue'], width=3),
    fill='tozeroy',
    fillcolor=f'rgba(0, 122, 255, 0.3)'
))

# Spin down state (opposite)
mag_down = 1 - mag_up
fig3.add_trace(go.Scatter(
    x=theta,
    y=mag_down,
    mode='lines',
    name='自旋向下 |↓⟩',
    line=dict(color=colors['red'], width=3),
    fill='tozeroy',
    fillcolor=f'rgba(255, 59, 48, 0.3)'
))

fig3.update_layout(
    title='自旋态的角依赖性',
    xaxis_title='方位角 θ',
    yaxis_title='概率密度 |⟨ψ|²|',
    xaxis=dict(tickvals=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
               ticktext=['0', 'π/2', 'π', '3π/2', '2π']),
    template=template,
    font=dict(family='Arial, sans-serif', size=14),
    height=500,
    width=800
)

fig3.write_image(f'{output_dir}/dirac-spin-states.png', width=800, height=500, scale=2)
print(f"  ✓ Saved: dirac-spin-states.png")

# 4. Probability Density Comparison
print("\n[4/5] Probability Density Comparison...")
x = np.linspace(-5, 5, 500)

# Schrödinger equation solution (Gaussian wave packet)
psi_schrodinger = np.exp(-x**2 / 2)
rho_schrodinger = psi_schrodinger**2

# Klein-Gordon probability density (with sign changes)
psi_klein_gordon = np.exp(-x**2 / 2)
rho_kg = psi_klein_gordon**2 * np.cos(x)  # Simulating the sign issue

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=x,
    y=rho_schrodinger,
    mode='lines',
    name='薛定谔方程概率密度 |ψ|²',
    line=dict(color=colors['blue'], width=3),
    fill='tozeroy',
    fillcolor=f'rgba(0, 122, 255, 0.2)'
))

fig4.add_trace(go.Scatter(
    x=x,
    y=rho_kg,
    mode='lines',
    name='克莱因-戈尔登方程 (有问题)',
    line=dict(color=colors['red'], width=3),
    fill='tozeroy',
    fillcolor=f'rgba(255, 59, 48, 0.2)'
))

# Add zero line
fig4.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

fig4.update_layout(
    title='概率密度对比：薛定谔 vs 克莱因-戈尔登方程',
    xaxis_title='位置 x',
    yaxis_title='概率密度 ρ',
    template=template,
    font=dict(family='Arial, sans-serif', size=14),
    height=500,
    width=800
)

fig4.write_image(f'{output_dir}/dirac-probability-density.png', width=800, height=500, scale=2)
print(f"  ✓ Saved: dirac-probability-density.png")

# 5. Fine Structure Energy Levels
print("\n[5/5] Hydrogen Fine Structure...")
n_values = [1, 2, 3, 4]
j_values = [0.5, 1.5]

alpha = 1/137  # Fine structure constant
E_nj = []

fig5 = go.Figure()

for n in n_values:
    for j in j_values:
        if j < n:
            E = 1 + alpha**2 / (n - (j + 0.5) + np.sqrt((j + 0.5)**2 - alpha**2))**2
            E_nj.append(E)
            fig5.add_trace(go.Scatter(
                x=[n],
                y=[E],
                mode='markers',
                name=f'n={n}, j={j}',
                marker=dict(size=10, color=colors['blue']),
                text=f'j={j}',
                textposition='top center'
            ))

fig5.update_layout(
    title='氢原子精细结构能级',
    xaxis_title='主量子数 n',
    yaxis_title='相对能量 E/mc²',
    template=template,
    font=dict(family='Arial, sans-serif', size=14),
    xaxis=dict(tickmode='linear', dtick=1),
    height=500,
    width=600
)

fig5.write_image(f'{output_dir}/dirac-fine-structure.png', width=600, height=500, scale=2)
print(f"  ✓ Saved: dirac-fine-structure.png")

print("\n" + "=" * 60)
print("All Dirac Equation visualizations completed!")
print("=" * 60)
