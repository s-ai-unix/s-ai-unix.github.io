#!/usr/bin/env python3
"""
Generate Plotly visualizations for Fourier Series article.
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
print("Generating Plotly visualizations for Fourier Series")
print("=" * 60)

# 1. Square Wave Fourier Approximation
print("\n[1/4] Square Wave Fourier Approximation...")
x = np.linspace(-np.pi, np.pi, 1000)

# Exact square wave
def square_wave(x):
    return np.where((x % (2*np.pi)) < np.pi, 1, -1)

# Fourier series approximations for different N
N_values = [1, 3, 5, 10, 20, 50]

fig1 = go.Figure()

for N in N_values:
    # Compute partial sum: f_N(x) = 4/π * Σ sin((2k-1)x) / (2k-1)
    y = np.zeros_like(x)
    for k in range(1, (N + 1) // 2 + 1):
        y += (4/np.pi) * np.sin((2*k - 1) * x) / (2*k - 1)

    fig1.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=f'N={N} 项',
        line=dict(width=2)
    ))

# Add exact square wave
fig1.add_trace(go.Scatter(
    x=x,
    y=square_wave(x),
    mode='lines',
    name='方波 (精确)',
    line=dict(color=colors['red'], width=3, dash='dash')
))

fig1.update_layout(
    title='方波的傅里叶级数逼近 f(x) = 4/π Σ sin((2k-1)x)/(2k-1)',
    xaxis_title='位置 x',
    yaxis_title='幅值 f(x)',
    template=template,
    font=dict(family='Arial, sans-serif', size=14),
    hovermode='x unified',
    height=500,
    width=800
)

fig1.write_image(f'{output_dir}/fourier-square-wave-approximation.png', width=800, height=500, scale=2)
print(f"  ✓ Saved: fourier-square-wave-approximation.png")

# 2. Gibbs Phenomenon
print("\n[2/4] Gibbs Phenomenon Visualization...")
x_zoom = np.linspace(-0.5, 0.5, 200)

fig2 = go.Figure()

for N in N_values:
    y = np.zeros_like(x_zoom)
    for k in range(1, (N + 1) // 2 + 1):
        y += (4/np.pi) * np.sin((2*k - 1) * x_zoom) / (2*k - 1)

    max_val = np.max(y[np.abs(x_zoom) < 0.4])
    overshoot = (max_val - 1) / 1 * 100

    fig2.add_trace(go.Scatter(
        x=[0.9],
        y=[max_val],
        mode='markers+text',
        name=f'N={N}',
        text=[f'{overshoot:.1f}%'],
        textposition='top center',
        marker=dict(size=10)
    ))

# Add exact square wave
fig2.add_trace(go.Scatter(
    x=x_zoom,
    y=square_wave(x_zoom),
    mode='lines',
    name='方波',
    line=dict(color=colors['red'], width=3, dash='dash')
))

fig2.update_layout(
    title='吉布斯现象：间断点处的过冲',
    xaxis_title='位置 x (在x=0附近)',
    yaxis_title='幅值',
    template=template,
    font=dict(family='Arial, sans-serif', size=14),
    xaxis=dict(range=[-0.45, 0.45]),
    yaxis=dict(range=[0.5, 1.15]),
    height=500,
    width=700,
    annotations=[
        dict(
            x=-0.3, y=1.1,
            text="理论值: 1.089 (过冲约9%)",
            showarrow=False,
            font=dict(size=11, color=colors['orange'])
        )
    ]
)

fig2.write_image(f'{output_dir}/fourier-gibbs-phenomenon.png', width=700, height=500, scale=2)
print(f"  ✓ Saved: fourier-gibbs-phenomenon.png")

# 3. Frequency Spectrum
print("\n[3/4] Frequency Spectrum...")
n = np.arange(1, 11, 1)
b_n = np.zeros_like(n, dtype=float)

# Fourier coefficients for square wave: b_n = 4/(nπ) for odd n, 0 for even n
for i, ni in enumerate(n):
    if ni % 2 == 1:  # odd
        b_n[i] = 4 / (ni * np.pi)
    else:
        b_n[i] = 0

fig3 = go.Figure()

fig3.add_trace(go.Bar(
    x=n,
    y=np.abs(b_n),
    name='幅值 |b_n|',
    marker=dict(color=colors['blue']),
    text=[f'{abs(b):.4f}' for b in b_n],
    textposition='outside',
))

# Add theoretical envelope (1/(nπ))
envelope = 4 / (n * np.pi)
fig3.add_trace(go.Scatter(
    x=n,
    y=envelope,
    mode='lines',
    name='理论包络 4/(nπ)',
    line=dict(color=colors['red'], width=2, dash='dot')
))

fig3.update_layout(
    title='方波的频谱分布',
    xaxis_title='频率 n',
    yaxis_title='幅值 |b_n|',
    template=template,
    font=dict(family='Arial, sans-serif', size=14),
    xaxis=dict(tickmode='linear'),
    yaxis=dict(range=[0, 1.4]),
    height=500,
    width=800
)

fig3.write_image(f'{output_dir}/fourier-frequency-spectrum.png', width=800, height=500, scale=2)
print(f"  ✓ Saved: fourier-frequency-spectrum.png")

# 4. Wave Superposition
print("\n[4/4] Wave Superposition Animation...")
t = np.linspace(0, 2*np.pi, 200)

# Individual harmonic waves
freqs = [1, 3, 5, 7]
amplitudes = [1, 1/3, 1/5, 1/7]

fig4 = go.Figure()

# Plot individual components
for freq, amp in zip(freqs, amplitudes):
    fig4.add_trace(go.Scatter(
        x=t,
        y=amp * np.sin(freq * t),
        mode='lines',
        name=f'{freq}次谐波 (幅值={amp:.2f})',
        line=dict(width=1),
        visible='legendonly'
    ))

# Plot sum (first 4 terms)
y_sum = sum(amp * np.sin(freq * t) for amp, freq in zip(amplitudes, freqs))
fig4.add_trace(go.Scatter(
    x=t,
    y=y_sum,
    mode='lines',
    name='叠加波形 (前4项)',
    line=dict(color=colors['blue'], width=3)
))

# Plot square wave for comparison
fig4.add_trace(go.Scatter(
    x=t,
    y=square_wave(t),
    mode='lines',
    name='方波 (目标)',
    line=dict(color=colors['red'], width=2, dash='dash'),
    visible='legendonly'
))

fig4.update_layout(
    title='波形叠加：用正弦波构建方波',
    xaxis_title='时间 t',
    yaxis_title='幅值',
    template=template,
    font=dict(family='Arial, sans-serif', size=14),
    height=500,
    width=800
)

fig4.write_image(f'{output_dir}/fourier-wave-superposition.png', width=800, height=500, scale=2)
print(f"  ✓ Saved: fourier-wave-superposition.png")

print("\n" + "=" * 60)
print("All Fourier Series visualizations completed!")
print("=" * 60)
