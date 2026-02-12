#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
柯西积分定理相关数学图形绘制
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
output_dir = 'static/images/posts/cauchy-theorem'
os.makedirs(output_dir, exist_ok=True)

# ============================================
# 图1: 复平面展示
# ============================================
print("绘制图1: 复平面展示...")
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制坐标轴
ax.axhline(y=0, color='k', linewidth=1, linestyle='-')
ax.axvline(x=0, color='k', linewidth=1, linestyle='-')

# 设置坐标范围
ax.set_xlim(-2, 4)
ax.set_ylim(-1, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlabel('实轴 Re', fontsize=14, fontweight='bold')
ax.set_ylabel('虚轴 Im', fontsize=14, fontweight='bold')
ax.set_title('复平面与复数表示', fontsize=16, fontweight='bold', pad=20)

# 标注原点
ax.text(0.1, 0.1, 'O', fontsize=14, fontweight='bold')

# 绘制复数点 z1 = 3 + 2i
z1_x, z1_y = 3, 2
ax.plot(z1_x, z1_y, 'ro', markersize=10, zorder=5)
ax.annotate(r'$z_1 = 3 + 2i$',
            xy=(z1_x, z1_y), xytext=(3.3, 2.3),
            fontsize=14, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

# 绘制从原点到z1的向量
ax.arrow(0, 0, z1_x, z1_y, head_width=0.15, head_length=0.15,
         fc='blue', ec='blue', linewidth=2, alpha=0.7, zorder=3)

# 绘制模
r1 = np.sqrt(z1_x**2 + z1_y**2)
ax.text(z1_x/2, z1_y/2 + 0.2, f'|z₁| = {r1:.2f}',
        fontsize=12, color='blue', bbox=dict(boxstyle='round,pad=0.3',
        facecolor='lightblue', alpha=0.5))

# 绘制复数点 z2 = -1 + 3i
z2_x, z2_y = -1, 3
ax.plot(z2_x, z2_y, 'go', markersize=10, zorder=5)
ax.annotate(r'$z_2 = -1 + 3i$',
            xy=(z2_x, z2_y), xytext=(-1.8, 3.3),
            fontsize=14, color='darkgreen',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

# 绘制从原点到z2的向量
ax.arrow(0, 0, z2_x, z2_y, head_width=0.15, head_length=0.15,
         fc='green', ec='green', linewidth=2, alpha=0.7, zorder=3)

plt.tight_layout()
plt.savefig(f'{output_dir}/complex-plane.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir}/complex-plane.png")
plt.close()

# ============================================
# 图2: Δz 的趋近方式
# ============================================
print("绘制图2: Δz 的趋近方式...")
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制坐标轴
ax.axhline(y=0, color='k', linewidth=1)
ax.axvline(x=0, color='k', linewidth=1)

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlabel('实轴 Re', fontsize=14, fontweight='bold')
ax.set_ylabel('虚轴 Im', fontsize=14, fontweight='bold')
ax.set_title('复导数的定义：Δz 从不同方向趋近于 0', fontsize=16, fontweight='bold', pad=20)

# 点 z 的位置
z_x, z_y = 1, 1
ax.plot(z_x, z_y, 'o', color='darkblue', markersize=15, zorder=5,
        markeredgecolor='black', markeredgewidth=2)
ax.text(z_x + 0.1, z_y + 0.1, 'z', fontsize=16, fontweight='bold')

# 方向1：从实轴方向（水平）
ax.arrow(z_x - 1, z_y, 0.8, 0, head_width=0.08, head_length=0.1,
         fc='orange', ec='orange', linewidth=3, alpha=0.8, zorder=3)
ax.text(z_x - 0.7, z_y + 0.15, 'Δz = Δx', fontsize=13, color='darkorange',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

# 方向2：从虚轴方向（垂直）
ax.arrow(z_x, z_y - 1, 0, 0.8, head_width=0.08, head_length=0.1,
         fc='purple', ec='purple', linewidth=3, alpha=0.8, zorder=3)
ax.text(z_x + 0.15, z_y - 0.5, 'Δz = iΔy', fontsize=13, color='darkviolet',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.6))

# 方向3：对角线方向
theta = np.pi/4
ax.arrow(z_x - 0.7*np.cos(theta), z_y - 0.7*np.sin(theta),
         0.6*np.cos(theta), 0.6*np.sin(theta),
         head_width=0.08, head_length=0.1,
         fc='brown', ec='brown', linewidth=3, alpha=0.8, zorder=3)
ax.text(z_x - 0.5, z_y - 0.6, 'Δz = re^{iθ}', fontsize=13, color='maroon',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.6))

# 圆圈表示极限过程
circle = Circle((z_x, z_y), 0.3, fill=False, edgecolor='red',
                linewidth=2, linestyle='--', alpha=0.5)
ax.add_patch(circle)

plt.tight_layout()
plt.savefig(f'{output_dir}/delta-z-approaches.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir}/delta-z-approaches.png")
plt.close()

# ============================================
# 图3: 格林定理 - 区域和边界
# ============================================
print("绘制图3: 格林定理...")
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_xlabel('x', fontsize=14, fontweight='bold')
ax.set_ylabel('y', fontsize=14, fontweight='bold')
ax.set_title('格林定理：区域 D 与边界曲线 ∂D', fontsize=16, fontweight='bold', pad=20)

# 绘制区域 D（椭圆）
theta = np.linspace(0, 2*np.pi, 200)
x_ellipse = 2.5 + 1.8*np.cos(theta)
y_ellipse = 2.5 + 1.5*np.sin(theta)
ax.fill(x_ellipse, y_ellipse, alpha=0.2, color='lightblue', label='区域 D')

# 绘制边界 ∂D（逆时针方向）
ax.plot(x_ellipse, y_ellipse, 'b-', linewidth=3, label='边界 ∂D')

# 绘制方向箭头（逆时针）
for t_arrow in [np.pi/4, np.pi, 5*np.pi/4]:
    x_arrow = 2.5 + 1.8*np.cos(t_arrow)
    y_arrow = 2.5 + 1.5*np.sin(t_arrow)
    dx = -1.8*np.sin(t_arrow)
    dy = 1.5*np.cos(t_arrow)
    ax.arrow(x_arrow, y_arrow, dx*0.1, dy*0.1,
             head_width=0.15, head_length=0.15, fc='blue', ec='blue', linewidth=2)

# 标注
ax.text(2.5, 2.5, 'D', fontsize=20, fontweight='bold',
        ha='center', va='center', color='darkblue')

# 添加说明文字
ax.text(2.5, 0.5, '∮_∂D (P dx + Q dy) = ∬_D (∂Q/∂x - ∂P/∂y) dx dy',
        fontsize=14, ha='center', bbox=dict(boxstyle='round,pad=0.5',
        facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{output_dir}/greens-theorem.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir}/greens-theorem.png")
plt.close()

# ============================================
# 图4: 柯西积分定理 - 单连通区域
# ============================================
print("绘制图4: 柯西积分定理（单连通区域）...")
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_xlabel('实轴 Re', fontsize=14, fontweight='bold')
ax.set_ylabel('虚轴 Im', fontsize=14, fontweight='bold')
ax.set_title('柯西积分定理：f 在 D 内解析 ⇒ ∮_γ f(z)dz = 0', fontsize=16, fontweight='bold', pad=20)

# 绘制区域 D
theta = np.linspace(0, 2*np.pi, 200)
x_circle = 2.5 + 2*np.cos(theta)
y_circle = 2.5 + 2*np.sin(theta)
ax.fill(x_circle, y_circle, alpha=0.15, color='lightgreen')

# 绘制闭合曲线 γ
ax.plot(x_circle, y_circle, 'g-', linewidth=3, label='曲线 γ')

# 绘制方向箭头（逆时针）
for t_arrow in [0, 2*np.pi/3, 4*np.pi/3]:
    x_arrow = 2.5 + 2*np.cos(t_arrow)
    y_arrow = 2.5 + 2*np.sin(t_arrow)
    dx = -2*np.sin(t_arrow)
    dy = 2*np.cos(t_arrow)
    ax.arrow(x_arrow, y_arrow, dx*0.08, dy*0.08,
             head_width=0.12, head_length=0.12, fc='green', ec='green', linewidth=2)

# 标注区域
ax.text(2.5, 2.5, 'D', fontsize=20, fontweight='bold',
        ha='center', va='center', color='darkgreen')
ax.text(2.5, 4.8, 'f 在 D 内解析', fontsize=13, color='darkgreen',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))

# 添加积分公式
ax.text(2.5, 0.3, r'$\oint_\gamma f(z)dz = 0$', fontsize=18, ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='honeydew', alpha=0.9))

plt.tight_layout()
plt.savefig(f'{output_dir}/cauchy-theorem-simple.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir}/cauchy-theorem-simple.png")
plt.close()

# ============================================
# 图5: 多连通区域
# ============================================
print("绘制图5: 多连通区域...")
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlim(-1, 8)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_xlabel('实轴 Re', fontsize=14, fontweight='bold')
ax.set_ylabel('虚轴 Im', fontsize=14, fontweight='bold')
ax.set_title('多连通区域：∮_γ₀ f(z)dz = Σ∮_γₖ f(z)dz', fontsize=16, fontweight='bold', pad=20)

# 外边界 γ₀（大椭圆）
theta_out = np.linspace(0, 2*np.pi, 200)
x_out = 3.5 + 2.5*np.cos(theta_out)
y_out = 2.5 + 2*np.sin(theta_out)
ax.fill(x_out, y_out, alpha=0.1, color='orange')
ax.plot(x_out, y_out, 'orange', linewidth=3, label='外边界 γ₀')

# 外边界箭头
for t_arrow in [0, np.pi]:
    x_arrow = 3.5 + 2.5*np.cos(t_arrow)
    y_arrow = 2.5 + 2*np.sin(t_arrow)
    dx = -2.5*np.sin(t_arrow)
    dy = 2*np.cos(t_arrow)
    ax.arrow(x_arrow, y_arrow, dx*0.08, dy*0.08,
             head_width=0.12, head_length=0.12, fc='orange', ec='orange', linewidth=2)

# 内边界 γ₁（左边的洞）
theta_in1 = np.linspace(0, 2*np.pi, 200)
x_in1 = 2 + 0.6*np.cos(theta_in1)
y_in1 = 2.5 + 0.6*np.sin(theta_in1)
ax.fill(x_in1, y_in1, alpha=0.3, color='white')
ax.plot(x_in1, y_in1, 'r-', linewidth=2.5, label='内边界 γ₁')

# 内边界 γ₁ 箭头（逆时针）
for t_arrow in [np.pi/2]:
    x_arrow = 2 + 0.6*np.cos(t_arrow)
    y_arrow = 2.5 + 0.6*np.sin(t_arrow)
    dx = -0.6*np.sin(t_arrow)
    dy = 0.6*np.cos(t_arrow)
    ax.arrow(x_arrow, y_arrow, dx*0.15, dy*0.15,
             head_width=0.08, head_length=0.08, fc='red', ec='red', linewidth=1.5)

# 内边界 γ₂（右边的洞）
theta_in2 = np.linspace(0, 2*np.pi, 200)
x_in2 = 5 + 0.7*np.cos(theta_in2)
y_in2 = 2.5 + 0.7*np.sin(theta_in2)
ax.fill(x_in2, y_in2, alpha=0.3, color='white')
ax.plot(x_in2, y_in2, 'purple', linewidth=2.5, label='内边界 γ₂')

# 内边界 γ₂ 箭头（逆时针）
for t_arrow in [3*np.pi/2]:
    x_arrow = 5 + 0.7*np.cos(t_arrow)
    y_arrow = 2.5 + 0.7*np.sin(t_arrow)
    dx = -0.7*np.sin(t_arrow)
    dy = 0.7*np.cos(t_arrow)
    ax.arrow(x_arrow, y_arrow, dx*0.15, dy*0.15,
             head_width=0.08, head_length=0.08, fc='purple', ec='purple', linewidth=1.5)

# 标注
ax.text(3.5, 4.5, '多连通区域 D', fontsize=14, fontweight='bold',
        ha='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='orange', alpha=0.7))
ax.text(2, 2.5, 'γ₁', fontsize=12, fontweight='bold', ha='center', va='center', color='red')
ax.text(5, 2.5, 'γ₂', fontsize=12, fontweight='bold', ha='center', va='center', color='purple')

# 添加公式
ax.text(6.5, 0.8, r'$\oint_{\gamma_0} f(z)dz$' + '\n' + r'$= \oint_{\gamma_1} f(z)dz + \oint_{\gamma_2} f(z)dz$',
        fontsize=13, va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='moccasin', alpha=0.8))

plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.savefig(f'{output_dir}/multi-connected-domain.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir}/multi-connected-domain.png")
plt.close()

# ============================================
# 图6: 1/z 沿单位圆积分
# ============================================
print("绘制图6: 1/z 沿单位圆积分...")
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_xlabel('实轴 Re', fontsize=14, fontweight='bold')
ax.set_ylabel('虚轴 Im', fontsize=14, fontweight='bold')
ax.set_title(r'经典例子：$\oint_{|z|=1} \frac{1}{z}dz = 2\pi i$',
             fontsize=16, fontweight='bold', pad=20)

# 单位圆
theta = np.linspace(0, 2*np.pi, 200)
x_unit = np.cos(theta)
y_unit = np.sin(theta)
ax.plot(x_unit, y_unit, 'b-', linewidth=3, label='单位圆 |z|=1')

# 标注奇点 z=0
ax.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label='奇点 z=0')
ax.text(0.1, -0.15, '奇点', fontsize=12, color='red')

# 方向箭头（逆时针）
for t_arrow in [0, 2*np.pi/3, 4*np.pi/3]:
    x_arrow = np.cos(t_arrow)
    y_arrow = np.sin(t_arrow)
    dx = -np.sin(t_arrow)
    dy = np.cos(t_arrow)
    ax.arrow(x_arrow, y_arrow, dx*0.15, dy*0.15,
             head_width=0.08, head_length=0.08, fc='blue', ec='blue', linewidth=2)

# 标注几个关键点
ax.plot(1, 0, 'go', markersize=8, zorder=5)
ax.text(1.1, 0.1, 'z=1', fontsize=12)

ax.plot(-1, 0, 'go', markersize=8, zorder=5)
ax.text(-1.3, 0.1, 'z=-1', fontsize=12)

ax.plot(0, 1, 'go', markersize=8, zorder=5)
ax.text(0.1, 1.15, 'z=i', fontsize=12)

ax.plot(0, -1, 'go', markersize=8, zorder=5)
ax.text(0.1, -1.2, 'z=-i', fontsize=12)

# 添加积分结果
ax.text(0, -2, r'$\oint_{|z|=1} \frac{1}{z}dz = 2\pi i \neq 0$',
        fontsize=16, ha='center', color='darkblue',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', alpha=0.8))

ax.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig(f'{output_dir}/unit-circle-integral.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir}/unit-circle-integral.png")
plt.close()

print("\n✅ 所有图形绘制完成！")
print(f"保存位置: {output_dir}/")
