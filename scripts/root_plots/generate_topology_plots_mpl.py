"""
使用 Matplotlib 生成拓扑学配图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arc, FancyArrowPatch
import subprocess
import os

# 设置中文字体
plt.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 苹果风格配色
APPLE_BLUE = '#007AFF'
APPLE_GREEN = '#34C759'
APPLE_ORANGE = '#FF9500'
APPLE_RED = '#FF3B30'
APPLE_PURPLE = '#AF52DE'
APPLE_GRAY = '#8E8E93'


def compress_png(filepath):
    """压缩 PNG 图片"""
    subprocess.run([
        'pngquant', '--quality=70-85', '--force', 
        '--output', filepath, filepath
    ], check=False)
    print(f"✅ 已压缩: {filepath}")


def plot_mobius_strip():
    """绘制莫比乌斯带"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 莫比乌斯带的参数曲线
    theta = np.linspace(0, 4*np.pi, 400)
    # 外轮廓
    r = 1 + 0.3 * np.cos(theta/2)
    x_outer = r * np.cos(theta)
    y_outer = r * np.sin(theta)
    
    # 内轮廓
    r_inner = 1 - 0.3 * np.cos(theta/2)
    x_inner = r_inner * np.cos(theta)
    y_inner = r_inner * np.sin(theta)
    
    # 绘制填充
    ax.fill(x_outer, y_outer, alpha=0.3, color=APPLE_BLUE)
    ax.fill(x_inner, y_inner, alpha=1, color='white')
    
    # 中心线
    theta_c = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta_c), np.sin(theta_c), '--', color=APPLE_RED, linewidth=2, label='中心线')
    
    # 箭头表示方向
    ax.annotate('', xy=(0.7, 0.7), xytext=(0.4, 0.9),
                arrowprops=dict(arrowstyle='->', color=APPLE_ORANGE, lw=2))
    
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title('莫比乌斯带：单侧不可定向曲面', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('../static/images/plots/mobius-strip.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('../static/images/plots/mobius-strip.png')


def plot_open_sets():
    """绘制开集概念"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：拓扑空间中的开集
    ax = axes[0]
    
    # 空间X
    circle_X = Circle((0, 0), 3, fill=False, edgecolor=APPLE_GRAY, linewidth=2, linestyle='--')
    ax.add_patch(circle_X)
    ax.annotate('X', xy=(2.5, 2.5), fontsize=12, color=APPLE_GRAY)
    
    # 开集U
    circle_U = Circle((-0.5, 0.5), 1.5, fill=True, facecolor=APPLE_BLUE, alpha=0.3, 
                      edgecolor=APPLE_BLUE, linewidth=2)
    ax.add_patch(circle_U)
    ax.annotate('U', xy=(-0.5, 2.2), fontsize=12, color=APPLE_BLUE)
    
    # 开集V
    circle_V = Circle((1.5, -0.5), 1.2, fill=True, facecolor=APPLE_GREEN, alpha=0.3,
                      edgecolor=APPLE_GREEN, linewidth=2)
    ax.add_patch(circle_V)
    ax.annotate('V', xy=(2.8, -0.5), fontsize=12, color=APPLE_GREEN)
    
    # 交集中的点
    ax.plot(0.8, 0.2, 'o', color=APPLE_RED, markersize=8)
    ax.annotate('p', xy=(0.8, 0.2), xytext=(1.2, 0.5), fontsize=12, color=APPLE_RED)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('拓扑空间中的开集', fontsize=12, fontweight='bold')
    
    # 右图：开集性质
    ax = axes[1]
    
    # 任意并
    circle1 = Circle((-1.5, 0), 2, fill=True, facecolor=APPLE_BLUE, alpha=0.2,
                     edgecolor=APPLE_BLUE, linewidth=2)
    ax.add_patch(circle1)
    ax.annotate('$U_1$', xy=(-3.5, 1.5), fontsize=10, color=APPLE_BLUE)
    
    circle2 = Circle((1.5, 0), 1.5, fill=True, facecolor=APPLE_GREEN, alpha=0.2,
                     edgecolor=APPLE_GREEN, linewidth=2)
    ax.add_patch(circle2)
    ax.annotate('$U_2$', xy=(2.8, 1), fontsize=10, color=APPLE_GREEN)
    
    # 有限交
    circle3 = Circle((0, -1.5), 1.2, fill=True, facecolor=APPLE_ORANGE, alpha=0.3,
                     edgecolor=APPLE_ORANGE, linewidth=2)
    ax.add_patch(circle3)
    ax.annotate('$U_1 \\cap U_2$', xy=(1.5, -2.5), fontsize=10, color=APPLE_ORANGE)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('开集的基本性质：任意并、有限交', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../static/images/plots/open-sets.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('../static/images/plots/open-sets.png')


def plot_continuity():
    """绘制连续性概念"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：连续函数
    ax = axes[0]
    x = np.linspace(-2, 2, 200)
    y = np.sin(x) + 0.3*x
    ax.plot(x, y, color=APPLE_BLUE, linewidth=2.5, label='$f(x) = \\sin x + 0.3x$')
    
    x0, y0 = 0.5, np.sin(0.5) + 0.3*0.5
    epsilon = 0.4
    delta = 0.3
    
    # epsilon 带
    ax.axhspan(y0-epsilon, y0+epsilon, alpha=0.2, color=APPLE_ORANGE)
    ax.axvspan(x0-delta, x0+delta, alpha=0.15, color=APPLE_BLUE)
    
    # 关键点
    ax.plot(x0, y0, 'D', color=APPLE_RED, markersize=10, label=f'点 $({x0}, {y0:.2f})$')
    
    ax.annotate('$\\epsilon$', xy=(1.8, y0+epsilon/2), fontsize=12, color=APPLE_ORANGE)
    ax.annotate('$\\delta$', xy=(x0+delta/2, -1.5), fontsize=12, color=APPLE_BLUE)
    
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$f(x)$', fontsize=12)
    ax.set_title('连续函数：拓扑学视角', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 右图：拓扑定义
    ax = axes[1]
    
    # 定义域 X
    circle_X = Circle((-2, 0), 1.5, fill=True, facecolor=APPLE_BLUE, alpha=0.2,
                      edgecolor=APPLE_BLUE, linewidth=2)
    ax.add_patch(circle_X)
    ax.annotate('X', xy=(-3.5, 1.2), fontsize=12, color=APPLE_BLUE)
    
    # 开集 U
    circle_U = Circle((-2, 0), 0.8, fill=True, facecolor=APPLE_GREEN, alpha=0.4,
                      edgecolor=APPLE_GREEN, linewidth=2)
    ax.add_patch(circle_U)
    ax.annotate('U', xy=(-1.3, 0.5), fontsize=11, color=APPLE_GREEN)
    
    # 值域 Y
    circle_Y = Circle((2, 0), 1.5, fill=True, facecolor=APPLE_ORANGE, alpha=0.2,
                      edgecolor=APPLE_ORANGE, linewidth=2)
    ax.add_patch(circle_Y)
    ax.annotate('Y', xy=(3.2, 1.2), fontsize=12, color=APPLE_ORANGE)
    
    # f(U) 是开集
    ellipse_fU = patches.Ellipse((2, 0), 1.2, 1.6, fill=True, facecolor=APPLE_PURPLE, alpha=0.4,
                                  edgecolor=APPLE_PURPLE, linewidth=2)
    ax.add_patch(ellipse_fU)
    ax.annotate('$f(U)$', xy=(2.8, 0.6), fontsize=11, color=APPLE_PURPLE)
    
    # 映射箭头
    ax.annotate('', xy=(1.2, 0), xytext=(-1.2, 0),
                arrowprops=dict(arrowstyle='->', color=APPLE_RED, lw=2.5))
    ax.annotate('$f$', xy=(0, 0.3), fontsize=14, color=APPLE_RED, ha='center')
    
    # 连续性说明
    ax.annotate('开集的原像仍是开集', xy=(0, -2.5), fontsize=11, ha='center')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('连续性的拓扑定义', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../static/images/plots/continuity.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('../static/images/plots/continuity.png')


def plot_homeomorphism():
    """绘制同胚映射"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 空间 X：圆
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = 1.5 * np.cos(theta) - 3
    y_circle = 1.5 * np.sin(theta)
    ax.fill(x_circle, y_circle, alpha=0.2, color=APPLE_BLUE)
    ax.plot(x_circle, y_circle, color=APPLE_BLUE, linewidth=2.5)
    ax.annotate('X（圆）', xy=(-3, 2), fontsize=12, color=APPLE_BLUE, ha='center')
    
    # 随机点
    np.random.seed(42)
    for _ in range(8):
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.3, 1.2)
        x_pt = -3 + r * np.cos(angle)
        y_pt = r * np.sin(angle)
        ax.plot(x_pt, y_pt, 'o', color=APPLE_GREEN, markersize=5)
    
    # 映射箭头
    ax.annotate('', xy=(1, 0), xytext=(-1.5, 0),
                arrowprops=dict(arrowstyle='<->', color=APPLE_RED, lw=2.5))
    ax.annotate('$f$（双射且双向连续）', xy=(0, 0.4), fontsize=11, color=APPLE_RED, ha='center')
    
    # 空间 Y：正方形
    square = patches.Rectangle((1.5, -1.5), 3, 3, fill=True, facecolor=APPLE_ORANGE, alpha=0.2,
                               edgecolor=APPLE_ORANGE, linewidth=2.5)
    ax.add_patch(square)
    ax.annotate('Y（正方形）', xy=(3, 2), fontsize=12, color=APPLE_ORANGE, ha='center')
    
    # 对应的点
    for _ in range(8):
        x_pt = np.random.uniform(2, 4)
        y_pt = np.random.uniform(-1, 1)
        ax.plot(x_pt, y_pt, 'o', color=APPLE_GREEN, markersize=5)
    
    ax.annotate('拓扑等价：\n洞的数量相同', xy=(0, -2.5), fontsize=11, ha='center', 
                style='italic', color=APPLE_GRAY)
    
    ax.set_xlim(-5, 6)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('同胚映射：圆与正方形拓扑等价', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../static/images/plots/homeomorphism.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('../static/images/plots/homeomorphism.png')


def plot_compactness():
    """绘制紧致性概念"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：紧致 [0, 4]
    ax = axes[0]
    ax.hlines(0, 0, 4, colors=APPLE_BLUE, linewidth=4)
    ax.plot([0, 4], [0, 0], 'o', color=APPLE_BLUE, markersize=12)
    
    # 有限覆盖
    covers = [(0, 1.8), (1.2, 3), (2.5, 4)]
    colors = [APPLE_ORANGE, APPLE_GREEN, APPLE_PURPLE]
    y_offset = [0.15, 0.25, 0.15]
    
    for (a, b), c, yo in zip(covers, colors, y_offset):
        ax.hlines(yo, a, b, colors=c, linewidth=6, alpha=0.6)
        ax.plot([a, b], [yo, yo], 's', color=c, markersize=6)
    
    ax.annotate('[0, 4]', xy=(2, -0.35), fontsize=12, ha='center', color=APPLE_BLUE, fontweight='bold')
    ax.annotate('有限子覆盖存在', xy=(2, 0.5), fontsize=11, ha='center', style='italic', color=APPLE_GRAY)
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 0.8)
    ax.axis('off')
    ax.set_title('紧致：闭区间 [a,b]', fontsize=12, fontweight='bold')
    
    # 右图：非紧致 (0, 4)
    ax = axes[1]
    x = np.linspace(0.01, 3.99, 200)
    ax.hlines(0, 0.01, 3.99, colors=APPLE_RED, linewidth=4)
    ax.plot([0, 4], [0, 0], 'o', color=APPLE_RED, markersize=12, markerfacecolor='white')
    
    # 无限覆盖（示意）
    for i in range(6):
        a = 0.5 + i * 0.6
        b = a + 0.8
        if b < 4:
            ax.hlines(0.15, a, b, colors=APPLE_ORANGE, linewidth=5, alpha=0.4)
    
    ax.annotate('(0, 4)', xy=(2, -0.35), fontsize=12, ha='center', color=APPLE_RED, fontweight='bold')
    ax.annotate('不存在有限子覆盖', xy=(2, 0.5), fontsize=11, ha='center', style='italic', color=APPLE_GRAY)
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 0.8)
    ax.axis('off')
    ax.set_title('非紧致：开区间 (a,b)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../static/images/plots/compactness.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('../static/images/plots/compactness.png')


def plot_connectedness():
    """绘制连通性概念"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 左图：连通空间
    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 100)
    x_donut = 2 * np.cos(theta)
    y_donut = 2 * np.sin(theta)
    ax.fill(x_donut, y_donut, alpha=0.2, color=APPLE_BLUE)
    ax.plot(x_donut, y_donut, color=APPLE_BLUE, linewidth=2.5)
    
    # 连接路径
    t = np.linspace(0, 1, 50)
    x_path = 1.5 * np.cos(t * np.pi) - 0.5
    y_path = 1.5 * np.sin(t * np.pi)
    ax.plot(x_path, y_path, '--', color=APPLE_ORANGE, linewidth=2)
    ax.plot([-0.5, 1], [0, 0], 'o', color=APPLE_RED, markersize=8)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('连通空间', fontsize=12, fontweight='bold')
    
    # 中图：道路连通
    ax = axes[1]
    t = np.linspace(0, 1, 100)
    x_path = 4 * t - 2
    y_path = 0.8 * np.sin(3 * np.pi * t)
    ax.plot(x_path, y_path, color=APPLE_GREEN, linewidth=3)
    
    ax.plot(-2, 0, 'o', color=APPLE_BLUE, markersize=12)
    ax.plot(2, 0, 'o', color=APPLE_BLUE, markersize=12)
    ax.annotate('A', xy=(-2, -0.4), fontsize=12, ha='center')
    ax.annotate('B', xy=(2, -0.4), fontsize=12, ha='center')
    
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title('道路连通', fontsize=12, fontweight='bold')
    
    # 右图：不连通空间
    ax = axes[2]
    circle1_x = 0.8 * np.cos(theta) - 1.5
    circle1_y = 0.8 * np.sin(theta)
    ax.fill(circle1_x, circle1_y, alpha=0.2, color=APPLE_RED)
    ax.plot(circle1_x, circle1_y, color=APPLE_RED, linewidth=2.5)
    
    circle2_x = 0.8 * np.cos(theta) + 1.5
    circle2_y = 0.8 * np.sin(theta)
    ax.fill(circle2_x, circle2_y, alpha=0.2, color=APPLE_RED)
    ax.plot(circle2_x, circle2_y, color=APPLE_RED, linewidth=2.5)
    
    # 断开标记
    ax.annotate('', xy=(0.5, 0), xytext=(-0.5, 0),
                arrowprops=dict(arrowstyle='<->', color=APPLE_GRAY, lw=2))
    ax.annotate('分离', xy=(0, -0.3), fontsize=10, ha='center', color=APPLE_GRAY)
    
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.axis('off')
    ax.set_title('不连通空间', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../static/images/plots/connectedness.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('../static/images/plots/connectedness.png')


def plot_manifold_concept():
    """绘制流形概念"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 左图：流形 M（圆）
    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = 2 * np.cos(theta)
    y_circle = 2 * np.sin(theta)
    ax.fill(x_circle, y_circle, alpha=0.15, color=APPLE_BLUE)
    ax.plot(x_circle, y_circle, color=APPLE_BLUE, linewidth=2.5)
    ax.annotate('M', xy=(2.3, 2.3), fontsize=12, color=APPLE_BLUE)
    
    # 高亮局部区域
    theta_local = np.linspace(np.pi/4, 3*np.pi/4, 50)
    x_local = 2 * np.cos(theta_local)
    y_local = 2 * np.sin(theta_local)
    ax.plot(x_local, y_local, color=APPLE_ORANGE, linewidth=4)
    ax.annotate('U', xy=(0, 2.4), fontsize=11, color=APPLE_ORANGE)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('流形 M', fontsize=12, fontweight='bold')
    
    # 中图：同胚映射
    ax = axes[1]
    ax.plot(x_local, y_local, color=APPLE_ORANGE, linewidth=3)
    ax.annotate('$\\phi: U \\to \\mathbb{R}^n$', xy=(0, 2.5), fontsize=12, ha='center', color=APPLE_PURPLE)
    
    # 映射箭头
    ax.annotate('', xy=(2, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=APPLE_RED, lw=3))
    ax.annotate('$\\phi$', xy=(1, 0.3), fontsize=14, color=APPLE_RED)
    
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    ax.set_title('坐标映射', fontsize=12, fontweight='bold')
    
    # 右图：欧氏空间中的像
    ax = axes[2]
    rect = patches.Rectangle((-2, -0.8), 4, 1.6, fill=True, facecolor=APPLE_GREEN, alpha=0.3,
                             edgecolor=APPLE_GREEN, linewidth=2.5)
    ax.add_patch(rect)
    ax.annotate('$\\phi(U) \\subset \\mathbb{R}^n$', xy=(0, 1.2), fontsize=12, ha='center', color=APPLE_GREEN)
    ax.annotate('开集', xy=(0, -1.2), fontsize=10, ha='center', style='italic', color=APPLE_GRAY)
    
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title('坐标卡像', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../static/images/plots/manifold-concept.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('../static/images/plots/manifold-concept.png')


def plot_euler_characteristic():
    """绘制欧拉示性数的历史示例"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # 正四面体
    ax = axes[0]
    # 顶点
    verts = [(0, 1), (-0.866, -0.5), (0.866, -0.5), (0, 0.2)]
    for i, (x, y) in enumerate(verts):
        ax.plot(x, y, 'o', color=APPLE_BLUE, markersize=12)
        ax.annotate(f'V{i+1}', xy=(x, y), xytext=(x+0.15, y+0.15), fontsize=10)
    
    # 边
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in edges:
        ax.plot([verts[i][0], verts[j][0]], [verts[i][1], verts[j][1]], 
                color=APPLE_GRAY, linewidth=2)
    
    ax.annotate('V=4, E=6, F=4\n$\\chi = 4-6+4 = 2$', xy=(0, -1.5), fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor=APPLE_BLUE, alpha=0.2))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('正四面体', fontsize=12, fontweight='bold')
    
    # 正方体投影
    ax = axes[1]
    # 简化的正方体投影
    square_outer = patches.Rectangle((-1, -1), 2, 2, fill=False, edgecolor=APPLE_BLUE, linewidth=2)
    ax.add_patch(square_outer)
    square_inner = patches.Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor=APPLE_BLUE, linewidth=2)
    ax.add_patch(square_inner)
    # 连接线
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        ax.plot([dx*0.5, dx], [dy*0.5, dy], color=APPLE_BLUE, linewidth=2)
    
    ax.annotate('V=8, E=12, F=6\n$\\chi = 8-12+6 = 2$', xy=(0, -1.6), fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor=APPLE_GREEN, alpha=0.2))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('正方体', fontsize=12, fontweight='bold')
    
    # 环面
    ax = axes[2]
    theta = np.linspace(0, 2*np.pi, 100)
    # 外圆
    x_outer = 2 * np.cos(theta)
    y_outer = 2 * np.sin(theta)
    ax.plot(x_outer, y_outer, color=APPLE_ORANGE, linewidth=2.5)
    # 内圆
    x_inner = 1 * np.cos(theta)
    y_inner = 1 * np.sin(theta)
    ax.plot(x_inner, y_inner, color=APPLE_ORANGE, linewidth=2.5)
    # 连接线
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        ax.plot([np.cos(angle), 2*np.cos(angle)], 
                [np.sin(angle), 2*np.sin(angle)], 
                color=APPLE_ORANGE, linewidth=2)
    
    ax.annotate('V=4, E=8, F=4\n$\\chi = 4-8+4 = 0$', xy=(0, -2.5), fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor=APPLE_ORANGE, alpha=0.2))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('环面', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../static/images/plots/euler-characteristic.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('../static/images/plots/euler-characteristic.png')


def main():
    """生成所有配图"""
    output_dir = '../static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成拓扑学配图...")
    
    plot_mobius_strip()
    print("✅ 莫比乌斯带")
    
    plot_open_sets()
    print("✅ 开集概念")
    
    plot_continuity()
    print("✅ 连续性概念")
    
    plot_homeomorphism()
    print("✅ 同胚映射")
    
    plot_compactness()
    print("✅ 紧致性")
    
    plot_connectedness()
    print("✅ 连通性")
    
    plot_manifold_concept()
    print("✅ 流形概念")
    
    plot_euler_characteristic()
    print("✅ 欧拉示性数")
    
    print("\n✅ 所有配图生成完成！")


if __name__ == '__main__':
    main()
