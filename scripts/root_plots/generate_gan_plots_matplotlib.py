#!/usr/bin/env python3
"""
使用 Matplotlib 为 GAN 论文解读文章生成配图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
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
APPLE_CYAN = '#5AC8FA'


def save_and_compress(fig, filepath):
    """保存并压缩图片"""
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # 压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存: {filepath}")


def plot_gan_architecture():
    """绘制 GAN 架构示意图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # 节点绘制函数
    def draw_circle(x, y, size, color, text, text_color='white'):
        circle = Circle((x, y), size, facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, 
                color=text_color, fontweight='bold')
    
    def draw_square(x, y, size, color, text):
        rect = FancyBboxPatch((x-size, y-size), 2*size, 2*size,
                              boxstyle="round,pad=0.02", 
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, 
                color='white', fontweight='bold')
    
    def draw_diamond(x, y, size, color, text):
        diamond = plt.Polygon([[x, y+size], [x+size, y], [x, y-size], [x-size, y]], 
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, 
                color='white', fontweight='bold')
    
    def draw_arrow(x1, y1, x2, y2, color):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # 绘制节点
    draw_circle(1, 5, 0.4, APPLE_GRAY, 'z')
    draw_square(3, 5, 0.4, APPLE_BLUE, 'G')
    draw_circle(5, 5, 0.35, APPLE_GREEN, 'G(z)')
    draw_circle(5, 2.5, 0.35, APPLE_ORANGE, 'x')
    draw_diamond(7, 3.75, 0.4, APPLE_PURPLE, 'D')
    draw_circle(9, 3.75, 0.3, APPLE_RED, '0/1')
    
    # 绘制箭头
    draw_arrow(1.5, 5, 2.5, 5, APPLE_GRAY)
    draw_arrow(3.5, 5, 4.5, 5, APPLE_BLUE)
    draw_arrow(5.3, 4.7, 6.5, 4.0, APPLE_GREEN)
    draw_arrow(5.3, 2.8, 6.5, 3.5, APPLE_ORANGE)
    draw_arrow(7.5, 3.75, 8.6, 3.75, APPLE_PURPLE)
    
    # 添加标签
    ax.text(1, 5.8, '潜在噪声', ha='center', fontsize=10, color=APPLE_GRAY)
    ax.text(3, 5.8, '生成器网络', ha='center', fontsize=10, color=APPLE_BLUE)
    ax.text(7, 4.8, '判别器网络', ha='center', fontsize=10, color=APPLE_PURPLE)
    ax.text(5, 1.7, '真实数据分布', ha='center', fontsize=10, color=APPLE_ORANGE)
    
    # 虚线反向传播箭头
    ax.annotate('', xy=(3, 4.3), xytext=(8.5, 3.3),
               arrowprops=dict(arrowstyle='->', color=APPLE_RED, lw=2, ls='--'))
    ax.text(5.5, 3.2, '反向传播', fontsize=9, color=APPLE_RED)
    
    # 图例
    ax.text(5, 6.5, 'GAN 对抗架构示意图', ha='center', fontsize=14, fontweight='bold')
    
    return fig


def plot_training_dynamics():
    """绘制训练损失曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    iterations = np.arange(0, 1000, 10)
    
    # 判别器损失
    d_loss = 2.0 * np.exp(-iterations / 200) + 0.3 + 0.1 * np.sin(iterations / 50)
    # 生成器损失
    g_loss = np.where(iterations < 200, 
                      1.0 + iterations / 200,
                      1.5 * np.exp(-(iterations - 200) / 300) + 0.2)
    
    # 左图：损失曲线
    ax = axes[0]
    ax.plot(iterations, d_loss, color=APPLE_PURPLE, linewidth=2, label='Discriminator Loss')
    ax.plot(iterations, g_loss, color=APPLE_GREEN, linewidth=2, label='Generator Loss')
    ax.set_xlabel('Iterations', fontsize=11)
    ax.set_ylabel('Loss Value', fontsize=11)
    ax.set_title('Training Loss Curves', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 右图：JS 散度
    ax = axes[1]
    js_div = 0.7 * np.exp(-iterations / 250)
    ax.plot(iterations, js_div, color=APPLE_BLUE, linewidth=2)
    ax.fill_between(iterations, js_div, alpha=0.2, color=APPLE_BLUE)
    ax.set_xlabel('Iterations', fontsize=11)
    ax.set_ylabel('JS Divergence', fontsize=11)
    ax.set_title('Distribution Distance Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_distribution_evolution():
    """绘制分布演化过程"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    x = np.linspace(-4, 4, 200)
    
    # 真实分布：双峰高斯
    def real_dist(x):
        return 0.5 * (1/np.sqrt(2*np.pi*0.5)) * np.exp(-(x-1.5)**2/0.5) + \
               0.5 * (1/np.sqrt(2*np.pi*0.5)) * np.exp(-(x+1.5)**2/0.5)
    
    real = real_dist(x)
    
    # 四个阶段的生成分布
    stages = [
        ('Initial Stage', (1/np.sqrt(2*np.pi*2)) * np.exp(-x**2/2), APPLE_GRAY),
        ('Mid Training', 
         0.6 * (1/np.sqrt(2*np.pi*1.2)) * np.exp(-(x-0.5)**2/1.2) + \
         0.4 * (1/np.sqrt(2*np.pi*1.2)) * np.exp(-(x+0.5)**2/1.2), APPLE_ORANGE),
        ('Late Training',
         0.5 * (1/np.sqrt(2*np.pi*0.7)) * np.exp(-(x-1.2)**2/0.7) + \
         0.5 * (1/np.sqrt(2*np.pi*0.7)) * np.exp(-(x+1.2)**2/0.7), APPLE_BLUE),
        ('Converged',
         0.5 * (1/np.sqrt(2*np.pi*0.5)) * np.exp(-(x-1.5)**2/0.5) + \
         0.5 * (1/np.sqrt(2*np.pi*0.5)) * np.exp(-(x+1.5)**2/0.5) + 0.01, APPLE_GREEN)
    ]
    
    for i, (title, gen_dist, color) in enumerate(stages):
        ax = axes[i]
        ax.plot(x, real, color=APPLE_RED, linewidth=2, linestyle='--', label='Real Data')
        ax.plot(x, gen_dist, color=color, linewidth=2, label='Generated')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_optimal_discriminator():
    """绘制最优判别器的几何解释"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    x = np.linspace(-4, 4, 500)
    
    # 真实分布和生成分布
    p_data = (1/np.sqrt(2*np.pi*0.8)) * np.exp(-(x-1)**2/0.8)
    p_g = (1/np.sqrt(2*np.pi*0.8)) * np.exp(-(x+0.5)**2/0.8)
    
    # 左图：概率密度
    ax = axes[0]
    ax.plot(x, p_data, color=APPLE_RED, linewidth=2, label=r'$p_{data}(x)$')
    ax.plot(x, p_g, color=APPLE_BLUE, linewidth=2, label=r'$p_g(x)$')
    ax.fill_between(x, p_data, alpha=0.2, color=APPLE_RED)
    ax.fill_between(x, p_g, alpha=0.2, color=APPLE_BLUE)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Probability Density Functions', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 右图：最优判别器
    ax = axes[1]
    D_optimal = p_data / (p_data + p_g + 1e-8)
    ax.plot(x, D_optimal, color=APPLE_PURPLE, linewidth=2.5)
    ax.axhline(y=0.5, color=APPLE_GRAY, linestyle='--', linewidth=1.5, label='Uncertainty Boundary')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel(r'$D^*(x)$', fontsize=11)
    ax.set_title('Optimal Discriminator Output', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_value_landscape():
    """绘制价值函数的等高线图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 创建网格
    theta_g = np.linspace(0, 4, 100)
    theta_d = np.linspace(0, 4, 100)
    Theta_G, Theta_D = np.meshgrid(theta_g, theta_d)
    
    # 简化的价值函数
    V = np.log(Theta_D + 0.1) + np.log(1.1 - Theta_D) * (1 - np.exp(-(Theta_G - 2)**2/2))
    
    # 等高线
    contour = ax.contour(Theta_G, Theta_D, V, levels=15, colors='gray', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 填充等高线
    ax.contourf(Theta_G, Theta_D, V, levels=20, cmap='RdYlBu', alpha=0.7)
    
    # 标记纳什均衡点
    ax.plot(2, 0.5, '*', markersize=20, color=APPLE_RED, markeredgecolor='white', markeredgewidth=2)
    ax.annotate('Nash Equilibrium\n(G optimal, D cannot distinguish)', 
                xy=(2, 0.5), xytext=(2.8, 1.5),
                fontsize=10, color=APPLE_RED,
                arrowprops=dict(arrowstyle='->', color=APPLE_RED))
    
    ax.set_xlabel(r'$\theta_G$ (Generator Parameter)', fontsize=11)
    ax.set_ylabel(r'$\theta_D$ (Discriminator Parameter)', fontsize=11)
    ax.set_title('GAN Value Function Landscape', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """生成所有配图"""
    output_dir = '../static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始生成 GAN 文章配图...\n")
    
    # 1. GAN 架构图
    fig1 = plot_gan_architecture()
    save_and_compress(fig1, f'{output_dir}/gan-architecture.png')
    
    # 2. 训练动态
    fig2 = plot_training_dynamics()
    save_and_compress(fig2, f'{output_dir}/gan-training-dynamics.png')
    
    # 3. 分布演化
    fig3 = plot_distribution_evolution()
    save_and_compress(fig3, f'{output_dir}/gan-distribution-evolution.png')
    
    # 4. 最优判别器
    fig4 = plot_optimal_discriminator()
    save_and_compress(fig4, f'{output_dir}/gan-optimal-discriminator.png')
    
    # 5. 价值函数 landscape
    fig5 = plot_value_landscape()
    save_and_compress(fig5, f'{output_dir}/gan-value-landscape.png')
    
    print("\n✅ 所有配图生成完成!")
    
    # 检查文件大小
    print("\n文件大小统计:")
    for fname in ['gan-architecture.png', 'gan-training-dynamics.png', 
                  'gan-distribution-evolution.png', 'gan-optimal-discriminator.png',
                  'gan-value-landscape.png']:
        fpath = f'{output_dir}/{fname}'
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  {fname}: {size/1024:.1f} KB")


if __name__ == '__main__':
    main()
