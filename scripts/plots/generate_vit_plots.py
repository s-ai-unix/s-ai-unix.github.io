#!/usr/bin/env python3
"""
生成 Vision Transformer (ViT) 论文解读的配图
使用 Matplotlib 生成专业的数理图形
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
import subprocess
import os

# 设置中文字体和样式
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 苹果风格配色
APPLE_BLUE = '#007AFF'
APPLE_GREEN = '#34C759'
APPLE_ORANGE = '#FF9500'
APPLE_RED = '#FF3B30'
APPLE_PURPLE = '#AF52DE'
APPLE_GRAY = '#8E8E93'
APPLE_TEAL = '#5AC8FA'

def save_and_compress(fig, filepath, dpi=150):
    """保存并压缩图片"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False, capture_output=True)
    print(f"✅ 已保存: {filepath}")

def plot_patch_embedding():
    """图像分块和嵌入示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    colors = [APPLE_BLUE, APPLE_GREEN, APPLE_ORANGE, APPLE_PURPLE]
    
    # 绘制原始图像网格
    grid_size = 4
    for i in range(grid_size):
        for j in range(grid_size):
            rect = FancyBboxPatch(
                (0.5 + j * 0.6, 3 - i * 0.6), 0.55, 0.55,
                boxstyle="round,pad=0.02",
                facecolor=colors[(i + j) % len(colors)],
                edgecolor='white', linewidth=2
            )
            ax.add_patch(rect)
    
    ax.text(1.7, 0.5, '原始图像 (224×224)', ha='center', fontsize=12, color=APPLE_GRAY)
    
    # 箭头
    ax.annotate('', xy=(4.5, 2.5), xytext=(3.2, 2.5),
                arrowprops=dict(arrowstyle='->', color=APPLE_GRAY, lw=2))
    ax.text(3.85, 2.8, '分块', ha='center', fontsize=10, color=APPLE_GRAY)
    
    # 绘制序列
    for i in range(6):
        y_pos = 3.8 - i * 0.5
        rect = FancyBboxPatch(
            (5, y_pos), 0.8, 0.4,
            boxstyle="round,pad=0.02",
            facecolor=colors[i % len(colors)],
            edgecolor='white', linewidth=1
        )
        ax.add_patch(rect)
        
        if i == 3:
            ax.text(5.4, y_pos + 0.2, '...', ha='center', va='center', fontsize=12, color='white')
        else:
            label = str(i+1) if i < 3 else str(193+i)
            ax.text(5.4, y_pos + 0.2, label, ha='center', va='center', fontsize=9, color='white')
    
    ax.text(5.4, 0.5, '序列 (196 tokens)', ha='center', fontsize=12, color=APPLE_GRAY)
    
    # 投影说明
    ax.text(7.5, 2.5, '线性投影\n(256 → D)', ha='center', fontsize=10, color=APPLE_GRAY,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=APPLE_GRAY, alpha=0.8))
    
    ax.annotate('', xy=(8.5, 2.5), xytext=(7.5, 2.5),
                arrowprops=dict(arrowstyle='->', color=APPLE_GRAY, lw=2))
    
    # 最终嵌入
    for i in range(4):
        y_pos = 3.3 - i * 0.5
        rect = FancyBboxPatch(
            (9, y_pos), 0.6, 0.35,
            boxstyle="round,pad=0.02",
            facecolor=APPLE_TEAL,
            edgecolor='white', linewidth=1
        )
        ax.add_patch(rect)
    ax.text(9.3, 0.5, '嵌入向量', ha='center', fontsize=11, color=APPLE_GRAY)
    
    ax.set_title('图像分块嵌入 (Patch Embedding)', fontsize=16, fontweight='bold', pad=20)
    return fig

def plot_vit_architecture():
    """ViT 架构简图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    stages = [
        ("图像块\n嵌入", 1, APPLE_BLUE),
        ("[CLS] +\n位置编码", 3, APPLE_GREEN),
        ("Transformer\nEncoder ×L", 5.5, APPLE_PURPLE),
        ("MLP\nHead", 8, APPLE_RED),
        ("分类\n输出", 10, APPLE_ORANGE),
    ]
    
    for label, x, color in stages:
        circle = Circle((x, 2), 0.8, facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 2, label, ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # 箭头连接
    arrow_pairs = [(1.8, 2.2), (3.8, 4.7), (6.3, 7.2), (8.8, 9.2)]
    for x1, x2 in arrow_pairs:
        ax.annotate('', xy=(x2, 2), xytext=(x1, 2),
                    arrowprops=dict(arrowstyle='->', color=APPLE_GRAY, lw=2.5))
    
    ax.set_title('Vision Transformer 架构', fontsize=16, fontweight='bold', pad=20)
    return fig

def plot_data_scaling():
    """数据规模影响对比"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    dataset_sizes = np.array([1.3e6, 14e6, 300e6])
    size_labels = ['ImageNet-1k\n(1.3M)', 'ImageNet-21k\n(14M)', 'JFT-300M\n(300M)']
    x_pos = np.array([1, 2, 3])
    
    resnet_acc = [77.3, 81.5, 83.5]
    vit_acc = [75.0, 82.5, 88.5]
    
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, resnet_acc, width, label='ResNet-50', 
                   color=APPLE_RED, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, vit_acc, width, label='ViT-Base/16',
                   color=APPLE_GREEN, edgecolor='white', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('ImageNet Top-1 准确率 (%)', fontsize=12)
    ax.set_xlabel('预训练数据集', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(size_labels)
    ax.set_ylim(70, 92)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加说明文字
    ax.text(2, 78, 'Transformer 需要\n更多数据才能\n超越 CNN', 
            ha='center', fontsize=10, color=APPLE_GRAY,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=APPLE_GRAY, alpha=0.8))
    
    ax.set_title('数据规模对模型性能的影响', fontsize=16, fontweight='bold', pad=20)
    return fig

def plot_attention_visualization():
    """注意力可视化热力图"""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    grid_size = 7
    attention_weights = np.zeros((grid_size, grid_size))
    center = grid_size // 2
    
    for i in range(grid_size):
        for j in range(grid_size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            attention_weights[i, j] = np.exp(-dist / 2)
    
    attention_weights = attention_weights / attention_weights.max()
    
    # 绘制热力图
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='注意力权重')
    cbar.ax.yaxis.label.set_size(11)
    
    # 添加网格线
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1)
        ax.axvline(i - 0.5, color='white', linewidth=1)
    
    # 添加 CLS token 标注
    ax.text(-0.8, 0, '[CLS] →', ha='right', va='center', fontsize=11, color=APPLE_RED, fontweight='bold')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('ViT 自注意力可视化示例', fontsize=16, fontweight='bold', pad=20)
    
    return fig

def main():
    """主函数：生成所有图表"""
    output_dir = 'static/images/plots'
    
    print("开始生成 Vision Transformer 论文配图...")
    print("=" * 50)
    
    plots = [
        (plot_patch_embedding, 'vit-patch-embedding.png', '图像分块嵌入'),
        (plot_vit_architecture, 'vit-architecture.png', 'ViT 架构'),
        (plot_data_scaling, 'vit-data-scaling.png', '数据规模影响'),
        (plot_attention_visualization, 'vit-attention-visualization.png', '注意力可视化'),
    ]
    
    for i, (plot_func, filename, desc) in enumerate(plots, 1):
        print(f"\n生成 {i}/{len(plots)}: {desc}...")
        try:
            fig = plot_func()
            filepath = f'{output_dir}/{filename}'
            save_and_compress(fig, filepath)
        except Exception as e:
            print(f"❌ 生成失败: {e}")
    
    print("\n" + "=" * 50)
    print("✅ 图表生成完成！")
    
    # 显示文件大小
    print("\n文件大小统计:")
    for _, filename, desc in plots:
        fpath = f'{output_dir}/{filename}'
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  {desc}: {size/1024:.1f} KB")

if __name__ == '__main__':
    main()
