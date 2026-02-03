"""
生成最大熵原理相关的配图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Arc
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func
from scipy.special import gammaln
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


def plot_entropy_comparison():
    """绘制不同分布的熵比较"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：不同分布的概率密度
    ax = axes[0]
    
    x = np.linspace(-4, 4, 500)
    mu, sigma = 0, 1
    
    # 高斯分布
    gaussian = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, gaussian, color=APPLE_BLUE, linewidth=2.5, label='高斯分布 (熵=1.42)')
    ax.fill_between(x, gaussian, alpha=0.2, color=APPLE_BLUE)
    
    # 拉普拉斯分布（双指数）
    laplace = stats.laplace.pdf(x, mu, sigma/np.sqrt(2))
    ax.plot(x, laplace, color=APPLE_ORANGE, linewidth=2.5, label='拉普拉斯 (熵=1.35)')
    ax.fill_between(x, laplace, alpha=0.2, color=APPLE_ORANGE)
    
    # 均匀分布（截断）
    uniform_range = np.sqrt(12) * sigma / 2  # 使方差为1
    uniform = np.where(np.abs(x) <= uniform_range, 1/(2*uniform_range), 0)
    ax.plot(x, uniform, color=APPLE_GREEN, linewidth=2.5, label='均匀分布 (熵=1.39)')
    ax.fill_between(x, uniform, alpha=0.2, color=APPLE_GREEN)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('相同均值和方差下的不同分布', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.45)
    
    # 右图：熵的比较
    ax = axes[1]
    
    distributions = ['均匀', '拉普拉斯', '高斯', '指数']
    entropies = [1.39, 1.35, 1.42, 1.00]  # 给定均值0，方差1的条件下
    colors = [APPLE_GREEN, APPLE_ORANGE, APPLE_BLUE, APPLE_RED]
    
    bars = ax.bar(distributions, entropies, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
    
    # 标记最大值
    max_idx = np.argmax(entropies)
    bars[max_idx].set_edgecolor(APPLE_BLUE)
    bars[max_idx].set_linewidth(3)
    
    ax.axhline(np.max(entropies), color=APPLE_BLUE, linestyle='--', linewidth=2, alpha=0.7)
    ax.annotate('最大熵', xy=(max_idx, np.max(entropies)), xytext=(max_idx+0.3, np.max(entropies)+0.05),
               fontsize=10, color=APPLE_BLUE)
    
    ax.set_ylabel('熵 (nats)', fontsize=12)
    ax.set_title('各分布的熵值比较', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.8, 1.6)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/entropy-comparison.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/entropy-comparison.png')


def plot_max_entropy_proof():
    """绘制最大熵证明的关键步骤"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 左图：约束条件
    ax = axes[0]
    
    # 绘制可行域
    x = np.linspace(-3, 3, 100)
    
    # 约束1：归一化（自动满足）
    # 约束2：均值 = 0
    # 约束3：方差 = 1
    
    # 绘制高斯分布作为最优解
    gaussian = stats.norm.pdf(x, 0, 1)
    ax.plot(x, gaussian, color=APPLE_BLUE, linewidth=3, label='高斯分布 (最优解)')
    ax.fill_between(x, gaussian, alpha=0.3, color=APPLE_BLUE)
    
    # 绘制其他满足约束的分布
    # 截断均匀
    uniform_trunc = np.where(np.abs(x) <= np.sqrt(3), 1/(2*np.sqrt(3)), 0)
    ax.plot(x, uniform_trunc, '--', color=APPLE_ORANGE, linewidth=2, label='其他可行解')
    
    # 标记约束
    ax.axvline(0, color=APPLE_GREEN, linestyle=':', linewidth=2, alpha=0.7)
    ax.annotate('均值约束', xy=(0.1, 0.35), fontsize=9, color=APPLE_GREEN)
    
    # 方差约束的可视化（置信区间）
    ax.axvline(-1, color=APPLE_RED, linestyle=':', linewidth=2, alpha=0.5)
    ax.axvline(1, color=APPLE_RED, linestyle=':', linewidth=2, alpha=0.5)
    ax.annotate('方差约束', xy=(1.1, 0.3), fontsize=9, color=APPLE_RED)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('步骤1：约束可行域', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 中图：拉格朗日函数
    ax = axes[1]
    
    # 绘制熵函数和约束
    lambda_range = np.linspace(-2, 2, 100)
    
    # 简化的拉格朗日函数示意
    # L = -sum(p log p) + lambda0(sum(p)-1) + lambda1(sum(px)) + lambda2(sum(px^2)-sigma^2)
    
    lagrangian = -lambda_range**2 + 0.5 * lambda_range + 2  # 示意曲线
    
    ax.plot(lambda_range, lagrangian, color=APPLE_PURPLE, linewidth=2.5)
    ax.fill_between(lambda_range, lagrangian, alpha=0.2, color=APPLE_PURPLE)
    
    # 标记最优解
    opt_idx = np.argmax(lagrangian)
    ax.plot(lambda_range[opt_idx], lagrangian[opt_idx], 'D', color=APPLE_RED, markersize=12)
    ax.annotate('最优拉格朗日乘子', xy=(lambda_range[opt_idx], lagrangian[opt_idx]),
               xytext=(lambda_range[opt_idx]+0.3, lagrangian[opt_idx]-0.5),
               fontsize=10, color=APPLE_RED,
               arrowprops=dict(arrowstyle='->', color=APPLE_RED))
    
    ax.set_xlabel('拉格朗日乘子 $\\lambda$', fontsize=12)
    ax.set_ylabel('拉格朗日函数 $\\mathcal{L}$', fontsize=12)
    ax.set_title('步骤2：变分优化', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 右图：结果
    ax = axes[2]
    
    x = np.linspace(-4, 4, 200)
    gaussian = stats.norm.pdf(x, 0, 1)
    
    ax.plot(x, gaussian, color=APPLE_BLUE, linewidth=3, label='$p(x) \\propto e^{-x^2/2}$')
    ax.fill_between(x, gaussian, alpha=0.3, color=APPLE_BLUE)
    
    # 标记关键性质
    ax.axhline(1/np.sqrt(2*np.pi), color=APPLE_GREEN, linestyle='--', linewidth=2, alpha=0.7)
    ax.annotate('最大值 $\\frac{1}{\\sqrt{2\\pi}\\sigma}$', xy=(0, 1/np.sqrt(2*np.pi)+0.02),
               fontsize=10, color=APPLE_GREEN, ha='center')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('步骤3：高斯分布', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/max-entropy-proof.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/max-entropy-proof.png')


def plot_jaynes_principle():
    """绘制Jaynes最大熵原理的示意图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：从约束到分布
    ax = axes[0]
    
    # 绘制流程图
    # 约束 -> 最大熵 -> 分布
    
    # 约束框
    constraint_box = FancyBboxPatch((0.1, 0.7), 0.2, 0.15, 
                                     boxstyle="round,pad=0.02",
                                     facecolor=APPLE_BLUE, alpha=0.3,
                                     edgecolor=APPLE_BLUE, linewidth=2)
    ax.add_patch(constraint_box)
    ax.text(0.2, 0.775, '约束\n(均值、方差)', ha='center', va='center', fontsize=10)
    
    # 箭头1
    ax.annotate('', xy=(0.45, 0.775), xytext=(0.32, 0.775),
               arrowprops=dict(arrowstyle='->', color=APPLE_GRAY, lw=2))
    ax.text(0.385, 0.82, 'MaxEnt', fontsize=9, color=APPLE_GRAY)
    
    # 最大熵框
    maxent_box = FancyBboxPatch((0.45, 0.7), 0.2, 0.15,
                               boxstyle="round,pad=0.02",
                               facecolor=APPLE_GREEN, alpha=0.3,
                               edgecolor=APPLE_GREEN, linewidth=2)
    ax.add_patch(maxent_box)
    ax.text(0.55, 0.775, '最大熵\n原理', ha='center', va='center', fontsize=10)
    
    # 箭头2
    ax.annotate('', xy=(0.8, 0.775), xytext=(0.67, 0.775),
               arrowprops=dict(arrowstyle='->', color=APPLE_GRAY, lw=2))
    
    # 分布框
    dist_box = FancyBboxPatch((0.8, 0.7), 0.15, 0.15,
                             boxstyle="round,pad=0.02",
                             facecolor=APPLE_ORANGE, alpha=0.3,
                             edgecolor=APPLE_ORANGE, linewidth=2)
    ax.add_patch(dist_box)
    ax.text(0.875, 0.775, '高斯\n分布', ha='center', va='center', fontsize=10)
    
    # 下方示例图
    x = np.linspace(-3, 3, 100)
    y_pos = 0.35
    
    # 绘制几个不同约束下的最优分布
    # 只有均值约束 -> 指数分布（在正半轴）
    # 均值和方差约束 -> 高斯分布
    # 有限支撑约束 -> 均匀分布
    
    gaussian = stats.norm.pdf(x, 0, 1) * 0.3
    ax.plot(x, gaussian + y_pos, color=APPLE_BLUE, linewidth=2)
    ax.fill_between(x, gaussian + y_pos, y_pos, alpha=0.3, color=APPLE_BLUE)
    ax.text(0, y_pos + 0.35, '约束: 均值, 方差', fontsize=9, ha='center')
    ax.text(0, y_pos - 0.05, '→ 高斯分布', fontsize=9, ha='center', color=APPLE_BLUE)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Jaynes最大熵原理', fontsize=12, fontweight='bold')
    
    # 右图：不同约束条件下的最大熵分布
    ax = axes[1]
    
    x = np.linspace(-4, 4, 200)
    
    # 约束1：只有归一化和均值
    # 这在连续情况下不是良定义的
    
    # 约束2：均值和方差 -> 高斯
    gaussian = stats.norm.pdf(x, 0, 1)
    ax.plot(x, gaussian, color=APPLE_BLUE, linewidth=2.5, label='均值+方差约束 → 高斯')
    
    # 约束3：有限支撑 [a, b] -> 均匀
    a, b = -2, 2
    uniform = np.where((x >= a) & (x <= b), 1/(b-a), 0)
    ax.plot(x, uniform, color=APPLE_GREEN, linewidth=2.5, label='有限支撑约束 → 均匀')
    
    # 约束4：均值（正半轴）-> 指数
    exponential = np.exp(-np.abs(x)) * (x >= 0).astype(float)
    ax.plot(x, exponential, color=APPLE_ORANGE, linewidth=2.5, label='正半轴+均值 → 指数')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('不同约束下的最大熵分布', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/jaynes-principle.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/jaynes-principle.png')


def plot_gaussian_natural():
    """绘制高斯分布在自然界中的普遍性"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：中心极限定理示意
    ax = axes[0]
    
    np.random.seed(42)
    n_samples = 10000
    
    # 单个均匀分布
    x = np.linspace(-4, 4, 200)
    
    # 1个均匀分布
    uniform_1 = np.random.uniform(-1, 1, n_samples)
    ax.hist(uniform_1, bins=50, density=True, alpha=0.3, color=APPLE_GRAY, 
           label='1个均匀分布', range=(-4, 4))
    
    # 2个均匀分布的和
    uniform_2 = np.random.uniform(-1, 1, n_samples) + np.random.uniform(-1, 1, n_samples)
    ax.hist(uniform_2, bins=50, density=True, alpha=0.4, color=APPLE_ORANGE,
           label='2个均匀之和', range=(-4, 4))
    
    # 10个均匀分布的和
    uniform_10 = sum([np.random.uniform(-1, 1, n_samples) for _ in range(10)])
    uniform_10 = (uniform_10 - np.mean(uniform_10)) / np.std(uniform_10)
    ax.hist(uniform_10, bins=50, density=True, alpha=0.5, color=APPLE_BLUE,
           label='10个均匀之和', range=(-4, 4))
    
    # 标准正态参考
    gaussian = stats.norm.pdf(x, 0, 1)
    ax.plot(x, gaussian, color=APPLE_RED, linewidth=2.5, label='标准正态', linestyle='--')
    
    ax.set_xlabel('标准化值', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('中心极限定理：高斯分布的出现', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 右图：物理系统中的高斯噪声
    ax = axes[1]
    
    x = np.linspace(-4, 4, 200)
    
    # 布朗运动
    brownian = stats.norm.pdf(x, 0, 1)
    ax.plot(x, brownian, color=APPLE_BLUE, linewidth=2.5, label='布朗运动 (热噪声)')
    
    # 电阻热噪声（Johnson-Nyquist噪声）
    johnson = stats.norm.pdf(x, 0, 1)
    ax.plot(x, johnson + 0.02, color=APPLE_GREEN, linewidth=2.5, label='Johnson-Nyquist噪声')
    
    # 测量误差
    measurement = stats.norm.pdf(x, 0, 1)
    ax.plot(x, measurement + 0.04, color=APPLE_ORANGE, linewidth=2.5, label='测量误差')
    
    ax.fill_between(x, brownian, alpha=0.2, color=APPLE_BLUE)
    
    # 标记最大熵原理的解释
    ax.annotate('最大熵原理解释：\n在已知方差条件下，\n高斯分布是最"自然"的假设',
               xy=(0.95, 0.95), xycoords='axes fraction',
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlabel('噪声幅度', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('自然界中的高斯噪声', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/gaussian-natural.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/gaussian-natural.png')


def plot_physics_applications():
    """绘制统计物理中的应用"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：玻尔兹曼分布
    ax = axes[1]
    
    # 能级
    E = np.linspace(0, 5, 100)
    
    # 不同温度下的玻尔兹曼分布
    T_values = [0.5, 1.0, 2.0]
    colors_temp = [APPLE_BLUE, APPLE_GREEN, APPLE_ORANGE]
    
    for T, color in zip(T_values, colors_temp):
        boltzmann = np.exp(-E / T)
        boltzmann = boltzmann / np.sum(boltzmann)  # 归一化
        ax.plot(E, boltzmann, color=color, linewidth=2.5, label=f'T = {T}')
        ax.fill_between(E, boltzmann, alpha=0.2, color=color)
    
    ax.set_xlabel('能量 E', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('玻尔兹曼分布 (最大熵分布)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 右图：熵与能量约束
    ax = axes[0]
    
    # 微观状态数与熵
    E_range = np.linspace(0.1, 5, 100)
    
    # 熵随能量的变化（示意）
    entropy = np.log(E_range) + 2
    
    ax.plot(E_range, entropy, color=APPLE_BLUE, linewidth=2.5, label='熵 $S(E)$')
    ax.fill_between(E_range, entropy, alpha=0.2, color=APPLE_BLUE)
    
    # 温度 = dS/dE
    temperature = 1 / (E_range)  # 简化的关系
    ax2 = ax.twinx()
    ax2.plot(E_range, temperature, color=APPLE_RED, linewidth=2.5, 
            linestyle='--', label='温度 $T = dS/dE$')
    
    ax.set_xlabel('平均能量 $\\langle E \\rangle$', fontsize=12)
    ax.set_ylabel('熵 $S$', fontsize=12, color=APPLE_BLUE)
    ax2.set_ylabel('温度 $T$', fontsize=12, color=APPLE_RED)
    ax.set_title('统计物理中的最大熵', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('static/images/plots/physics-applications.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/physics-applications.png')


def plot_ml_applications():
    """绘制机器学习中的应用"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：Dropout作为高斯近似
    ax = axes[0]
    
    np.random.seed(42)
    n_neurons = 1000
    dropout_rate = 0.5
    
    # 模拟Dropout的输出分布
    weights = np.random.randn(n_neurons)
    mask = np.random.binomial(1, 1-dropout_rate, n_neurons)
    output = np.sum(weights * mask) / (1-dropout_rate)
    
    # 多次采样
    outputs = []
    for _ in range(1000):
        mask = np.random.binomial(1, 1-dropout_rate, n_neurons)
        out = np.sum(weights * mask) / (1-dropout_rate)
        outputs.append(out)
    
    ax.hist(outputs, bins=50, density=True, alpha=0.5, color=APPLE_BLUE, label='Dropout输出')
    
    # 拟合高斯
    mu_fit, sigma_fit = np.mean(outputs), np.std(outputs)
    x = np.linspace(min(outputs), max(outputs), 200)
    gaussian_fit = stats.norm.pdf(x, mu_fit, sigma_fit)
    ax.plot(x, gaussian_fit, color=APPLE_RED, linewidth=2.5, 
           label='高斯近似 (最大熵)', linestyle='--')
    
    ax.set_xlabel('网络输出', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('Dropout的高斯近似', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 右图：变分推断中的最大熵
    ax = axes[1]
    
    # 绘制后验近似的过程
    iterations = np.arange(0, 101)
    
    # ELBO随迭代的变化
    elbo = -5 * np.exp(-iterations / 20) - 2 + np.random.normal(0, 0.05, len(iterations))
    
    # 熵随迭代的变化
    entropy = 1.5 - 0.5 * np.exp(-iterations / 20)
    
    ax.plot(iterations, elbo, color=APPLE_BLUE, linewidth=2.5, label='ELBO')
    ax.fill_between(iterations, elbo, alpha=0.2, color=APPLE_BLUE)
    
    ax2 = ax.twinx()
    ax2.plot(iterations, entropy, color=APPLE_GREEN, linewidth=2.5, 
            label='熵 $H(q)$', linestyle='--')
    ax2.fill_between(iterations, entropy, alpha=0.2, color=APPLE_GREEN)
    
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('ELBO', fontsize=12, color=APPLE_BLUE)
    ax2.set_ylabel('熵', fontsize=12, color=APPLE_GREEN)
    ax.set_title('变分推断：最大化ELBO ≈ 最大化熵', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('static/images/plots/ml-maximum-entropy.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/ml-maximum-entropy.png')


def main():
    """生成所有配图"""
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成最大熵原理配图...")
    
    plot_entropy_comparison()
    print("✅ 熵的比较")
    
    plot_max_entropy_proof()
    print("✅ 最大熵证明")
    
    plot_jaynes_principle()
    print("✅ Jaynes原理")
    
    plot_gaussian_natural()
    print("✅ 高斯分布的自然选择")
    
    plot_physics_applications()
    print("✅ 统计物理应用")
    
    plot_ml_applications()
    print("✅ 机器学习应用")
    
    print("\n✅ 所有配图生成完成！")


if __name__ == '__main__':
    main()
