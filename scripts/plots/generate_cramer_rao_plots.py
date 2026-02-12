"""
生成Cramér-Rao下界相关的配图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from scipy.optimize import minimize
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


def plot_estimator_variance():
    """绘制估计量的方差比较图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：不同估计量的抽样分布
    ax = axes[0]
    
    # 真实参数值
    theta_true = 5
    
    # 生成不同估计量的分布
    x = np.linspace(3, 7, 500)
    
    # 有效估计量（方差小，无偏）
    y_efficient = stats.norm.pdf(x, theta_true, 0.3)
    ax.plot(x, y_efficient, color=APPLE_GREEN, linewidth=2.5, label='有效估计量')
    ax.fill_between(x, y_efficient, alpha=0.2, color=APPLE_GREEN)
    
    # 低效估计量（方差大）
    y_inefficient = stats.norm.pdf(x, theta_true, 0.8)
    ax.plot(x, y_inefficient, color=APPLE_ORANGE, linewidth=2.5, label='低效估计量')
    ax.fill_between(x, y_inefficient, alpha=0.2, color=APPLE_ORANGE)
    
    # 有偏估计量
    y_biased = stats.norm.pdf(x, theta_true + 0.5, 0.3)
    ax.plot(x, y_biased, color=APPLE_RED, linewidth=2.5, label='有偏估计量')
    ax.fill_between(x, y_biased, alpha=0.2, color=APPLE_RED)
    
    # 标记真实值
    ax.axvline(theta_true, color=APPLE_GRAY, linestyle='--', linewidth=2, alpha=0.7)
    ax.annotate('$\\theta_0$ (真值)', xy=(theta_true, 1.4), fontsize=11, color=APPLE_GRAY)
    
    ax.set_xlabel('$\\hat{\\theta}$', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('估计量的抽样分布', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 右图：样本量与方差关系
    ax = axes[1]
    
    n_samples = np.arange(10, 210, 10)
    
    # CRLB = 1/(n*I(theta))
    # 假设I(theta) = 1
    crlb = 1.0 / n_samples
    
    # 模拟的估计量方差（接近CRLB）
    np.random.seed(42)
    sample_var = crlb * (1 + np.random.normal(0, 0.05, len(n_samples)))
    sample_var = np.maximum(sample_var, crlb * 0.9)  # 不低于CRLB
    
    ax.plot(n_samples, crlb, color=APPLE_RED, linewidth=2.5, label='Cramér-Rao下界')
    ax.plot(n_samples, sample_var, 'o', color=APPLE_BLUE, markersize=6, label='估计量方差')
    
    ax.fill_between(n_samples, crlb, alpha=0.15, color=APPLE_RED)
    
    ax.set_xlabel('样本量 $n$', fontsize=12)
    ax.set_ylabel('方差', fontsize=12)
    ax.set_title('方差随样本量增加而减小', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/estimator-variance.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/estimator-variance.png')


def plot_fisher_information():
    """绘制Fisher信息的概念图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：对数似然函数的曲率
    ax = axes[0]
    
    theta = np.linspace(3, 7, 200)
    theta_mle = 5
    
    # 高曲率（大Fisher信息）
    loglik_high = -2 * (theta - theta_mle)**2
    ax.plot(theta, loglik_high, color=APPLE_GREEN, linewidth=2.5, label='高Fisher信息')
    
    # 低曲率（小Fisher信息）
    loglik_low = -0.5 * (theta - theta_mle)**2
    ax.plot(theta, loglik_low, color=APPLE_ORANGE, linewidth=2.5, label='低Fisher信息')
    
    # 标记MLE
    ax.plot(theta_mle, 0, 'D', color=APPLE_RED, markersize=10)
    ax.annotate('MLE', xy=(theta_mle, 0.3), fontsize=11, ha='center', color=APPLE_RED)
    
    # 曲率示意
    ax.annotate('', xy=(4.5, -0.5), xytext=(5.5, -0.5),
                arrowprops=dict(arrowstyle='<->', color=APPLE_GRAY, lw=1.5))
    ax.annotate('曲率大', xy=(5, -0.8), fontsize=10, ha='center', color=APPLE_GRAY)
    
    ax.annotate('', xy=(4.5, -2), xytext=(5.5, -2),
                arrowprops=dict(arrowstyle='<->', color=APPLE_GRAY, lw=1.5))
    ax.annotate('曲率小', xy=(5, -2.4), fontsize=10, ha='center', color=APPLE_GRAY)
    
    ax.set_xlabel('$\\theta$', fontsize=12)
    ax.set_ylabel('$\\ell(\\theta)$ (对数似然)', fontsize=12)
    ax.set_title('对数似然函数的曲率与Fisher信息', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # 右图：Fisher信息与方差的关系
    ax = axes[1]
    
    I_theta = np.linspace(0.5, 5, 100)
    var_crlb = 1 / I_theta
    
    ax.plot(I_theta, var_crlb, color=APPLE_BLUE, linewidth=2.5)
    ax.fill_between(I_theta, var_crlb, alpha=0.2, color=APPLE_BLUE)
    
    ax.set_xlabel('Fisher信息 $\\mathcal{I}(\\theta)$', fontsize=12)
    ax.set_ylabel('Cramér-Rao下界', fontsize=12)
    ax.set_title('Fisher信息越大，方差下界越小', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加标注
    ax.annotate('信息量↑\n估计精度↑', xy=(4, 0.4), fontsize=10, 
                color=APPLE_GREEN, ha='center',
                bbox=dict(boxstyle='round', facecolor=APPLE_GREEN, alpha=0.2))
    
    plt.tight_layout()
    plt.savefig('static/images/plots/fisher-information.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/fisher-information.png')


def plot_crlb_illustration():
    """绘制CRLB的几何解释"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 生成估计量的散点（模拟多次抽样）
    np.random.seed(42)
    n_estimators = 4
    
    # 不同效率的估计量
    estimators = [
        ('有效估计量', 0.3, 0, APPLE_GREEN),
        ('低效估计量', 0.6, 0, APPLE_ORANGE),
        ('有偏估计量1', 0.3, 0.5, APPLE_RED),
        ('有偏估计量2', 0.6, -0.4, APPLE_PURPLE)
    ]
    
    for i, (name, std, bias, color) in enumerate(estimators):
        x = np.random.normal(bias, std, 200) + 5  # 真实值在5
        y = np.random.normal(0, 0.1, 200) + i * 1.5
        ax.scatter(x, y, c=color, alpha=0.5, s=20)
        
        # 绘制置信椭圆
        ellipse = patches.Ellipse((5 + bias, i * 1.5), 2 * 1.96 * std, 0.5, 
                                   fill=True, facecolor=color, alpha=0.1,
                                   edgecolor=color, linewidth=2)
        ax.add_patch(ellipse)
        
        # 标签
        ax.annotate(name, xy=(5 + bias + 1.5, i * 1.5), fontsize=10, color=color)
    
    # 真实值线
    ax.axvline(5, color=APPLE_GRAY, linestyle='--', linewidth=2, alpha=0.7)
    ax.annotate('真值 $\\theta_0$', xy=(5, -0.8), fontsize=11, ha='center', color=APPLE_GRAY)
    
    # CRLB界限
    crlb_std = 0.3
    ax.axvline(5 + 1.96 * crlb_std, color=APPLE_RED, linestyle=':', linewidth=2, alpha=0.5)
    ax.axvline(5 - 1.96 * crlb_std, color=APPLE_RED, linestyle=':', linewidth=2, alpha=0.5)
    ax.annotate('CRLB界限', xy=(5 + 1.96 * crlb_std, 5.5), fontsize=10, color=APPLE_RED, rotation=90)
    
    ax.set_xlim(3, 7)
    ax.set_ylim(-1, 6)
    ax.set_xlabel('估计值', fontsize=12)
    ax.set_yticks([])
    ax.set_title('Cramér-Rao下界：估计量方差的理论极限', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/crlb-illustration.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/crlb-illustration.png')


def plot_normal_example():
    """绘制正态分布的例子"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：正态分布的Fisher信息
    ax = axes[0]
    
    sigma = np.linspace(0.5, 3, 100)
    I_mu = 1 / sigma**2  # 对于均值估计
    
    ax.plot(sigma, I_mu, color=APPLE_BLUE, linewidth=2.5, label='$\\mathcal{I}(\\mu) = 1/\\sigma^2$')
    ax.fill_between(sigma, I_mu, alpha=0.2, color=APPLE_BLUE)
    
    ax.set_xlabel('标准差 $\\sigma$', fontsize=12)
    ax.set_ylabel('Fisher信息', fontsize=12)
    ax.set_title('正态分布：均值估计的Fisher信息', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 添加标注
    ax.annotate('方差越小\n信息越多', xy=(0.8, 3), fontsize=10, 
                color=APPLE_GREEN, ha='center',
                bbox=dict(boxstyle='round', facecolor=APPLE_GREEN, alpha=0.2))
    
    # 右图：样本均值与样本方差的效率比较
    ax = axes[1]
    
    n_samples = np.arange(10, 210, 10)
    
    # 样本均值的方差 = sigma^2/n
    var_mean = 1.0 / n_samples
    
    # 样本方差的方差（对于正态分布）= 2*sigma^4/(n-1)
    var_variance = 2.0 / (n_samples - 1)
    
    ax.plot(n_samples, var_mean, color=APPLE_GREEN, linewidth=2.5, label='样本均值 (达到CRLB)')
    ax.plot(n_samples, var_variance, color=APPLE_ORANGE, linewidth=2.5, label='样本方差 (达到CRLB)')
    
    ax.fill_between(n_samples, var_mean, alpha=0.15, color=APPLE_GREEN)
    ax.fill_between(n_samples, var_variance, alpha=0.15, color=APPLE_ORANGE)
    
    ax.set_xlabel('样本量 $n$', fontsize=12)
    ax.set_ylabel('方差', fontsize=12)
    ax.set_title('正态分布：充分统计量的效率', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('static/images/plots/normal-example.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/normal-example.png')


def plot_efficiency_comparison():
    """绘制效率比较图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    estimators = ['样本均值', '样本中位数', '样本方差', 'MME*', 'MLE']
    efficiency = [1.0, 0.637, 1.0, 0.75, 1.0]  # 相对效率
    colors = [APPLE_GREEN if e == 1.0 else APPLE_ORANGE for e in efficiency]
    
    bars = ax.barh(estimators, efficiency, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bar, eff in zip(bars, efficiency):
        width = bar.get_width()
        ax.annotate(f'{eff:.3f}',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=11, va='center', fontweight='bold')
    
    # 添加参考线
    ax.axvline(1.0, color=APPLE_RED, linestyle='--', linewidth=2, alpha=0.7, label='CRLB (效率=1)')
    
    ax.set_xlabel('相对效率', fontsize=12)
    ax.set_title('不同估计量的相对效率（正态分布，位置参数）', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加说明
    ax.annotate('* MME: 矩估计方法', xy=(0.02, -0.12), xycoords='axes fraction', 
                fontsize=9, style='italic', color=APPLE_GRAY)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/efficiency-comparison.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/efficiency-comparison.png')


def plot_crlb_derivation():
    """绘制CRLB推导的关键步骤"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 图1：得分函数
    ax = axes[0]
    theta = np.linspace(3, 7, 200)
    theta0 = 5
    
    # 对数似然函数
    loglik = -0.5 * (theta - theta0)**2
    # 得分函数（导数）
    score = -(theta - theta0)
    
    ax.plot(theta, score, color=APPLE_BLUE, linewidth=2.5)
    ax.axhline(0, color=APPLE_GRAY, linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(theta0, color=APPLE_RED, linestyle='--', linewidth=2, alpha=0.7)
    
    ax.plot(theta0, 0, 'D', color=APPLE_RED, markersize=10)
    ax.annotate('E[S]=0', xy=(theta0, 0.5), fontsize=11, ha='center', color=APPLE_RED)
    
    ax.set_xlabel('$\\theta$', fontsize=12)
    ax.set_ylabel('$S(\\theta) = \\partial \\ell/\\partial \\theta$', fontsize=12)
    ax.set_title('步骤1：得分函数\n(期望为0)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 图2：方差关系
    ax = axes[1]
    
    x = np.linspace(-3, 3, 200)
    y_cov = np.exp(-x**2/2) / np.sqrt(2*np.pi)  # 协方差图示
    
    ax.plot(x, y_cov, color=APPLE_GREEN, linewidth=2.5)
    ax.fill_between(x, y_cov, alpha=0.3, color=APPLE_GREEN)
    
    # 方差标记
    ax.annotate('Var($\\hat{\\theta}$)', xy=(1.5, 0.15), fontsize=11, color=APPLE_BLUE)
    ax.annotate('Var(S) = $\\mathcal{I}(\\theta)$', xy=(-2, 0.35), fontsize=11, color=APPLE_ORANGE)
    
    ax.set_xlabel('偏离程度', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title('步骤2：Cauchy-Schwarz不等式\n方差关系', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 图3：CRLB结果
    ax = axes[2]
    
    n_range = np.arange(1, 21)
    crlb = 1.0 / n_range
    
    ax.plot(n_range, crlb, 'o-', color=APPLE_RED, linewidth=2.5, markersize=6, label='CRLB')
    ax.fill_between(n_range, crlb, alpha=0.2, color=APPLE_RED)
    
    # 渐近线
    ax.axhline(0, color=APPLE_GRAY, linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('样本量 $n$', fontsize=12)
    ax.set_ylabel('方差下界', fontsize=12)
    ax.set_title('步骤3：CRLB\n$\\mathrm{Var}(\\hat{\\theta}) \\geq \\frac{1}{n\\mathcal{I}(\\theta)}$', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('static/images/plots/crlb-derivation.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/crlb-derivation.png')


def plot_multivariate_crlb():
    """绘制多元CRLB的示意图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：Fisher信息矩阵
    ax = axes[0]
    
    # 绘制2x2 Fisher信息矩阵的热力图
    I = np.array([[2.0, 0.8], [0.8, 1.5]])
    
    im = ax.imshow(I, cmap='Blues', aspect='auto', vmin=0, vmax=2.5)
    
    # 添加数值
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{I[i, j]:.1f}', ha='center', va='center', 
                           color='white' if I[i, j] > 1.5 else 'black', fontsize=16, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['$\\theta_1$', '$\\theta_2$'])
    ax.set_yticklabels(['$\\theta_1$', '$\\theta_2$'])
    ax.set_title('Fisher信息矩阵 $\\mathcal{I}(\\theta)$', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('信息量', fontsize=10)
    
    # 右图：CRLB矩阵（逆矩阵）
    ax = axes[1]
    
    CRLB = np.linalg.inv(I)
    
    im2 = ax.imshow(CRLB, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{CRLB[i, j]:.3f}', ha='center', va='center', 
                           color='white' if CRLB[i, j] > 0.5 else 'black', fontsize=16, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['$\\theta_1$', '$\\theta_2$'])
    ax.set_yticklabels(['$\\theta_1$', '$\\theta_2$'])
    ax.set_title('CRLB矩阵 $\\mathcal{I}(\\theta)^{-1}$', fontsize=12, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax, shrink=0.8)
    cbar2.set_label('方差下界', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/multivariate-crlb.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/multivariate-crlb.png')


def plot_rao_blackwell():
    """绘制Rao-Blackwell定理的示意"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 生成原始估计量的分布
    np.random.seed(42)
    n_points = 500
    
    # 原始估计量（方差大）
    theta_original = np.random.normal(5, 1.2, n_points)
    y_original = np.random.normal(2, 0.3, n_points)
    ax.scatter(theta_original, y_original, c=APPLE_ORANGE, alpha=0.4, s=30, label='原始估计量')
    
    # Rao-Blackwell改进后的估计量（方差小）
    theta_improved = np.random.normal(5, 0.5, n_points)
    y_improved = np.random.normal(0, 0.3, n_points)
    ax.scatter(theta_improved, y_improved, c=APPLE_GREEN, alpha=0.4, s=30, label='Rao-Blackwell改进')
    
    # 添加置信椭圆
    from matplotlib.patches import Ellipse
    
    # 原始椭圆
    ellipse_orig = Ellipse((5, 2), 2*1.96*1.2, 0.8, fill=True, facecolor=APPLE_ORANGE, 
                           alpha=0.1, edgecolor=APPLE_ORANGE, linewidth=2)
    ax.add_patch(ellipse_orig)
    
    # 改进后椭圆
    ellipse_improved = Ellipse((5, 0), 2*1.96*0.5, 0.8, fill=True, facecolor=APPLE_GREEN, 
                               alpha=0.1, edgecolor=APPLE_GREEN, linewidth=2)
    ax.add_patch(ellipse_improved)
    
    # 真实值
    ax.axvline(5, color=APPLE_RED, linestyle='--', linewidth=2, alpha=0.7)
    ax.plot(5, 2, 'D', color=APPLE_RED, markersize=10)
    ax.plot(5, 0, 'D', color=APPLE_RED, markersize=10)
    
    ax.annotate('真值 $\\theta_0$', xy=(5, 3.2), fontsize=11, ha='center', color=APPLE_RED)
    ax.annotate('方差↓', xy=(7, 1), fontsize=12, color=APPLE_GREEN, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=APPLE_GREEN, lw=2))
    
    ax.set_xlim(1, 9)
    ax.set_ylim(-1, 4)
    ax.set_xlabel('估计值', fontsize=12)
    ax.set_yticks([])
    ax.set_title('Rao-Blackwell定理：充分统计量降低方差', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/rao-blackwell.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/rao-blackwell.png')


def main():
    """生成所有配图"""
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成Cramér-Rao下界配图...")
    
    plot_estimator_variance()
    print("✅ 估计量方差图")
    
    plot_fisher_information()
    print("✅ Fisher信息图")
    
    plot_crlb_illustration()
    print("✅ CRLB概念图")
    
    plot_normal_example()
    print("✅ 正态分布例子")
    
    plot_efficiency_comparison()
    print("✅ 效率比较图")
    
    plot_crlb_derivation()
    print("✅ CRLB推导图")
    
    plot_multivariate_crlb()
    print("✅ 多元CRLB图")
    
    plot_rao_blackwell()
    print("✅ Rao-Blackwell图")
    
    print("\n✅ 所有配图生成完成！")


if __name__ == '__main__':
    main()
