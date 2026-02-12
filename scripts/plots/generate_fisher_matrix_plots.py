"""
生成Fisher信息矩阵相关的配图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, FancyArrowPatch
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


def plot_fisher_vs_variance():
    """绘制Fisher信息与方差的关系"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：单参数Fisher信息与方差
    ax = axes[0]
    
    I_range = np.linspace(0.5, 5, 100)
    variance_bound = 1 / I_range
    
    ax.plot(I_range, variance_bound, color=APPLE_BLUE, linewidth=2.5, 
            label='CRLB = $1/\\mathcal{I}(\\theta)$')
    ax.fill_between(I_range, variance_bound, alpha=0.2, color=APPLE_BLUE)
    
    # 标记几个点
    for I_val in [1, 2, 4]:
        var_val = 1 / I_val
        ax.plot(I_val, var_val, 'o', color=APPLE_RED, markersize=10)
        ax.annotate(f'I={I_val}\nVar≤{var_val:.2f}', 
                   xy=(I_val, var_val), xytext=(10, 10), 
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Fisher信息 $\\mathcal{I}(\\theta)$', fontsize=12)
    ax.set_ylabel('方差下界', fontsize=12)
    ax.set_title('信息越大方差越小', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 右图：样本量对Fisher信息的影响
    ax = axes[1]
    
    n_samples = np.arange(1, 51)
    I_single = 1.0  # 单样本Fisher信息
    I_total = n_samples * I_single
    
    ax.plot(n_samples, I_total, color=APPLE_GREEN, linewidth=2.5, 
            label='$\\mathcal{I}_n(\\theta) = n \\cdot \\mathcal{I}_1(\\theta)$')
    ax.fill_between(n_samples, I_total, alpha=0.2, color=APPLE_GREEN)
    
    ax.set_xlabel('样本量 $n$', fontsize=12)
    ax.set_ylabel('Fisher信息', fontsize=12)
    ax.set_title('信息随样本线性增长', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/fisher-vs-variance.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/fisher-vs-variance.png')


def plot_loglikelihood_curvature():
    """绘制对数似然函数的曲率与Fisher信息"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：不同曲率的对数似然函数
    ax = axes[0]
    
    theta = np.linspace(2, 8, 200)
    theta_mle = 5
    
    # 高曲率（尖峰）
    loglik_sharp = -2 * (theta - theta_mle)**2
    ax.plot(theta, loglik_sharp, color=APPLE_GREEN, linewidth=2.5, 
            label='高曲率 (I=4)')
    
    # 中等曲率
    loglik_med = -0.5 * (theta - theta_mle)**2
    ax.plot(theta, loglik_med, color=APPLE_BLUE, linewidth=2.5, 
            label='中等曲率 (I=1)')
    
    # 低曲率（平坦）
    loglik_flat = -0.2 * (theta - theta_mle)**2
    ax.plot(theta, loglik_flat, color=APPLE_ORANGE, linewidth=2.5, 
            label='低曲率 (I=0.4)')
    
    # 标记MLE
    ax.plot(theta_mle, 0, 'D', color=APPLE_RED, markersize=12, zorder=5)
    ax.annotate('MLE', xy=(theta_mle, 0.5), fontsize=11, ha='center', color=APPLE_RED)
    
    ax.set_xlabel('$\\theta$', fontsize=12)
    ax.set_ylabel('$\\ell(\\theta)$', fontsize=12)
    ax.set_title('对数似然函数的曲率', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # 右图：曲率与置信区间
    ax = axes[1]
    
    # 绘制三个不同曲率的高斯近似
    x = np.linspace(3, 7, 200)
    
    # 高曲率 -> 窄置信区间
    y1 = stats.norm.pdf(x, 5, 0.5)
    ax.plot(x, y1, color=APPLE_GREEN, linewidth=2.5, label='高曲率: CI窄')
    ax.fill_between(x, y1, alpha=0.2, color=APPLE_GREEN)
    
    # 低曲率 -> 宽置信区间
    y2 = stats.norm.pdf(x, 5, 1.2)
    ax.plot(x, y2, color=APPLE_ORANGE, linewidth=2.5, label='低曲率: CI宽')
    ax.fill_between(x, y2, alpha=0.2, color=APPLE_ORANGE)
    
    # 标记95%置信区间
    ax.axvline(5 - 1.96*0.5, color=APPLE_GREEN, linestyle='--', alpha=0.5)
    ax.axvline(5 + 1.96*0.5, color=APPLE_GREEN, linestyle='--', alpha=0.5)
    ax.axvline(5 - 1.96*1.2, color=APPLE_ORANGE, linestyle='--', alpha=0.5)
    ax.axvline(5 + 1.96*1.2, color=APPLE_ORANGE, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('参数值', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('曲率决定置信区间宽度', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/loglikelihood-curvature.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/loglikelihood-curvature.png')


def plot_fisher_matrix_geometry():
    """绘制Fisher信息矩阵的几何解释"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：参数空间的度量
    ax = axes[0]
    
    # 创建网格
    theta1 = np.linspace(-2, 2, 20)
    theta2 = np.linspace(-2, 2, 20)
    T1, T2 = np.meshgrid(theta1, theta2)
    
    # Fisher信息矩阵 (单位矩阵代表欧氏度量)
    I = np.array([[2, 0.5], [0.5, 1]])
    
    # 计算每个点的距离（马氏距离）
    Z = np.zeros_like(T1)
    for i in range(len(theta1)):
        for j in range(len(theta2)):
            v = np.array([T1[i,j], T2[i,j]])
            Z[i,j] = np.sqrt(v.T @ I @ v)
    
    # 绘制等高线
    contour = ax.contour(T1, T2, Z, levels=5, colors=APPLE_BLUE, linewidths=2)
    ax.clabel(contour, inline=True, fontsize=9)
    
    # 填充等高线
    ax.contourf(T1, T2, Z, levels=5, cmap='Blues', alpha=0.3)
    
    # 绘制向量场表示度量
    # 计算局部尺度
    scale = 0.3
    for i in range(0, len(theta1), 3):
        for j in range(0, len(theta2), 3):
            # 局部椭圆的半轴
            eigvals, eigvecs = np.linalg.eig(I)
            for k in range(2):
                vec = eigvecs[:, k] * scale / np.sqrt(eigvals[k])
                ax.annotate('', xy=(T1[i,j] + vec[0], T2[i,j] + vec[1]),
                           xytext=(T1[i,j], T2[i,j]),
                           arrowprops=dict(arrowstyle='->', color=APPLE_ORANGE, lw=1, alpha=0.5))
    
    ax.set_xlabel('$\\theta_1$', fontsize=12)
    ax.set_ylabel('$\\theta_2$', fontsize=12)
    ax.set_title('Fisher度量下的参数空间', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 右图：置信椭圆
    ax = axes[1]
    
    # 绘制不同Fisher信息矩阵的置信椭圆
    matrices = [
        (np.array([[2, 0], [0, 2]]), '各向同性', APPLE_GREEN),
        (np.array([[3, 0], [0, 1]]), '各向异性', APPLE_BLUE),
        (np.array([[2, 1.5], [1.5, 2]]), '相关参数', APPLE_PURPLE)
    ]
    
    for I, label, color in matrices:
        # CRLB矩阵
        CRLB = np.linalg.inv(I)
        
        # 计算椭圆的参数
        eigvals, eigvecs = np.linalg.eig(CRLB)
        
        # 95%置信椭圆的半轴 (卡方分布临界值)
        chi2_val = 5.991  # chi2(0.95, 2)
        a = np.sqrt(eigvals[0] * chi2_val)
        b = np.sqrt(eigvals[1] * chi2_val)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        
        ellipse = Ellipse((0, 0), 2*a, 2*b, angle=angle, 
                         fill=True, facecolor=color, alpha=0.2,
                         edgecolor=color, linewidth=2, label=label)
        ax.add_patch(ellipse)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('$\\theta_1$', fontsize=12)
    ax.set_ylabel('$\\theta_2$', fontsize=12)
    ax.set_title('不同Fisher矩阵的置信椭圆', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/fisher-matrix-geometry.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/fisher-matrix-geometry.png')


def plot_linear_regression_fisher():
    """绘制线性回归中的Fisher信息矩阵"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：实验设计的影响
    ax = axes[0]
    
    np.random.seed(42)
    n = 20
    
    # 设计1：x集中在均值附近（差的设计）
    x1 = np.random.normal(0, 0.5, n)
    y1 = 2 + 3*x1 + np.random.normal(0, 1, n)
    ax.scatter(x1, y1, c=APPLE_ORANGE, alpha=0.6, s=50, label='差的设计 (x集中)')
    
    # 设计2：x分散（好的设计）
    x2 = np.linspace(-3, 3, n)
    y2 = 2 + 3*x2 + np.random.normal(0, 1, n)
    ax.scatter(x2, y2, c=APPLE_GREEN, alpha=0.6, s=50, label='好的设计 (x分散)')
    
    # 拟合线
    z1 = np.polyfit(x1, y1, 1)
    z2 = np.polyfit(x2, y2, 1)
    x_line = np.linspace(-3, 3, 100)
    ax.plot(x_line, z1[0]*x_line + z1[1], '--', color=APPLE_ORANGE, linewidth=2)
    ax.plot(x_line, z2[0]*x_line + z2[1], '--', color=APPLE_GREEN, linewidth=2)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('实验设计影响Fisher信息', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 右图：Fisher信息矩阵的数值
    ax = axes[1]
    ax.axis('off')
    
    # 计算Fisher信息矩阵
    # 对于线性回归 y = beta0 + beta1*x + epsilon, epsilon ~ N(0, sigma^2)
    # Fisher信息矩阵 = (1/sigma^2) * X^T X
    
    sigma2 = 1
    X1 = np.column_stack([np.ones(n), x1])
    I1 = (X1.T @ X1) / sigma2
    
    X2 = np.column_stack([np.ones(n), x2])
    I2 = (X2.T @ X2) / sigma2
    
    # 显示矩阵
    ax.text(0.5, 0.8, '设计1的Fisher矩阵:', fontsize=12, ha='center', fontweight='bold', 
           transform=ax.transAxes)
    matrix_text1 = f'[[{I1[0,0]:.1f}, {I1[0,1]:.1f}],\n [{I1[1,0]:.1f}, {I1[1,1]:.1f}]]'
    ax.text(0.5, 0.65, matrix_text1, fontsize=11, ha='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor=APPLE_ORANGE, alpha=0.3),
           transform=ax.transAxes)
    
    ax.text(0.5, 0.45, '设计2的Fisher矩阵:', fontsize=12, ha='center', fontweight='bold',
           transform=ax.transAxes)
    matrix_text2 = f'[[{I2[0,0]:.1f}, {I2[1,0]:.1f}],\n [{I2[0,1]:.1f}, {I2[1,1]:.1f}]]'
    ax.text(0.5, 0.3, matrix_text2, fontsize=11, ha='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor=APPLE_GREEN, alpha=0.3),
           transform=ax.transAxes)
    
    # 方差比较
    var_beta1_design1 = np.linalg.inv(I1)[1, 1]
    var_beta1_design2 = np.linalg.inv(I2)[1, 1]
    
    ax.text(0.5, 0.1, f'Var(β₁)对比: {var_beta1_design1:.3f} vs {var_beta1_design2:.3f}', 
           fontsize=11, ha='center', style='italic',
           transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/linear-regression-fisher.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/linear-regression-fisher.png')


def plot_information_geometry():
    """绘制信息几何的概念图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：统计流形
    ax = axes[0]
    
    # 绘制参数化的分布族（如正态分布族）
    mu_range = np.linspace(-2, 2, 50)
    sigma_range = np.linspace(0.5, 2, 50)
    MU, SIGMA = np.meshgrid(mu_range, sigma_range)
    
    # 绘制等高线表示某个统计量（如熵）
    entropy = 0.5 * np.log(2 * np.pi * np.e * SIGMA**2)
    
    contour = ax.contour(MU, SIGMA, entropy, levels=8, colors=APPLE_BLUE, linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.contourf(MU, SIGMA, entropy, levels=8, cmap='Blues', alpha=0.3)
    
    # 绘制一条测地线（参数更新路径）
    mu_path = np.linspace(-1.5, 1.5, 50)
    sigma_path = 0.8 + 0.3 * mu_path**2
    ax.plot(mu_path, sigma_path, 'o-', color=APPLE_RED, markersize=3, linewidth=2, 
           label='自然梯度路径')
    
    # 起点和终点
    ax.plot(mu_path[0], sigma_path[0], 's', color=APPLE_GREEN, markersize=10, label='起点')
    ax.plot(mu_path[-1], sigma_path[-1], '^', color=APPLE_PURPLE, markersize=10, label='终点')
    
    ax.set_xlabel('$\\mu$ (均值)', fontsize=12)
    ax.set_ylabel('$\\sigma$ (标准差)', fontsize=12)
    ax.set_title('正态分布族的统计流形', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 右图：Fisher度量的局部几何
    ax = axes[1]
    
    # 绘制局部切空间
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 单位圆（欧氏度量）
    ax.plot(np.cos(theta), np.sin(theta), '--', color=APPLE_GRAY, linewidth=2, 
           label='欧氏单位圆', alpha=0.7)
    
    # Fisher度量下的单位圆（椭圆）
    I_local = np.array([[4, 1], [1, 2]])
    I_inv = np.linalg.inv(I_local)
    
    # 参数化椭圆
    t = np.linspace(0, 2*np.pi, 100)
    # v^T I v = 1 定义的单位球
    eigvals, eigvecs = np.linalg.eig(I_local)
    ellipse_x = eigvecs[0,0] * np.cos(t) / np.sqrt(eigvals[0]) + eigvecs[0,1] * np.sin(t) / np.sqrt(eigvals[1])
    ellipse_y = eigvecs[1,0] * np.cos(t) / np.sqrt(eigvals[0]) + eigvecs[1,1] * np.sin(t) / np.sqrt(eigvals[1])
    
    ax.plot(ellipse_x, ellipse_y, color=APPLE_BLUE, linewidth=2.5, 
           label='Fisher单位圆')
    ax.fill(ellipse_x, ellipse_y, alpha=0.2, color=APPLE_BLUE)
    
    # 绘制主轴
    for i in range(2):
        vec = eigvecs[:, i] / np.sqrt(eigvals[i])
        ax.annotate('', xy=(vec[0], vec[1]), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=APPLE_ORANGE, lw=2))
        ax.annotate(f'$\\lambda_{i+1}^{{-1/2}}$', 
                   xy=(vec[0]*1.2, vec[1]*1.2), fontsize=10, color=APPLE_ORANGE)
    
    ax.set_xlabel('$d\\theta_1$', fontsize=12)
    ax.set_ylabel('$d\\theta_2$', fontsize=12)
    ax.set_title('Fisher度量下的局部几何', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/information-geometry.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/information-geometry.png')


def plot_observed_vs_expected():
    """绘制观测Fisher信息与期望Fisher信息的比较"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：单次观测的对比
    ax = axes[0]
    
    np.random.seed(42)
    n = 50
    theta_true = 2
    
    # 生成数据（来自指数族）
    x = np.random.exponential(1/theta_true, n)
    
    # 计算对数似然和观测信息
    theta_range = np.linspace(0.5, 4, 200)
    loglik = np.zeros_like(theta_range)
    obs_info = np.zeros_like(theta_range)
    
    for i, th in enumerate(theta_range):
        loglik[i] = n * np.log(th) - th * np.sum(x)
        # 观测Fisher信息 = -d²ℓ/dθ²
        obs_info[i] = n / th**2
    
    # 期望Fisher信息
    exp_info = n / theta_range**2
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(theta_range, loglik, color=APPLE_BLUE, linewidth=2.5, label='对数似然')
    line2 = ax2.plot(theta_range, obs_info, '--', color=APPLE_GREEN, linewidth=2, label='观测信息')
    line3 = ax2.plot(theta_range, exp_info, ':', color=APPLE_ORANGE, linewidth=2, label='期望信息')
    
    # 标记MLE
    theta_mle = n / np.sum(x)
    ax.axvline(theta_mle, color=APPLE_RED, linestyle='--', alpha=0.7)
    ax.plot(theta_mle, np.interp(theta_mle, theta_range, loglik), 'D', color=APPLE_RED, markersize=10)
    ax.annotate('MLE', xy=(theta_mle, np.interp(theta_mle, theta_range, loglik)), 
               xytext=(10, 10), textcoords='offset points', fontsize=10, color=APPLE_RED)
    
    ax.set_xlabel('$\\theta$', fontsize=12)
    ax.set_ylabel('$\\ell(\\theta)$', fontsize=12, color=APPLE_BLUE)
    ax2.set_ylabel('Fisher信息', fontsize=12)
    ax.set_title('观测信息与期望信息（指数分布）', fontsize=12, fontweight='bold')
    
    # 合并图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 右图：大样本下的收敛
    ax = axes[1]
    
    sample_sizes = np.arange(10, 1010, 10)
    n_reps = 100
    
    relative_errors = []
    for n in sample_sizes:
        errors = []
        for _ in range(n_reps):
            x = np.random.exponential(1/theta_true, n)
            theta_mle = n / np.sum(x)
            obs_info = n / theta_mle**2
            exp_info = n / theta_true**2
            errors.append(abs(obs_info - exp_info) / exp_info)
        relative_errors.append(np.mean(errors))
    
    ax.plot(sample_sizes, relative_errors, color=APPLE_BLUE, linewidth=2)
    ax.fill_between(sample_sizes, relative_errors, alpha=0.2, color=APPLE_BLUE)
    
    # 绘制1/sqrt(n)参考线
    ref_line = relative_errors[0] * np.sqrt(sample_sizes[0]) / np.sqrt(sample_sizes)
    ax.plot(sample_sizes, ref_line, '--', color=APPLE_RED, linewidth=2, 
           label='$O(1/\\sqrt{n})$', alpha=0.7)
    
    ax.set_xlabel('样本量 $n$', fontsize=12)
    ax.set_ylabel('相对误差 $|\\mathcal{J} - \\mathcal{I}| / \\mathcal{I}$', fontsize=12)
    ax.set_title('观测信息收敛于期望信息', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('static/images/plots/observed-vs-expected.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/observed-vs-expected.png')


def plot_neural_network_natural_gradient():
    """绘制神经网络中自然梯度的概念"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 参数空间
    theta1 = np.linspace(-3, 3, 100)
    theta2 = np.linspace(-3, 3, 100)
    T1, T2 = np.meshgrid(theta1, theta2)
    
    # 损失函数（二次型）
    L = 0.5 * (T1**2 + 2*T2**2 + 0.5*T1*T2)
    
    # 绘制损失等高线
    contour = ax.contour(T1, T2, L, levels=15, colors=APPLE_BLUE, linewidths=1, alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 起始点
    start = np.array([2.5, 2.5])
    ax.plot(start[0], start[1], 'o', color=APPLE_RED, markersize=12, label='起点', zorder=5)
    
    # 普通梯度下降方向
    grad = np.array([start[0] + 0.25*start[1], 4*start[1] + 0.25*start[0]])
    grad_norm = grad / np.linalg.norm(grad) * 1.5
    ax.annotate('', xy=(start[0] - grad_norm[0], start[1] - grad_norm[1]),
               xytext=(start[0], start[1]),
               arrowprops=dict(arrowstyle='->', color=APPLE_ORANGE, lw=3),
               label='普通梯度')
    ax.text(start[0] - grad_norm[0] - 0.3, start[1] - grad_norm[1] + 0.3, 
           '普通梯度', fontsize=10, color=APPLE_ORANGE)
    
    # 自然梯度方向（考虑Fisher信息）
    # 假设Fisher矩阵在某个区域近似为 [[2, 0.5], [0.5, 3]]
    F = np.array([[2, 0.5], [0.5, 3]])
    F_inv = np.linalg.inv(F)
    natural_grad = F_inv @ grad
    natural_grad_norm = natural_grad / np.linalg.norm(natural_grad) * 1.5
    ax.annotate('', xy=(start[0] - natural_grad_norm[0], start[1] - natural_grad_norm[1]),
               xytext=(start[0], start[1]),
               arrowprops=dict(arrowstyle='->', color=APPLE_GREEN, lw=3))
    ax.text(start[0] - natural_grad_norm[0] + 0.2, start[1] - natural_grad_norm[1] - 0.3,
           '自然梯度', fontsize=10, color=APPLE_GREEN)
    
    # 最优方向（直接指向最小值）
    ax.annotate('', xy=(0, 0), xytext=(start[0], start[1]),
               arrowprops=dict(arrowstyle='->', color=APPLE_PURPLE, lw=2, ls='--'),
               label='最优方向')
    
    # 标记最小值
    ax.plot(0, 0, '*', color=APPLE_RED, markersize=15, label='全局最小值', zorder=5)
    
    ax.set_xlabel('$\\theta_1$', fontsize=12)
    ax.set_ylabel('$\\theta_2$', fontsize=12)
    ax.set_title('自然梯度下降 vs 普通梯度下降', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/neural-network-natural-gradient.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/neural-network-natural-gradient.png')


def main():
    """生成所有配图"""
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成Fisher信息矩阵配图...")
    
    plot_fisher_vs_variance()
    print("✅ Fisher信息与方差关系")
    
    plot_loglikelihood_curvature()
    print("✅ 对数似然曲率")
    
    plot_fisher_matrix_geometry()
    print("✅ Fisher矩阵几何")
    
    plot_linear_regression_fisher()
    print("✅ 线性回归Fisher信息")
    
    plot_information_geometry()
    print("✅ 信息几何")
    
    plot_observed_vs_expected()
    print("✅ 观测vs期望信息")
    
    plot_neural_network_natural_gradient()
    print("✅ 神经网络自然梯度")
    
    print("\n✅ 所有配图生成完成！")


if __name__ == '__main__':
    main()
