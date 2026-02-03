"""
生成KL散度相关的配图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from scipy import stats
from scipy.special import kl_div
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


def plot_kl_divergence_intuition():
    """绘制KL散度的直观解释"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：两个分布的差异
    ax = axes[0]
    
    x = np.linspace(-5, 5, 500)
    
    # 真实分布 P
    p = stats.norm.pdf(x, 0, 1)
    ax.plot(x, p, color=APPLE_BLUE, linewidth=2.5, label='真实分布 $P$')
    ax.fill_between(x, p, alpha=0.2, color=APPLE_BLUE)
    
    # 近似分布 Q
    q = stats.norm.pdf(x, 1, 1.2)
    ax.plot(x, q, color=APPLE_ORANGE, linewidth=2.5, label='近似分布 $Q$')
    ax.fill_between(x, q, alpha=0.2, color=APPLE_ORANGE)
    
    # 标记差异较大的区域
    diff_region = (x > -1) & (x < 2)
    ax.fill_between(x[diff_region], p[diff_region], q[diff_region], 
                    alpha=0.3, color=APPLE_RED, label='差异区域')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('KL散度度量分布差异', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 右图：KL散度的计算可视化
    ax = axes[1]
    
    x_subset = np.linspace(-4, 4, 200)
    p_subset = stats.norm.pdf(x_subset, 0, 1)
    q_subset = stats.norm.pdf(x_subset, 1, 1.2)
    
    # 计算比率的对数
    ratio = np.log(p_subset / q_subset)
    
    # KL散度的被积函数: p(x) * log(p(x)/q(x))
    integrand = p_subset * ratio
    
    ax.plot(x_subset, integrand, color=APPLE_GREEN, linewidth=2.5)
    ax.fill_between(x_subset, integrand, alpha=0.3, color=APPLE_GREEN)
    ax.axhline(0, color=APPLE_GRAY, linestyle='-', linewidth=1)
    
    # 标记积分区域
    ax.annotate('正贡献', xy=(-2, 0.1), fontsize=10, color=APPLE_GREEN)
    ax.annotate('负贡献', xy=(2.5, -0.05), fontsize=10, color=APPLE_ORANGE)
    
    # 计算并显示KL值
    kl_value = np.trapz(integrand, x_subset)
    ax.text(0.95, 0.95, f'$D_{{KL}}(P||Q) \\approx {kl_value:.3f}$', 
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           horizontalalignment='right', bbox=dict(boxstyle='round', 
           facecolor='white', alpha=0.8))
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('$p(x) \\log \\frac{p(x)}{q(x)}$', fontsize=12)
    ax.set_title('KL散度的被积函数', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/kl-divergence-intuition.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/kl-divergence-intuition.png')


def plot_kl_asymmetry():
    """绘制KL散度的非对称性"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：D_KL(P||Q)
    ax = axes[0]
    
    x = np.linspace(-5, 5, 500)
    p = stats.norm.pdf(x, 0, 1)
    q = stats.norm.pdf(x, 2, 1.5)
    
    ax.plot(x, p, color=APPLE_BLUE, linewidth=2.5, label='$P = N(0,1)$')
    ax.plot(x, q, color=APPLE_ORANGE, linewidth=2.5, label='$Q = N(2,1.5)$')
    
    # 用颜色填充差异
    ax.fill_between(x, p, q, where=(p >= q), alpha=0.3, color=APPLE_GREEN, 
                   interpolate=True, label='P > Q')
    ax.fill_between(x, p, q, where=(p < q), alpha=0.3, color=APPLE_RED, 
                   interpolate=True, label='P < Q')
    
    ax.set_title('$D_{KL}(P||Q)$：用P的视角看差异', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 计算KL值
    kl_pq = np.trapz(p * np.log(p / q), x)
    ax.text(0.95, 0.95, f'$D_{{KL}}(P||Q) = {kl_pq:.3f}$', 
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           horizontalalignment='right', bbox=dict(boxstyle='round', 
           facecolor=APPLE_GREEN, alpha=0.3))
    
    # 右图：D_KL(Q||P)
    ax = axes[1]
    
    ax.plot(x, q, color=APPLE_ORANGE, linewidth=2.5, label='$Q = N(2,1.5)$')
    ax.plot(x, p, color=APPLE_BLUE, linewidth=2.5, label='$P = N(0,1)$')
    
    ax.fill_between(x, q, p, where=(q >= p), alpha=0.3, color=APPLE_GREEN, 
                   interpolate=True, label='Q > P')
    ax.fill_between(x, q, p, where=(q < p), alpha=0.3, color=APPLE_RED, 
                   interpolate=True, label='Q < P')
    
    ax.set_title('$D_{KL}(Q||P)$：用Q的视角看差异', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 计算KL值
    kl_qp = np.trapz(q * np.log(q / p), x)
    ax.text(0.95, 0.95, f'$D_{{KL}}(Q||P) = {kl_qp:.3f}$', 
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           horizontalalignment='right', bbox=dict(boxstyle='round', 
           facecolor=APPLE_RED, alpha=0.3))
    
    # 添加非对称性说明
    fig.text(0.5, 0.02, f'非对称性: $D_{{KL}}(P||Q) \\neq D_{{KL}}(Q||P)$ ({kl_pq:.3f} vs {kl_qp:.3f})', 
            ha='center', fontsize=11, style='italic', color=APPLE_PURPLE)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('static/images/plots/kl-asymmetry.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/kl-asymmetry.png')


def plot_kl_properties():
    """绘制KL散度的性质"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：非负性 (Gibbs不等式)
    ax = axes[0]
    
    # 绘制KL散度随参数变化
    mu_range = np.linspace(-3, 3, 100)
    kl_values = []
    
    p = stats.norm(0, 1)
    for mu in mu_range:
        q = stats.norm(mu, 1)
        # 数值计算KL
        x = np.linspace(-10, 10, 1000)
        px = p.pdf(x)
        qx = q.pdf(x)
        kl = np.trapz(px * np.log(px / qx), x)
        kl_values.append(kl)
    
    ax.plot(mu_range, kl_values, color=APPLE_BLUE, linewidth=2.5)
    ax.axhline(0, color=APPLE_GRAY, linestyle='--', linewidth=1)
    ax.axvline(0, color=APPLE_RED, linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 标记最小值
    ax.plot(0, 0, 'D', color=APPLE_RED, markersize=10)
    ax.annotate('$D_{KL}(P||P) = 0$', xy=(0, 0.1), fontsize=10, ha='center', color=APPLE_RED)
    
    ax.fill_between(mu_range, kl_values, alpha=0.2, color=APPLE_BLUE)
    
    ax.set_xlabel('$\\mu_Q$ (Q的均值)', fontsize=12)
    ax.set_ylabel('$D_{KL}(P||Q)$', fontsize=12)
    ax.set_title('KL散度的非负性：$D_{KL} \\geq 0$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 右图：凸性
    ax = axes[1]
    
    # 参数空间中的KL等高线
    mu1_range = np.linspace(-2, 2, 50)
    sigma_range = np.linspace(0.5, 2, 50)
    MU1, SIGMA = np.meshgrid(mu1_range, sigma_range)
    
    p = stats.norm(0, 1)
    KL = np.zeros_like(MU1)
    
    for i in range(len(mu1_range)):
        for j in range(len(sigma_range)):
            q = stats.norm(MU1[j, i], SIGMA[j, i])
            x = np.linspace(-10, 10, 500)
            px = p.pdf(x)
            qx = q.pdf(x)
            KL[j, i] = np.trapz(px * np.log(px / qx), x)
    
    contour = ax.contour(MU1, SIGMA, KL, levels=10, colors=APPLE_BLUE, linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.contourf(MU1, SIGMA, KL, levels=20, cmap='Blues', alpha=0.4)
    
    # 标记最小值点
    ax.plot(0, 1, 'D', color=APPLE_RED, markersize=10)
    ax.annotate('最小值点', xy=(0, 1), xytext=(0.5, 0.7), fontsize=10, color=APPLE_RED)
    
    ax.set_xlabel('$\\mu_Q$', fontsize=12)
    ax.set_ylabel('$\\sigma_Q$', fontsize=12)
    ax.set_title('KL散度的凸性', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/kl-properties.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/kl-properties.png')


def plot_variational_inference():
    """绘制变分推断中的KL散度"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：复杂后验与简单近似
    ax = axes[0]
    
    x = np.linspace(-5, 5, 500)
    
    # 复杂的真实后验（双峰）
    p_posterior = 0.4 * stats.norm.pdf(x, -1.5, 0.5) + 0.6 * stats.norm.pdf(x, 1.5, 0.7)
    ax.plot(x, p_posterior, color=APPLE_BLUE, linewidth=2.5, label='真实后验 $p(z|x)$')
    ax.fill_between(x, p_posterior, alpha=0.2, color=APPLE_BLUE)
    
    # 简单的变分近似（单峰高斯）
    q_approx = stats.norm.pdf(x, 0.5, 1.2)
    ax.plot(x, q_approx, color=APPLE_ORANGE, linewidth=2.5, label='变分近似 $q(z)$')
    ax.fill_between(x, q_approx, alpha=0.2, color=APPLE_ORANGE)
    
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('变分推断：用简单分布近似复杂后验', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 右图：优化过程
    ax = axes[1]
    
    # 模拟优化过程中的KL散度下降
    iterations = np.arange(0, 101)
    kl_trajectory = 2.0 * np.exp(-iterations / 20) + 0.1 + np.random.normal(0, 0.02, len(iterations))
    
    ax.plot(iterations, kl_trajectory, color=APPLE_GREEN, linewidth=2)
    ax.fill_between(iterations, kl_trajectory, alpha=0.2, color=APPLE_GREEN)
    
    # 标记收敛点
    ax.axhline(0.1, color=APPLE_RED, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.annotate('收敛值', xy=(80, 0.12), fontsize=10, color=APPLE_RED)
    
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('$D_{KL}(q||p)$', fontsize=12)
    ax.set_title('变分推断优化过程', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/variational-inference.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/variational-inference.png')


def plot_kl_derivation():
    """绘制KL散度的推导和等价形式"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：对数似然比的几何
    ax = axes[0]
    
    x = np.linspace(-4, 4, 200)
    p = stats.norm.pdf(x, 0, 1)
    q = stats.norm.pdf(x, 1.5, 1.2)
    
    # 对数似然比
    log_ratio = np.log(p / q)
    
    ax.plot(x, log_ratio, color=APPLE_BLUE, linewidth=2.5, label='$\\log \\frac{p(x)}{q(x)}$')
    ax.axhline(0, color=APPLE_GRAY, linestyle='-', linewidth=1, alpha=0.5)
    ax.fill_between(x, log_ratio, where=(log_ratio > 0), alpha=0.3, color=APPLE_GREEN)
    ax.fill_between(x, log_ratio, where=(log_ratio < 0), alpha=0.3, color=APPLE_RED)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('$\\log \\frac{p(x)}{q(x)}$', fontsize=12)
    ax.set_title('对数似然比：区分P和Q的证据', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 右图：熵与交叉熵
    ax = axes[1]
    
    # 二分类例子
    p_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q_vals = np.array([0.2, 0.3, 0.25, 0.5, 0.5, 0.55, 0.75, 0.7, 0.85])
    
    # 计算各分量
    kl_components = p_vals * np.log(p_vals / q_vals)
    
    x_pos = np.arange(len(p_vals))
    
    bars = ax.bar(x_pos, kl_components, color=[APPLE_GREEN if k > 0 else APPLE_RED for k in kl_components],
                  alpha=0.7, edgecolor='white', linewidth=1.5)
    
    ax.axhline(0, color=APPLE_GRAY, linestyle='-', linewidth=1)
    ax.set_xlabel('事件索引', fontsize=12)
    ax.set_ylabel('$p_i \\log \\frac{p_i}{q_i}$', fontsize=12)
    ax.set_title('KL散度的分量贡献', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加总和
    total_kl = np.sum(kl_components)
    ax.text(0.95, 0.95, f'$D_{{KL}} = {total_kl:.3f}$', 
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           horizontalalignment='right', bbox=dict(boxstyle='round', 
           facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('static/images/plots/kl-derivation.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/kl-derivation.png')


def plot_information_geometry_kl():
    """绘制KL散度与信息几何"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：统计流形上的测地线
    ax = axes[0]
    
    # 正态分布族的二维截面
    mu_range = np.linspace(-2, 2, 50)
    sigma_range = np.linspace(0.5, 2.5, 50)
    MU, SIGMA = np.meshgrid(mu_range, sigma_range)
    
    # 参考点
    mu0, sigma0 = 0, 1
    
    # 计算到参考点的KL散度
    KL_mesh = 0.5 * ((SIGMA**2 + (MU - mu0)**2) / sigma0**2 - 1 - 2*np.log(SIGMA/sigma0))
    
    contour = ax.contour(MU, SIGMA, KL_mesh, levels=12, colors=APPLE_BLUE, linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.contourf(MU, SIGMA, KL_mesh, levels=20, cmap='Blues', alpha=0.4)
    
    # 标记参考点
    ax.plot(mu0, sigma0, 'D', color=APPLE_RED, markersize=10)
    ax.annotate('参考分布', xy=(mu0, sigma0), xytext=(0.5, 0.7), 
               fontsize=10, color=APPLE_RED)
    
    # 绘制一条测地线路径
    mu_path = np.linspace(-1.5, 1.5, 50)
    # 近似测地线：保持特定关系的曲线
    sigma_path = np.sqrt(1 + mu_path**2/2)
    ax.plot(mu_path, sigma_path, 'o-', color=APPLE_GREEN, markersize=3, 
           linewidth=2, label='测地线（近似）')
    
    ax.set_xlabel('$\\mu$', fontsize=12)
    ax.set_ylabel('$\\sigma$', fontsize=12)
    ax.set_title('统计流形上的KL散度等高线', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 右图：Fisher度量与KL的关系
    ax = axes[1]
    
    # 在小邻域内，KL散度 ≈ 0.5 * d^T I d
    d_theta = np.linspace(-1, 1, 100)
    
    # Fisher信息矩阵（对于N(mu, sigma)，在(0,1)处）
    I = np.array([[1, 0], [0, 2]])  # 简化的Fisher矩阵
    
    # 沿不同方向的Fisher距离
    # 方向1：只变mu
    fisher_dist1 = 0.5 * I[0,0] * d_theta**2
    # 方向2：只变sigma
    fisher_dist2 = 0.5 * I[1,1] * d_theta**2
    
    ax.plot(d_theta, fisher_dist1, color=APPLE_BLUE, linewidth=2.5, 
           label='Fisher距离 ($\\Delta \\mu$)')
    ax.plot(d_theta, fisher_dist2, color=APPLE_ORANGE, linewidth=2.5, 
           label='Fisher距离 ($\\Delta \\sigma$)')
    
    # 实际的KL散度（二阶近似）
    ax.plot(d_theta, fisher_dist1, 'o', color=APPLE_BLUE, markersize=4, alpha=0.5)
    ax.plot(d_theta, fisher_dist2, 's', color=APPLE_ORANGE, markersize=4, alpha=0.5)
    
    ax.fill_between(d_theta, fisher_dist1, alpha=0.1, color=APPLE_BLUE)
    ax.fill_between(d_theta, fisher_dist2, alpha=0.1, color=APPLE_ORANGE)
    
    ax.set_xlabel('参数变化 $\\Delta \\theta$', fontsize=12)
    ax.set_ylabel('距离度量', fontsize=12)
    ax.set_title('小邻域内：$D_{KL} \\approx \\frac{1}{2} d^T \\mathcal{I} d$', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/information-geometry-kl.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/information-geometry-kl.png')


def plot_machine_learning_applications():
    """绘制机器学习中的应用"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：VAE中的KL散度
    ax = axes[0]
    
    # 潜在空间的可视化
    np.random.seed(42)
    n_samples = 200
    
    # 先验分布（标准正态）
    z_prior = np.random.randn(n_samples, 2)
    ax.scatter(z_prior[:, 0], z_prior[:, 1], c=APPLE_BLUE, alpha=0.4, s=30, label='先验 $p(z)$')
    
    # 后验近似（变分分布）
    z_posterior = np.random.randn(n_samples, 2) * 0.6 + np.array([1, 0.5])
    ax.scatter(z_posterior[:, 0], z_posterior[:, 1], c=APPLE_ORANGE, alpha=0.4, s=30, 
              label='后验近似 $q(z|x)$')
    
    # 置信椭圆
    from matplotlib.patches import Ellipse
    
    # 先验椭圆
    prior_ellipse = Ellipse((0, 0), 2*1.96*2, 2*1.96*2, fill=False, 
                           edgecolor=APPLE_BLUE, linewidth=2, linestyle='--')
    ax.add_patch(prior_ellipse)
    
    # 后验椭圆
    posterior_ellipse = Ellipse((1, 0.5), 2*1.96*0.6*2, 2*1.96*0.6*2, fill=False,
                               edgecolor=APPLE_ORANGE, linewidth=2, linestyle='--')
    ax.add_patch(posterior_ellipse)
    
    ax.set_xlabel('$z_1$', fontsize=12)
    ax.set_ylabel('$z_2$', fontsize=12)
    ax.set_title('VAE中的潜在空间与KL正则化', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # 右图：强化学习中的策略优化
    ax = axes[1]
    
    # 策略更新前后的分布
    actions = np.linspace(-3, 3, 200)
    
    # 旧策略
    old_policy = stats.norm.pdf(actions, 0, 1)
    ax.plot(actions, old_policy, color=APPLE_GRAY, linewidth=2.5, 
           label='旧策略 $\\pi_{old}$', linestyle='--')
    
    # 新策略（不同KL约束下的更新）
    new_policy_small = stats.norm.pdf(actions, 0.3, 0.9)
    ax.plot(actions, new_policy_small, color=APPLE_GREEN, linewidth=2.5, 
           label='新策略 (KL小)')
    ax.fill_between(actions, new_policy_small, alpha=0.2, color=APPLE_GREEN)
    
    new_policy_large = stats.norm.pdf(actions, 0.8, 0.7)
    ax.plot(actions, new_policy_large, color=APPLE_RED, linewidth=2.5, 
           label='新策略 (KL大)')
    ax.fill_between(actions, new_policy_large, alpha=0.2, color=APPLE_RED)
    
    ax.set_xlabel('动作 $a$', fontsize=12)
    ax.set_ylabel('策略概率 $\\pi(a)$', fontsize=12)
    ax.set_title('TRPO：KL约束下的策略更新', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/plots/ml-applications-kl.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    compress_png('static/images/plots/ml-applications-kl.png')


def main():
    """生成所有配图"""
    output_dir = 'static/images/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成KL散度配图...")
    
    plot_kl_divergence_intuition()
    print("✅ KL散度直观解释")
    
    plot_kl_asymmetry()
    print("✅ KL散度非对称性")
    
    plot_kl_properties()
    print("✅ KL散度性质")
    
    plot_variational_inference()
    print("✅ 变分推断")
    
    plot_kl_derivation()
    print("✅ KL散度推导")
    
    plot_information_geometry_kl()
    print("✅ 信息几何与KL")
    
    plot_machine_learning_applications()
    print("✅ 机器学习应用")
    
    print("\n✅ 所有配图生成完成！")


if __name__ == '__main__':
    main()
