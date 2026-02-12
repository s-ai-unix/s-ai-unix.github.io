#!/usr/bin/env python3
"""
生成微分几何曲面论的 Plotly 图形
保存为 PNG 图片格式
"""

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from PIL import Image
import io

def save_plotly_as_png(fig, filename, width=800, height=600, scale=2):
    """将 Plotly 图形保存为 PNG 图片"""
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    img = Image.open(io.BytesIO(img_bytes))
    img.save(filename)
    print(f"✅ 已生成: {filename}")

# ============================================
# 图1: 球面的参数化表示
# ============================================
def plot_sphere_parametrization():
    """球面的参数化表示"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    U, V = np.meshgrid(u, v)
    
    # 球面参数方程
    R = 1
    X = R * np.sin(V) * np.cos(U)
    Y = R * np.sin(V) * np.sin(U)
    Z = R * np.cos(V)
    
    fig = go.Figure()
    
    # 绘制球面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Blues',
        showscale=False,
        opacity=0.7,
        name='球面'
    ))
    
    # 添加坐标曲线（u = 常数，即经线）
    for u_val in np.linspace(0, 2*np.pi, 8, endpoint=False):
        x_line = R * np.sin(v) * np.cos(u_val)
        y_line = R * np.sin(v) * np.sin(u_val)
        z_line = R * np.cos(v)
        fig.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode='lines',
            line=dict(color='#FF9500', width=3),
            showlegend=False
        ))
    
    # 添加坐标曲线（v = 常数，即纬线）
    for v_val in np.linspace(0.2, np.pi-0.2, 5):
        u_line = np.linspace(0, 2*np.pi, 100)
        x_line = R * np.sin(v_val) * np.cos(u_line)
        y_line = R * np.sin(v_val) * np.sin(u_line)
        z_line = R * np.cos(v_val) * np.ones_like(u_line)
        fig.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode='lines',
            line=dict(color='#34C759', width=3),
            showlegend=False
        ))
    
    # 标记一个特定点及其切向量
    u0, v0 = np.pi/4, np.pi/3
    x0 = R * np.sin(v0) * np.cos(u0)
    y0 = R * np.sin(v0) * np.sin(u0)
    z0 = R * np.cos(v0)
    
    # 计算切向量
    dx_du = R * np.sin(v0) * (-np.sin(u0))
    dy_du = R * np.sin(v0) * np.cos(u0)
    dz_du = 0
    
    dx_dv = R * np.cos(v0) * np.cos(u0)
    dy_dv = R * np.cos(v0) * np.sin(u0)
    dz_dv = -R * np.sin(v0)
    
    # 归一化并缩放
    scale = 0.5
    fig.add_trace(go.Scatter3d(
        x=[x0, x0 + scale*dx_du], y=[y0, y0 + scale*dy_du], z=[z0, z0 + scale*dz_du],
        mode='lines+markers',
        line=dict(color='#007AFF', width=5),
        marker=dict(size=4, color='#007AFF'),
        name=r'$\mathbf{r}_u$'
    ))
    fig.add_trace(go.Scatter3d(
        x=[x0, x0 + scale*dx_dv], y=[y0, y0 + scale*dy_dv], z=[z0, z0 + scale*dz_dv],
        mode='lines+markers',
        line=dict(color='#FF3B30', width=5),
        marker=dict(size=4, color='#FF3B30'),
        name=r'$\mathbf{r}_v$'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=8, color='#000000'),
        name='点 P'
    ))
    
    fig.update_layout(
        title=dict(text='球面的参数化与切向量', font=dict(size=16)),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='cube'
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.02, y=0.98)
    )
    
    save_plotly_as_png(fig, 'static/images/plots/surface_parametrization.png', width=900, height=700)

# ============================================
# 图2: 第一基本型的度量解释 - 曲面片上的度量
# ============================================
def plot_first_fundamental_form():
    """第一基本型的度量解释"""
    # 创建一个抛物面 z = x^2 + y^2 的例子
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    fig = go.Figure()
    
    # 绘制曲面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title=dict(text='高度', side='right'))
    ))
    
    # 标记一个点
    x0, y0 = 0.3, 0.3
    z0 = x0**2 + y0**2
    
    # 在参数平面上的无穷小矩形
    du, dv = 0.3, 0.2
    rect_x = [x0, x0+du, x0+du, x0, x0]
    rect_y = [y0, y0, y0+dv, y0+dv, y0]
    rect_z = [z0, z0, z0, z0, z0]
    
    # 计算曲面上的对应点
    def surface_point(x, y):
        return x**2 + y**2
    
    # 参数平面上的点
    fig.add_trace(go.Scatter3d(
        x=rect_x, y=rect_y, z=[z0+0.05]*5,
        mode='lines',
        line=dict(color='#007AFF', width=4, dash='dash'),
        name='参数平面微元'
    ))
    
    # 曲面上的实际微元（变形后的平行四边形）
    # 切向量
    rx = np.array([1, 0, 2*x0])  # r_x = (1, 0, 2x)
    ry = np.array([0, 1, 2*y0])  # r_y = (0, 1, 2y)
    
    p0 = np.array([x0, y0, z0])
    p1 = p0 + du * rx
    p2 = p0 + du * rx + dv * ry
    p3 = p0 + dv * ry
    
    surf_x = [p0[0], p1[0], p2[0], p3[0], p0[0]]
    surf_y = [p0[1], p1[1], p2[1], p3[1], p0[1]]
    surf_z = [p0[2], p1[2], p2[2], p3[2], p0[2]]
    
    fig.add_trace(go.Scatter3d(
        x=surf_x, y=surf_y, z=surf_z,
        mode='lines',
        line=dict(color='#FF9500', width=4),
        name='曲面上微元'
    ))
    
    # 标记中心点
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=8, color='#FF3B30'),
        name='点 P'
    ))
    
    fig.update_layout(
        title=dict(text='第一基本型：度量曲面上的无穷小距离', font=dict(size=16)),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.02, y=0.98)
    )
    
    save_plotly_as_png(fig, 'static/images/plots/first_fundamental_form.png', width=900, height=700)

# ============================================
# 图3: 曲率的概念 - 法曲率的几何意义
# ============================================
def plot_normal_curvature():
    """法曲率的几何解释"""
    # 创建一个鞍面的例子 z = x^2 - y^2
    x = np.linspace(-1.5, 1.5, 50)
    y = np.linspace(-1.5, 1.5, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2
    
    fig = go.Figure()
    
    # 绘制曲面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='RdBu',
        showscale=False,
        opacity=0.8
    ))
    
    # 在原点处的法向量
    x0, y0, z0 = 0, 0, 0
    # 对于 z = x^2 - y^2，法向量是 (0, 0, 1) 在原点
    normal = np.array([0, 0, 1])
    
    # 绘制法向量
    fig.add_trace(go.Scatter3d(
        x=[x0, x0], y=[y0, y0], z=[z0, z0+0.8],
        mode='lines+markers',
        line=dict(color='#FF3B30', width=5),
        marker=dict(size=[6, 6], color='#FF3B30'),
        name='法向量 n'
    ))
    
    # 绘制不同方向的法截线
    # 方向1: x方向 (曲率为正)
    t1 = np.linspace(-1, 1, 50)
    x1 = t1
    y1 = np.zeros_like(t1)
    z1 = t1**2
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='lines',
        line=dict(color='#007AFF', width=4),
        name='x方向 (k>0)'
    ))
    
    # 方向2: y方向 (曲率为负)
    t2 = np.linspace(-1, 1, 50)
    x2 = np.zeros_like(t2)
    y2 = t2
    z2 = -t2**2
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='lines',
        line=dict(color='#34C759', width=4),
        name='y方向 (k<0)'
    ))
    
    # 方向3: 45度方向 (曲率为零)
    t3 = np.linspace(-1.2, 1.2, 50)
    x3 = t3 / np.sqrt(2)
    y3 = t3 / np.sqrt(2)
    z3 = np.zeros_like(t3)
    fig.add_trace(go.Scatter3d(
        x=x3, y=y3, z=z3,
        mode='lines',
        line=dict(color='#FF9500', width=4),
        name='45°方向 (k=0)'
    ))
    
    # 标记原点
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=10, color='#000000'),
        name='点 P'
    ))
    
    fig.update_layout(
        title=dict(text='法曲率：不同方向的弯曲程度', font=dict(size=16)),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.02, y=0.98)
    )
    
    save_plotly_as_png(fig, 'static/images/plots/normal_curvature.png', width=900, height=700)

# ============================================
# 图4: 主曲率和杜邦指标线
# ============================================
def plot_dupin_indicatrix():
    """杜邦指标线和主曲率方向"""
    from plotly.subplots import make_subplots
    
    # 创建2D图展示杜邦指标线
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 情况1: 椭圆点 (k1 > 0, k2 > 0)
    k1, k2 = 2, 1
    r1 = 1 / np.sqrt(np.abs(k1 * np.cos(theta)**2 + k2 * np.sin(theta)**2))
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)
    
    # 情况2: 双曲点 (k1 > 0, k2 < 0)
    k1, k2 = 2, -1
    # 处理渐近线附近的问题
    denom = k1 * np.cos(theta)**2 + k2 * np.sin(theta)**2
    r2 = np.where(np.abs(denom) > 0.01, 1 / np.sqrt(np.abs(denom)), np.nan)
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta)
    
    # 情况3: 抛物点 (k1 > 0, k2 = 0)
    k1 = 2
    r3 = 1 / np.sqrt(k1 * np.cos(theta)**2 + 0.001)  # 避免除零
    x3 = r3 * np.cos(theta)
    y3 = r3 * np.sin(theta)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('椭圆点 (K>0)', '双曲点 (K<0)', '抛物点 (K=0)'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # 椭圆点
    fig.add_trace(go.Scatter(
        x=x1, y=y1,
        mode='lines',
        line=dict(color='#007AFF', width=2),
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        name='椭圆点'
    ), row=1, col=1)
    
    # 双曲点 - 分为两部分
    mask_pos = denom > 0
    mask_neg = denom < 0
    fig.add_trace(go.Scatter(
        x=np.where(mask_pos, x2, np.nan),
        y=np.where(mask_pos, y2, np.nan),
        mode='lines',
        line=dict(color='#34C759', width=2),
        name='双曲点(+)'
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=np.where(mask_neg, x2, np.nan),
        y=np.where(mask_neg, y2, np.nan),
        mode='lines',
        line=dict(color='#34C759', width=2),
        name='双曲点(-)'
    ), row=1, col=2)
    
    # 抛物点
    fig.add_trace(go.Scatter(
        x=x3, y=y3,
        mode='lines',
        line=dict(color='#FF9500', width=2),
        name='抛物点'
    ), row=1, col=3)
    
    # 添加坐标轴
    for i in range(1, 4):
        fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dash'), row=1, col=i)
        fig.add_vline(x=0, line=dict(color='gray', width=1, dash='dash'), row=1, col=i)
        fig.update_xaxes(range=[-2, 2], row=1, col=i)
        fig.update_yaxes(range=[-2, 2], row=1, col=i, scaleanchor='x', scaleratio=1)
    
    fig.update_layout(
        title=dict(text='杜邦指标线：不同点类型的曲率分布', font=dict(size=16), x=0.5),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        showlegend=False,
        width=1200,
        height=450
    )
    
    save_plotly_as_png(fig, 'static/images/plots/dupin_indicatrix.png', width=1200, height=450)

# ============================================
# 图5: 高斯曲率的几何意义 - 球面映射
# ============================================
def plot_gauss_map():
    """高斯映射的几何解释"""
    # 创建一个波纹曲面
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    U, V = np.meshgrid(u, v)
    
    # 变形的球面 (类似椭球)
    a, b, c = 1.5, 1, 0.8
    X = a * np.sin(V) * np.cos(U)
    Y = b * np.sin(V) * np.sin(U)
    Z = c * np.cos(V)
    
    # 计算高斯曲率
    # 对于椭球面，高斯曲率 K = (abc)^2 / (某复杂表达式)
    # 这里我们使用一个简化的高斯曲率估计
    K = np.ones_like(X)  # 简化为常数
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('原始曲面', '高斯球面映射'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )
    
    # 原始曲面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        showscale=False,
        name='曲面'
    ), row=1, col=1)
    
    # 高斯映射后的单位球面（法向量的端点）
    # 对于椭球面，单位法向量
    nx = (X/a**2) / np.sqrt((X/a**2)**2 + (Y/b**2)**2 + (Z/c**2)**2)
    ny = (Y/b**2) / np.sqrt((X/a**2)**2 + (Y/b**2)**2 + (Z/c**2)**2)
    nz = (Z/c**2) / np.sqrt((X/a**2)**2 + (Y/b**2)**2 + (Z/c**2)**2)
    
    fig.add_trace(go.Surface(
        x=nx, y=ny, z=nz,
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(title='K', x=0.95),
        name='高斯映射'
    ), row=1, col=2)
    
    fig.update_scenes(
        aspectmode='cube',
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'
    )
    
    fig.update_layout(
        title=dict(text='高斯映射：曲面上每点的单位法向量', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1100,
        height=550
    )
    
    save_plotly_as_png(fig, 'static/images/plots/gauss_map.png', width=1100, height=550)

# ============================================
# 图6: 可展曲面与不可展曲面对比
# ============================================
def plot_developable_surfaces():
    """可展曲面与不可展曲面"""
    # 圆柱面（可展）
    theta = np.linspace(0, 2*np.pi, 50)
    z_cyl = np.linspace(-1, 1, 30)
    Theta, Z_cyl = np.meshgrid(theta, z_cyl)
    R = 1
    X_cyl = R * np.cos(Theta)
    Y_cyl = R * np.sin(Theta)
    Z_cyl_grid = Z_cyl
    
    # 球面（不可展）
    phi = np.linspace(0, np.pi, 30)
    theta_sph = np.linspace(0, 2*np.pi, 50)
    Phi, Theta_sph = np.meshgrid(phi, theta_sph)
    X_sph = np.sin(Phi) * np.cos(Theta_sph)
    Y_sph = np.sin(Phi) * np.sin(Theta_sph)
    Z_sph = np.cos(Phi)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('圆柱面 (K=0, 可展)', '球面 (K>0, 不可展)'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )
    
    # 圆柱面
    fig.add_trace(go.Surface(
        x=X_cyl, y=Y_cyl, z=Z_cyl_grid,
        colorscale='Blues',
        showscale=False,
        opacity=0.9
    ), row=1, col=1)
    
    # 添加圆柱的母线
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        fig.add_trace(go.Scatter3d(
            x=[np.cos(angle)]*2, y=[np.sin(angle)]*2, z=[-1, 1],
            mode='lines',
            line=dict(color='#FF9500', width=3),
            showlegend=False
        ), row=1, col=1)
    
    # 球面
    fig.add_trace(go.Surface(
        x=X_sph, y=Y_sph, z=Z_sph,
        colorscale='Reds',
        showscale=False,
        opacity=0.9
    ), row=1, col=2)
    
    fig.update_scenes(
        aspectmode='cube',
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'
    )
    
    fig.update_layout(
        title=dict(text='可展曲面与不可展曲面的对比', font=dict(size=16)),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        width=1100,
        height=550
    )
    
    save_plotly_as_png(fig, 'static/images/plots/developable_surfaces.png', width=1100, height=550)

# ============================================
# 图7: 测地线的直观理解
# ============================================
def plot_geodesic():
    """球面上的测地线"""
    # 球面
    phi = np.linspace(0, np.pi, 40)
    theta = np.linspace(0, 2*np.pi, 60)
    Phi, Theta = np.meshgrid(phi, theta)
    R = 1
    X = R * np.sin(Phi) * np.cos(Theta)
    Y = R * np.sin(Phi) * np.sin(Theta)
    Z = R * np.cos(Phi)
    
    fig = go.Figure()
    
    # 绘制球面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Blues',
        showscale=False,
        opacity=0.5
    ))
    
    # 大圆（测地线）- 赤道
    theta_eq = np.linspace(0, 2*np.pi, 100)
    x_eq = R * np.cos(theta_eq)
    y_eq = R * np.sin(theta_eq)
    z_eq = np.zeros_like(theta_eq)
    fig.add_trace(go.Scatter3d(
        x=x_eq, y=y_eq, z=z_eq,
        mode='lines',
        line=dict(color='#007AFF', width=5),
        name='测地线（大圆）'
    ))
    
    # 另一条测地线（经线）
    phi_mer = np.linspace(0, np.pi, 50)
    theta_mer = np.pi/4
    x_mer = R * np.sin(phi_mer) * np.cos(theta_mer)
    y_mer = R * np.sin(phi_mer) * np.sin(theta_mer)
    z_mer = R * np.cos(phi_mer)
    fig.add_trace(go.Scatter3d(
        x=x_mer, y=y_mer, z=z_mer,
        mode='lines',
        line=dict(color='#34C759', width=5),
        name='另一条测地线'
    ))
    
    # 非测地线的小圆
    phi_small = np.pi/4  # 固定纬度
    theta_small = np.linspace(0, 2*np.pi, 100)
    r_small = R * np.sin(phi_small)
    x_small = r_small * np.cos(theta_small)
    y_small = r_small * np.sin(theta_small)
    z_small = R * np.cos(phi_small) * np.ones_like(theta_small)
    fig.add_trace(go.Scatter3d(
        x=x_small, y=y_small, z=z_small,
        mode='lines',
        line=dict(color='#FF3B30', width=3, dash='dash'),
        name='非测地线（小圆）'
    ))
    
    # 标记点
    fig.add_trace(go.Scatter3d(
        x=[R/np.sqrt(2)], y=[R/np.sqrt(2)], z=[0],
        mode='markers',
        marker=dict(size=10, color='#FF9500'),
        name='交点'
    ))
    
    fig.update_layout(
        title=dict(text='球面上的测地线：最短路径的弯曲程度', font=dict(size=16)),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='cube'
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(x=0.02, y=0.98)
    )
    
    save_plotly_as_png(fig, 'static/images/plots/geodesic.png', width=900, height=700)


if __name__ == '__main__':
    from plotly.subplots import make_subplots
    
    print("开始生成曲面论的 Plotly 图形...")
    
    plot_sphere_parametrization()
    plot_first_fundamental_form()
    plot_normal_curvature()
    plot_dupin_indicatrix()
    plot_gauss_map()
    plot_developable_surfaces()
    plot_geodesic()
    
    print("\n✅ 所有图形生成完成！")
