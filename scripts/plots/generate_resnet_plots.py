#!/usr/bin/env python3
"""
生成 ResNet 论文解读的配图
使用 Plotly 生成专业的数理图形，并保存为 PNG 格式
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import subprocess
import os

# 苹果风格配色
APPLE_BLUE = '#007AFF'
APPLE_GREEN = '#34C759'
APPLE_ORANGE = '#FF9500'
APPLE_RED = '#FF3B30'
APPLE_PURPLE = '#AF52DE'
APPLE_GRAY = '#8E8E93'
APPLE_TEAL = '#5AC8FA'

def save_and_compress(fig, filepath, scale=2):
    """保存并压缩图片"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存为 PNG
    fig.write_image(filepath, scale=scale)
    
    # 压缩 PNG
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")

def plot_degradation_problem():
    """
    绘制退化问题示意图：展示网络深度增加时，训练误差反而上升的现象
    """
    # 模拟数据
    depths = [20, 32, 44, 56, 110]
    
    # 普通网络：随着深度增加，训练误差上升（退化问题）
    plain_train_error = [8.0, 9.5, 14.0, 20.0, 35.0]
    plain_val_error = [10.0, 11.5, 16.0, 22.0, 38.0]
    
    # ResNet：随着深度增加，误差持续下降
    resnet_train_error = [8.0, 7.0, 6.0, 5.0, 4.0]
    resnet_val_error = [10.0, 9.0, 8.0, 7.5, 6.5]
    
    fig = go.Figure()
    
    # 普通网络训练误差
    fig.add_trace(go.Scatter(
        x=depths, y=plain_train_error,
        mode='lines+markers',
        name='Plain Net (Train)',
        line=dict(color=APPLE_RED, width=2, dash='solid'),
        marker=dict(size=10, symbol='circle')
    ))
    
    # 普通网络验证误差
    fig.add_trace(go.Scatter(
        x=depths, y=plain_val_error,
        mode='lines+markers',
        name='Plain Net (Val)',
        line=dict(color=APPLE_RED, width=2, dash='dash'),
        marker=dict(size=10, symbol='circle')
    ))
    
    # ResNet 训练误差
    fig.add_trace(go.Scatter(
        x=depths, y=resnet_train_error,
        mode='lines+markers',
        name='ResNet (Train)',
        line=dict(color=APPLE_GREEN, width=2, dash='solid'),
        marker=dict(size=10, symbol='diamond')
    ))
    
    # ResNet 验证误差
    fig.add_trace(go.Scatter(
        x=depths, y=resnet_val_error,
        mode='lines+markers',
        name='ResNet (Val)',
        line=dict(color=APPLE_GREEN, width=2, dash='dash'),
        marker=dict(size=10, symbol='diamond')
    ))
    
    # 添加退化问题的标注
    fig.add_annotation(
        x=56, y=20,
        text='退化问题:<br>深层网络误差更高',
        showarrow=True,
        arrowhead=2,
        arrowcolor=APPLE_RED,
        ax=60, ay=-40,
        font=dict(size=12, color=APPLE_RED),
        bgcolor='rgba(255,255,255,0.8)'
    )
    
    fig.update_layout(
        title=dict(
            text='退化问题：深层网络的训练困境',
            font=dict(size=18, family='Arial, sans-serif')
        ),
        xaxis_title='网络深度（层数）',
        yaxis_title='错误率 (%)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=APPLE_GRAY,
            borderwidth=1
        ),
        width=900,
        height=500,
        margin=dict(l=80, r=50, t=80, b=60)
    )
    
    # 使用线性刻度，确保数据点均匀分布
    fig.update_xaxes(tickvals=depths, ticktext=[str(d) for d in depths])
    
    return fig

def plot_residual_block():
    """
    绘制残差块结构示意图 - 修复布局
    """
    fig = go.Figure()
    
    # 节点位置 - 均匀分布
    node_size = 55
    
    # 输入节点 x
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=node_size, color=APPLE_BLUE, line=dict(width=2, color='white')),
        name='输入',
        hoverinfo='text',
        hovertext='输入特征 x'
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='text',
        text=['x'],
        textposition='middle center',
        textfont=dict(size=18, color='white', family='Arial')
    ))
    
    # 卷积层 1 - 使用更简洁的标签
    fig.add_trace(go.Scatter(
        x=[2], y=[0],
        mode='markers',
        marker=dict(size=node_size, color=APPLE_PURPLE, line=dict(width=2, color='white')),
        name='卷积层1',
        hoverinfo='text',
        hovertext='3×3 Conv + BN + ReLU'
    ))
    fig.add_trace(go.Scatter(
        x=[2], y=[0],
        mode='text',
        text=['Conv1'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial')
    ))
    
    # 卷积层 2
    fig.add_trace(go.Scatter(
        x=[4], y=[0],
        mode='markers',
        marker=dict(size=node_size, color=APPLE_PURPLE, line=dict(width=2, color='white')),
        name='卷积层2',
        hoverinfo='text',
        hovertext='3×3 Conv + BN'
    ))
    fig.add_trace(go.Scatter(
        x=[4], y=[0],
        mode='text',
        text=['Conv2'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial')
    ))
    
    # 加法节点 (菱形)
    fig.add_trace(go.Scatter(
        x=[6], y=[0],
        mode='markers',
        marker=dict(size=node_size, color=APPLE_ORANGE, symbol='diamond', line=dict(width=2, color='white')),
        name='逐元素相加',
        hoverinfo='text',
        hovertext='F(x) + x'
    ))
    fig.add_trace(go.Scatter(
        x=[6], y=[0],
        mode='text',
        text=['+'],
        textposition='middle center',
        textfont=dict(size=22, color='white', family='Arial')
    ))
    
    # 输出节点 y
    fig.add_trace(go.Scatter(
        x=[8], y=[0],
        mode='markers',
        marker=dict(size=node_size, color=APPLE_GREEN, line=dict(width=2, color='white')),
        name='输出',
        hoverinfo='text',
        hovertext='输出 y = F(x) + x'
    ))
    fig.add_trace(go.Scatter(
        x=[8], y=[0],
        mode='text',
        text=['y'],
        textposition='middle center',
        textfont=dict(size=18, color='white', family='Arial')
    ))
    
    # 残差路径箭头 (主路径)
    arrow_style = dict(arrowhead=2, arrowsize=1.2, arrowwidth=2.5, arrowcolor='#8E8E93')
    
    # x -> Conv1
    fig.add_annotation(x=1.3, y=0, ax=0.4, ay=0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, **arrow_style)
    # Conv1 -> Conv2
    fig.add_annotation(x=3.3, y=0, ax=2.4, ay=0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, **arrow_style)
    # Conv2 -> Add
    fig.add_annotation(x=5.3, y=0, ax=4.4, ay=0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, **arrow_style)
    # Add -> y
    fig.add_annotation(x=7.3, y=0, ax=6.4, ay=0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, **arrow_style)
    
    # 跳跃连接路径 (shortcut) - 使用贝塞尔曲线效果
    shortcut_color = APPLE_TEAL
    
    # 绘制跳跃连接线
    fig.add_trace(go.Scatter(
        x=[0, 0, 6, 6],
        y=[0, 1.5, 1.5, 0.2],
        mode='lines',
        line=dict(color=shortcut_color, width=4),
        fill='none',
        hoverinfo='skip'
    ))
    
    # 跳跃连接箭头
    fig.add_annotation(x=6, y=0.2, ax=6, ay=1.3,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3,
                       arrowcolor=shortcut_color)
    
    # 添加标签
    fig.add_annotation(x=3, y=-0.9, text='残差路径 F(x)', showarrow=False,
                       font=dict(size=14, color=APPLE_PURPLE, family='Arial'))
    fig.add_annotation(x=3, y=1.75, text='跳跃连接 (Identity)', showarrow=False,
                       font=dict(size=14, color=shortcut_color, family='Arial'))
    
    fig.update_layout(
        title=dict(
            text='残差块 (Residual Block) 结构',
            font=dict(size=20, family='Arial, sans-serif')
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        showlegend=False,
        width=900,
        height=480,
        margin=dict(l=60, r=60, t=80, b=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, 
                   range=[-1, 9.5], fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, 
                   range=[-1.8, 2.2], fixedrange=True),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_gradient_flow():
    """
    绘制梯度流动对比图
    """
    # 模拟梯度随层数的变化
    layers = np.arange(1, 51)
    
    # 传统网络：梯度消失
    plain_grad = np.exp(-layers / 10) * 100
    
    # ResNet：梯度保持
    resnet_grad = 80 + 20 * np.exp(-layers / 30) + np.random.randn(50) * 2
    
    fig = go.Figure()
    
    # 传统网络梯度
    fig.add_trace(go.Scatter(
        x=layers, y=plain_grad,
        mode='lines',
        name='传统网络',
        line=dict(color=APPLE_RED, width=3),
        fill='tozeroy',
        fillcolor='rgba(255,59,48,0.1)'
    ))
    
    # ResNet 梯度
    fig.add_trace(go.Scatter(
        x=layers, y=resnet_grad,
        mode='lines',
        name='ResNet',
        line=dict(color=APPLE_GREEN, width=3),
        fill='tozeroy',
        fillcolor='rgba(52,199,89,0.1)'
    ))
    
    # 添加梯度消失区域标注
    fig.add_annotation(
        x=40, y=2,
        text='梯度消失',
        showarrow=True,
        arrowhead=2,
        arrowcolor=APPLE_RED,
        ax=0, ay=-30,
        font=dict(size=12, color=APPLE_RED),
        bgcolor='rgba(255,255,255,0.8)'
    )
    
    # 添加梯度高速公路标注
    fig.add_annotation(
        x=25, y=90,
        text='梯度高速公路',
        showarrow=True,
        arrowhead=2,
        arrowcolor=APPLE_GREEN,
        ax=0, ay=-40,
        font=dict(size=12, color=APPLE_GREEN),
        bgcolor='rgba(255,255,255,0.8)'
    )
    
    fig.update_layout(
        title=dict(
            text='反向传播梯度流动对比',
            font=dict(size=18, family='Arial, sans-serif')
        ),
        xaxis_title='网络层数（从输入到输出）',
        yaxis_title='梯度大小（相对值）',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14),
        legend=dict(
            x=0.65, y=0.95,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=APPLE_GRAY,
            borderwidth=1
        ),
        width=900,
        height=500,
        margin=dict(l=80, r=50, t=80, b=60)
    )
    
    return fig

def plot_imagenet_results():
    """
    绘制 ImageNet 分类结果对比
    """
    models = ['VGG-16', 'VGG-19', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152']
    top1_error = [28.07, 27.44, 26.73, 24.01, 22.44, 21.31]
    top5_error = [9.93, 9.05, 8.74, 7.02, 6.21, 5.54]
    params = [138, 144, 22, 26, 45, 60]  # 百万参数
    
    fig = go.Figure()
    
    # Top-1 错误率
    fig.add_trace(go.Bar(
        x=models,
        y=top1_error,
        name='Top-1 错误率 (%)',
        marker_color=[APPLE_RED, APPLE_RED, APPLE_TEAL, APPLE_BLUE, APPLE_BLUE, APPLE_GREEN],
        text=[f'{v:.2f}%' for v in top1_error],
        textposition='outside',
        textfont=dict(size=11)
    ))
    
    # Top-5 错误率（次坐标轴）
    fig.add_trace(go.Scatter(
        x=models,
        y=top5_error,
        name='Top-5 错误率 (%)',
        mode='lines+markers',
        line=dict(color=APPLE_ORANGE, width=2),
        marker=dict(size=10, symbol='diamond'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=dict(
            text='ImageNet 验证集错误率对比',
            font=dict(size=18, family='Arial, sans-serif')
        ),
        xaxis_title='模型',
        yaxis_title='Top-1 错误率 (%)',
        yaxis2=dict(
            title='Top-5 错误率 (%)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=13),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=APPLE_GRAY,
            borderwidth=1
        ),
        width=950,
        height=500,
        margin=dict(l=80, r=80, t=80, b=80),
        barmode='group'
    )
    
    # 添加注释说明参数量
    annotations = [
        dict(x='VGG-16', y=28.5, text='138M 参数', showarrow=False, font=dict(size=10, color=APPLE_GRAY)),
        dict(x='ResNet-34', y=27, text='22M 参数', showarrow=False, font=dict(size=10, color=APPLE_GRAY)),
        dict(x='ResNet-152', y=21.8, text='60M 参数', showarrow=False, font=dict(size=10, color=APPLE_GRAY)),
    ]
    fig.update_layout(annotations=annotations)
    
    return fig

def main():
    """主函数：生成所有图表"""
    output_dir = 'static/images/plots'
    
    print("开始生成 ResNet 论文配图...")
    print("=" * 50)
    
    # 1. 退化问题图
    print("\n生成 1/4: 退化问题示意图...")
    fig1 = plot_degradation_problem()
    save_and_compress(fig1, f'{output_dir}/resnet-degradation.png')
    
    # 2. 残差块结构图
    print("\n生成 2/4: 残差块结构图...")
    fig2 = plot_residual_block()
    save_and_compress(fig2, f'{output_dir}/resnet-block.png')
    
    # 3. 梯度流动对比图
    print("\n生成 3/4: 梯度流动对比图...")
    fig3 = plot_gradient_flow()
    save_and_compress(fig3, f'{output_dir}/resnet-gradient.png')
    
    # 4. ImageNet 结果对比图
    print("\n生成 4/4: ImageNet 结果对比图...")
    fig4 = plot_imagenet_results()
    save_and_compress(fig4, f'{output_dir}/resnet-imagenet-results.png')
    
    print("\n" + "=" * 50)
    print("✅ 所有图表生成完成！")
    print(f"输出目录: {output_dir}/")
    
    # 显示文件大小
    print("\n文件大小统计:")
    for fname in ['resnet-degradation.png', 'resnet-block.png', 'resnet-gradient.png', 'resnet-imagenet-results.png']:
        fpath = f'{output_dir}/{fname}'
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  {fname}: {size/1024:.1f} KB")

if __name__ == '__main__':
    main()
