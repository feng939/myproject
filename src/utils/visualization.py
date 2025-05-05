"""
可视化工具模块

该模块提供了用于可视化视差图和立体匹配结果的工具。
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm
import torch
from PIL import Image


def colorize_disparity(disparity, colormap=cv2.COLORMAP_JET, invalid_color=(0, 0, 0)):
    """
    将视差图转换为彩色图像
    
    参数:
        disparity (numpy.ndarray): 视差图
        colormap: 颜色映射，默认为JET
        invalid_color (tuple): 无效区域的颜色
        
    返回:
        numpy.ndarray: 彩色视差图，形状为 [H, W, 3]，RGB格式
    """
    # 转换为NumPy数组
    if isinstance(disparity, torch.Tensor):
        disparity = disparity.detach().cpu().numpy()
    
    # 如果视差图是多维的，取第一个通道
    if len(disparity.shape) > 2:
        disparity = disparity.squeeze()
    
    # 创建无效区域掩码
    invalid_mask = (disparity <= 0)
    
    # 确定视差范围
    vmin = disparity[~invalid_mask].min() if np.any(~invalid_mask) else 0
    vmax = disparity[~invalid_mask].max() if np.any(~invalid_mask) else 1
    
    # 归一化视差到0-1范围
    if vmax > vmin:
        normalized_disp = (disparity - vmin) / (vmax - vmin)
    else:
        normalized_disp = np.zeros_like(disparity)
    
    # 设置无效区域为0
    normalized_disp[invalid_mask] = 0
    
    # 转换为0-255范围
    disp_uint8 = (normalized_disp * 255).astype(np.uint8)
    
    # 应用颜色映射
    colored_disp = cv2.applyColorMap(disp_uint8, colormap)
    
    # 转换为RGB格式
    colored_disp = cv2.cvtColor(colored_disp, cv2.COLOR_BGR2RGB)
    
    # 设置无效区域的颜色
    if invalid_color is not None:
        colored_disp[invalid_mask] = invalid_color
    
    return colored_disp


def save_disparity_image(disparity, output_path, colormap=cv2.COLORMAP_JET, invalid_color=(0, 0, 0)):
    """
    保存视差图为彩色图像
    
    参数:
        disparity (numpy.ndarray): 视差图
        output_path (str): 输出路径
        colormap: 颜色映射，默认为JET
        invalid_color (tuple): 无效区域的颜色
    """
    # 生成彩色视差图
    colored_disp = colorize_disparity(disparity, colormap, invalid_color)
    
    # 转换为BGR格式
    bgr_disp = cv2.cvtColor(colored_disp, cv2.COLOR_RGB2BGR)
    
    # 保存图像
    cv2.imwrite(output_path, bgr_disp)


def create_disparity_visualization(left_img, right_img, disparity, gt_disparity=None, error=None):
    """
    创建视差可视化图像
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        disparity (numpy.ndarray): 预测视差图
        gt_disparity (numpy.ndarray, optional): 真值视差图
        error (numpy.ndarray, optional): 误差图
        
    返回:
        numpy.ndarray: 包含所有可视化的图像，RGB格式
    """
    # 确保所有输入都是NumPy数组
    if isinstance(left_img, torch.Tensor):
        left_img = left_img.detach().cpu().numpy()
    if isinstance(right_img, torch.Tensor):
        right_img = right_img.detach().cpu().numpy()
    if isinstance(disparity, torch.Tensor):
        disparity = disparity.detach().cpu().numpy()
    if gt_disparity is not None and isinstance(gt_disparity, torch.Tensor):
        gt_disparity = gt_disparity.detach().cpu().numpy()
    
    # 转换图像格式
    if left_img.max() <= 1.0:
        left_img = (left_img * 255).astype(np.uint8)
    if right_img.max() <= 1.0:
        right_img = (right_img * 255).astype(np.uint8)
    
    # 对于多通道图像，转换为RGB
    if len(left_img.shape) > 2 and left_img.shape[0] == 3:  # [C, H, W] 格式
        left_img = np.transpose(left_img, (1, 2, 0))
    if len(right_img.shape) > 2 and right_img.shape[0] == 3:
        right_img = np.transpose(right_img, (1, 2, 0))
    
    # 生成彩色视差图
    colored_disp = colorize_disparity(disparity)
    
    # 生成真值视差图（如果有）
    if gt_disparity is not None:
        colored_gt = colorize_disparity(gt_disparity)
    else:
        colored_gt = np.zeros_like(colored_disp)
    
    # 生成误差图（如果有）
    if error is not None:
        # 使用热图显示误差
        colored_error = colorize_disparity(error, colormap=cv2.COLORMAP_HOT)
    else:
        colored_error = np.zeros_like(colored_disp)
    
    # 确保所有图像大小一致
    h, w = colored_disp.shape[:2]
    left_img_resized = cv2.resize(left_img, (w, h))
    right_img_resized = cv2.resize(right_img, (w, h))
    
    # 组合图像
    if gt_disparity is not None and error is not None:
        # 2x3布局：左图、右图、预测视差、真值视差、误差
        vis_img = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        
        # 第一行
        vis_img[:h, :w] = left_img_resized
        vis_img[:h, w:2*w] = right_img_resized
        vis_img[:h, 2*w:] = colored_disp
        
        # 第二行
        vis_img[h:, :w] = colored_gt
        vis_img[h:, w:2*w] = colored_error
        # 最后一个位置可以留空或放置其他信息
    
    elif gt_disparity is not None:
        # 2x2布局：左图、右图、预测视差、真值视差
        vis_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # 第一行
        vis_img[:h, :w] = left_img_resized
        vis_img[:h, w:] = right_img_resized
        
        # 第二行
        vis_img[h:, :w] = colored_disp
        vis_img[h:, w:] = colored_gt
    
    else:
        # 1x3布局：左图、右图、预测视差
        vis_img = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        vis_img[:, :w] = left_img_resized
        vis_img[:, w:2*w] = right_img_resized
        vis_img[:, 2*w:] = colored_disp
    
    return vis_img


def create_anaglyph(left_img, right_img):
    """
    创建红青立体图（anaglyph）
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        
    返回:
        numpy.ndarray: 红青立体图，RGB格式
    """
    # 确保所有输入都是NumPy数组
    if isinstance(left_img, torch.Tensor):
        left_img = left_img.detach().cpu().numpy()
    if isinstance(right_img, torch.Tensor):
        right_img = right_img.detach().cpu().numpy()
    
    # 转换图像格式
    if left_img.max() <= 1.0:
        left_img = (left_img * 255).astype(np.uint8)
    if right_img.max() <= 1.0:
        right_img = (right_img * 255).astype(np.uint8)
    
    # 对于多通道图像，转换为BGR
    if len(left_img.shape) < 3:
        left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    elif left_img.shape[0] == 3:  # [C, H, W] 格式
        left_img = np.transpose(left_img, (1, 2, 0))
        
    if len(right_img.shape) < 3:
        right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    elif right_img.shape[0] == 3:  # [C, H, W] 格式
        right_img = np.transpose(right_img, (1, 2, 0))
    
    # 确保图像是BGR格式
    if left_img.shape[2] != 3:
        left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    if right_img.shape[2] != 3:
        right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    
    # 转换为RGB
    left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    
    # 创建红青立体图
    anaglyph = np.zeros_like(left_img_rgb)
    
    # 左图提取红色通道
    anaglyph[:, :, 0] = left_img_rgb[:, :, 0]
    
    # 右图提取青色部分（绿色和蓝色通道）
    anaglyph[:, :, 1] = right_img_rgb[:, :, 1]
    anaglyph[:, :, 2] = right_img_rgb[:, :, 2]
    
    return anaglyph


def create_disparity_map_3d(disparity, left_img=None, scale_factor=2.0):
    """
    创建视差图的3D表示
    
    参数:
        disparity (numpy.ndarray): 视差图
        left_img (numpy.ndarray, optional): 左图像，用于纹理
        scale_factor (float): 深度缩放因子
        
    返回:
        tuple: (fig, ax) matplotlib图形对象
    """
    # 确保所有输入都是NumPy数组
    if isinstance(disparity, torch.Tensor):
        disparity = disparity.detach().cpu().numpy()
    if left_img is not None and isinstance(left_img, torch.Tensor):
        left_img = left_img.detach().cpu().numpy()
    
    # 如果视差图是多维的，取第一个通道
    if len(disparity.shape) > 2:
        disparity = disparity.squeeze()
    
    # 创建网格
    h, w = disparity.shape
    y, x = np.mgrid[0:h, 0:w]
    
    # 计算Z坐标（深度）
    z = disparity * scale_factor
    
    # 创建3D图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 如果有左图像，使用它作为颜色映射
    if left_img is not None:
        # 确保左图像大小与视差图一致
        if left_img.shape[:2] != disparity.shape:
            left_img = cv2.resize(left_img, (w, h))
        
        # 转换为RGB
        if len(left_img.shape) < 3:
            # 灰度图
            colors = plt.cm.gray(left_img.flatten() / 255.0)
        else:
            # 多通道图像
            if left_img.shape[0] == 3:  # [C, H, W] 格式
                left_img = np.transpose(left_img, (1, 2, 0))
            
            # 归一化到[0, 1]
            if left_img.max() > 1.0:
                left_img = left_img / 255.0
            
            colors = left_img.reshape(-1, 3)
    else:
        # 使用视差值作为颜色
        norm_disparity = disparity - disparity.min()
        if norm_disparity.max() > 0:
            norm_disparity = norm_disparity / norm_disparity.max()
        colors = plt.cm.viridis(norm_disparity.flatten())
    
    # 绘制3D表面
    surf = ax.plot_surface(x, y, z, facecolors=colors.reshape(*disparity.shape, -1)[:, :, :3],
                         rstride=1, cstride=1, shade=False)
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    
    # 设置视角
    ax.view_init(elev=30, azim=-45)
    
    return fig, ax


def plot_cost_volume_slice(cost_volume, slice_idx, title=None):
    """
    绘制代价体的一个切片
    
    参数:
        cost_volume (numpy.ndarray): 代价体，形状为 [D, H, W] 或 [B, D, H, W]
        slice_idx (int): 要绘制的视差索引
        title (str, optional): 图表标题
        
    返回:
        matplotlib.figure.Figure: 图形对象
    """
    # 确保输入是NumPy数组
    if isinstance(cost_volume, torch.Tensor):
        cost_volume = cost_volume.detach().cpu().numpy()
    
    # 处理形状
    if len(cost_volume.shape) == 4:  # [B, D, H, W]
        cost_volume = cost_volume[0]  # 取第一个批次
    
    # 提取指定视差的切片
    cost_slice = cost_volume[slice_idx]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制代价切片
    im = ax.imshow(cost_slice, cmap='viridis')
    fig.colorbar(im, ax=ax, label='Cost')
    
    # 设置标题
    if title is None:
        title = f'Cost Volume Slice at Disparity = {slice_idx}'
    ax.set_title(title)
    
    # 设置坐标轴标签
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    # 紧凑布局
    plt.tight_layout()
    
    return fig


def visualize_matching_cost(left_img, right_img, y_coord):
    """
    可视化立体匹配成本
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        y_coord (int): 要分析的行坐标
        
    返回:
        tuple: (fig, axes) matplotlib图形对象
    """
    # 确保所有输入都是NumPy数组
    if isinstance(left_img, torch.Tensor):
        left_img = left_img.detach().cpu().numpy()
    if isinstance(right_img, torch.Tensor):
        right_img = right_img.detach().cpu().numpy()
    
    # 转换图像格式
    if left_img.max() <= 1.0:
        left_img = (left_img * 255).astype(np.uint8)
    if right_img.max() <= 1.0:
        right_img = (right_img * 255).astype(np.uint8)
    
    # 对于多通道图像，转换为RGB
    if len(left_img.shape) > 2 and left_img.shape[0] == 3:  # [C, H, W] 格式
        left_img = np.transpose(left_img, (1, 2, 0))
    if len(right_img.shape) > 2 and right_img.shape[0] == 3:
        right_img = np.transpose(right_img, (1, 2, 0))
    
    # 转换为灰度图
    if len(left_img.shape) > 2:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img
    
    # 获取图像尺寸
    h, w = left_gray.shape
    y_coord = min(max(0, y_coord), h - 1)
    
    # 提取指定行
    left_row = left_gray[y_coord, :]
    right_row = right_gray[y_coord, :]
    
    # 计算简单的匹配成本（SSD）
    max_disp = min(w // 2, 64)
    cost = np.zeros((w, max_disp))
    
    for x in range(w):
        for d in range(min(max_disp, x + 1)):
            # 左边图像的当前像素与右边图像的偏移像素的差异
            cost[x, d] = (int(left_row[x]) - int(right_row[x - d])) ** 2
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1, 2]})
    
    # 绘制左图像行
    axes[0].plot(left_row, 'b-', linewidth=2)
    axes[0].set_title(f'Left Image Row at y={y_coord}')
    axes[0].set_xlim(0, w - 1)
    axes[0].set_ylabel('Intensity')
    
    # 绘制右图像行
    axes[1].plot(right_row, 'r-', linewidth=2)
    axes[1].set_title(f'Right Image Row at y={y_coord}')
    axes[1].set_xlim(0, w - 1)
    axes[1].set_ylabel('Intensity')
    
    # 绘制匹配成本
    im = axes[2].imshow(cost.T, cmap='viridis', aspect='auto', 
                      extent=[0, w - 1, max_disp - 1, 0])
    axes[2].set_title('Matching Cost')
    axes[2].set_xlabel('X Coordinate')
    axes[2].set_ylabel('Disparity')
    fig.colorbar(im, ax=axes[2], label='SSD Cost')
    
    # 紧凑布局
    plt.tight_layout()
    
    return fig, axes


def visualize_belief_propagation(disparity, iterations=5, step=1):
    """
    可视化信念传播过程
    
    参数:
        disparity (list): 每次迭代的视差结果列表
        iterations (int): 迭代次数
        step (int): 展示迭代的步长
        
    返回:
        matplotlib.figure.Figure: 图形对象
    """
    # 确定迭代轮次
    n_iter = min(iterations, len(disparity))
    
    # 计算子图行列数
    cols = min(5, n_iter)
    rows = (n_iter + cols - 1) // cols
    
    # 创建图表
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    # 展平轴数组（如果只有一行或一列）
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    if rows == 1 or cols == 1:
        axes = axes.flatten()
    
    # 绘制每次迭代的视差图
    for i in range(n_iter):
        # 确定子图位置
        if rows > 1 and cols > 1:
            ax = axes[i // cols, i % cols]
        else:
            ax = axes[i]
        
        # 获取当前视差图
        disp = disparity[i * step]
        
        # 绘制视差图
        if isinstance(disp, torch.Tensor):
            disp = disp.detach().cpu().numpy()
        
        if len(disp.shape) > 2:
            disp = disp.squeeze()
        
        im = ax.imshow(disp, cmap='viridis')
        ax.set_title(f'Iteration {i * step}')
        ax.axis('off')
    
    # 隐藏空白子图
    for i in range(n_iter, rows * cols):
        if rows > 1 and cols > 1:
            axes[i // cols, i % cols].axis('off')
        else:
            axes[i].axis('off')
    
    # 添加颜色条
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Disparity')
    
    return fig


def plot_convergence(loss_history, metric_history=None):
    """
    绘制训练收敛曲线
    
    参数:
        loss_history (list): 损失历史记录
        metric_history (dict, optional): 指标历史记录
        
    返回:
        matplotlib.figure.Figure: 图形对象
    """
    # 创建图表
    if metric_history is not None:
        fig, axes = plt.subplots(1 + len(metric_history), 1, figsize=(10, 4 * (1 + len(metric_history))), sharex=True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        axes = [axes]
    
    # 绘制损失曲线
    epochs = range(1, len(loss_history) + 1)
    axes[0].plot(epochs, loss_history, 'b-', linewidth=2)
    axes[0].set_title('Training Loss')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # 绘制指标曲线（如果有）
    if metric_history is not None:
        for i, (metric_name, metric_values) in enumerate(metric_history.items(), 1):
            axes[i].plot(epochs, metric_values, 'r-', linewidth=2)
            axes[i].set_title(f'Metric: {metric_name}')
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True)
    
    # 设置X轴标签
    axes[-1].set_xlabel('Epochs')
    
    # 紧凑布局
    plt.tight_layout()
    
    return fig