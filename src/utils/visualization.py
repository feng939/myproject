"""
可视化工具模块

该模块提供了用于可视化视差图和立体匹配结果的工具。
"""

import numpy as np
import cv2
import torch


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


def create_disparity_visualization(left_img, right_img, disparity, gt_disparity=None):
    """
    创建视差可视化图像
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        disparity (numpy.ndarray): 预测视差图
        gt_disparity (numpy.ndarray, optional): 真值视差图
        
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
    
    # 确保所有图像大小一致
    h, w = colored_disp.shape[:2]
    left_img_resized = cv2.resize(left_img, (w, h))
    right_img_resized = cv2.resize(right_img, (w, h))
    
    # 组合图像
    if gt_disparity is not None:
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
    if len(left_img.shape) > 2:
        if left_img.shape[0] == 3:  # [C, H, W] 格式
            left_img = np.transpose(left_img, (1, 2, 0))
        
        if len(left_img.shape) == 3 and left_img.shape[2] == 3:
            left_img_bgr = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
        else:
            left_img_bgr = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    else:
        left_img_bgr = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        
    if len(right_img.shape) > 2:
        if right_img.shape[0] == 3:  # [C, H, W] 格式
            right_img = np.transpose(right_img, (1, 2, 0))
        
        if len(right_img.shape) == 3 and right_img.shape[2] == 3:
            right_img_bgr = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)
        else:
            right_img_bgr = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    else:
        right_img_bgr = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    
    # 创建红青立体图
    anaglyph = np.zeros_like(left_img_bgr)
    
    # 左图提取红色通道
    anaglyph[:, :, 2] = left_img_bgr[:, :, 2]
    
    # 右图提取青色部分（绿色和蓝色通道）
    anaglyph[:, :, 0] = right_img_bgr[:, :, 0]
    anaglyph[:, :, 1] = right_img_bgr[:, :, 1]
    
    # 转换为RGB
    anaglyph_rgb = cv2.cvtColor(anaglyph, cv2.COLOR_BGR2RGB)
    
    return anaglyph_rgb