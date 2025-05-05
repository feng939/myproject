"""
图像预处理模块

该模块提供了立体图像对的预处理和数据增强功能。
"""

import numpy as np
import cv2
import random
import torch


def normalize_image(img):
    """
    归一化图像到0-1范围
    
    参数:
        img (numpy.ndarray): 输入图像
        
    返回:
        numpy.ndarray: 归一化后的图像
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    else:
        return img.astype(np.float32)


def random_color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    随机颜色抖动
    
    参数:
        img (numpy.ndarray): 输入图像
        brightness (float): 亮度抖动范围
        contrast (float): 对比度抖动范围
        saturation (float): 饱和度抖动范围
        hue (float): 色调抖动范围
        
    返回:
        numpy.ndarray: 颜色抖动后的图像
    """
    # 转换为HSV颜色空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # 亮度抖动
    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        img_hsv[:, :, 2] = img_hsv[:, :, 2] * brightness_factor
    
    # 对比度抖动
    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        img_hsv[:, :, 2] = (img_hsv[:, :, 2] - 128) * contrast_factor + 128
    
    # 饱和度抖动
    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * saturation_factor
    
    # 色调抖动
    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_factor * 180) % 180
    
    # 裁剪值范围
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 179)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)
    
    # 转换回RGB颜色空间
    img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return img_rgb


def augment_stereo_pair(left_img, right_img, disparity=None):
    """
    对立体图像对进行数据增强
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        disparity (numpy.ndarray, optional): 视差图
        
    返回:
        tuple: (左图像, 右图像, 视差图)
    """
    # 检查图像格式并进行归一化
    if left_img.max() > 1.0:
        left_img = normalize_image(left_img)
        right_img = normalize_image(right_img)
    
    # 随机颜色抖动（分别对左右图像应用不同的抖动）
    if random.random() < 0.5:
        left_img = random_color_jitter(left_img)
        right_img = random_color_jitter(right_img)
    
    # 简化处理，只保留必要的增强
    # 不再需要复杂的变换，因为我们的重点是降低内存使用
    
    return left_img, right_img, disparity


def preprocess_for_prediction(left_img, right_img, target_size=None):
    """
    预处理用于预测的图像对
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        target_size (tuple, optional): 目标大小 (height, width)
        
    返回:
        tuple: (左图像张量, 右图像张量, 原始大小)
    """
    # 保存原始大小
    original_size = left_img.shape[:2]
    
    # 调整大小（如果需要）
    if target_size is not None:
        left_img = cv2.resize(left_img, (target_size[1], target_size[0]))
        right_img = cv2.resize(right_img, (target_size[1], target_size[0]))
    
    # 归一化
    left_img = normalize_image(left_img)
    right_img = normalize_image(right_img)
    
    # 转换为张量 [C, H, W]
    left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).float()
    right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).float()
    
    # 标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    left_tensor = (left_tensor - mean) / std
    right_tensor = (right_tensor - mean) / std
    
    return left_tensor, right_tensor, original_size


def postprocess_disparity(disparity, original_size=None):
    """
    对预测的视差图进行后处理
    
    参数:
        disparity (torch.Tensor): 预测的视差图
        original_size (tuple, optional): 原始图像大小 (height, width)
        
    返回:
        numpy.ndarray: 后处理的视差图
    """
    # 转换为NumPy数组
    if isinstance(disparity, torch.Tensor):
        disparity = disparity.detach().cpu().numpy()
    
    # 如果是批量数据，取第一个
    if len(disparity.shape) == 4:
        disparity = disparity[0]
    
    # 如果有通道维度，去除它
    if len(disparity.shape) == 3 and disparity.shape[0] == 1:
        disparity = disparity[0]
    
    # 调整大小为原始尺寸
    if original_size is not None:
        h, w = disparity.shape
        orig_h, orig_w = original_size
        
        # 计算缩放因子
        scale_w = orig_w / w
        
        # 调整视差图大小
        disparity = cv2.resize(disparity, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # 调整视差值
        disparity = disparity * scale_w
    
    return disparity