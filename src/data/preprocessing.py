"""
图像预处理模块

该模块提供了立体图像对的预处理和数据增强功能。
"""

import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F


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


def denormalize_image(img):
    """
    反归一化图像到0-255范围
    
    参数:
        img (numpy.ndarray or torch.Tensor): 输入图像
        
    返回:
        numpy.ndarray: 反归一化后的图像
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    if img.max() <= 1.0:
        img = img * 255.0
    
    return np.clip(img, 0, 255).astype(np.uint8)


def resize_image_pair(left_img, right_img, target_size, disparity=None):
    """
    调整立体图像对的大小
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        target_size (tuple): 目标大小 (height, width)
        disparity (numpy.ndarray, optional): 视差图
        
    返回:
        tuple: (左图像, 右图像, 视差图)
    """
    # 获取原始大小
    h, w = left_img.shape[:2]
    target_h, target_w = target_size
    
    # 计算调整系数
    scale_h = target_h / h
    scale_w = target_w / w
    
    # 调整图像大小
    left_img_resized = cv2.resize(left_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    right_img_resized = cv2.resize(right_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # 调整视差图大小
    disparity_resized = None
    if disparity is not None:
        # 视差值需要按宽度比例调整
        disparity_resized = cv2.resize(disparity, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        disparity_resized = disparity_resized * scale_w
    
    return left_img_resized, right_img_resized, disparity_resized


def pad_image_pair(left_img, right_img, pad_size, disparity=None):
    """
    对立体图像对进行填充
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        pad_size (tuple): 填充大小 (top, bottom, left, right)
        disparity (numpy.ndarray, optional): 视差图
        
    返回:
        tuple: (左图像, 右图像, 视差图)
    """
    top, bottom, left, right = pad_size
    
    # 填充图像
    left_img_padded = cv2.copyMakeBorder(left_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    right_img_padded = cv2.copyMakeBorder(right_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    # 填充视差图
    disparity_padded = None
    if disparity is not None:
        disparity_padded = cv2.copyMakeBorder(disparity, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    return left_img_padded, right_img_padded, disparity_padded


def random_crop_image_pair(left_img, right_img, crop_size, disparity=None):
    """
    随机裁剪立体图像对
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        crop_size (tuple): 裁剪大小 (height, width)
        disparity (numpy.ndarray, optional): 视差图
        
    返回:
        tuple: (左图像, 右图像, 视差图)
    """
    h, w = left_img.shape[:2]
    crop_h, crop_w = crop_size
    
    # 确保裁剪大小不超过图像大小
    crop_h = min(h, crop_h)
    crop_w = min(w, crop_w)
    
    # 随机选择裁剪位置
    start_h = random.randint(0, h - crop_h)
    start_w = random.randint(0, w - crop_w)
    
    # 裁剪图像
    left_img_cropped = left_img[start_h:start_h+crop_h, start_w:start_w+crop_w]
    right_img_cropped = right_img[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    # 裁剪视差图
    disparity_cropped = None
    if disparity is not None:
        disparity_cropped = disparity[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    return left_img_cropped, right_img_cropped, disparity_cropped


def random_flip_image_pair(left_img, right_img, disparity=None, p=0.5):
    """
    随机水平翻转立体图像对
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        disparity (numpy.ndarray, optional): 视差图
        p (float): 翻转概率
        
    返回:
        tuple: (左图像, 右图像, 视差图)
    """
    if random.random() < p:
        # 交换左右图像
        left_img, right_img = right_img, left_img
        
        # 翻转图像
        left_img = cv2.flip(left_img, 1)
        right_img = cv2.flip(right_img, 1)
        
        # 处理视差图
        if disparity is not None:
            # 翻转视差图
            disparity = cv2.flip(disparity, 1)
            # 反转视差值
            disparity = -disparity
    
    return left_img, right_img, disparity


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


def random_gaussian_blur(img, kernel_size=5, sigma=1.5, p=0.5):
    """
    随机高斯模糊
    
    参数:
        img (numpy.ndarray): 输入图像
        kernel_size (int): 高斯核大小
        sigma (float): 高斯核标准差
        p (float): 应用模糊的概率
        
    返回:
        numpy.ndarray: 模糊后的图像
    """
    if random.random() < p:
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return img


def random_noise(img, noise_type='gaussian', mean=0, std=0.01, p=0.5):
    """
    添加随机噪声
    
    参数:
        img (numpy.ndarray): 输入图像
        noise_type (str): 噪声类型，'gaussian' 或 'salt_pepper'
        mean (float): 高斯噪声均值
        std (float): 高斯噪声标准差
        p (float): 应用噪声的概率
        
    返回:
        numpy.ndarray: 添加噪声后的图像
    """
    if random.random() < p:
        if noise_type == 'gaussian':
            # 高斯噪声
            noise = np.random.normal(mean, std, img.shape)
            noisy_img = img + noise
            return np.clip(noisy_img, 0, 1.0 if img.max() <= 1.0 else 255).astype(img.dtype)
        
        elif noise_type == 'salt_pepper':
            # 椒盐噪声
            s_vs_p = 0.5  # 盐噪声比例
            amount = 0.004  # 噪声总量
            noisy_img = img.copy()
            
            # 盐噪声
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
            noisy_img[coords[0], coords[1], :] = 1.0 if img.max() <= 1.0 else 255
            
            # 椒噪声
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
            noisy_img[coords[0], coords[1], :] = 0
            
            return noisy_img
    
    return img


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
    left_img = random_color_jitter(left_img)
    right_img = random_color_jitter(right_img)
    
    # 随机高斯模糊
    if random.random() < 0.5:
        # 对左右图像应用相同的模糊
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.1, 1.5)
        left_img = random_gaussian_blur(left_img, kernel_size, sigma, p=1.0)
        right_img = random_gaussian_blur(right_img, kernel_size, sigma, p=1.0)
    
    # 随机噪声
    if random.random() < 0.5:
        # 对左右图像应用不同的噪声
        noise_type = random.choice(['gaussian', 'salt_pepper'])
        left_img = random_noise(left_img, noise_type=noise_type, p=1.0)
        right_img = random_noise(right_img, noise_type=noise_type, p=1.0)
    
    # 随机水平翻转
    # 注意：对于立体匹配，水平翻转需要特殊处理
    if random.random() < 0.5:
        left_img, right_img, disparity = random_flip_image_pair(left_img, right_img, disparity)
    
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
        left_img, right_img, _ = resize_image_pair(left_img, right_img, target_size)
    
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
        disparity (torch.Tensor): 预测的视差图，形状为 [B, 1, H, W]
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
    
    # 填充无效区域
    invalid_mask = disparity <= 0
    if np.any(invalid_mask):
        # 使用中值滤波填充无效区域
        valid_disp = disparity.copy()
        valid_disp[invalid_mask] = np.median(disparity[~invalid_mask])
        disparity = cv2.medianBlur(valid_disp, 5)
    
    return disparity


def generate_disparity_colormap(disparity, max_disp=None):
    """
    生成视差图的彩色映射
    
    参数:
        disparity (numpy.ndarray): 视差图
        max_disp (float, optional): 最大视差值
        
    返回:
        numpy.ndarray: 彩色视差图
    """
    # 确定最大视差值
    if max_disp is None:
        max_disp = np.max(disparity) if np.max(disparity) > 0 else 1.0
    
    # 归一化视差值到0-1范围
    normalized_disp = np.clip(disparity / max_disp, 0, 1)
    
    # 使用jet颜色映射
    colormap = cv2.applyColorMap((normalized_disp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 设置无效区域为黑色
    invalid_mask = disparity <= 0
    colormap[invalid_mask] = [0, 0, 0]
    
    return colormap


def compute_stereo_confidence(left_img, right_img, disparity, window_size=5):
    """
    计算立体匹配的置信度
    
    参数:
        left_img (numpy.ndarray): 左图像
        right_img (numpy.ndarray): 右图像
        disparity (numpy.ndarray): 视差图
        window_size (int): 窗口大小
        
    返回:
        numpy.ndarray: 置信度图
    """
    h, w = disparity.shape[:2]
    confidence = np.zeros((h, w), dtype=np.float32)
    
    # 确保图像已归一化
    if left_img.max() > 1.0:
        left_img = normalize_image(left_img)
        right_img = normalize_image(right_img)
    
    # 将图像转换为灰度图
    if len(left_img.shape) == 3 and left_img.shape[2] == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img
    
    # 计算每个像素的置信度
    half_window = window_size // 2
    
    for y in range(half_window, h - half_window):
        for x in range(half_window, w - half_window):
            d = disparity[y, x]
            
            # 跳过无效视差
            if d <= 0:
                continue
            
            # 计算对应点在右图中的位置
            x_right = int(x - d)
            
            # 如果对应点超出图像边界，置信度为0
            if x_right < half_window:
                confidence[y, x] = 0
                continue
            
            # 提取左右图像的窗口
            left_window = left_gray[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
            right_window = right_gray[y-half_window:y+half_window+1, x_right-half_window:x_right+half_window+1]
            
            # 计算归一化交叉相关 (NCC)
            left_mean = np.mean(left_window)
            right_mean = np.mean(right_window)
            
            left_std = np.std(left_window)
            right_std = np.std(right_window)
            
            if left_std == 0 or right_std == 0:
                confidence[y, x] = 0
                continue
            
            ncc = np.mean((left_window - left_mean) * (right_window - right_mean)) / (left_std * right_std)
            
            # 将NCC值映射到[0, 1]范围，作为置信度
            confidence[y, x] = (ncc + 1) / 2
    
    return confidence