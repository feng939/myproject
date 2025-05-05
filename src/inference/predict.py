"""
模型推理模块

该模块提供了使用训练好的BPNN模型进行视差预测的功能。
"""

import os
import torch
import numpy as np
import time
import cv2
from PIL import Image

from ..data.preprocessing import preprocess_for_prediction, postprocess_disparity
from ..utils.visualization import colorize_disparity


class DisparityPredictor:
    """
    视差预测器
    
    该类使用训练好的模型进行视差预测。
    """
    
    def __init__(self, model, device=None, target_size=None):
        """
        初始化视差预测器
        
        参数:
            model (torch.nn.Module): 训练好的模型
            device (torch.device, optional): 运行设备，默认为GPU（如果可用）
            target_size (tuple, optional): 目标图像大小 (height, width)
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 加载模型
        self.model = model.to(self.device)
        self.model.eval()
        
        # 设置目标大小
        self.target_size = target_size
        
        # 检查模型是否支持半精度
        self.use_half_precision = hasattr(model, 'use_half_precision') and model.use_half_precision
    
    def _load_image(self, image_path):
        """
        加载图像
        
        参数:
            image_path (str): 图像路径
            
        返回:
            numpy.ndarray: 加载的图像
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 使用PIL加载图像
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        
        return img
    
    def predict(self, left_img, right_img):
        """
        预测视差图
        
        参数:
            left_img (numpy.ndarray or str): 左图像或图像路径
            right_img (numpy.ndarray or str): 右图像或图像路径
            
        返回:
            dict: 包含视差图和处理时间的字典
        """
        # 加载图像（如果输入是路径）
        if isinstance(left_img, str):
            left_img = self._load_image(left_img)
        if isinstance(right_img, str):
            right_img = self._load_image(right_img)
        
        # 预处理图像
        start_time = time.time()
        left_tensor, right_tensor, original_size = preprocess_for_prediction(left_img, right_img, self.target_size)
        
        # 添加批次维度并移动到设备
        left_tensor = left_tensor.unsqueeze(0).to(self.device)
        right_tensor = right_tensor.unsqueeze(0).to(self.device)
        
        # 使用半精度（如果支持）
        if self.use_half_precision and self.device.type == 'cuda':
            left_tensor = left_tensor.half()
            right_tensor = right_tensor.half()
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(left_tensor, right_tensor)
        
        # 如果模型返回字典，提取视差图
        if isinstance(outputs, dict) and 'disparity' in outputs:
            disparity = outputs['disparity']
        else:
            disparity = outputs
        
        # 后处理视差图
        disparity = postprocess_disparity(disparity, original_size)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        return {
            'disparity': disparity,
            'processing_time': processing_time
        }
    
    def predict_and_save(self, left_path, right_path, output_path, colorize=True):
        """
        预测视差图并保存
        
        参数:
            left_path (str): 左图像路径
            right_path (str): 右图像路径
            output_path (str): 输出路径
            colorize (bool): 是否生成彩色视差图
            
        返回:
            dict: 包含视差图和处理时间的字典
        """
        # 预测视差图
        result = self.predict(left_path, right_path)
        disparity = result['disparity']
        
        # 保存视差图
        if output_path:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 默认保存为PNG
            if colorize:
                # 生成彩色视差图
                color_disparity = colorize_disparity(disparity)
                cv2.imwrite(output_path, cv2.cvtColor(color_disparity, cv2.COLOR_RGB2BGR))
            else:
                # 缩放视差值到0-65535（16位）
                max_disp = np.max(disparity) if np.max(disparity) > 0 else 1.0
                scaled_disp = np.clip(disparity / max_disp * 65535, 0, 65535).astype(np.uint16)
                cv2.imwrite(output_path, scaled_disp)
        
        return result