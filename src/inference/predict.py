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
    
    def _preprocess_images(self, left_img, right_img):
        """
        预处理图像
        
        参数:
            left_img (numpy.ndarray): 左图像
            right_img (numpy.ndarray): 右图像
            
        返回:
            tuple: (left_tensor, right_tensor, original_size)
        """
        return preprocess_for_prediction(left_img, right_img, self.target_size)
    
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
        left_tensor, right_tensor, original_size = self._preprocess_images(left_img, right_img)
        
        # 添加批次维度并移动到设备
        left_tensor = left_tensor.unsqueeze(0).to(self.device)
        right_tensor = right_tensor.unsqueeze(0).to(self.device)
        
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
            
            # 保存原始视差图
            if output_path.endswith('.pfm'):
                # 保存为PFM格式
                self._write_pfm(disparity, output_path)
            elif output_path.endswith('.npy'):
                # 保存为NumPy数组
                np.save(output_path, disparity)
            else:
                # 默认保存为PNG
                # 视差值通常需要缩放以适应8位或16位图像
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
    
    def process_video(self, left_video_path, right_video_path, output_video_path, fps=30):
        """
        处理立体视频
        
        参数:
            left_video_path (str): 左视频路径
            right_video_path (str): 右视频路径
            output_video_path (str): 输出视频路径
            fps (int): 输出视频帧率
            
        返回:
            float: 平均处理时间
        """
        # 打开左右视频
        left_cap = cv2.VideoCapture(left_video_path)
        right_cap = cv2.VideoCapture(right_video_path)
        
        # 检查视频是否成功打开
        if not left_cap.isOpened() or not right_cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        # 获取视频参数
        width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建输出视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), True)
        
        processing_times = []
        
        # 逐帧处理
        for _ in range(frame_count):
            # 读取左右帧
            left_ret, left_frame = left_cap.read()
            right_ret, right_frame = right_cap.read()
            
            if not left_ret or not right_ret:
                break
            
            # 转换BGR到RGB
            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
            
            # 预测视差
            result = self.predict(left_frame, right_frame)
            disparity = result['disparity']
            processing_time = result['processing_time']
            processing_times.append(processing_time)
            
            # 生成彩色视差图
            color_disparity = colorize_disparity(disparity)
            
            # 转换RGB到BGR并写入输出视频
            out_frame = cv2.cvtColor(color_disparity, cv2.COLOR_RGB2BGR)
            out.write(out_frame)
        
        # 释放资源
        left_cap.release()
        right_cap.release()
        out.release()
        
        # 计算平均处理时间
        avg_time = sum(processing_times) / len(processing_times)
        
        return avg_time
    
    def _write_pfm(self, data, path):
        """
        将视差图保存为PFM格式
        
        参数:
            data (numpy.ndarray): 视差图
            path (str): 输出路径
        """
        with open(path, 'wb') as f:
            # 写入头部信息
            f.write(b'Pf\n')
            f.write(f"{data.shape[1]} {data.shape[0]}\n".encode())
            
            # 确定字节序
            scale = -1.0 if data.dtype.byteorder == '<' or (data.dtype.byteorder == '=' and np.array([1.0], dtype=np.float32).tobytes()[0] == 0) else 1.0
            f.write(f"{scale}\n".encode())
            
            # 写入数据（按行倒序）
            data = data.astype(np.float32)
            data = np.flipud(data)
            data.tofile(f)


class BatchPredictor(DisparityPredictor):
    """
    批量视差预测器
    
    该类用于批量预测视差图。
    """
    
    def predict_batch(self, left_images, right_images, batch_size=4):
        """
        批量预测视差图
        
        参数:
            left_images (list): 左图像列表
            right_images (list): 右图像列表
            batch_size (int): 批大小
            
        返回:
            list: 视差图列表
        """
        num_images = len(left_images)
        disparities = []
        
        # 批量处理
        for i in range(0, num_images, batch_size):
            # 获取当前批次
            batch_left = left_images[i:i+batch_size]
            batch_right = right_images[i:i+batch_size]
            
            # 处理当前批次
            batch_disparities = self._process_batch(batch_left, batch_right)
            disparities.extend(batch_disparities)
        
        return disparities
    
    def _process_batch(self, batch_left, batch_right):
        """
        处理一个批次
        
        参数:
            batch_left (list): 左图像批次
            batch_right (list): 右图像批次
            
        返回:
            list: 视差图批次
        """
        batch_size = len(batch_left)
        batch_disparities = []
        original_sizes = []
        
        # 预处理所有图像
        left_tensors = []
        right_tensors = []
        
        for i in range(batch_size):
            # 加载图像（如果输入是路径）
            left_img = batch_left[i]
            right_img = batch_right[i]
            
            if isinstance(left_img, str):
                left_img = self._load_image(left_img)
            if isinstance(right_img, str):
                right_img = self._load_image(right_img)
            
            # 预处理图像
            left_tensor, right_tensor, original_size = self._preprocess_images(left_img, right_img)
            
            left_tensors.append(left_tensor)
            right_tensors.append(right_tensor)
            original_sizes.append(original_size)
        
        # 组合成批次
        left_batch = torch.stack(left_tensors).to(self.device)
        right_batch = torch.stack(right_tensors).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(left_batch, right_batch)
        
        # 提取视差图
        if isinstance(outputs, dict) and 'disparity' in outputs:
            batch_disparity = outputs['disparity']
        else:
            batch_disparity = outputs
        
        # 后处理每个视差图
        for i in range(batch_size):
            disparity = batch_disparity[i:i+1]  # 保持批次维度
            post_disparity = postprocess_disparity(disparity, original_sizes[i])
            batch_disparities.append(post_disparity)
        
        return batch_disparities
    
    def predict_directory(self, left_dir, right_dir, output_dir, file_ext='.png'):
        """
        预测目录中的所有图像对
        
        参数:
            left_dir (str): 左图像目录
            right_dir (str): 右图像目录
            output_dir (str): 输出目录
            file_ext (str): 输出文件扩展名
            
        返回:
            dict: 处理统计信息
        """
        # 检查目录是否存在
        if not os.path.exists(left_dir) or not os.path.exists(right_dir):
            raise ValueError(f"目录不存在: {left_dir} 或 {right_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取左图像列表
        left_files = sorted([f for f in os.listdir(left_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # 匹配右图像
        left_paths = []
        right_paths = []
        
        for left_file in left_files:
            left_path = os.path.join(left_dir, left_file)
            
            # 查找对应的右图像
            right_file = left_file
            right_path = os.path.join(right_dir, right_file)
            
            if os.path.exists(right_path):
                left_paths.append(left_path)
                right_paths.append(right_path)
        
        # 批量预测
        start_time = time.time()
        disparities = self.predict_batch(left_paths, right_paths)
        total_time = time.time() - start_time
        
        # 保存结果
        for i, (left_path, disparity) in enumerate(zip(left_paths, disparities)):
            # 生成输出文件名
            base_name = os.path.basename(left_path)
            name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(output_dir, name_without_ext + file_ext)
            
            # 保存视差图
            if file_ext == '.npy':
                np.save(output_path, disparity)
            elif file_ext == '.pfm':
                self._write_pfm(disparity, output_path)
            else:
                # 生成彩色视差图并保存
                color_disparity = colorize_disparity(disparity)
                cv2.imwrite(output_path, cv2.cvtColor(color_disparity, cv2.COLOR_RGB2BGR))
        
        # 计算统计信息
        num_images = len(disparities)
        avg_time = total_time / num_images if num_images > 0 else 0
        
        stats = {
            'num_images': num_images,
            'total_time': total_time,
            'avg_time': avg_time
        }
        
        return stats


def load_model_for_inference(model_path, model_class, config=None):
    """
    加载模型用于推理
    
    参数:
        model_path (str): 模型路径
        model_class (type): 模型类
        config (dict, optional): 模型配置
        
    返回:
        torch.nn.Module: 加载的模型
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    if config is None:
        model = model_class()
    else:
        model = model_class(**config)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 提取模型状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 加载权重
    model.load_state_dict(state_dict)
    
    # 设置为评估模式
    model.eval()
    
    return model