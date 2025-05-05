"""
信念传播算法实现模块

该模块实现了用于立体匹配的信念传播算法。
信念传播是一种在概率图模型上进行推理的消息传递算法。
在立体匹配中，我们将图像表示为一个网格图，每个像素是一个节点，
相邻像素之间有边连接。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefPropagation:
    """
    经典信念传播算法实现
    
    该类实现了用于立体匹配的经典信念传播算法。
    算法将立体匹配表示为马尔可夫随机场(MRF)上的能量最小化问题，
    并使用消息传递来寻找最优解。
    """
    
    def __init__(self, max_disp, iterations=5, damping=0.5, 
                 data_weight=1.0, smooth_weight=0.1, 
                 use_cuda=torch.cuda.is_available()):
        """
        初始化信念传播算法
        
        参数:
            max_disp (int): 最大视差值
            iterations (int): 消息传递迭代次数
            damping (float): 消息更新的阻尼系数，用于稳定收敛
            data_weight (float): 数据项权重
            smooth_weight (float): 平滑项权重
            use_cuda (bool): 是否使用GPU加速
        """
        self.max_disp = max_disp
        self.iterations = iterations
        self.damping = damping
        self.data_weight = data_weight
        self.smooth_weight = smooth_weight
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
    def compute_data_cost(self, left_img, right_img):
        """
        计算数据项成本（像素匹配代价）
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [H, W, C]
            right_img (torch.Tensor): 右图像，形状为 [H, W, C]
            
        返回:
            torch.Tensor: 数据项成本体，形状为 [H, W, max_disp]
        """
        height, width = left_img.shape[:2]
        data_cost = torch.zeros((height, width, self.max_disp), 
                               device=self.device)
        
        for d in range(self.max_disp):
            # 将右图像向右移动d个像素
            shifted_right = torch.zeros_like(right_img)
            if d == 0:
                shifted_right = right_img.clone()
            else:
                shifted_right[:, d:, :] = right_img[:, :-d, :]
            
            # 计算左图像和移位右图像之间的绝对差异
            diff = torch.abs(left_img - shifted_right)
            
            # 对于多通道图像，对所有通道求和
            if len(diff.shape) > 2:
                diff = diff.sum(dim=2)
                
            # 存储当前视差的数据成本
            data_cost[:, :, d] = diff
            
        return data_cost * self.data_weight
    
    def compute_smoothness_cost(self):
        """
        计算平滑项成本（相邻像素之间的惩罚）
        
        返回:
            torch.Tensor: 平滑项成本矩阵，形状为 [max_disp, max_disp]
        """
        # 创建二次惩罚矩阵：视差差异的平方
        disp_range = torch.arange(self.max_disp, device=self.device)
        disp_diff = (disp_range.unsqueeze(0) - disp_range.unsqueeze(1)) ** 2
        
        # 应用截断的二次惩罚，防止过大的惩罚
        max_penalty = 5.0  # 最大惩罚阈值
        smooth_cost = torch.minimum(disp_diff, 
                                   torch.tensor(max_penalty, device=self.device))
        
        return smooth_cost * self.smooth_weight
    
    def init_messages(self, height, width):
        """
        初始化信念传播消息
        
        参数:
            height (int): 图像高度
            width (int): 图像宽度
            
        返回:
            tuple: 包含四个方向的消息张量，每个形状为 [H, W, max_disp]
        """
        # 初始化四个方向的消息：上、下、左、右
        msg_shape = (height, width, self.max_disp)
        msg_up = torch.zeros(msg_shape, device=self.device)
        msg_down = torch.zeros(msg_shape, device=self.device)
        msg_left = torch.zeros(msg_shape, device=self.device)
        msg_right = torch.zeros(msg_shape, device=self.device)
        
        return msg_up, msg_down, msg_left, msg_right
    
    def update_message(self, data_cost, smooth_cost, msg_incoming1, 
                      msg_incoming2, msg_old, direction):
        """
        更新信念传播消息
        
        参数:
            data_cost (torch.Tensor): 数据项成本
            smooth_cost (torch.Tensor): 平滑项成本
            msg_incoming1 (torch.Tensor): 第一个传入消息
            msg_incoming2 (torch.Tensor): 第二个传入消息
            msg_old (torch.Tensor): 旧消息
            direction (str): 消息传递方向 ('up', 'down', 'left', 'right')
            
        返回:
            torch.Tensor: 更新后的消息
        """
        height, width, max_disp = data_cost.shape
        msg_new = torch.zeros_like(msg_old)
        
        # 根据方向确定遍历顺序和索引偏移
        if direction == 'up':
            start, end, step = 1, height, 1
            offset_y, offset_x = -1, 0
        elif direction == 'down':
            start, end, step = height-2, -1, -1
            offset_y, offset_x = 1, 0
        elif direction == 'left':
            start, end, step = 1, width, 1
            offset_y, offset_x = 0, -1
        elif direction == 'right':
            start, end, step = width-2, -1, -1
            offset_y, offset_x = 0, 1
        else:
            raise ValueError(f"未知方向: {direction}")
        
        # 横向消息更新
        if direction in ['left', 'right']:
            for x in range(start, end, step):
                for y in range(height):
                    # 计算当前像素的总信息
                    total_msg = (data_cost[y, x] + 
                                msg_incoming1[y, x] + 
                                msg_incoming2[y, x])
                    
                    # 对每个可能的视差标签计算消息
                    for d in range(max_disp):
                        # 计算将当前节点标记为d时的总成本
                        # smooth_cost[d, :] 表示当前标签d与所有可能标签之间的平滑成本
                        msg_min = float('inf')
                        for d_other in range(max_disp):
                            cost = total_msg[d_other] + smooth_cost[d, d_other]
                            msg_min = min(msg_min, cost)
                        
                        msg_new[y, x+offset_x, d] = msg_min
                    
                    # 归一化消息
                    msg_new[y, x+offset_x] -= msg_new[y, x+offset_x].min()
        
        # 纵向消息更新
        else:
            for y in range(start, end, step):
                for x in range(width):
                    # 计算当前像素的总信息
                    total_msg = (data_cost[y, x] + 
                                msg_incoming1[y, x] + 
                                msg_incoming2[y, x])
                    
                    # 对每个可能的视差标签计算消息
                    for d in range(max_disp):
                        # 计算将当前节点标记为d时的总成本
                        msg_min = float('inf')
                        for d_other in range(max_disp):
                            cost = total_msg[d_other] + smooth_cost[d, d_other]
                            msg_min = min(msg_min, cost)
                        
                        msg_new[y+offset_y, x, d] = msg_min
                    
                    # 归一化消息
                    msg_new[y+offset_y, x] -= msg_new[y+offset_y, x].min()
        
        # 应用阻尼系数
        msg_new = self.damping * msg_new + (1 - self.damping) * msg_old
        
        return msg_new
    
    def compute_beliefs(self, data_cost, msg_up, msg_down, msg_left, msg_right):
        """
        计算每个像素的信念（最终的视差估计）
        
        参数:
            data_cost (torch.Tensor): 数据项成本
            msg_up (torch.Tensor): 从上方传来的消息
            msg_down (torch.Tensor): 从下方传来的消息
            msg_left (torch.Tensor): 从左方传来的消息
            msg_right (torch.Tensor): 从右方传来的消息
            
        返回:
            torch.Tensor: 每个像素的信念，形状为 [H, W, max_disp]
        """
        # 对每个像素，合并所有传入消息和数据成本
        beliefs = (data_cost + 
                  msg_up + 
                  msg_down + 
                  msg_left + 
                  msg_right)
        
        return beliefs
    
    def get_disparity_map(self, beliefs):
        """
        从信念中获取视差图
        
        参数:
            beliefs (torch.Tensor): 信念张量，形状为 [H, W, max_disp]
            
        返回:
            torch.Tensor: 视差图，形状为 [H, W]
        """
        # 对每个像素，选择具有最小能量（最大信念）的视差
        disparity_map = torch.argmin(beliefs, dim=2)
        
        return disparity_map
    
    def run(self, left_img, right_img):
        """
        运行信念传播算法生成视差图
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [H, W, C]
            right_img (torch.Tensor): 右图像，形状为 [H, W, C]
            
        返回:
            torch.Tensor: 视差图，形状为 [H, W]
        """
        height, width = left_img.shape[:2]
        
        # 计算数据项成本
        data_cost = self.compute_data_cost(left_img, right_img)
        
        # 计算平滑项成本
        smooth_cost = self.compute_smoothness_cost()
        
        # 初始化消息
        msg_up, msg_down, msg_left, msg_right = self.init_messages(height, width)
        
        # 迭代信念传播
        for _ in range(self.iterations):
            # 更新从下到上的消息
            msg_up = self.update_message(
                data_cost, smooth_cost, msg_left, msg_right, msg_up, 'up')
            
            # 更新从上到下的消息
            msg_down = self.update_message(
                data_cost, smooth_cost, msg_left, msg_right, msg_down, 'down')
            
            # 更新从右到左的消息
            msg_left = self.update_message(
                data_cost, smooth_cost, msg_up, msg_down, msg_left, 'left')
            
            # 更新从左到右的消息
            msg_right = self.update_message(
                data_cost, smooth_cost, msg_up, msg_down, msg_right, 'right')
        
        # 计算信念
        beliefs = self.compute_beliefs(data_cost, msg_up, msg_down, msg_left, msg_right)
        
        # 获取视差图
        disparity_map = self.get_disparity_map(beliefs)
        
        return disparity_map


class HierarchicalBeliefPropagation(BeliefPropagation):
    """
    层次化信念传播算法实现
    
    该类扩展了基本的信念传播算法，使用多尺度方法来加速收敛和提高准确性。
    算法首先在低分辨率图像上运行BP，然后将结果上采样并用作高分辨率图像的初始猜测。
    """
    
    def __init__(self, max_disp, iterations=5, damping=0.5, 
                 data_weight=1.0, smooth_weight=0.1,
                 num_scales=3, scale_factor=0.5,
                 use_cuda=torch.cuda.is_available()):
        """
        初始化层次化信念传播算法
        
        参数:
            max_disp (int): 最大视差值
            iterations (int): 每个尺度的消息传递迭代次数
            damping (float): 消息更新的阻尼系数
            data_weight (float): 数据项权重
            smooth_weight (float): 平滑项权重
            num_scales (int): 层次金字塔的尺度数量
            scale_factor (float): 相邻尺度之间的缩放因子
            use_cuda (bool): 是否使用GPU加速
        """
        super().__init__(max_disp, iterations, damping, 
                         data_weight, smooth_weight, use_cuda)
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        
    def downsample_image(self, img, scale_factor):
        """
        对图像进行下采样
        
        参数:
            img (torch.Tensor): 输入图像，形状为 [H, W, C]
            scale_factor (float): 缩放因子
            
        返回:
            torch.Tensor: 下采样后的图像
        """
        # 转换为[C, H, W]格式进行处理
        if len(img.shape) == 3:
            img = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        else:
            img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
        # 使用双线性插值进行下采样
        h, w = img.shape[2:]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        downsampled = F.interpolate(img, size=(new_h, new_w), 
                                   mode='bilinear', align_corners=False)
        
        # 转换回原始格式
        if len(downsampled.shape) == 4 and downsampled.shape[1] > 1:
            downsampled = downsampled.squeeze(0).permute(1, 2, 0)  # [H, W, C]
        else:
            downsampled = downsampled.squeeze(0).squeeze(0)  # [H, W]
            
        return downsampled
    
    def upsample_disparity(self, disp, target_shape):
        """
        对视差图进行上采样
        
        参数:
            disp (torch.Tensor): 视差图，形状为 [H, W]
            target_shape (tuple): 目标形状 (H, W)
            
        返回:
            torch.Tensor: 上采样后的视差图
        """
        # 转换为[1, 1, H, W]格式进行处理
        disp = disp.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 使用最近邻插值进行上采样，保持视差值不变
        upsampled = F.interpolate(disp, size=target_shape, 
                                 mode='nearest')
        
        # 根据尺度调整视差值
        scale_h = target_shape[0] / disp.shape[2]
        upsampled = upsampled * scale_h
        
        return upsampled.squeeze(0).squeeze(0)  # 返回[H, W]格式
    
    def run(self, left_img, right_img):
        """
        运行层次化信念传播算法生成视差图
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [H, W, C]
            right_img (torch.Tensor): 右图像，形状为 [H, W, C]
            
        返回:
            torch.Tensor: 视差图，形状为 [H, W]
        """
        original_shape = left_img.shape[:2]
        
        # 创建图像金字塔
        left_pyramid = [left_img]
        right_pyramid = [right_img]
        
        for s in range(1, self.num_scales):
            # 对上一层图像进行下采样
            left_scaled = self.downsample_image(left_pyramid[-1], self.scale_factor)
            right_scaled = self.downsample_image(right_pyramid[-1], self.scale_factor)
            
            left_pyramid.append(left_scaled)
            right_pyramid.append(right_scaled)
        
        # 从最粗糙的尺度开始处理
        disparity = None
        
        for s in range(self.num_scales - 1, -1, -1):
            current_left = left_pyramid[s]
            current_right = right_pyramid[s]
            current_shape = current_left.shape[:2]
            
            # 如果是最粗糙的尺度，直接运行BP
            if disparity is None:
                disparity = super().run(current_left, current_right)
            else:
                # 上采样上一尺度的视差作为初始化
                disparity = self.upsample_disparity(disparity, current_shape)
                
                # 使用上采样的视差来初始化消息
                # 这里可以添加代码，使用上一尺度的结果来初始化当前尺度的消息
                
                # 运行BP进行细化
                disparity = super().run(current_left, current_right)
        
        return disparity


class AdaptiveBeliefPropagation(BeliefPropagation):
    """
    自适应信念传播算法实现
    
    该类扩展了基本的信念传播算法，使用自适应的权重和参数来处理不同的图像区域。
    算法根据图像内容自适应地调整数据项和平滑项的权重。
    """
    
    def __init__(self, max_disp, iterations=5, damping=0.5, 
                 base_data_weight=1.0, base_smooth_weight=0.1,
                 edge_threshold=30, edge_data_scale=0.5, edge_smooth_scale=2.0,
                 use_cuda=torch.cuda.is_available()):
        """
        初始化自适应信念传播算法
        
        参数:
            max_disp (int): 最大视差值
            iterations (int): 消息传递迭代次数
            damping (float): 消息更新的阻尼系数
            base_data_weight (float): 基础数据项权重
            base_smooth_weight (float): 基础平滑项权重
            edge_threshold (float): 边缘检测阈值
            edge_data_scale (float): 边缘区域数据项缩放因子
            edge_smooth_scale (float): 边缘区域平滑项缩放因子
            use_cuda (bool): 是否使用GPU加速
        """
        super().__init__(max_disp, iterations, damping, 
                         base_data_weight, base_smooth_weight, use_cuda)
        self.edge_threshold = edge_threshold
        self.edge_data_scale = edge_data_scale
        self.edge_smooth_scale = edge_smooth_scale
        
    def detect_edges(self, img):
        """
        检测图像中的边缘
        
        参数:
            img (torch.Tensor): 输入图像，形状为 [H, W, C]
            
        返回:
            torch.Tensor: 边缘图，形状为 [H, W]，值为0或1
        """
        # 将图像转换为灰度图
        if len(img.shape) == 3 and img.shape[2] > 1:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            gray = img.squeeze(-1) if len(img.shape) == 3 else img
        
        # 计算水平和垂直梯度
        h, w = gray.shape
        grad_x = torch.zeros_like(gray)
        grad_y = torch.zeros_like(gray)
        
        grad_x[:, 1:w-1] = gray[:, 2:] - gray[:, :w-2]
        grad_y[1:h-1, :] = gray[2:, :] - gray[:h-2, :]
        
        # 计算梯度幅度
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 应用阈值得到边缘图
        edges = (grad_mag > self.edge_threshold).float()
        
        return edges
    
    def compute_adaptive_weights(self, left_img):
        """
        计算自适应权重
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [H, W, C]
            
        返回:
            tuple: 包含数据项权重和平滑项权重的张量，每个形状为 [H, W]
        """
        # 检测边缘
        edges = self.detect_edges(left_img)
        
        # 调整数据项权重：在边缘处减小数据项权重
        data_weights = torch.ones_like(edges) * self.data_weight
        data_weights[edges == 1] *= self.edge_data_scale
        
        # 调整平滑项权重：在边缘处增加平滑项权重
        smooth_weights = torch.ones_like(edges) * self.smooth_weight
        smooth_weights[edges == 1] *= self.edge_smooth_scale
        
        return data_weights, smooth_weights
    
    def compute_data_cost(self, left_img, right_img):
        """
        计算自适应数据项成本
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [H, W, C]
            right_img (torch.Tensor): 右图像，形状为 [H, W, C]
            
        返回:
            torch.Tensor: 数据项成本体，形状为 [H, W, max_disp]
        """
        # 计算标准数据成本
        data_cost = super().compute_data_cost(left_img, right_img)
        
        # 计算自适应权重
        data_weights, _ = self.compute_adaptive_weights(left_img)
        
        # 应用自适应权重
        for d in range(self.max_disp):
            data_cost[:, :, d] *= data_weights
        
        return data_cost
    
    def compute_smoothness_cost(self, left_img):
        """
        计算自适应平滑项成本
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [H, W, C]
            
        返回:
            torch.Tensor: 平滑项成本矩阵，形状为 [max_disp, max_disp]
        """
        # 计算标准平滑成本
        smooth_cost = super().compute_smoothness_cost()
        
        # 计算自适应权重
        _, smooth_weights = self.compute_adaptive_weights(left_img)
        
        # 平滑权重是全局的，我们可以取平均值
        avg_smooth_weight = smooth_weights.mean().item()
        smooth_cost *= avg_smooth_weight
        
        return smooth_cost
    
    def run(self, left_img, right_img):
        """
        运行自适应信念传播算法生成视差图
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [H, W, C]
            right_img (torch.Tensor): 右图像，形状为 [H, W, C]
            
        返回:
            torch.Tensor: 视差图，形状为 [H, W]
        """
        height, width = left_img.shape[:2]
        
        # 计算自适应数据项成本
        data_cost = self.compute_data_cost(left_img, right_img)
        
        # 计算自适应平滑项成本
        smooth_cost = self.compute_smoothness_cost(left_img)
        
        # 初始化消息
        msg_up, msg_down, msg_left, msg_right = self.init_messages(height, width)
        
        # 迭代信念传播
        for _ in range(self.iterations):
            # 更新从下到上的消息
            msg_up = self.update_message(
                data_cost, smooth_cost, msg_left, msg_right, msg_up, 'up')
            
            # 更新从上到下的消息
            msg_down = self.update_message(
                data_cost, smooth_cost, msg_left, msg_right, msg_down, 'down')
            
            # 更新从右到左的消息
            msg_left = self.update_message(
                data_cost, smooth_cost, msg_up, msg_down, msg_left, 'left')
            
            # 更新从左到右的消息
            msg_right = self.update_message(
                data_cost, smooth_cost, msg_up, msg_down, msg_right, 'right')
        
        # 计算信念
        beliefs = self.compute_beliefs(data_cost, msg_up, msg_down, msg_left, msg_right)
        
        # 获取视差图
        disparity_map = self.get_disparity_map(beliefs)
        
        return disparity_map