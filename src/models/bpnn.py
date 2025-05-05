"""
信念传播神经网络模型

该模块实现了结合信念传播和神经网络的BPNN模型，
用于立体视觉的视差图生成。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neural_net import (
    FeatureExtractorNetwork,
    CostVolumeNetwork,
    MessagePassingNetwork,
    DisparityRefinementNetwork,
    AttentionGuidedMessagePassing
)


class BPNN(nn.Module):
    """
    信念传播神经网络（BPNN）模型
    
    结合了神经网络特征提取和信念传播算法的端到端可学习的模型。
    """
    
    def __init__(self, max_disp=64, feature_channels=32, iterations=5, 
                use_attention=False, use_refinement=True):
        """
        初始化BPNN模型
        
        参数:
            max_disp (int): 最大视差值
            feature_channels (int): 特征通道数
            iterations (int): 消息传递迭代次数
            use_attention (bool): 是否使用注意力引导的消息传递
            use_refinement (bool): 是否使用视差细化网络
        """
        super(BPNN, self).__init__()
        self.max_disp = max_disp
        self.iterations = iterations
        self.use_refinement = use_refinement
        
        # 特征提取网络（左右图像共享权重）
        self.feature_extractor = FeatureExtractorNetwork(
            in_channels=3, 
            base_channels=feature_channels
        )
        
        # 代价体计算网络
        self.cost_volume_net = CostVolumeNetwork(
            in_channels=feature_channels, 
            max_disp=max_disp
        )
        
        # 消息传递网络（选择使用标准或注意力引导的消息传递）
        if use_attention:
            self.message_passing = AttentionGuidedMessagePassing(
                feature_channels=feature_channels, 
                hidden_channels=64
            )
        else:
            self.message_passing = MessagePassingNetwork(
                feature_channels=feature_channels, 
                hidden_channels=64
            )
        
        # 视差细化网络（可选）
        if use_refinement:
            self.refinement = DisparityRefinementNetwork(in_channels=4)
        
    def estimate_disparity(self, cost_volume):
        """
        从代价体估计视差图
        
        参数:
            cost_volume (torch.Tensor): 代价体，形状为 [B, 1, D, H, W]
            
        返回:
            torch.Tensor: 视差图，形状为 [B, 1, H, W]
        """
        # 对D维度求softmax，使其成为概率分布
        prob_volume = F.softmax(-cost_volume, dim=2)
        
        # 创建视差索引，形状为 [D]
        disp_indices = torch.arange(
            0, self.max_disp, device=cost_volume.device, dtype=torch.float32
        )
        
        # 扩展视差索引的形状为 [B, 1, D, H, W]
        disp_indices = disp_indices.view(1, 1, self.max_disp, 1, 1).expand_as(prob_volume)
        
        # 计算期望视差（加权和）
        disparity = torch.sum(disp_indices * prob_volume, dim=2, keepdim=False)
        
        return disparity
    
    def forward(self, left_img, right_img):
        """
        前向传播
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [B, C, H, W]
            right_img (torch.Tensor): 右图像，形状为 [B, C, H, W]
            
        返回:
            dict: 包含视差估计和中间结果的字典
        """
        # 特征提取
        left_feature = self.feature_extractor(left_img)
        right_feature = self.feature_extractor(right_img)
        
        # 构建代价体
        cost_volume = self.cost_volume_net(left_feature, right_feature)
        
        # 通过消息传递网络处理代价体
        # 对每个视差平面单独处理
        batch_size, _, max_disp, height, width = cost_volume.shape
        refined_cost = torch.zeros_like(cost_volume)
        
        for d in range(max_disp):
            # 提取当前视差平面的代价
            data_term = cost_volume[:, :, d, :, :]
            
            # 应用消息传递
            belief = self.message_passing(left_feature, data_term, self.iterations)
            
            # 存储优化后的代价
            refined_cost[:, :, d, :, :] = belief
        
        # 从优化后的代价体估计视差
        disparity = self.estimate_disparity(refined_cost)
        
        # 视差细化（可选）
        if self.use_refinement:
            disparity = self.refinement(left_img, disparity)
        
        # 返回视差图和中间结果
        results = {
            'disparity': disparity,              # 最终视差图
            'left_feature': left_feature,        # 左图像特征
            'right_feature': right_feature,      # 右图像特征
            'cost_volume': cost_volume,          # 原始代价体
            'refined_cost': refined_cost         # 优化后的代价体
        }
        
        return results


class DeepBPNetModule(nn.Module):
    """
    深度BP网络模块

    一个端到端可训练的模块，结合了深度神经网络和BP算法。
    此模块是BP层的可微分实现，用于优化视差估计。
    """
    
    def __init__(self, channels=32, iterations=5):
        """
        初始化DeepBPNetModule
        
        参数:
            channels (int): 通道数
            iterations (int): BP迭代次数
        """
        super(DeepBPNetModule, self).__init__()
        self.iterations = iterations
        
        # 数据项网络：学习数据项权重
        self.data_weight_net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 平滑项网络：学习平滑项权重
        self.smooth_weight_net = nn.Sequential(
            nn.Conv2d(channels*2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 消息更新网络：学习消息的更新规则
        self.msg_update_net = nn.Sequential(
            nn.Conv2d(channels + 1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        
        # 信念更新网络：学习信念的更新规则
        self.belief_update_net = nn.Sequential(
            nn.Conv2d(4 + 1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
    
    def compute_data_weight(self, feature):
        """
        计算数据项权重
        
        参数:
            feature (torch.Tensor): 特征图，形状为 [B, C, H, W]
            
        返回:
            torch.Tensor: 数据项权重，形状为 [B, 1, H, W]
        """
        return self.data_weight_net(feature)
    
    def compute_smooth_weight(self, feature1, feature2):
        """
        计算平滑项权重
        
        参数:
            feature1 (torch.Tensor): 第一个像素特征，形状为 [B, C, H, W]
            feature2 (torch.Tensor): 第二个像素特征，形状为 [B, C, H, W]
            
        返回:
            torch.Tensor: 平滑项权重，形状为 [B, 1, H, W]
        """
        concat_feat = torch.cat([feature1, feature2], dim=1)
        return self.smooth_weight_net(concat_feat)
    
    def message_update(self, feature, cost, msg1, msg2):
        """
        更新消息
        
        参数:
            feature (torch.Tensor): 特征图，形状为 [B, C, H, W]
            cost (torch.Tensor): 数据项代价，形状为 [B, 1, H, W]
            msg1 (torch.Tensor): 第一个传入消息，形状为 [B, 1, H, W]
            msg2 (torch.Tensor): 第二个传入消息，形状为 [B, 1, H, W]
            
        返回:
            torch.Tensor: 更新后的消息，形状为 [B, 1, H, W]
        """
        # 组合特征和代价
        input_tensor = torch.cat([feature, cost + msg1 + msg2], dim=1)
        
        # 使用网络预测更新后的消息
        updated_msg = self.msg_update_net(input_tensor)
        
        return updated_msg
    
    def belief_update(self, cost, msg_up, msg_down, msg_left, msg_right):
        """
        更新信念
        
        参数:
            cost (torch.Tensor): 数据项代价，形状为 [B, 1, H, W]
            msg_up, msg_down, msg_left, msg_right: 各方向的消息
            
        返回:
            torch.Tensor: 更新后的信念，形状为 [B, 1, H, W]
        """
        # 组合所有消息和代价
        input_tensor = torch.cat([msg_up, msg_down, msg_left, msg_right, cost], dim=1)
        
        # 使用网络预测更新后的信念
        updated_belief = self.belief_update_net(input_tensor)
        
        return updated_belief
    
    def forward(self, feature, cost_volume):
        """
        前向传播
        
        参数:
            feature (torch.Tensor): 特征图，形状为 [B, C, H, W]
            cost_volume (torch.Tensor): 代价体，形状为 [B, 1, D, H, W]
            
        返回:
            torch.Tensor: 优化后的代价体，形状为 [B, 1, D, H, W]
        """
        batch_size, _, max_disp, height, width = cost_volume.shape
        device = feature.device
        
        # 计算数据项权重
        data_weight = self.compute_data_weight(feature)
        
        # 初始化优化后的代价体
        refined_cost = torch.zeros_like(cost_volume)
        
        # 对每个视差平面单独处理
        for d in range(max_disp):
            # 提取当前视差平面的代价
            cost = cost_volume[:, :, d, :, :] * data_weight
            
            # 初始化消息
            msg_up = torch.zeros((batch_size, 1, height, width), device=device)
            msg_down = torch.zeros((batch_size, 1, height, width), device=device)
            msg_left = torch.zeros((batch_size, 1, height, width), device=device)
            msg_right = torch.zeros((batch_size, 1, height, width), device=device)
            
            # 迭代BP
            for _ in range(self.iterations):
                # 更新消息（可以并行处理）
                # 更新从下到上的消息
                msg_up_new = torch.zeros_like(msg_up)
                for i in range(1, height):
                    i_feature = feature[:, :, i, :]
                    i_cost = cost[:, :, i, :]
                    smooth_weight = self.compute_smooth_weight(
                        feature[:, :, i, :], feature[:, :, i-1, :]
                    )
                    msg_up_new[:, :, i-1, :] = self.message_update(
                        i_feature, i_cost, msg_left[:, :, i, :], msg_right[:, :, i, :]
                    ) * smooth_weight
                
                # 更新从上到下的消息
                msg_down_new = torch.zeros_like(msg_down)
                for i in range(height-2, -1, -1):
                    i_feature = feature[:, :, i, :]
                    i_cost = cost[:, :, i, :]
                    smooth_weight = self.compute_smooth_weight(
                        feature[:, :, i, :], feature[:, :, i+1, :]
                    )
                    msg_down_new[:, :, i+1, :] = self.message_update(
                        i_feature, i_cost, msg_left[:, :, i, :], msg_right[:, :, i, :]
                    ) * smooth_weight
                
                # 更新从右到左的消息
                msg_left_new = torch.zeros_like(msg_left)
                for j in range(1, width):
                    j_feature = feature[:, :, :, j]
                    j_cost = cost[:, :, :, j]
                    smooth_weight = self.compute_smooth_weight(
                        feature[:, :, :, j], feature[:, :, :, j-1]
                    )
                    msg_left_new[:, :, :, j-1] = self.message_update(
                        j_feature, j_cost, msg_up[:, :, :, j], msg_down[:, :, :, j]
                    ) * smooth_weight
                
                # 更新从左到右的消息
                msg_right_new = torch.zeros_like(msg_right)
                for j in range(width-2, -1, -1):
                    j_feature = feature[:, :, :, j]
                    j_cost = cost[:, :, :, j]
                    smooth_weight = self.compute_smooth_weight(
                        feature[:, :, :, j], feature[:, :, :, j+1]
                    )
                    msg_right_new[:, :, :, j+1] = self.message_update(
                        j_feature, j_cost, msg_up[:, :, :, j], msg_down[:, :, :, j]
                    ) * smooth_weight
                
                # 更新消息
                msg_up = msg_up_new
                msg_down = msg_down_new
                msg_left = msg_left_new
                msg_right = msg_right_new
            
            # 更新信念
            belief = self.belief_update(cost, msg_up, msg_down, msg_left, msg_right)
            
            # 存储更新后的代价
            refined_cost[:, :, d, :, :] = belief
        
        return refined_cost


class HierarchicalBPNN(nn.Module):
    """
    层次化信念传播神经网络

    使用多尺度方法来加速BP的收敛并提高准确性。
    """
    
    def __init__(self, max_disp=64, feature_channels=32, num_scales=3, scale_factor=0.5):
        """
        初始化层次化BPNN模型
        
        参数:
            max_disp (int): 最大视差值
            feature_channels (int): 特征通道数
            num_scales (int): 尺度级别数量
            scale_factor (float): 相邻尺度之间的缩放因子
        """
        super(HierarchicalBPNN, self).__init__()
        self.max_disp = max_disp
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        
        # 特征提取网络
        self.feature_net = FeatureExtractorNetwork(
            in_channels=3, 
            base_channels=feature_channels
        )
        
        # 代价体网络
        self.cost_net = CostVolumeNetwork(
            in_channels=feature_channels, 
            max_disp=max_disp
        )
        
        # 多个尺度的BP模块
        self.bp_modules = nn.ModuleList([
            DeepBPNetModule(channels=feature_channels, iterations=5)
            for _ in range(num_scales)
        ])
        
        # 视差细化网络
        self.refinement = DisparityRefinementNetwork(in_channels=4)
    
    def downsample(self, x, scale):
        """
        下采样张量
        
        参数:
            x (torch.Tensor): 输入张量
            scale (float): 缩放因子
            
        返回:
            torch.Tensor: 下采样后的张量
        """
        if scale == 1.0:
            return x
        
        # 根据x的维度确定处理方式
        if len(x.shape) == 5:  # [B, C, D, H, W]
            b, c, d, h, w = x.shape
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w)
            x_down = F.interpolate(
                x_reshaped, scale_factor=scale, mode='bilinear', align_corners=False
            )
            _, _, new_h, new_w = x_down.shape
            x_down = x_down.reshape(b, d, c, new_h, new_w).permute(0, 2, 1, 3, 4)
            return x_down
        else:  # [B, C, H, W]
            return F.interpolate(
                x, scale_factor=scale, mode='bilinear', align_corners=False
            )
    
    def upsample(self, x, target_size):
        """
        上采样张量到目标尺寸
        
        参数:
            x (torch.Tensor): 输入张量
            target_size (tuple): 目标尺寸 (H, W)
            
        返回:
            torch.Tensor: 上采样后的张量
        """
        # 根据x的维度确定处理方式
        if len(x.shape) == 5:  # [B, C, D, H, W]
            b, c, d, _, _ = x.shape
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, x.shape[3], x.shape[4])
            x_up = F.interpolate(
                x_reshaped, size=target_size, mode='bilinear', align_corners=False
            )
            x_up = x_up.reshape(b, d, c, target_size[0], target_size[1]).permute(0, 2, 1, 3, 4)
            return x_up
        else:  # [B, C, H, W]
            return F.interpolate(
                x, size=target_size, mode='bilinear', align_corners=False
            )
    
    def forward(self, left_img, right_img):
        """
        前向传播
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [B, C, H, W]
            right_img (torch.Tensor): 右图像，形状为 [B, C, H, W]
            
        返回:
            dict: 包含视差估计和中间结果的字典
        """
        original_size = (left_img.shape[2], left_img.shape[3])
        
        # 创建图像和特征的金字塔
        left_pyramid = [left_img]
        right_pyramid = [right_img]
        
        # 下采样创建图像金字塔
        for s in range(1, self.num_scales):
            scale = self.scale_factor ** s
            left_pyramid.append(self.downsample(left_img, scale))
            right_pyramid.append(self.downsample(right_img, scale))
        
        # 提取每个尺度的特征
        left_feature_pyramid = []
        right_feature_pyramid = []
        
        for s in range(self.num_scales):
            left_feature = self.feature_net(left_pyramid[s])
            right_feature = self.feature_net(right_pyramid[s])
            left_feature_pyramid.append(left_feature)
            right_feature_pyramid.append(right_feature)
        
        # 从最粗糙的尺度开始处理
        cost = None
        
        for s in range(self.num_scales-1, -1, -1):
            current_size = (left_pyramid[s].shape[2], left_pyramid[s].shape[3])
            left_feature = left_feature_pyramid[s]
            right_feature = right_feature_pyramid[s]
            
            # 计算当前尺度的代价体
            current_cost = self.cost_net(left_feature, right_feature)
            
            # 如果有来自粗糙尺度的代价，则上采样并融合
            if cost is not None:
                cost = self.upsample(cost, current_size)
                # 融合粗糙尺度的代价和当前尺度的代价
                current_cost = current_cost + cost
            
            # 应用当前尺度的BP模块
            cost = self.bp_modules[s](left_feature, current_cost)
        
        # 从优化后的代价体估计视差
        disparity = self.estimate_disparity(cost)
        
        # 视差细化
        disparity = self.refinement(left_img, disparity)
        
        return {
            'disparity': disparity,
            'cost_volume': cost
        }
    
    def estimate_disparity(self, cost_volume):
        """
        从代价体估计视差图
        
        参数:
            cost_volume (torch.Tensor): 代价体，形状为 [B, 1, D, H, W]
            
        返回:
            torch.Tensor: 视差图，形状为 [B, 1, H, W]
        """
        # 对D维度求softmax，使其成为概率分布
        prob_volume = F.softmax(-cost_volume, dim=2)
        
        # 创建视差索引，形状为 [D]
        disp_indices = torch.arange(
            0, self.max_disp, device=cost_volume.device, dtype=torch.float32
        )
        
        # 扩展视差索引的形状为 [B, 1, D, H, W]
        disp_indices = disp_indices.view(1, 1, self.max_disp, 1, 1).expand_as(prob_volume)
        
        # 计算期望视差（加权和）
        disparity = torch.sum(disp_indices * prob_volume, dim=2, keepdim=False)
        
        return disparity


class DualBPNN(nn.Module):
    """
    双通道BPNN模型
    
    该模型在左右图像上同时应用BP，并通过一致性检查改进结果。
    """
    
    def __init__(self, max_disp=64, feature_channels=32, iterations=5):
        """
        初始化双通道BPNN模型
        
        参数:
            max_disp (int): 最大视差值
            feature_channels (int): 特征通道数
            iterations (int): BP迭代次数
        """
        super(DualBPNN, self).__init__()
        self.max_disp = max_disp
        
        # 共享的特征提取网络
        self.feature_net = FeatureExtractorNetwork(
            in_channels=3, 
            base_channels=feature_channels
        )
        
        # 左右代价体网络
        self.left_cost_net = CostVolumeNetwork(
            in_channels=feature_channels, 
            max_disp=max_disp
        )
        self.right_cost_net = CostVolumeNetwork(
            in_channels=feature_channels, 
            max_disp=max_disp
        )
        
        # 左右BP模块
        self.left_bp_module = DeepBPNetModule(
            channels=feature_channels, 
            iterations=iterations
        )
        self.right_bp_module = DeepBPNetModule(
            channels=feature_channels, 
            iterations=iterations
        )
        
        # 视差融合网络
        self.fusion_net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
        
        # 视差细化网络
        self.refinement = DisparityRefinementNetwork(in_channels=4)
    
    def left_right_consistency_check(self, left_disp, right_disp):
        """
        左右一致性检查
        
        参数:
            left_disp (torch.Tensor): 左视差图，形状为 [B, 1, H, W]
            right_disp (torch.Tensor): 右视差图，形状为 [B, 1, H, W]
            
        返回:
            torch.Tensor: 一致性掩码，形状为 [B, 1, H, W]，值为0或1
        """
        batch_size, _, height, width = left_disp.shape
        device = left_disp.device
        
        # 对于左图中的每个像素，在右图中找到对应点
        x_left = torch.arange(0, width, device=device).view(1, 1, 1, width).expand(batch_size, 1, height, width)
        y_left = torch.arange(0, height, device=device).view(1, 1, height, 1).expand(batch_size, 1, height, width)
        
        # 计算对应右图中的x坐标
        x_right = x_left - left_disp
        x_right = torch.clamp(x_right, 0, width-1)
        
        # 对右视差图进行采样
        x_right_int = x_right.long().squeeze(1)
        y_left_int = y_left.long().squeeze(1)
        
        right_disp_at_left = torch.zeros_like(left_disp)
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    xr = x_right_int[b, h, w]
                    if 0 <= xr < width:
                        right_disp_at_left[b, 0, h, w] = right_disp[b, 0, h, xr]
        
        # 计算左右视差的差异
        diff = torch.abs(left_disp + right_disp_at_left)
        
        # 一致性掩码：差异小于阈值的像素
        consistency_mask = (diff < 1.0).float()
        
        return consistency_mask
    
    def forward(self, left_img, right_img):
        """
        前向传播
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [B, C, H, W]
            right_img (torch.Tensor): 右图像，形状为 [B, C, H, W]
            
        返回:
            dict: 包含视差估计和中间结果的字典
        """
        # 提取特征
        left_feature = self.feature_net(left_img)
        right_feature = self.feature_net(right_img)
        
        # 计算左右代价体
        left_cost = self.left_cost_net(left_feature, right_feature)
        right_cost = self.right_cost_net(right_feature, left_feature)
        
        # 应用BP模块
        left_refined_cost = self.left_bp_module(left_feature, left_cost)
        right_refined_cost = self.right_bp_module(right_feature, right_cost)
        
        # 估计左右视差
        left_disp = self.estimate_disparity(left_refined_cost)
        right_disp = self.estimate_disparity(right_refined_cost)
        
        # 左右一致性检查
        consistency_mask = self.left_right_consistency_check(left_disp, right_disp)
        
        # 融合左右视差
        fused_disp = self.fusion_net(torch.cat([left_disp, right_disp], dim=1))
        
        # 使用一致性掩码过滤结果
        # 对于一致性检查不通过的像素，可以使用更复杂的后处理
        fused_disp = fused_disp * consistency_mask + left_disp * (1 - consistency_mask)
        
        # 视差细化
        refined_disp = self.refinement(left_img, fused_disp)
        
        return {
            'disparity': refined_disp,
            'left_disparity': left_disp,
            'right_disparity': right_disp,
            'consistency_mask': consistency_mask
        }
    
    def estimate_disparity(self, cost_volume):
        """
        从代价体估计视差图
        
        参数:
            cost_volume (torch.Tensor): 代价体，形状为 [B, 1, D, H, W]
            
        返回:
            torch.Tensor: 视差图，形状为 [B, 1, H, W]
        """
        # 对D维度求softmax，使其成为概率分布
        prob_volume = F.softmax(-cost_volume, dim=2)
        
        # 创建视差索引，形状为 [D]
        disp_indices = torch.arange(
            0, self.max_disp, device=cost_volume.device, dtype=torch.float32
        )
        
        # 扩展视差索引的形状为 [B, 1, D, H, W]
        disp_indices = disp_indices.view(1, 1, self.max_disp, 1, 1).expand_as(prob_volume)
        
        # 计算期望视差（加权和）
        disparity = torch.sum(disp_indices * prob_volume, dim=2, keepdim=False)
        
        return disparity