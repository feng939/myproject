"""
信念传播神经网络模型

该模块实现了结合信念传播和神经网络的BPNN模型，
用于立体视觉的视差图生成。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractorNetwork(nn.Module):
    """特征提取网络"""
    
    def __init__(self, in_channels=3, base_channels=16):
        super(FeatureExtractorNetwork, self).__init__()
        
        # 轻量级特征提取网络
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CostVolumeNetwork(nn.Module):
    """代价体计算网络 - 优化显存使用"""
    
    def __init__(self, in_channels, max_disp):
        super(CostVolumeNetwork, self).__init__()
        self.max_disp = max_disp
        
        # 减少通道数的卷积层
        self.cost_conv = nn.Sequential(
            nn.Conv3d(in_channels*2, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, left_feature, right_feature):
        """
        前向传播 - 分块处理代价体以减少显存使用
        
        参数:
            left_feature (torch.Tensor): 左图像特征
            right_feature (torch.Tensor): 右图像特征
            
        返回:
            torch.Tensor: 代价体
        """
        batch_size, channels, height, width = left_feature.shape
        
        # 分块计算代价体，以节省内存
        cost_list = []
        
        # 每次处理8个视差
        for d_start in range(0, self.max_disp, 8):
            d_end = min(d_start + 8, self.max_disp)
            d_range = d_end - d_start
            
            # 初始化当前块的代价体
            cost_block = torch.zeros((batch_size, channels*2, d_range, height, width), 
                                    device=left_feature.device)
            
            # 填充代价体块
            for d_idx, d in enumerate(range(d_start, d_end)):
                if d > 0:
                    # 沿x轴偏移右图像特征
                    shifted_right = torch.zeros_like(right_feature)
                    shifted_right[:, :, :, 0:width-d] = right_feature[:, :, :, d:width]
                else:
                    shifted_right = right_feature
                
                # 连接左右特征
                cost_block[:, :channels, d_idx, :, :] = left_feature
                cost_block[:, channels:, d_idx, :, :] = shifted_right
            
            # 应用卷积层
            processed_block = self.cost_conv(cost_block)
            cost_list.append(processed_block)
        
        # 合并所有块
        cost_volume = torch.cat(cost_list, dim=2)
        
        return cost_volume


class SelfAttentionModule(nn.Module):
    """自注意力模块 - 用于增强重要特征"""
    
    def __init__(self, in_channels):
        super(SelfAttentionModule, self).__init__()
        
        # 降维以节省计算资源
        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))  # 控制注意力的强度
    
    def forward(self, x):
        """前向传播"""
        batch_size, C, H, W = x.shape
        
        # 计算查询、键、值
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B, HW, C//2
        key = self.key_conv(x).view(batch_size, -1, H * W)  # B, C//2, HW
        value = self.value_conv(x).view(batch_size, -1, H * W)  # B, C, HW
        
        # 计算注意力图
        energy = torch.bmm(query, key)  # B, HW, HW
        attention = F.softmax(energy / (C ** 0.5), dim=2)  # 缩放点积注意力
        
        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, HW
        out = out.view(batch_size, C, H, W)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


class AttentionBPNN(nn.Module):
    """
    注意力增强的信念传播神经网络
    
    使用自注意力机制来加强BP过程中的特征表示和消息传递。
    """
    
    def __init__(self, max_disp=32, feature_channels=16, iterations=3, 
                use_refinement=True, use_half_precision=True,
                block_size=64, overlap=8):
        """
        初始化注意力增强的BPNN模型
        
        参数:
            max_disp (int): 最大视差值
            feature_channels (int): 特征通道数
            iterations (int): BP迭代次数
            use_refinement (bool): 是否使用视差细化
            use_half_precision (bool): 是否使用半精度浮点数
            block_size (int): BP处理时的块大小
            overlap (int): 块重叠大小
        """
        super(AttentionBPNN, self).__init__()
        self.max_disp = max_disp
        self.iterations = iterations
        self.use_refinement = use_refinement
        self.use_half_precision = use_half_precision
        self.block_size = block_size
        self.overlap = overlap
        
        # 特征提取网络
        self.feature_extractor = FeatureExtractorNetwork(
            in_channels=3, 
            base_channels=feature_channels
        )
        
        # 自注意力模块
        self.attention = SelfAttentionModule(feature_channels)
        
        # 代价体计算网络
        self.cost_volume_net = CostVolumeNetwork(
            in_channels=feature_channels, 
            max_disp=max_disp
        )
        
        # BP消息传递网络的四个方向卷积
        self.msg_up = nn.Sequential(
            nn.Conv2d(feature_channels + 1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        
        self.msg_down = nn.Sequential(
            nn.Conv2d(feature_channels + 1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        
        self.msg_left = nn.Sequential(
            nn.Conv2d(feature_channels + 1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        
        self.msg_right = nn.Sequential(
            nn.Conv2d(feature_channels + 1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        
        # 信念更新网络
        self.belief_update = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        
        # 视差细化网络
        if use_refinement:
            self.refinement = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1)
            )
    
    def forward(self, left_img, right_img):
        """
        前向传播
        
        参数:
            left_img (torch.Tensor): 左图像
            right_img (torch.Tensor): 右图像
            
        返回:
            dict: 包含视差估计的字典
        """
        # 使用半精度浮点数以节省显存
        if self.use_half_precision and left_img.is_cuda:
            left_img = left_img.half()
            right_img = right_img.half()
            self.to(torch.float16)
        
        # 特征提取
        left_feature = self.feature_extractor(left_img)
        right_feature = self.feature_extractor(right_img)
        
        # 应用自注意力
        left_feature = self.attention(left_feature)
        
        # 构建代价体
        cost_volume = self.cost_volume_net(left_feature, right_feature)
        
        # 分块BP处理以节省显存
        _, _, _, height, width = cost_volume.shape
        
        # 将处理分成多个块
        h_blocks = (height + self.block_size - 1) // self.block_size
        w_blocks = (width + self.block_size - 1) // self.block_size
        
        # 初始化输出视差图
        disparity = torch.zeros((left_img.shape[0], 1, height, width), device=left_img.device)
        
        for h_idx in range(h_blocks):
            for w_idx in range(w_blocks):
                # 计算当前块的范围（带重叠区域）
                h_start = max(0, h_idx * self.block_size - self.overlap)
                h_end = min(height, (h_idx + 1) * self.block_size + self.overlap)
                w_start = max(0, w_idx * self.block_size - self.overlap)
                w_end = min(width, (w_idx + 1) * self.block_size + self.overlap)
                
                # 提取特征块
                feat_block = left_feature[:, :, h_start:h_end, w_start:w_end]
                cost_block = cost_volume[:, :, :, h_start:h_end, w_start:w_end]
                
                # 信念传播处理
                disp_block = self._process_block(feat_block, cost_block)
                
                # 计算有效区域（去除重叠部分）
                valid_h_start = 0 if h_idx == 0 else self.overlap
                valid_h_end = h_end - h_start if h_idx == h_blocks - 1 else min(h_end - h_start, (h_idx + 1) * self.block_size - h_start)
                valid_w_start = 0 if w_idx == 0 else self.overlap
                valid_w_end = w_end - w_start if w_idx == w_blocks - 1 else min(w_end - w_start, (w_idx + 1) * self.block_size - w_start)
                
                # 填充到输出视差图
                h_offset = h_idx * self.block_size
                w_offset = w_idx * self.block_size
                output_h_start = max(0, h_offset)
                output_h_end = min(height, h_offset + valid_h_end - valid_h_start)
                output_w_start = max(0, w_offset)
                output_w_end = min(width, w_offset + valid_w_end - valid_w_start)
                
                disparity[:, :, output_h_start:output_h_end, output_w_start:output_w_end] = \
                    disp_block[:, :, valid_h_start:valid_h_end, valid_w_start:valid_w_end]
        
        # 视差细化（如果需要）
        if self.use_refinement:
            # 将原始图像与视差连接起来
            refined_input = torch.cat([left_img, disparity], dim=1)
            residual = self.refinement(refined_input)
            disparity = disparity + residual
        
        # 恢复为单精度（如有必要）
        if self.use_half_precision and disparity.is_cuda:
            disparity = disparity.float()
            self.to(torch.float32)
        
        return {'disparity': disparity}
    
    def _process_block(self, feature_block, cost_block):
        """
        处理一个块的信念传播
        
        参数:
            feature_block (torch.Tensor): 特征块
            cost_block (torch.Tensor): 代价块
            
        返回:
            torch.Tensor: 该块的视差估计
        """
        batch_size, _, max_disp, height, width = cost_block.shape
        
        # 初始化视差图
        disparity = torch.zeros((batch_size, 1, height, width), device=feature_block.device)
        
        # 对每个视差平面进行BP处理
        for d in range(max_disp):
            # 提取当前视差平面的代价
            cost = cost_block[:, :, d]
            
            # 信念传播
            belief = self._belief_propagation(feature_block, cost)
            
            # 更新视差图（根据最小代价）
            disparity = torch.where(belief < disparity, torch.full_like(disparity, d), disparity)
        
        return disparity
    
    def _belief_propagation(self, feature, cost):
        """
        执行信念传播
        
        参数:
            feature (torch.Tensor): 特征图
            cost (torch.Tensor): 代价图
            
        返回:
            torch.Tensor: 更新后的信念
        """
        batch_size, _, height, width = feature.shape
        
        # 初始化消息为零
        msg_up = torch.zeros_like(cost)
        msg_down = torch.zeros_like(cost)
        msg_left = torch.zeros_like(cost)
        msg_right = torch.zeros_like(cost)
        
        # 迭代BP
        for _ in range(self.iterations):
            # 更新消息 (UP)
            for i in range(1, height):
                # 附加特征信息
                msg_input = torch.cat([feature[:, :, i-1:i, :], 
                                     cost[:, :, i:i+1, :] + msg_down[:, :, i-1:i, :] + 
                                     msg_left[:, :, i:i+1, 1:] + msg_right[:, :, i:i+1, :-1]], dim=1)
                msg_up[:, :, i-1:i, :] = self.msg_up(msg_input)
            
            # 更新消息 (DOWN)
            for i in range(height - 2, -1, -1):
                msg_input = torch.cat([feature[:, :, i+1:i+2, :], 
                                     cost[:, :, i:i+1, :] + msg_up[:, :, i+1:i+2, :] + 
                                     msg_left[:, :, i:i+1, 1:] + msg_right[:, :, i:i+1, :-1]], dim=1)
                msg_down[:, :, i+1:i+2, :] = self.msg_down(msg_input)
            
            # 更新消息 (LEFT)
            for j in range(1, width):
                msg_input = torch.cat([feature[:, :, :, j-1:j], 
                                     cost[:, :, :, j:j+1] + msg_up[:, :, 1:, j:j+1] + 
                                     msg_down[:, :, :-1, j:j+1] + msg_right[:, :, :, j-1:j]], dim=1)
                msg_left[:, :, :, j-1:j] = self.msg_left(msg_input)
            
            # 更新消息 (RIGHT)
            for j in range(width - 2, -1, -1):
                msg_input = torch.cat([feature[:, :, :, j+1:j+2], 
                                     cost[:, :, :, j:j+1] + msg_up[:, :, 1:, j:j+1] + 
                                     msg_down[:, :, :-1, j:j+1] + msg_left[:, :, :, j+1:j+2]], dim=1)
                msg_right[:, :, :, j+1:j+2] = self.msg_right(msg_input)
        
        # 计算最终信念
        belief_input = torch.cat([cost, msg_up, msg_down, msg_left, msg_right], dim=1)
        belief = self.belief_update(belief_input)
        
        return belief


class BPNN(nn.Module):
    """
    基础信念传播神经网络
    
    简化的BPNN模型，不使用注意力机制，适用于极小显存设备。
    """
    
    def __init__(self, max_disp=32, feature_channels=16, iterations=3, 
                use_attention=False, use_refinement=True, use_half_precision=True,
                block_size=64, overlap=8):
        """
        初始化BPNN模型
        
        参数同AttentionBPNN，但忽略attention参数
        """
        super(BPNN, self).__init__()
        
        # 如果启用了注意力，使用AttentionBPNN
        if use_attention:
            self.model = AttentionBPNN(
                max_disp=max_disp,
                feature_channels=feature_channels,
                iterations=iterations,
                use_refinement=use_refinement,
                use_half_precision=use_half_precision,
                block_size=block_size,
                overlap=overlap
            )
        else:
            # 创建一个不带注意力的简化模型
            self.max_disp = max_disp
            self.iterations = iterations
            self.use_refinement = use_refinement
            self.use_half_precision = use_half_precision
            self.block_size = block_size
            self.overlap = overlap
            
            # 特征提取网络
            self.feature_extractor = FeatureExtractorNetwork(
                in_channels=3, 
                base_channels=feature_channels
            )
            
            # 代价体计算网络
            self.cost_volume_net = CostVolumeNetwork(
                in_channels=feature_channels, 
                max_disp=max_disp
            )
            
            # 简化的消息传递卷积（所有方向共享权重）
            self.msg_pass = nn.Sequential(
                nn.Conv2d(feature_channels + 1, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1)
            )
            
            # 简化的信念更新网络
            self.belief_update = nn.Sequential(
                nn.Conv2d(5, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1)
            )
            
            # 视差细化网络
            if use_refinement:
                self.refinement = nn.Sequential(
                    nn.Conv2d(4, 16, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 1, kernel_size=3, padding=1)
                )
    
    def forward(self, left_img, right_img):
        """前向传播"""
        # 使用注意力模型
        if hasattr(self, 'model'):
            return self.model(left_img, right_img)
        
        # 使用半精度浮点数以节省显存
        if self.use_half_precision and left_img.is_cuda:
            left_img = left_img.half()
            right_img = right_img.half()
            self.to(torch.float16)
        
        # 特征提取
        left_feature = self.feature_extractor(left_img)
        right_feature = self.feature_extractor(right_img)
        
        # 构建代价体
        cost_volume = self.cost_volume_net(left_feature, right_feature)
        
        # 分块BP处理以节省显存
        _, _, _, height, width = cost_volume.shape
        
        # 将处理分成多个块
        h_blocks = (height + self.block_size - 1) // self.block_size
        w_blocks = (width + self.block_size - 1) // self.block_size
        
        # 初始化输出视差图
        disparity = torch.zeros((left_img.shape[0], 1, height, width), device=left_img.device)
        
        for h_idx in range(h_blocks):
            for w_idx in range(w_blocks):
                # 计算当前块的范围
                h_start = max(0, h_idx * self.block_size - self.overlap)
                h_end = min(height, (h_idx + 1) * self.block_size + self.overlap)
                w_start = max(0, w_idx * self.block_size - self.overlap)
                w_end = min(width, (w_idx + 1) * self.block_size + self.overlap)
                
                # 提取特征块
                feat_block = left_feature[:, :, h_start:h_end, w_start:w_end]
                cost_block = cost_volume[:, :, :, h_start:h_end, w_start:w_end]
                
                # 信念传播处理
                disp_block = self._process_block(feat_block, cost_block)
                
                # 计算有效区域（去除重叠部分）
                valid_h_start = 0 if h_idx == 0 else self.overlap
                valid_h_end = h_end - h_start if h_idx == h_blocks - 1 else min(h_end - h_start, (h_idx + 1) * self.block_size - h_start)
                valid_w_start = 0 if w_idx == 0 else self.overlap
                valid_w_end = w_end - w_start if w_idx == w_blocks - 1 else min(w_end - w_start, (w_idx + 1) * self.block_size - w_start)
                
                # 填充到输出视差图
                h_offset = h_idx * self.block_size
                w_offset = w_idx * self.block_size
                output_h_start = max(0, h_offset)
                output_h_end = min(height, h_offset + valid_h_end - valid_h_start)
                output_w_start = max(0, w_offset)
                output_w_end = min(width, w_offset + valid_w_end - valid_w_start)
                
                disparity[:, :, output_h_start:output_h_end, output_w_start:output_w_end] = \
                    disp_block[:, :, valid_h_start:valid_h_end, valid_w_start:valid_w_end]
        
        # 视差细化（如果需要）
        if self.use_refinement:
            # 将原始图像与视差连接起来
            refined_input = torch.cat([left_img, disparity], dim=1)
            residual = self.refinement(refined_input)
            disparity = disparity + residual
        
        # 恢复为单精度（如有必要）
        if self.use_half_precision and disparity.is_cuda:
            disparity = disparity.float()
            self.to(torch.float32)
        
        return {'disparity': disparity}
    
    def _process_block(self, feature_block, cost_block):
        """处理一个块的BP"""
        batch_size, _, max_disp, height, width = cost_block.shape
        
        # 初始化视差图
        disparity = torch.zeros((batch_size, 1, height, width), device=feature_block.device)
        
        # 对每个视差平面进行BP处理
        for d in range(max_disp):
            # 提取当前视差平面的代价
            cost = cost_block[:, :, d]
            
            # 简化的信念传播
            belief = self._belief_propagation_simple(feature_block, cost)
            
            # 更新视差图（根据最小代价）
            disparity = torch.where(belief < disparity, torch.full_like(disparity, d), disparity)
        
        return disparity
    
    def _belief_propagation_simple(self, feature, cost):
        """简化的BP过程"""
        batch_size, _, height, width = feature.shape
        
        # 初始化消息为零
        msg_up = torch.zeros_like(cost)
        msg_down = torch.zeros_like(cost)
        msg_left = torch.zeros_like(cost)
        msg_right = torch.zeros_like(cost)
        
        # 迭代BP（减少迭代次数）
        for _ in range(self.iterations):
            # 更新所有消息（简化版本）
            # 更新上方消息
            for i in range(1, height):
                msg_input = torch.cat([feature[:, :, i-1:i, :], 
                                     cost[:, :, i:i+1, :] + msg_down[:, :, i-1:i, :]], dim=1)
                msg_up[:, :, i-1:i, :] = self.msg_pass(msg_input)
            
            # 更新下方消息
            for i in range(height - 2, -1, -1):
                msg_input = torch.cat([feature[:, :, i+1:i+2, :], 
                                     cost[:, :, i:i+1, :] + msg_up[:, :, i+1:i+2, :]], dim=1)
                msg_down[:, :, i+1:i+2, :] = self.msg_pass(msg_input)
            
            # 更新左侧消息
            for j in range(1, width):
                msg_input = torch.cat([feature[:, :, :, j-1:j], 
                                     cost[:, :, :, j:j+1] + msg_right[:, :, :, j-1:j]], dim=1)
                msg_left[:, :, :, j-1:j] = self.msg_pass(msg_input)
            
            # 更新右侧消息
            for j in range(width - 2, -1, -1):
                msg_input = torch.cat([feature[:, :, :, j+1:j+2], 
                                     cost[:, :, :, j:j+1] + msg_left[:, :, :, j+1:j+2]], dim=1)
                msg_right[:, :, :, j+1:j+2] = self.msg_pass(msg_input)
        
        # 计算最终信念
        belief_input = torch.cat([cost, msg_up, msg_down, msg_left, msg_right], dim=1)
        belief = self.belief_update(belief_input)
        
        return belief