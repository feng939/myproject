"""
神经网络组件模块

该模块实现了用于立体匹配的神经网络组件，包括特征提取网络、
代价计算网络和消息传递网络等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractorNetwork(nn.Module):
    """
    特征提取网络
    
    该网络用于从输入图像中提取丰富的特征表示，
    这些特征将用于后续的视差估计。
    """
    
    def __init__(self, in_channels=3, base_channels=32):
        """
        初始化特征提取网络
        
        参数:
            in_channels (int): 输入图像的通道数
            base_channels (int): 基础通道数
        """
        super(FeatureExtractorNetwork, self).__init__()
        
        self.conv1a = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        self.conv2a = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1, stride=2)
        self.conv2b = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels*2)
        
        self.conv3a = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1, stride=2)
        self.conv3b = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=3, padding=1, stride=2, output_padding=1)
        
        self.conv_out = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入图像，形状为 [B, C, H, W]
            
        返回:
            torch.Tensor: 提取的特征，形状为 [B, base_channels, H, W]
        """
        # 第一层
        conv1 = F.relu(self.bn1(self.conv1b(F.relu(self.conv1a(x)))))
        
        # 第二层（下采样）
        conv2 = F.relu(self.bn2(self.conv2b(F.relu(self.conv2a(conv1)))))
        
        # 第三层（下采样）
        conv3 = F.relu(self.bn3(self.conv3b(F.relu(self.conv3a(conv2)))))
        
        # 上采样和跳跃连接
        upconv2 = F.relu(self.upconv2(conv3)) + conv2
        upconv1 = F.relu(self.upconv1(upconv2)) + conv1
        
        # 输出层
        out = self.conv_out(upconv1)
        
        return out


class CostVolumeNetwork(nn.Module):
    """
    代价体计算网络
    
    该网络用于计算左右图像特征之间的匹配代价，
    生成用于视差估计的三维代价体。
    """
    
    def __init__(self, in_channels, max_disp):
        """
        初始化代价体计算网络
        
        参数:
            in_channels (int): 输入特征的通道数
            max_disp (int): 最大视差值
        """
        super(CostVolumeNetwork, self).__init__()
        self.max_disp = max_disp
        self.in_channels = in_channels
        
        # 用于处理代价体的3D卷积网络
        self.conv3d_1 = nn.Conv3d(in_channels*2, 32, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn3d_1 = nn.BatchNorm3d(32)
        
        self.conv3d_3 = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3d_4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn3d_2 = nn.BatchNorm3d(64)
        
        self.deconv3d_1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3d_2 = nn.Conv3d(32, 1, kernel_size=3, padding=1)
        
    def build_cost_volume(self, left_feature, right_feature):
        """
        构建代价体
        
        参数:
            left_feature (torch.Tensor): 左图像特征，形状为 [B, C, H, W]
            right_feature (torch.Tensor): 右图像特征，形状为 [B, C, H, W]
            
        返回:
            torch.Tensor: 代价体，形状为 [B, 2*C, D, H, W]
        """
        batch_size, channels, height, width = left_feature.shape
        cost_volume = torch.zeros((batch_size, 2*channels, self.max_disp, height, width), 
                                 device=left_feature.device)
        
        for d in range(self.max_disp):
            if d > 0:
                # 沿x轴（宽度）移动右图像特征
                shifted_right = torch.zeros_like(right_feature)
                shifted_right[:, :, :, 0:width-d] = right_feature[:, :, :, d:width]
            else:
                shifted_right = right_feature
            
            # 连接左特征和移位后的右特征
            cost_volume[:, :channels, d, :, :] = left_feature
            cost_volume[:, channels:, d, :, :] = shifted_right
        
        return cost_volume
    
    def forward(self, left_feature, right_feature):
        """
        前向传播
        
        参数:
            left_feature (torch.Tensor): 左图像特征，形状为 [B, C, H, W]
            right_feature (torch.Tensor): 右图像特征，形状为 [B, C, H, W]
            
        返回:
            torch.Tensor: 代价体，形状为 [B, 1, D, H, W]
        """
        cost_volume = self.build_cost_volume(left_feature, right_feature)
        
        # 3D卷积处理代价体
        cost = F.relu(self.bn3d_1(self.conv3d_2(F.relu(self.conv3d_1(cost_volume)))))
        cost = F.relu(self.bn3d_2(self.conv3d_4(F.relu(self.conv3d_3(cost)))))
        
        # 上采样回原始尺寸
        cost = F.relu(self.deconv3d_1(cost))
        cost = self.deconv3d_2(cost)
        
        return cost


class MessagePassingNetwork(nn.Module):
    """
    消息传递网络
    
    该网络实现了基于神经网络的消息传递，替代传统信念传播中的消息更新规则，
    使得消息传递过程可以通过反向传播进行学习。
    """
    
    def __init__(self, feature_channels, hidden_channels=64):
        """
        初始化消息传递网络
        
        参数:
            feature_channels (int): 特征通道数
            hidden_channels (int): 隐藏层通道数
        """
        super(MessagePassingNetwork, self).__init__()
        
        # 消息生成网络
        self.msg_gen = nn.Sequential(
            nn.Conv2d(feature_channels*2 + 1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        )
        
        # 消息聚合网络
        self.msg_agg = nn.Sequential(
            nn.Conv2d(4 + 1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        )
    
    def generate_message(self, node_feature, neighbor_feature, node_data):
        """
        生成从node到neighbor的消息
        
        参数:
            node_feature (torch.Tensor): 节点特征，形状为 [B, C, H, W]
            neighbor_feature (torch.Tensor): 邻居节点特征，形状为 [B, C, H, W]
            node_data (torch.Tensor): 节点数据项，形状为 [B, 1, H, W]
            
        返回:
            torch.Tensor: 消息，形状为 [B, 1, H, W]
        """
        # 检查并调整通道维度
        b, c, h, w = node_feature.shape

        # 确保node_data的通道数为1
        if node_data.shape[1] != 1:
            node_data = node_data.mean(dim=1, keepdim=True)

        # 连接节点特征、邻居特征和数据项
        concat_feature = torch.cat([node_feature, neighbor_feature, node_data], dim=1)
        
        # 生成消息
        message = self.msg_gen(concat_feature)
        
        return message
    
    def aggregate_messages(self, data_term, msg_up, msg_down, msg_left, msg_right):
        """
        聚合各方向的消息以更新信念
        
        参数:
            data_term (torch.Tensor): 数据项，形状为 [B, 1, H, W]
            msg_up (torch.Tensor): 上方消息，形状为 [B, 1, H, W]
            msg_down (torch.Tensor): 下方消息，形状为 [B, 1, H, W]
            msg_left (torch.Tensor): 左方消息，形状为 [B, 1, H, W]
            msg_right (torch.Tensor): 右方消息，形状为 [B, 1, H, W]
            
        返回:
            torch.Tensor: 更新后的信念，形状为 [B, 1, H, W]
        """
        # 连接所有消息和数据项
        concat_msgs = torch.cat([msg_up, msg_down, msg_left, msg_right, data_term], dim=1)
        
        # 聚合消息更新信念
        belief = self.msg_agg(concat_msgs)
        
        return belief
    
    def forward(self, features, data_term, iterations=5):
        """
        前向传播
    
        参数:
            features (torch.Tensor): 节点特征，形状为 [B, C, H, W]
           data_term (torch.Tensor): 数据项，形状为 [B, 1, H, W]
            iterations (int): 消息传递迭代次数
        
        返回:
            torch.Tensor: 最终的信念，形状为 [B, 1, H, W]
        """
        batch_size, channel, height, width = features.shape
        device = features.device

        # 确保data_term的通道数为1
        if data_term.shape[1] != 1:
            data_term = data_term.mean(dim=1, keepdim=True)

        # 初始化消息为零
        msg_up = torch.zeros((batch_size, 1, height, width), device=device)
        msg_down = torch.zeros((batch_size, 1, height, width), device=device)
        msg_left = torch.zeros((batch_size, 1, height, width), device=device)
        msg_right = torch.zeros((batch_size, 1, height, width), device=device)

        # 迭代消息传递
        for _ in range(iterations):
            # 向上传递的消息（从下到上）
            msg_up_new = torch.zeros_like(msg_up)
            for i in range(1, height):
                node_feat = features[:, :, i, :]
                nbr_feat = features[:, :, i-1, :]
                node_data = data_term[:, :, i, :]
            
                # 调整特征维度，确保正确的通道排列
                # 从[B, C, W]变为[B, C, 1, W]
                node_feat = node_feat.unsqueeze(2)
                nbr_feat = nbr_feat.unsqueeze(2)
                node_data = node_data.unsqueeze(2)

                msg_up_new[:, :, i-1, :] = self.generate_message(node_feat, nbr_feat, node_data).squeeze(2)
        
            # 其余方向类似修改...
            # 向下传递的消息（从上到下）
            msg_down_new = torch.zeros_like(msg_down)
            for i in range(height-2, -1, -1):
                node_feat = features[:, :, i, :].unsqueeze(2)
                nbr_feat = features[:, :, i+1, :].unsqueeze(2)
                node_data = data_term[:, :, i, :].unsqueeze(2)
                msg_down_new[:, :, i+1, :] = self.generate_message(node_feat, nbr_feat, node_data).squeeze(2)
        
            # 向左传递的消息（从右到左）
            msg_left_new = torch.zeros_like(msg_left)
            for j in range(1, width):
                node_feat = features[:, :, :, j].unsqueeze(3)
                nbr_feat = features[:, :, :, j-1].unsqueeze(3)
                node_data = data_term[:, :, :, j].unsqueeze(3)
                msg_left_new[:, :, :, j-1] = self.generate_message(node_feat, nbr_feat, node_data).squeeze(3)
        
            # 向右传递的消息（从左到右）
            msg_right_new = torch.zeros_like(msg_right)
            for j in range(width-2, -1, -1):
                node_feat = features[:, :, :, j].unsqueeze(3)
                nbr_feat = features[:, :, :, j+1].unsqueeze(3)
                node_data = data_term[:, :, :, j].unsqueeze(3)
                msg_right_new[:, :, :, j+1] = self.generate_message(node_feat, nbr_feat, node_data).squeeze(3)
        
            # 更新消息
            msg_up = msg_up_new
            msg_down = msg_down_new
            msg_left = msg_left_new
            msg_right = msg_right_new

        # 聚合所有消息
        belief = self.aggregate_messages(data_term, msg_up, msg_down, msg_left, msg_right)
    
        return belief


class DisparityRefinementNetwork(nn.Module):
    """
    视差细化网络
    
    该网络用于细化和改进初始视差估计，处理遮挡区域和错误匹配。
    """
    
    def __init__(self, in_channels=4):
        """
        初始化视差细化网络
        
        参数:
            in_channels (int): 输入通道数（原始图像通道数+视差通道）
        """
        super(DisparityRefinementNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, img, disparity):
        """
        前向传播
        
        参数:
            img (torch.Tensor): 原始图像，形状为 [B, C, H, W]
            disparity (torch.Tensor): 初始视差图，形状为 [B, 1, H, W]
            
        返回:
            torch.Tensor: 细化后的视差图，形状为 [B, 1, H, W]
        """
        # 连接图像和视差
        x = torch.cat([img, disparity], dim=1)
        
        # 特征提取
        x = F.relu(self.bn1(self.conv2(F.relu(self.conv1(x)))))
        x = F.relu(self.bn2(self.conv4(F.relu(self.conv3(x)))))
        
        # 特征上采样
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        
        # 残差学习：预测视差的修正值
        residual = self.conv_out(x)
        
        # 添加残差到原始视差
        refined_disparity = disparity + residual
        
        return refined_disparity


class AttentionGuidedMessagePassing(nn.Module):
    """
    注意力引导的消息传递网络
    
    该网络使用注意力机制来选择性地传递消息，根据像素的相似性加权消息的重要性。
    """
    
    def __init__(self, feature_channels, hidden_channels=64):
        """
        初始化注意力引导的消息传递网络
        
        参数:
            feature_channels (int): 特征通道数
            hidden_channels (int): 隐藏层通道数
        """
        super(AttentionGuidedMessagePassing, self).__init__()
        
        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Conv2d(feature_channels*2, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 消息生成网络
        self.msg_gen = nn.Sequential(
            nn.Conv2d(feature_channels*2 + 1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        )
        
        # 消息聚合网络
        self.msg_agg = nn.Sequential(
            nn.Conv2d(4 + 1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        )
    
    def compute_attention(self, node_feature, neighbor_feature):
        """
        计算注意力权重
        
        参数:
            node_feature (torch.Tensor): 节点特征，形状为 [B, C, H, W]
            neighbor_feature (torch.Tensor): 邻居节点特征，形状为 [B, C, H, W]
            
        返回:
            torch.Tensor: 注意力权重，形状为 [B, 1, H, W]
        """
        # 连接节点特征和邻居特征
        concat_feature = torch.cat([node_feature, neighbor_feature], dim=1)
        
        # 计算注意力权重
        attention = self.attention_net(concat_feature)
        
        return attention
    
    def generate_message(self, node_feature, neighbor_feature, node_data):
        """
        生成从node到neighbor的消息，并用注意力加权
        
        参数:
            node_feature (torch.Tensor): 节点特征，形状为 [B, C, H, W]
            neighbor_feature (torch.Tensor): 邻居节点特征，形状为 [B, C, H, W]
            node_data (torch.Tensor): 节点数据项，形状为 [B, 1, H, W]
            
        返回:
            torch.Tensor: 加权消息，形状为 [B, 1, H, W]
        """
        # 计算注意力权重
        attention = self.compute_attention(node_feature, neighbor_feature)
        
        # 连接特征和数据项
        concat_feature = torch.cat([node_feature, neighbor_feature, node_data], dim=1)
        
        # 生成消息
        message = self.msg_gen(concat_feature)
        
        # 用注意力加权消息
        weighted_message = message * attention
        
        return weighted_message
    
    def aggregate_messages(self, data_term, msg_up, msg_down, msg_left, msg_right):
        """
        聚合各方向的消息以更新信念
        
        参数与MessagePassingNetwork中的相同
        """
        concat_msgs = torch.cat([msg_up, msg_down, msg_left, msg_right, data_term], dim=1)
        belief = self.msg_agg(concat_msgs)
        return belief
    
    def forward(self, features, data_term, iterations=5):
        """
        前向传播
        
        参数与MessagePassingNetwork中的相同
        """
        batch_size, _, height, width = features.shape
        device = features.device
        
        # 初始化消息
        msg_up = torch.zeros((batch_size, 1, height, width), device=device)
        msg_down = torch.zeros((batch_size, 1, height, width), device=device)
        msg_left = torch.zeros((batch_size, 1, height, width), device=device)
        msg_right = torch.zeros((batch_size, 1, height, width), device=device)
        
        # 迭代消息传递
        for _ in range(iterations):
            # 更新消息，实现与MessagePassingNetwork中类似，但使用注意力机制
            # 向上传递的消息
            msg_up_new = torch.zeros_like(msg_up)
            for i in range(1, height):
                node_feat = features[:, :, i, :]
                nbr_feat = features[:, :, i-1, :]
                node_data = data_term[:, :, i, :]
                msg_up_new[:, :, i-1, :] = self.generate_message(node_feat, nbr_feat, node_data)
            
            # 向下、向左、向右传递的消息，实现类似
            # 向下传递
            msg_down_new = torch.zeros_like(msg_down)
            for i in range(height-2, -1, -1):
                node_feat = features[:, :, i, :]
                nbr_feat = features[:, :, i+1, :]
                node_data = data_term[:, :, i, :]
                msg_down_new[:, :, i+1, :] = self.generate_message(node_feat, nbr_feat, node_data)
            
            # 向左传递
            msg_left_new = torch.zeros_like(msg_left)
            for j in range(1, width):
                node_feat = features[:, :, :, j]
                nbr_feat = features[:, :, :, j-1]
                node_data = data_term[:, :, :, j]
                msg_left_new[:, :, :, j-1] = self.generate_message(node_feat, nbr_feat, node_data)
            
            # 向右传递
            msg_right_new = torch.zeros_like(msg_right)
            for j in range(width-2, -1, -1):
                node_feat = features[:, :, :, j]
                nbr_feat = features[:, :, :, j+1]
                node_data = data_term[:, :, :, j]
                msg_right_new[:, :, :, j+1] = self.generate_message(node_feat, nbr_feat, node_data)
            
            # 更新消息
            msg_up = msg_up_new
            msg_down = msg_down_new
            msg_left = msg_left_new
            msg_right = msg_right_new
        
        # 聚合消息
        belief = self.aggregate_messages(data_term, msg_up, msg_down, msg_left, msg_right)
        
        return belief