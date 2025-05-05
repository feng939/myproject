"""
立体匹配损失函数模块

该模块提供了用于训练立体匹配模型的损失函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothL1Loss(nn.Module):
    """
    平滑L1损失函数
    
    适合于视差估计的鲁棒损失函数，比L1损失对异常值更不敏感。
    """
    
    def __init__(self, beta=1.0):
        """
        初始化平滑L1损失
        
        参数:
            beta (float): 平滑参数，控制平滑区域的大小
        """
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
    
    def forward(self, pred, target, mask=None):
        """
        计算损失
        
        参数:
            pred (torch.Tensor): 预测视差，形状为 [B, 1, H, W]
            target (torch.Tensor): 目标视差，形状为 [B, 1, H, W]
            mask (torch.Tensor, optional): 有效区域掩码，形状为 [B, 1, H, W]
            
        返回:
            torch.Tensor: 损失值
        """
        diff = torch.abs(pred - target)
        
        # 应用掩码（如果有的话）
        if mask is None:
            # 创建一个默认掩码，忽略视差为0的区域（通常是无效区域）
            mask = (target > 0).float()
        
        diff = diff * mask
        valid_pixels = torch.sum(mask) + 1e-6
        
        # 应用平滑L1
        loss = torch.where(diff < self.beta,
                          0.5 * diff ** 2 / self.beta,
                          diff - 0.5 * self.beta)
        
        # 返回平均损失
        return torch.sum(loss) / valid_pixels


class SSIM(nn.Module):
    """
    结构相似性（SSIM）损失
    
    度量图像结构相似性的损失函数，适用于立体匹配中。
    """
    
    def __init__(self, window_size=5, size_average=True):
        """
        初始化SSIM损失
        
        参数:
            window_size (int): 滑动窗口大小
            size_average (bool): 是否对结果求平均
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _create_window(self, window_size, channel):
        """创建高斯窗口"""
        def _gaussian(window_size, sigma):
            coords = torch.arange(window_size, dtype=torch.float)
            center = window_size // 2
            gauss = torch.exp(-((coords - center)**2) / (2 * sigma**2))
            return gauss / gauss.sum()
            
        _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        """
        计算SSIM损失
        
        参数:
            img1 (torch.Tensor): 第一个图像，形状为 [B, C, H, W]
            img2 (torch.Tensor): 第二个图像，形状为 [B, C, H, W]
            
        返回:
            torch.Tensor: 1 - SSIM(img1, img2)
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device)
            self.window = window
            self.channel = channel
            
        # 计算均值
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        # SSIM计算
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            ssim_value = ssim_map.mean()
        else:
            ssim_value = ssim_map.mean(1).mean(1).mean(1)
            
        # 返回SSIM损失
        return 1 - ssim_value


class DisparitySmoothness(nn.Module):
    """
    视差平滑损失
    
    鼓励视差图在图像边缘以外的区域保持平滑。
    """
    
    def __init__(self, edge_weight=10.0):
        """
        初始化视差平滑损失
        
        参数:
            edge_weight (float): 边缘权重，控制边缘处平滑程度
        """
        super(DisparitySmoothness, self).__init__()
        self.edge_weight = edge_weight
    
    def forward(self, disparity, image):
        """
        计算视差平滑损失
        
        参数:
            disparity (torch.Tensor): 视差图，形状为 [B, 1, H, W]
            image (torch.Tensor): 原始图像，形状为 [B, 3, H, W]
            
        返回:
            torch.Tensor: 平滑损失值
        """
        # 计算视差梯度
        disp_gradients_x = torch.abs(disparity[:, :, :, :-1] - disparity[:, :, :, 1:])
        disp_gradients_y = torch.abs(disparity[:, :, :-1, :] - disparity[:, :, 1:, :])
        
        # 计算图像梯度
        image_gradients_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
        image_gradients_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)
        
        # 在图像边缘处降低平滑损失权重
        weights_x = torch.exp(-self.edge_weight * image_gradients_x)
        weights_y = torch.exp(-self.edge_weight * image_gradients_y)
        
        # 计算加权平滑损失
        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y
        
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)


class StereoLoss(nn.Module):
    """
    立体匹配综合损失
    
    结合多种损失函数来训练立体匹配模型。
    """
    
    def __init__(self, 
                 smooth_l1_weight=1.0, 
                 ssim_weight=0.1, 
                 smoothness_weight=0.1):
        """
        初始化立体匹配损失
        
        参数:
            smooth_l1_weight (float): 平滑L1损失权重
            ssim_weight (float): SSIM损失权重
            smoothness_weight (float): 平滑损失权重
        """
        super(StereoLoss, self).__init__()
        self.smooth_l1_loss = SmoothL1Loss()
        self.ssim_loss = SSIM()
        self.smoothness_loss = DisparitySmoothness()
        
        self.smooth_l1_weight = smooth_l1_weight
        self.ssim_weight = ssim_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(self, output, target, left_img=None):
        """
        计算综合损失
        
        参数:
            output (torch.Tensor): 预测视差，形状为 [B, 1, H, W]
            target (torch.Tensor): 目标视差，形状为 [B, 1, H, W]
            left_img (torch.Tensor, optional): 左图像，形状为 [B, 3, H, W]
            
        返回:
            torch.Tensor: 综合损失值
        """
        # 创建掩码，忽略视差为0的区域
        mask = (target > 0).float()
        
        # 计算平滑L1损失
        l1_loss = self.smooth_l1_loss(output, target, mask)
        total_loss = self.smooth_l1_weight * l1_loss
        
        # 如果左图像不为空，计算SSIM损失和平滑损失
        if left_img is not None and self.ssim_weight > 0:
            # 计算SSIM损失（使用预测视差和真值视差）
            ssim_loss = self.ssim_loss(output * mask, target * mask)
            total_loss += self.ssim_weight * ssim_loss
        
        if left_img is not None and self.smoothness_weight > 0:
            # 计算视差平滑损失
            smoothness_loss = self.smoothness_loss(output, left_img)
            total_loss += self.smoothness_weight * smoothness_loss
        
        return total_loss