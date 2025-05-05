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
        if mask is not None:
            diff = diff * mask
            valid_pixels = torch.sum(mask) + 1e-6
        else:
            # 创建一个默认掩码，忽略视差为0的区域（通常是无效区域）
            default_mask = (target > 0).float()
            diff = diff * default_mask
            valid_pixels = torch.sum(default_mask) + 1e-6
        
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
    
    def __init__(self, window_size=11, size_average=True):
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
            # 创建坐标数组
            coords = torch.arange(window_size, dtype=torch.float)
            # 计算中心点
            center = window_size // 2
            # 创建高斯权重 - 使用vectorized操作代替列表推导
            gauss = torch.exp(-((coords - center)**2) / (2 * sigma**2))
            # 归一化

            # gauss = torch.Tensor([
            #    torch.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            #    for x in range(window_size)
            # ])
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


class DistributionFocalLoss(nn.Module):
    """
    分布焦点损失
    
    用于视差概率分布的焦点损失，鼓励模型对正确视差更加确定。
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        初始化分布焦点损失
        
        参数:
            alpha (float): 正样本权重
            gamma (float): 聚焦参数
        """
        super(DistributionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, prob_volume, target_disp):
        """
        计算损失
        
        参数:
            prob_volume (torch.Tensor): 视差概率体，形状为 [B, D, H, W]
            target_disp (torch.Tensor): 目标视差，形状为 [B, 1, H, W]
            
        返回:
            torch.Tensor: 损失值
        """
        batch_size, disp_range, height, width = prob_volume.shape
        
        # 创建目标视差索引
        target_idx = target_disp.long()
        
        # 裁剪索引在有效范围内
        target_idx = torch.clamp(target_idx, 0, disp_range - 1)
        
        # 创建掩码，忽略视差为0的区域
        valid_mask = (target_disp > 0).float()
        
        # 获取目标视差的预测概率
        target_prob = torch.gather(
            prob_volume,
            dim=1,
            index=target_idx
        )
        
        # 计算焦点损失
        pt = target_prob
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-6)
        
        # 应用掩码
        loss = loss * valid_mask
        
        # 计算平均损失
        valid_pixels = torch.sum(valid_mask) + 1e-6
        return torch.sum(loss) / valid_pixels


class StereoLoss(nn.Module):
    """
    立体匹配综合损失
    
    结合多种损失函数来训练立体匹配模型。
    """
    
    def __init__(self, 
                 smooth_l1_weight=1.0, 
                 ssim_weight=0.0, 
                 smoothness_weight=0.0,
                 focal_weight=0.0):
        """
        初始化立体匹配损失
        
        参数:
            smooth_l1_weight (float): 平滑L1损失权重
            ssim_weight (float): SSIM损失权重
            smoothness_weight (float): 平滑损失权重
            focal_weight (float): 焦点损失权重
        """
        super(StereoLoss, self).__init__()
        self.smooth_l1_loss = SmoothL1Loss()
        self.ssim_loss = SSIM()
        self.smoothness_loss = DisparitySmoothness()
        self.focal_loss = DistributionFocalLoss()
        
        self.smooth_l1_weight = smooth_l1_weight
        self.ssim_weight = ssim_weight
        self.smoothness_weight = smoothness_weight
        self.focal_weight = focal_weight
    
    def forward(self, output, target, left_img=None, prob_volume=None):
        """
        计算综合损失
        
        参数:
            output (torch.Tensor): 预测视差，形状为 [B, 1, H, W]
            target (torch.Tensor): 目标视差，形状为 [B, 1, H, W]
            left_img (torch.Tensor, optional): 左图像，形状为 [B, 3, H, W]
            prob_volume (torch.Tensor, optional): 视差概率体，形状为 [B, D, H, W]
            
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
            # 计算SSIM损失（需要重建右图像）
            # 这里我们简化处理，直接使用视差图作为输入
            ssim_loss = self.ssim_loss(output * mask, target * mask)
            total_loss += self.ssim_weight * ssim_loss
        
        if left_img is not None and self.smoothness_weight > 0:
            # 计算视差平滑损失
            smoothness_loss = self.smoothness_loss(output, left_img)
            total_loss += self.smoothness_weight * smoothness_loss
        
        # 如果概率体不为空，计算焦点损失
        if prob_volume is not None and self.focal_weight > 0:
            focal_loss = self.focal_loss(prob_volume, target)
            total_loss += self.focal_weight * focal_loss
        
        return total_loss


class CensusLoss(nn.Module):
    """
    Census变换损失函数
    
    使用Census变换进行立体匹配的损失函数，对光照变化具有鲁棒性。
    """
    
    def __init__(self, window_size=9, max_disp=192):
        """
        初始化Census损失
        
        参数:
            window_size (int): Census窗口大小
            max_disp (int): 最大视差值
        """
        super(CensusLoss, self).__init__()
        self.window_size = window_size
        self.max_disp = max_disp
        
        # 窗口大小应为奇数
        assert window_size % 2 == 1, "窗口大小必须为奇数"
        
        # 计算半窗口大小
        self.radius = window_size // 2
    
    def census_transform(self, img):
        """
        计算Census变换
        
        参数:
            img (torch.Tensor): 输入图像，形状为 [B, 1, H, W]
            
        返回:
            torch.Tensor: Census变换结果，形状为 [B, window_size*window_size, H, W]
        """
        batch_size, _, height, width = img.shape
        center = img[:, :, self.radius:-self.radius, self.radius:-self.radius]
        
        # 初始化Census结果
        census = torch.zeros(
            (batch_size, self.window_size * self.window_size, 
             height - 2 * self.radius, width - 2 * self.radius),
            device=img.device
        )
        
        # 计算Census变换
        idx = 0
        for i in range(self.window_size):
            for j in range(self.window_size):
                if i == self.radius and j == self.radius:
                    continue  # 跳过中心像素
                
                # 提取当前窗口像素
                window_pixel = img[:, :, i:i+height-2*self.radius, j:j+width-2*self.radius]
                
                # 计算当前像素与中心像素的比较结果
                census[:, idx] = (window_pixel > center).float().squeeze(1)
                idx += 1
        
        return census
    
    def hamming_distance(self, census1, census2):
        """
        计算Hamming距离
        
        参数:
            census1 (torch.Tensor): 第一个Census变换结果
            census2 (torch.Tensor): 第二个Census变换结果
            
        返回:
            torch.Tensor: Hamming距离
        """
        # 计算XOR结果
        xor_result = torch.abs(census1 - census2)
        
        # 沿着通道维度求和得到Hamming距离
        distance = torch.sum(xor_result, dim=1, keepdim=True)
        
        return distance
    
    def forward(self, left_img, right_img, target_disp):
        """
        计算Census损失
        
        参数:
            left_img (torch.Tensor): 左图像，形状为 [B, 1, H, W]
            right_img (torch.Tensor): 右图像，形状为 [B, 1, H, W]
            target_disp (torch.Tensor): 目标视差，形状为 [B, 1, H, W]
            
        返回:
            torch.Tensor: Census损失值
        """
        batch_size, _, height, width = left_img.shape
        
        # 计算Census变换
        left_census = self.census_transform(left_img)
        right_census = self.census_transform(right_img)
        
        # 初始化代价体
        pad_width = self.radius * 2
        cost_volume = torch.zeros(
            (batch_size, 1, self.max_disp, height - pad_width, width - pad_width),
            device=left_img.device
        )
        
        # 计算所有视差的Hamming距离
        for d in range(self.max_disp):
            if d > 0:
                # 沿x轴（宽度）移动右Census
                right_census_d = torch.zeros_like(right_census)
                right_census_d[:, :, :, 0:width-pad_width-d] = right_census[:, :, :, d:width-pad_width]
            else:
                right_census_d = right_census
            
            # 计算Hamming距离
            cost_volume[:, :, d] = self.hamming_distance(left_census, right_census_d)
        
        # 提取目标视差处的代价
        target_idx = target_disp[:, :, self.radius:-self.radius, self.radius:-self.radius].long()
        target_idx = torch.clamp(target_idx, 0, self.max_disp - 1)
        
        # 提取每个像素的最佳代价
        best_cost = torch.zeros_like(target_idx, dtype=torch.float32)
        for b in range(batch_size):
            for h in range(height - pad_width):
                for w in range(width - pad_width):
                    d = target_idx[b, 0, h, w]
                    best_cost[b, 0, h, w] = cost_volume[b, 0, d, h, w]
        
        # 创建掩码，忽略视差为0的区域
        mask = (target_disp[:, :, self.radius:-self.radius, self.radius:-self.radius] > 0).float()
        
        # 计算平均损失
        valid_pixels = torch.sum(mask) + 1e-6
        loss = torch.sum(best_cost * mask) / valid_pixels
        
        return loss