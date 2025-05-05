"""
评估指标模块

该模块提供了立体匹配评估指标的计算功能。
"""

import numpy as np
import torch


def end_point_error(pred_disp, gt_disp, mask=None):
    """
    计算终点误差(End-Point Error, EPE)
    
    参数:
        pred_disp (torch.Tensor or numpy.ndarray): 预测视差
        gt_disp (torch.Tensor or numpy.ndarray): 真值视差
        mask (torch.Tensor or numpy.ndarray, optional): 有效区域掩码
        
    返回:
        float: 平均终点误差
    """
    # 转换为torch.Tensor
    if isinstance(pred_disp, np.ndarray):
        pred_disp = torch.from_numpy(pred_disp)
    if isinstance(gt_disp, np.ndarray):
        gt_disp = torch.from_numpy(gt_disp)
    if mask is None:
        # 创建默认掩码，忽略视差为0的区域
        mask = (gt_disp > 0).float()
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    
    # 计算误差
    error = torch.abs(pred_disp - gt_disp) * mask
    
    # 计算平均误差
    valid_pixels = torch.sum(mask)
    if valid_pixels > 0:
        epe = torch.sum(error) / valid_pixels
    else:
        epe = torch.tensor(0.0)
    
    return epe.item()


def bad_pixel_ratio(pred_disp, gt_disp, threshold=3.0, mask=None):
    """
    计算错误像素比率(Bad Pixel Ratio)
    
    参数:
        pred_disp (torch.Tensor or numpy.ndarray): 预测视差
        gt_disp (torch.Tensor or numpy.ndarray): 真值视差
        threshold (float): 错误像素阈值
        mask (torch.Tensor or numpy.ndarray, optional): 有效区域掩码
        
    返回:
        float: 错误像素比率
    """
    # 转换为torch.Tensor
    if isinstance(pred_disp, np.ndarray):
        pred_disp = torch.from_numpy(pred_disp)
    if isinstance(gt_disp, np.ndarray):
        gt_disp = torch.from_numpy(gt_disp)
    if mask is None:
        # 创建默认掩码，忽略视差为0的区域
        mask = (gt_disp > 0).float()
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    
    # 计算绝对误差和相对误差
    abs_err = torch.abs(pred_disp - gt_disp)
    rel_err = abs_err / (gt_disp + 1e-6)
    
    # 错误像素：绝对误差 > threshold 且 相对误差 > 0.05
    bad = ((abs_err > threshold) & (rel_err > 0.05)).float() * mask
    
    # 计算错误像素比率
    valid_pixels = torch.sum(mask)
    if valid_pixels > 0:
        bad_ratio = torch.sum(bad) / valid_pixels
    else:
        bad_ratio = torch.tensor(0.0)
    
    return bad_ratio.item()


def depth_accuracy(pred_disp, gt_disp, thresholds=[1.25, 1.25**2, 1.25**3], mask=None):
    """
    计算深度准确率(δ < threshold)
    
    参数:
        pred_disp (torch.Tensor or numpy.ndarray): 预测视差
        gt_disp (torch.Tensor or numpy.ndarray): 真值视差
        thresholds (list): 准确率阈值列表
        mask (torch.Tensor or numpy.ndarray, optional): 有效区域掩码
        
    返回:
        dict: 不同阈值下的准确率
    """
    # 转换为torch.Tensor
    if isinstance(pred_disp, np.ndarray):
        pred_disp = torch.from_numpy(pred_disp).float()
    if isinstance(gt_disp, np.ndarray):
        gt_disp = torch.from_numpy(gt_disp).float()
    if mask is None:
        # 创建默认掩码，忽略视差为0的区域
        mask = (gt_disp > 0).float()
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).float()
    
    # 防止除零
    pred_disp = torch.clamp(pred_disp, min=1e-6)
    gt_disp = torch.clamp(gt_disp, min=1e-6)
    
    # 计算视差比率
    # 视差与深度成反比，因此比率为反比
    # max(gt/pred, pred/gt)
    ratio = torch.max(gt_disp / pred_disp, pred_disp / gt_disp)
    
    # 计算不同阈值下的准确率
    results = {}
    valid_pixels = torch.sum(mask)
    
    for i, threshold in enumerate(thresholds):
        # 计算当前阈值下的准确率
        correct = (ratio < threshold).float() * mask
        if valid_pixels > 0:
            accuracy = torch.sum(correct) / valid_pixels
        else:
            accuracy = torch.tensor(0.0)
        
        results[f'delta{i+1}'] = accuracy.item()
    
    return results


def compute_error_metrics(pred_disp, gt_disp, mask=None):
    """
    计算所有误差指标
    
    参数:
        pred_disp (torch.Tensor or numpy.ndarray): 预测视差
        gt_disp (torch.Tensor or numpy.ndarray): 真值视差
        mask (torch.Tensor or numpy.ndarray, optional): 有效区域掩码
        
    返回:
        dict: 包含所有指标的字典
    """
    # 确保输入形状一致
    if isinstance(pred_disp, torch.Tensor) and len(pred_disp.shape) == 4:
        pred_disp = pred_disp.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    if isinstance(gt_disp, torch.Tensor) and len(gt_disp.shape) == 4:
        gt_disp = gt_disp.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    if mask is not None and isinstance(mask, torch.Tensor) and len(mask.shape) == 4:
        mask = mask.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    
    # 计算指标
    epe = end_point_error(pred_disp, gt_disp, mask)
    bad1 = bad_pixel_ratio(pred_disp, gt_disp, threshold=1.0, mask=mask)
    bad3 = bad_pixel_ratio(pred_disp, gt_disp, threshold=3.0, mask=mask)
    bad5 = bad_pixel_ratio(pred_disp, gt_disp, threshold=5.0, mask=mask)
    accuracies = depth_accuracy(pred_disp, gt_disp, mask=mask)
    
    # 合并结果
    metrics = {
        'epe': epe,
        'bad1': bad1,
        'bad3': bad3,
        'bad5': bad5,
        **accuracies
    }
    
    return metrics


def compute_error_for_batch(pred_disp, gt_disp, mask=None):
    """
    计算批次数据的误差指标
    
    参数:
        pred_disp (torch.Tensor): 预测视差，形状为 [B, 1, H, W]
        gt_disp (torch.Tensor): 真值视差，形状为 [B, 1, H, W]
        mask (torch.Tensor, optional): 有效区域掩码，形状为 [B, 1, H, W]
        
    返回:
        dict: 包含所有指标的字典
    """
    batch_size = pred_disp.shape[0]
    metrics = {
        'epe': 0.0,
        'bad1': 0.0,
        'bad3': 0.0,
        'bad5': 0.0,
        'delta1': 0.0,
        'delta2': 0.0,
        'delta3': 0.0
    }
    
    # 对每个样本计算指标
    for i in range(batch_size):
        sample_pred = pred_disp[i].squeeze(0)  # [1, H, W] -> [H, W]
        sample_gt = gt_disp[i].squeeze(0)
        sample_mask = None if mask is None else mask[i].squeeze(0)
        
        sample_metrics = compute_error_metrics(sample_pred, sample_gt, sample_mask)
        
        # 累加指标
        for key in metrics:
            metrics[key] += sample_metrics[key]
    
    # 计算平均值
    for key in metrics:
        metrics[key] /= batch_size
    
    return metrics


def mean_absolute_error(pred_disp, gt_disp, mask=None):
    """
    计算平均绝对误差(Mean Absolute Error, MAE)
    
    参数:
        pred_disp (torch.Tensor or numpy.ndarray): 预测视差
        gt_disp (torch.Tensor or numpy.ndarray): 真值视差
        mask (torch.Tensor or numpy.ndarray, optional): 有效区域掩码
        
    返回:
        float: 平均绝对误差
    """
    # 转换为torch.Tensor
    if isinstance(pred_disp, np.ndarray):
        pred_disp = torch.from_numpy(pred_disp)
    if isinstance(gt_disp, np.ndarray):
        gt_disp = torch.from_numpy(gt_disp)
    if mask is None:
        # 创建默认掩码，忽略视差为0的区域
        mask = (gt_disp > 0).float()
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    
    # 计算绝对误差
    error = torch.abs(pred_disp - gt_disp) * mask
    
    # 计算平均误差
    valid_pixels = torch.sum(mask)
    if valid_pixels > 0:
        mae = torch.sum(error) / valid_pixels
    else:
        mae = torch.tensor(0.0)
    
    return mae.item()


def root_mean_squared_error(pred_disp, gt_disp, mask=None):
    """
    计算均方根误差(Root Mean Squared Error, RMSE)
    
    参数:
        pred_disp (torch.Tensor or numpy.ndarray): 预测视差
        gt_disp (torch.Tensor or numpy.ndarray): 真值视差
        mask (torch.Tensor or numpy.ndarray, optional): 有效区域掩码
        
    返回:
        float: 均方根误差
    """
    # 转换为torch.Tensor
    if isinstance(pred_disp, np.ndarray):
        pred_disp = torch.from_numpy(pred_disp)
    if isinstance(gt_disp, np.ndarray):
        gt_disp = torch.from_numpy(gt_disp)
    if mask is None:
        # 创建默认掩码，忽略视差为0的区域
        mask = (gt_disp > 0).float()
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    
    # 计算平方误差
    squared_error = ((pred_disp - gt_disp) ** 2) * mask
    
    # 计算均方根误差
    valid_pixels = torch.sum(mask)
    if valid_pixels > 0:
        mse = torch.sum(squared_error) / valid_pixels
        rmse = torch.sqrt(mse)
    else:
        rmse = torch.tensor(0.0)
    
    return rmse.item()


def disparity_error_image(pred_disp, gt_disp, mask=None, max_error=10.0):
    """
    生成视差误差图像
    
    参数:
        pred_disp (numpy.ndarray): 预测视差
        gt_disp (numpy.ndarray): 真值视差
        mask (numpy.ndarray, optional): 有效区域掩码
        max_error (float): 最大误差值，用于归一化
        
    返回:
        numpy.ndarray: 误差图像，值范围为0-255
    """
    # 确保输入是NumPy数组
    if isinstance(pred_disp, torch.Tensor):
        pred_disp = pred_disp.detach().cpu().numpy()
    if isinstance(gt_disp, torch.Tensor):
        gt_disp = gt_disp.detach().cpu().numpy()
    if mask is None:
        # 创建默认掩码，忽略视差为0的区域
        mask = (gt_disp > 0).astype(np.float32)
    elif isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 计算绝对误差
    error = np.abs(pred_disp - gt_disp) * mask
    
    # 归一化误差到0-1范围
    normalized_error = np.clip(error / max_error, 0, 1)
    
    # 转换为0-255范围
    error_image = (normalized_error * 255).astype(np.uint8)
    
    # 无效区域设为0
    error_image[mask == 0] = 0
    
    return error_image