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
    
    # 合并结果
    metrics = {
        'epe': epe,
        'bad1': bad1,
        'bad3': bad3,
        'bad5': bad5
    }
    
    return metrics