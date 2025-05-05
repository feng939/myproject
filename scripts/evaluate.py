#!/usr/bin/env python
"""
评估脚本

该脚本用于评估BPNN模型在测试集上的性能。
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import cv2

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bpnn import BPNN
from src.data.dataset import get_data_loaders
from src.utils.metrics import compute_error_metrics
from src.utils.visualization import colorize_disparity


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估BPNN模型')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='结果保存目录')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID，-1表示使用CPU')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化视差图')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_model(model_path, config, device):
    """加载模型"""
    # 创建模型
    model_config = config['model']
    model = BPNN(
        max_disp=model_config.get('max_disp', 32),
        feature_channels=model_config.get('feature_channels', 16),
        iterations=model_config.get('iterations', 3),
        use_attention=model_config.get('use_attention', True),
        use_refinement=model_config.get('use_refinement', True),
        use_half_precision=model_config.get('use_half_precision', True),
        block_size=model_config.get('block_size', 64),
        overlap=model_config.get('overlap', 8)
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 如果启用半精度，立即转换模型
    if model_config.get('use_half_precision', True) and device.type == 'cuda':
        model = model.half()
    
    model.eval()
    return model


def evaluate(model, test_loader, device, save_dir=None, visualize=False):
    """评估模型"""
    model.eval()
    
    # 初始化指标
    metrics = {
        'epe': 0.0,
        'bad1': 0.0,
        'bad3': 0.0,
        'bad5': 0.0
    }
    
    # 创建保存目录（如果需要）
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if visualize:
            vis_dir = os.path.join(save_dir, 'visualization')
            os.makedirs(vis_dir, exist_ok=True)
    
    # 遍历测试集
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="评估中")):
            # 准备数据
            left_img = batch['left'].to(device)
            right_img = batch['right'].to(device)
            gt_disp = batch['disparity'].to(device)
            
            # 前向传播
            outputs = model(left_img, right_img)
            
            # 如果模型返回字典，提取视差图
            if isinstance(outputs, dict) and 'disparity' in outputs:
                pred_disp = outputs['disparity']
            else:
                pred_disp = outputs
            
            # 计算指标
            batch_metrics = compute_error_metrics(pred_disp, gt_disp)
            
            # 累加指标
            for key in metrics:
                metrics[key] += batch_metrics[key]
            
            # 可视化（如果需要）
            if visualize and save_dir:
                # 将tensor转换为numpy
                pred_np = pred_disp[0].squeeze().cpu().numpy()
                gt_np = gt_disp[0].squeeze().cpu().numpy()
                
                # 生成彩色视差图
                color_pred = colorize_disparity(pred_np)
                color_gt = colorize_disparity(gt_np)
                
                # 从batch获取图像名称
                left_filename = batch['left_filename'][0]
                name_without_ext = os.path.splitext(left_filename)[0]
                
                # 保存预测视差图
                pred_path = os.path.join(vis_dir, f'{name_without_ext}_pred.png')
                cv2.imwrite(pred_path, cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))
                
                # 保存真值视差图
                gt_path = os.path.join(vis_dir, f'{name_without_ext}_gt.png')
                cv2.imwrite(gt_path, cv2.cvtColor(color_gt, cv2.COLOR_RGB2BGR))
    
    # 计算平均指标
    num_samples = len(test_loader)
    for key in metrics:
        metrics[key] /= num_samples
    
    return metrics


def print_metrics(metrics):
    """打印评估指标"""
    print("\n评估结果:")
    print(f"平均终点误差 (EPE): {metrics['epe']:.4f}")
    print(f"错误像素比率 >1px: {metrics['bad1']:.4f}")
    print(f"错误像素比率 >3px: {metrics['bad3']:.4f}")
    print(f"错误像素比率 >5px: {metrics['bad5']:.4f}")


def save_metrics(metrics, save_dir):
    """保存评估指标"""
    metrics_path = os.path.join(save_dir, 'metrics.txt')
    
    with open(metrics_path, 'w') as f:
        f.write("评估指标:\n")
        f.write(f"平均终点误差 (EPE): {metrics['epe']:.4f}\n")
        f.write(f"错误像素比率 >1px: {metrics['bad1']:.4f}\n")
        f.write(f"错误像素比率 >3px: {metrics['bad3']:.4f}\n")
        f.write(f"错误像素比率 >5px: {metrics['bad5']:.4f}\n")
    
    print(f"指标已保存到 {metrics_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.model_path, config, device)
    print(f"模型已加载: {args.model_path}")
    
    # 加载数据
    _, _, test_loader = get_data_loaders(config['dataset'])
    print(f"数据集: {config['dataset'].get('type', 'unknown')}")
    print(f"测试集大小: {len(test_loader.dataset) if test_loader else 0}")
    
    # 评估模型
    metrics = evaluate(model, test_loader, device, args.save_dir, args.visualize)
    
    # 打印和保存指标
    print_metrics(metrics)
    if args.save_dir:
        save_metrics(metrics, args.save_dir)
    
    print("评估完成!")


if __name__ == '__main__':
    main()