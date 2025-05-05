#!/usr/bin/env python
"""
演示脚本

该脚本用于演示BPNN模型的视差估计效果。
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bpnn import BPNN
from src.inference.predict import DisparityPredictor
from src.utils.visualization import (
    colorize_disparity, 
    create_anaglyph, 
    create_disparity_visualization
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BPNN视差估计演示')
    parser.add_argument('--left', type=str, required=True,
                       help='左图像路径')
    parser.add_argument('--right', type=str, required=True,
                       help='右图像路径')
    parser.add_argument('--gt', type=str, default=None,
                       help='真值视差图路径（可选）')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径（可选）')
    parser.add_argument('--max_disp', type=int, default=32,
                       help='最大视差值')
    parser.add_argument('--output', type=str, default='demo_output.png',
                       help='输出图像路径')
    parser.add_argument('--save_disparity', type=str, default=None,
                       help='保存视差图的路径')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID，-1表示使用CPU')
    parser.add_argument('--use_attention', action='store_true',
                        help='使用注意力机制')
    parser.add_argument('--target_size', type=str, default=None,
                       help='目标大小，格式为HxW，例如：200x200')
    
    return parser.parse_args()


def create_model(max_disp, use_attention):
    """创建模型"""
    model = BPNN(
        max_disp=max_disp,
        feature_channels=16,
        iterations=3,
        use_attention=use_attention,
        use_refinement=True,
        use_half_precision=True
    )
    
    return model


def load_model(model, model_path, device):
    """加载模型权重"""
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def load_image(image_path):
    """加载图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    
    return img


def load_disparity(disp_path):
    """加载视差图"""
    if not os.path.exists(disp_path):
        raise FileNotFoundError(f"视差图文件不存在: {disp_path}")
    
    # 默认读取图像
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    
    # 处理不同格式的视差图
    if disp.dtype == np.uint16:
        # 16位图像，通常需要缩放
        return disp.astype(np.float32) / 256.0
    elif disp.dtype == np.uint8:
        # 8位图像，通常需要缩放
        return disp.astype(np.float32)
    else:
        return disp


def parse_target_size(target_size_str):
    """解析目标大小字符串"""
    if target_size_str:
        h, w = map(int, target_size_str.split('x'))
        return (h, w)
    return None


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 解析目标大小
    target_size = parse_target_size(args.target_size)
    
    # 创建模型
    model = create_model(args.max_disp, args.use_attention)
    print(f"最大视差: {args.max_disp}")
    print(f"使用注意力: {args.use_attention}")
    
    # 加载模型权重
    model = load_model(model, args.model_path, device)
    if args.model_path:
        print(f"模型已加载: {args.model_path}")
    else:
        print("使用未训练的模型")
    
    # 创建预测器
    predictor = DisparityPredictor(
        model=model,
        device=device,
        target_size=target_size
    )
    
    # 加载图像
    left_img = load_image(args.left)
    right_img = load_image(args.right)
    print(f"图像大小: {left_img.shape[:2]}")
    
    # 加载真值视差图（如果有）
    gt_disp = None
    if args.gt:
        gt_disp = load_disparity(args.gt)
        print(f"真值视差图已加载: {args.gt}")
    
    # 预测视差图
    start_time = time.time()
    result = predictor.predict(left_img, right_img)
    disparity = result['disparity']
    processing_time = result['processing_time']
    print(f"处理时间: {processing_time:.3f}秒")
    
    # 计算统计信息
    valid_mask = disparity > 0
    min_disp = disparity[valid_mask].min() if np.any(valid_mask) else 0
    max_disp = disparity[valid_mask].max() if np.any(valid_mask) else 0
    mean_disp = disparity[valid_mask].mean() if np.any(valid_mask) else 0
    print(f"最小视差: {min_disp:.2f}")
    print(f"最大视差: {max_disp:.2f}")
    print(f"平均视差: {mean_disp:.2f}")
    
    # 保存视差图（如果需要）
    if args.save_disparity:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(args.save_disparity)), exist_ok=True)
        
        # 保存为PNG
        # 生成彩色视差图
        color_disparity = colorize_disparity(disparity)
        cv2.imwrite(args.save_disparity, cv2.cvtColor(color_disparity, cv2.COLOR_RGB2BGR))
        
        print(f"视差图已保存到: {args.save_disparity}")
    
    # 创建可视化
    vis_img = create_disparity_visualization(left_img, right_img, disparity, gt_disp)
    
    # 保存可视化
    cv2.imwrite(args.output, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"可视化结果已保存到: {args.output}")
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    plt.imshow(vis_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()