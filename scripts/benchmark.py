#!/usr/bin/env python
"""
基准测试脚本

该脚本用于基准测试BPNN模型的性能，包括处理速度和内存使用。
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
import psutil
import gc
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bpnn import BPNN
from src.inference.predict import DisparityPredictor
from src.utils.visualization import colorize_disparity


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BPNN性能基准测试')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径')
    parser.add_argument('--left', type=str, required=True,
                       help='左图像路径')
    parser.add_argument('--right', type=str, required=True,
                       help='右图像路径')
    parser.add_argument('--max_disp', type=int, default=32,
                       help='最大视差值')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                       help='结果输出路径')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID，-1表示使用CPU')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化处理结果')
    parser.add_argument('--use_attention', action='store_true',
                       help='使用注意力机制')
    parser.add_argument('--resolution_test', action='store_true',
                       help='测试不同分辨率')
    
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


def get_memory_usage():
    """获取当前内存使用情况"""
    # 系统内存使用
    system_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
    
    # PyTorch内存使用（如果可用）
    torch_memory = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    
    return system_memory, torch_memory


def run_benchmark(model, left_img, right_img, device, num_runs=10, visualize=False):
    """
    运行基准测试
    
    参数:
        model: 模型
        left_img: 左图像
        right_img: 右图像
        device: 计算设备
        num_runs: 运行次数
        visualize: 是否可视化结果
    
    返回:
        dict: 包含性能指标的字典
    """
    model.eval()
    
    # 创建预测器
    predictor = DisparityPredictor(model=model, device=device)
    
    # 初始化性能指标
    processing_times = []
    memory_usages = []
    
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 测量初始内存
    initial_system_memory, initial_torch_memory = get_memory_usage()
    
    # 预热
    _ = predictor.predict(left_img, right_img)
    
    # 进行多次测试
    for i in range(num_runs):
        # 测量内存
        before_system_memory, before_torch_memory = get_memory_usage()
        
        # 进行预测
        start_time = time.time()
        result = predictor.predict(left_img, right_img)
        disparity = result['disparity']
        end_time = time.time()
        
        # 记录处理时间
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        # 测量内存
        after_system_memory, after_torch_memory = get_memory_usage()
        memory_usage = {
            'system_diff': after_system_memory - before_system_memory,
            'torch_diff': after_torch_memory - before_torch_memory,
            'system_total': after_system_memory,
            'torch_total': after_torch_memory
        }
        memory_usages.append(memory_usage)
    
    # 可视化（如果需要）
    if visualize:
        # 生成彩色视差图
        color_pred = colorize_disparity(disparity)
        
        # 显示结果
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(left_img)
        plt.title('输入左图像')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(color_pred)
        plt.title('预测视差图')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # 计算平均性能指标
    avg_time = sum(processing_times) / len(processing_times)
    fps = 1.0 / avg_time
    
    # 计算内存使用
    avg_system_memory_diff = sum(m['system_diff'] for m in memory_usages) / len(memory_usages)
    avg_torch_memory_diff = sum(m['torch_diff'] for m in memory_usages) / len(memory_usages)
    max_system_memory = max(m['system_total'] for m in memory_usages)
    max_torch_memory = max(m['torch_total'] for m in memory_usages)
    
    # 计算峰值内存使用
    peak_memory = max_system_memory - initial_system_memory + max_torch_memory - initial_torch_memory
    
    # 收集结果
    results = {
        'avg_time': avg_time,
        'min_time': min(processing_times),
        'max_time': max(processing_times),
        'fps': fps,
        'avg_system_memory_diff': avg_system_memory_diff,
        'avg_torch_memory_diff': avg_torch_memory_diff,
        'peak_memory': peak_memory,
        'max_system_memory': max_system_memory,
        'max_torch_memory': max_torch_memory
    }
    
    return results, disparity


def test_resolution_scaling(model, left_img, right_img, device, resolutions=None):
    """
    测试不同分辨率的性能
    
    参数:
        model: 模型
        left_img: 左图像
        right_img: 右图像
        device: 计算设备
        resolutions: 要测试的分辨率列表
    
    返回:
        list: 包含不同分辨率性能指标的列表
    """
    if resolutions is None:
        # 默认测试分辨率
        resolutions = [
            (120, 160),    # 非常小
            (180, 240),    # 小
            (200, 200),    # 默认配置
            (240, 320),    # QVGA
            (480, 640),    # VGA（可能在较小显存下仍然可行）
        ]
    
    # 初始化结果
    resolution_results = []
    
    for height, width in resolutions:
        print(f"测试分辨率: {height}x{width}")
        
        # 调整图像大小
        left_resized = cv2.resize(left_img, (width, height))
        right_resized = cv2.resize(right_img, (width, height))
        
        # 创建预测器并指定目标大小
        predictor = DisparityPredictor(
            model=model, 
            device=device,
            target_size=(height, width)
        )
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 预热
        _ = predictor.predict(left_img, right_img)
        
        # 测量性能
        start_time = time.time()
        result = predictor.predict(left_img, right_img)
        processing_time = result['processing_time']
        
        # 测量内存
        _, torch_memory = get_memory_usage()
        
        # 计算FPS
        fps = 1.0 / processing_time
        
        # 添加到结果
        resolution_results.append({
            'height': height,
            'width': width,
            'resolution': f"{height}x{width}",
            'pixels': height * width,
            'time': processing_time,
            'fps': fps,
            'memory': torch_memory
        })
    
    return resolution_results


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(args.max_disp, args.use_attention)
    print(f"最大视差: {args.max_disp}")
    print(f"使用注意力: {args.use_attention}")
    
    # 加载模型权重
    if args.model_path:
        model = load_model(model, args.model_path, device)
        print(f"模型已加载: {args.model_path}")
    else:
        print("使用未训练的模型")
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    # 加载图像
    left_img = np.array(Image.open(args.left).convert('RGB'))
    right_img = np.array(Image.open(args.right).convert('RGB'))
    
    if args.resolution_test:
        # 运行分辨率测试
        resolution_results = test_resolution_scaling(model, left_img, right_img, device)
        
        # 创建结果DataFrame
        df = pd.DataFrame(resolution_results)
        
        # 打印结果
        print("\n分辨率测试结果:")
        print(df.to_string(index=False))
        
        # 保存结果
        output_path = os.path.splitext(args.output)[0] + '_resolution.csv'
        df.to_csv(output_path, index=False)
        print(f"分辨率测试结果已保存到: {output_path}")
        
        # 绘制结果
        plt.figure(figsize=(12, 10))
        
        # 绘制处理时间与分辨率关系
        plt.subplot(2, 1, 1)
        plt.plot(df['pixels'], df['time'], 'o-', linewidth=2)
        plt.title('处理时间与分辨率关系')
        plt.xlabel('像素数')
        plt.ylabel('处理时间（秒）')
        plt.grid(True)
        
        # 绘制FPS与分辨率关系
        plt.subplot(2, 1, 2)
        plt.plot(df['pixels'], df['fps'], 'o-', linewidth=2)
        plt.title('帧率与分辨率关系')
        plt.xlabel('像素数')
        plt.ylabel('帧率（FPS）')
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.splitext(args.output)[0] + '_resolution_chart.png'
        plt.savefig(chart_path)
        print(f"分辨率测试图表已保存到: {chart_path}")
        
        # 显示图表
        plt.show()
    
    else:
        # 标准性能测试
        print("运行标准性能测试...")
        
        # 运行基准测试
        results, disparity = run_benchmark(
            model=model,
            left_img=left_img,
            right_img=right_img,
            device=device,
            visualize=args.visualize
        )
        
        # 打印结果
        print("\n基准测试结果:")
        print(f"平均处理时间: {results['avg_time']:.4f}秒")
        print(f"最小处理时间: {results['min_time']:.4f}秒")
        print(f"最大处理时间: {results['max_time']:.4f}秒")
        print(f"帧率 (FPS): {results['fps']:.2f}")
        print(f"平均系统内存增加: {results['avg_system_memory_diff']:.2f} MB")
        print(f"平均PyTorch内存增加: {results['avg_torch_memory_diff']:.2f} MB")
        print(f"峰值内存使用: {results['peak_memory']:.2f} MB")
        
        # 保存结果
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        # 创建结果字典
        result_dict = {
            'model_type': 'bpnn_attention' if args.use_attention else 'bpnn',
            'max_disp': args.max_disp,
            'device': device.type,
            'avg_time': results['avg_time'],
            'min_time': results['min_time'],
            'max_time': results['max_time'],
            'fps': results['fps'],
            'avg_system_memory_diff': results['avg_system_memory_diff'],
            'avg_torch_memory_diff': results['avg_torch_memory_diff'],
            'peak_memory': results['peak_memory'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 将结果保存为CSV
        df = pd.DataFrame([result_dict])
        df.to_csv(args.output, index=False)
        print(f"结果已保存到: {args.output}")


if __name__ == '__main__':
    main()