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
from tqdm import tqdm
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bpnn import BPNN, HierarchicalBPNN, DualBPNN
from src.inference.predict import DisparityPredictor
from src.utils.visualization import colorize_disparity
from src.data.dataset import get_data_loaders


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BPNN性能基准测试')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='数据集路径')
    parser.add_argument('--dataset_type', type=str, default='middlebury',
                       choices=['middlebury', 'kitti', 'eth3d', 'custom'],
                       help='数据集类型')
    parser.add_argument('--model_type', type=str, default='bpnn',
                       choices=['bpnn', 'hierarchical', 'dual'],
                       help='模型类型')
    parser.add_argument('--max_disp', type=int, default=64,
                       help='最大视差值')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='测试样本数量')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                       help='结果输出路径')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID，-1表示使用CPU')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化处理结果')
    parser.add_argument('--resolution_test', action='store_true',
                       help='测试不同分辨率')
    
    return parser.parse_args()


def create_model(model_type, max_disp):
    """创建模型"""
    if model_type == 'hierarchical':
        model = HierarchicalBPNN(
            max_disp=max_disp,
            feature_channels=32,
            num_scales=3,
            scale_factor=0.5
        )
    elif model_type == 'dual':
        model = DualBPNN(
            max_disp=max_disp,
            feature_channels=32,
            iterations=5
        )
    else:  # 默认BPNN
        model = BPNN(
            max_disp=max_disp,
            feature_channels=32,
            iterations=5,
            use_attention=False,
            use_refinement=True
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


def run_benchmark(model, test_loader, device, num_samples, visualize=False):
    """
    运行基准测试
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_samples: 测试样本数量
        visualize: 是否可视化结果
    
    返回:
        dict: 包含性能指标的字典
    """
    model.eval()
    
    # 初始化性能指标
    processing_times = []
    memory_usages = []
    
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 测量初始内存
    initial_system_memory, initial_torch_memory = get_memory_usage()
    
    # 遍历测试样本
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="基准测试中")):
            if i >= num_samples:
                break
            
            # 准备数据
            left_img = batch['left'].to(device)
            right_img = batch['right'].to(device)
            
            # 测量内存
            before_system_memory, before_torch_memory = get_memory_usage()
            
            # 测量处理时间
            start_time = time.time()
            
            # 前向传播
            outputs = model(left_img, right_img)
            
            # 如果模型返回字典，提取视差图
            if isinstance(outputs, dict) and 'disparity' in outputs:
                pred_disp = outputs['disparity']
            else:
                pred_disp = outputs
            
            # 同步GPU（如果使用）
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 记录处理时间
            processing_time = time.time() - start_time
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
                # 将tensor转换为numpy
                pred_np = pred_disp[0].squeeze().cpu().numpy()
                left_np = batch['left'][0].permute(1, 2, 0).cpu().numpy()
                
                # 如果值在0-1范围，转换为0-255
                if left_np.max() <= 1.0:
                    left_np = (left_np * 255).astype(np.uint8)
                
                # 生成彩色视差图
                color_pred = colorize_disparity(pred_np)
                
                # 显示结果
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.imshow(left_np)
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
    
    return results


def test_resolution_scaling(model, sample_pair, device, resolutions=None):
    """
    测试不同分辨率的性能
    
    参数:
        model: 模型
        sample_pair: (左图像, 右图像)元组
        device: 计算设备
        resolutions: 要测试的分辨率列表
    
    返回:
        dict: 包含不同分辨率性能指标的字典
    """
    if resolutions is None:
        # 默认测试分辨率
        resolutions = [
            (240, 320),    # QVGA
            (480, 640),    # VGA
            (720, 1280),   # 720p HD
            (1080, 1920)   # 1080p Full HD
        ]
    
    # 提取原始图像
    left_img, right_img = sample_pair
    
    # 初始化结果
    resolution_results = []
    
    for height, width in resolutions:
        print(f"测试分辨率: {height}x{width}")
        
        # 调整图像大小
        left_resized = cv2.resize(left_img, (width, height))
        right_resized = cv2.resize(right_img, (width, height))
        
        # 转换为张量
        left_tensor = torch.from_numpy(left_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        right_tensor = torch.from_numpy(right_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        
        # 归一化（如果需要）
        if left_tensor.max() > 1.0:
            left_tensor = left_tensor / 255.0
        if right_tensor.max() > 1.0:
            right_tensor = right_tensor / 255.0
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 测量内存和时间
        with torch.no_grad():
            # 测量初始内存
            initial_system_memory, initial_torch_memory = get_memory_usage()
            
            # 预热
            model(left_tensor, right_tensor)
            
            # 运行多次并测量时间
            num_runs = 5
            times = []
            
            for _ in range(num_runs):
                # 测量处理时间
                start_time = time.time()
                
                # 前向传播
                model(left_tensor, right_tensor)
                
                # 同步GPU（如果使用）
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # 记录处理时间
                times.append(time.time() - start_time)
            
            # 测量最终内存
            final_system_memory, final_torch_memory = get_memory_usage()
        
        # 计算结果
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        memory_diff = (final_system_memory - initial_system_memory) + (final_torch_memory - initial_torch_memory)
        
        # 添加到结果
        result = {
            'height': height,
            'width': width,
            'resolution': f"{height}x{width}",
            'pixels': height * width,
            'avg_time': avg_time,
            'fps': fps,
            'memory_diff': memory_diff
        }
        resolution_results.append(result)
    
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
    model = create_model(args.model_type, args.max_disp)
    print(f"模型类型: {args.model_type}")
    
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
    
    if args.resolution_test:
        # 为分辨率测试加载样本图像
        print("加载样本图像用于分辨率测试...")
        
        # 查找样本图像
        if args.dataset_path:
            # 尝试从数据集中找到样本
            sample_found = False
            
            # 尝试查找常见的测试图像目录结构
            test_paths = [
                os.path.join(args.dataset_path, 'test'),
                os.path.join(args.dataset_path, 'testing'),
                os.path.join(args.dataset_path, 'val'),
                os.path.join(args.dataset_path, 'validation'),
                args.dataset_path
            ]
            
            for test_path in test_paths:
                if os.path.exists(test_path):
                    # 查找图像文件
                    for root, _, files in os.walk(test_path):
                        image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
                        if len(image_files) >= 2:
                            # 假设前两个图像是左右图像对
                            left_img_path = os.path.join(root, image_files[0])
                            right_img_path = os.path.join(root, image_files[1])
                            
                            left_img = np.array(Image.open(left_img_path).convert('RGB'))
                            right_img = np.array(Image.open(right_img_path).convert('RGB'))
                            
                            sample_pair = (left_img, right_img)
                            sample_found = True
                            break
                
                if sample_found:
                    break
            
            if not sample_found:
                # 如果无法找到样本，使用随机生成的图像
                print("无法从数据集中找到样本图像，使用随机生成的图像")
                left_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                right_img = np.roll(left_img, shift=-20, axis=1)  # 简单移位模拟视差
                sample_pair = (left_img, right_img)
        else:
            # 使用随机生成的图像
            print("使用随机生成的图像")
            left_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            right_img = np.roll(left_img, shift=-20, axis=1)  # 简单移位模拟视差
            sample_pair = (left_img, right_img)
        
        # 运行分辨率测试
        resolution_results = test_resolution_scaling(model, sample_pair, device)
        
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
        plt.plot(df['pixels'], df['avg_time'], 'o-', linewidth=2)
        plt.title('处理时间与分辨率关系')
        plt.xlabel('像素数（百万）')
        plt.ylabel('处理时间（秒）')
        plt.grid(True)
        
        # 绘制FPS与分辨率关系
        plt.subplot(2, 1, 2)
        plt.plot(df['pixels'], df['fps'], 'o-', linewidth=2)
        plt.title('帧率与分辨率关系')
        plt.xlabel('像素数（百万）')
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
        # 创建数据集配置
        dataset_config = {
            'type': args.dataset_type,
            'root_dir': args.dataset_path,
            'batch_size': 1,
            'num_workers': 4
        }
        
        # 加载数据
        _, _, test_loader = get_data_loaders(dataset_config)
        
        if test_loader is None or len(test_loader) == 0:
            print("无法加载测试数据，请检查数据集路径和类型")
            return
        
        print(f"数据集: {args.dataset_type}")
        print(f"测试集大小: {len(test_loader.dataset)}")
        
        # 运行基准测试
        results = run_benchmark(
            model=model,
            test_loader=test_loader,
            device=device,
            num_samples=min(args.num_samples, len(test_loader)),
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
            'model_type': args.model_type,
            'max_disp': args.max_disp,
            'dataset': args.dataset_type,
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