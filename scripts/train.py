#!/usr/bin/env python
"""
训练脚本

该脚本用于训练BPNN模型。
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bpnn import BPNN
from src.data.dataset import get_data_loaders
from src.training.trainer import Trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练BPNN模型')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID，-1表示使用CPU')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_model(config):
    """创建模型"""
    model_config = config['model']
    
    # 创建BPNN模型
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
    
    # 移动到设备
    device = config['device']
    model = model.to(device)
    
    # 如果启用半精度，立即转换模型
    if model_config.get('use_half_precision', True) and device.type == 'cuda':
        model = model.half()
    
    return model


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 添加设备信息到配置中
    config['device'] = device
    
    # 创建模型
    model = create_model(config).to(device)
    print(f"模型类型: {config['model'].get('type', 'bpnn')}")
    
    # 加载数据
    train_loader, val_loader, test_loader = get_data_loaders(config['dataset'])
    print(f"数据集: {config['dataset'].get('type', 'unknown')}")
    print(f"训练集大小: {len(train_loader.dataset) if train_loader else 0}")
    print(f"验证集大小: {len(val_loader.dataset) if val_loader else 0}")
    print(f"测试集大小: {len(test_loader.dataset) if test_loader else 0}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training']
    )
    
    # 开始训练
    trainer.train()
    
    print("训练完成!")


if __name__ == '__main__':
    main()