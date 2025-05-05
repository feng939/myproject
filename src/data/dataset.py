"""
数据集加载模块

该模块提供了加载和处理立体视觉数据集的功能。
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision import transforms

from .preprocessing import normalize_image, augment_stereo_pair


class StereoDataset(Dataset):
    """
    立体视觉数据集基类
    
    该类定义了立体数据集的基本接口和通用操作。
    """
    
    def __init__(self, 
                 root_dir,
                 split='train',
                 transform=None, 
                 augment=False,
                 max_disp=32,
                 resize=None):
        """
        初始化数据集
        
        参数:
            root_dir (str): 数据集根目录
            split (str): 数据集分割，'train', 'val', 或 'test'
            transform (callable, optional): 应用于样本的可选变换
            augment (bool): 是否进行数据增强
            max_disp (int): 最大视差值
            resize (tuple): 调整后的尺寸，如 (height, width)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.augment = augment
        self.max_disp = max_disp
        self.resize = resize
        
        # 子类需要实现样本列表的生成
        self.samples = []
    
    def __len__(self):
        """返回数据集中样本数量"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            dict: 包含左右图像和视差图的字典
        """
        # 获取样本路径
        sample_paths = self.samples[idx]
        
        # 加载左右图像
        left_img = Image.open(sample_paths['left']).convert('RGB')
        right_img = Image.open(sample_paths['right']).convert('RGB')
        
        # 将图像转换为NumPy数组
        left_img = np.array(left_img)
        right_img = np.array(right_img)
        
        # 加载视差图（如果有）
        disparity = None
        if 'disparity' in sample_paths and sample_paths['disparity']:
            # 加载视差图
            disp_path = sample_paths['disparity']
            if disp_path.endswith('.pfm'):
                disparity = self._read_pfm(disp_path)
            elif disp_path.endswith('.png'):
                disparity = np.array(Image.open(disp_path))
                # 一些数据集中PNG格式的视差图需要缩放
                if disparity.max() > self.max_disp:
                    disparity = disparity / 256.0
            else:
                # 默认处理
                disparity = np.array(Image.open(disp_path))
        
        # 调整图像和视差图大小（如果指定了resize）
        if self.resize is not None:
            # 保存原始宽度（用于计算视差值缩放因子）
            orig_width = left_img.shape[1]
            
            # 调整左右图像大小
            left_img = cv2.resize(left_img, (self.resize[1], self.resize[0]))
            right_img = cv2.resize(right_img, (self.resize[1], self.resize[0]))
            
            # 调整视差图大小和视差值
            if disparity is not None:
                # 计算视差值缩放因子（基于宽度变化）
                new_width = left_img.shape[1]
                disp_scale_factor = new_width / orig_width
                
                # 调整视差图
                disparity = cv2.resize(disparity, (self.resize[1], self.resize[0]), 
                                     interpolation=cv2.INTER_NEAREST)
                print(disparity)
                disparity = disparity * disp_scale_factor
        
        # 数据增强（可选）
        if self.augment and self.split == 'train':
            left_img, right_img, disparity = augment_stereo_pair(
                left_img, right_img, disparity)
        
        # 转换为Tensor
        left_img = self._to_tensor(left_img)
        right_img = self._to_tensor(right_img)
        
        # 转换视差图（如果有）
        if disparity is not None:
            # 确保视差值在有效范围内
            valid_mask = (disparity > 0) & (disparity < self.max_disp)
            
            if np.sum(valid_mask) > 0:
                # 无效区域设为0
                disparity[~valid_mask] = 0
                disparity = torch.from_numpy(disparity.astype(np.float32))
            else:
                # 如果没有有效视差值，设为零张量
                disparity = torch.zeros((left_img.shape[1], left_img.shape[2]), dtype=torch.float32)
            
            # 增加通道维度
            disparity = disparity.unsqueeze(0)
        else:
            # 如果没有视差图，提供零张量
            disparity = torch.zeros((1, left_img.shape[1], left_img.shape[2]), 
                                  dtype=torch.float32)
        
        # 应用变换（如果有）
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        return {
            'left': left_img,
            'right': right_img,
            'disparity': disparity,
            'left_filename': os.path.basename(sample_paths['left']),
            'right_filename': os.path.basename(sample_paths['right'])
        }
    
    def _to_tensor(self, img):
        """
        将图像转换为Tensor
        
        参数:
            img (numpy.ndarray): 输入图像
            
        返回:
            torch.Tensor: 转换后的图像Tensor
        """
        # 归一化图像到0-1
        img = normalize_image(img)
        
        # 转换为Tensor [C, H, W]
        img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
        
        return img
    
    def _read_pfm(self, file_path):
        """
        读取PFM格式的视差图
        
        参数:
            file_path (str): PFM文件路径
            
        返回:
            numpy.ndarray: 视差图
        """
        with open(file_path, 'rb') as f:
            header = f.readline().decode('UTF-8').rstrip()
            if header not in ['PF', 'Pf']:
                raise Exception('Not a PFM file.')

            dim_match = f.readline().decode('UTF-8')
            width, height = map(int, dim_match.split())
            
            scale = float(f.readline().decode('UTF-8').rstrip())
            little_endian = scale < 0
            scale = abs(scale)
            
            channels = 3 if header == 'PF' else 1

            data = np.fromfile(f, np.float32)
            shape = (height, width, channels) if channels > 1 else (height, width)
            data = np.reshape(data, shape)
            #data = data.reshape(height, width, channels) if channels > 1 else data.reshape(height, width)
            
            if little_endian:
                data = data.byteswap()
                
            data = np.flipud(data)

            return data


class MiddleburyDataset(StereoDataset):
    """
    Middlebury立体视觉数据集
    
    该类加载Middlebury数据集，一个广泛使用的立体视觉基准数据集。
    """
    
    def __init__(self, 
                 root_dir,
                 split='train',
                 transform=None, 
                 augment=False,
                 max_disp=32,
                 resize=None):
        """
        初始化Middlebury数据集
        
        参数与StereoDataset基类相同
        """
        super().__init__(root_dir, split, transform, augment, max_disp, resize)
        
        # 生成样本列表
        self._generate_samples()
    
    def _generate_samples(self):
        """生成Middlebury数据集的样本列表"""
        # 根据分割选择适当的文件夹
        split_dir = os.path.join(self.root_dir, self.split)
        
        # 确保目录存在
        if not os.path.exists(split_dir):
            raise ValueError(f"目录不存在: {split_dir}")
        
        # 查找所有场景
        scenes = [d for d in os.listdir(split_dir) 
                 if os.path.isdir(os.path.join(split_dir, d))]
                
        # 对于每个场景，找到左右图像和视差图
        for scene in scenes:
            scene_dir = os.path.join(split_dir, scene)
            
            # 左图像
            left_img_path = os.path.join(scene_dir, 'im0.png')
            if not os.path.exists(left_img_path):
                continue
            
            # 右图像
            right_img_path = os.path.join(scene_dir, 'im1.png')
            if not os.path.exists(right_img_path):
                continue
            
            # 视差图（如果存在）
            disp_path = os.path.join(scene_dir, 'disp0.pfm')
            if not os.path.exists(disp_path):
                disp_path = None
            
            # 添加到样本列表
            self.samples.append({
                'left': left_img_path,
                'right': right_img_path,
                'disparity': disp_path,
                'scene': scene
            })


class CustomStereoDataset(StereoDataset):
    """
    自定义立体视觉数据集
    
    该类加载用户自定义的立体视觉数据。
    """
    
    def __init__(self, 
                 left_dir,
                 right_dir,
                 disp_dir=None,
                 transform=None, 
                 augment=False,
                 max_disp=32,
                 resize=None):
        """
        初始化自定义数据集
        
        参数:
            left_dir (str): 左图像目录
            right_dir (str): 右图像目录
            disp_dir (str, optional): 视差图目录
            transform (callable, optional): 应用于样本的可选变换
            augment (bool): 是否进行数据增强
            max_disp (int): 最大视差值
            resize (tuple): 调整尺寸
        """
        # 不使用StereoDataset的初始化，因为目录结构不同
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.disp_dir = disp_dir
        self.transform = transform
        self.augment = augment
        self.max_disp = max_disp
        self.resize = resize
        
        # 生成样本列表
        self.samples = []
        self._generate_samples()
    
    def _generate_samples(self):
        """生成自定义数据集的样本列表"""
        # 确保目录存在
        if not os.path.exists(self.left_dir) or not os.path.exists(self.right_dir):
            raise ValueError(f"目录不存在: {self.left_dir} 或 {self.right_dir}")
        
        # 查找所有左图像
        left_imgs = sorted(glob.glob(os.path.join(self.left_dir, '*.png')) + 
                          glob.glob(os.path.join(self.left_dir, '*.jpg')))
        
        # 对于每个左图像，找到对应的右图像和视差图（如果有）
        for left_img_path in left_imgs:
            # 提取图像ID
            img_id = os.path.basename(left_img_path)
            
            # 右图像
            right_img_path = os.path.join(self.right_dir, img_id)
            if not os.path.exists(right_img_path):
                # 尝试不同的扩展名
                base_name = os.path.splitext(img_id)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    right_img_path = os.path.join(self.right_dir, base_name + ext)
                    if os.path.exists(right_img_path):
                        break
                else:
                    # 如果没有找到匹配的右图像，跳过这个样本
                    continue
            
            # 视差图（如果存在）
            disp_path = None
            if self.disp_dir is not None:
                # 尝试不同的扩展名
                base_name = os.path.splitext(img_id)[0]
                for ext in ['.png', '.pfm', '.npy', '.tiff']:
                    disp_path = os.path.join(self.disp_dir, base_name + ext)
                    if os.path.exists(disp_path):
                        break
                else:
                    disp_path = None
            
            # 添加到样本列表
            self.samples.append({
                'left': left_img_path,
                'right': right_img_path,
                'disparity': disp_path,
                'id': img_id
            })
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        实现与StereoDataset基类相同
        """
        # 使用StereoDataset基类的__getitem__实现
        sample = super().__getitem__(idx)
        return sample


def get_dataset(config):
    """
    获取数据集
    
    参数:
        config (dict): 数据集配置
            
    返回:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # 基本变换
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
    
    # 数据集类型
    dataset_type = config.get('type', 'middlebury')
    
    # 通用参数
    common_params = {
        'transform': transform,
        'augment': config.get('augment', False),
        'max_disp': config.get('max_disp', 32),
        'resize': config.get('resize', None)
    }
    
    # 根据数据集类型创建数据集
    if dataset_type.lower() == 'middlebury':
        train_dataset = MiddleburyDataset(
            root_dir=config['root_dir'],
            split='train',
            **common_params
        )
        val_dataset = MiddleburyDataset(
            root_dir=config['root_dir'],
            split='val',
            augment=False,  # 验证集不增强
            **{k: v for k, v in common_params.items() if k != 'augment'}
        )
        test_dataset = MiddleburyDataset(
            root_dir=config['root_dir'],
            split='test',
            augment=False,  # 测试集不增强
            **{k: v for k, v in common_params.items() if k != 'augment'}
        )
    elif dataset_type.lower() == 'custom':
        # 自定义数据集只有一个分割
        dataset = CustomStereoDataset(
            left_dir=config['left_dir'],
            right_dir=config['right_dir'],
            disp_dir=config.get('disp_dir', None),
            **common_params
        )
        
        # 如果提供了分割比例，则分割数据集
        if 'split_ratio' in config:
            split_ratio = config['split_ratio']
            total_len = len(dataset)
            
            # 计算分割索引
            train_len = int(total_len * split_ratio[0])
            val_len = int(total_len * split_ratio[1])
            
            # 创建数据子集
            indices = list(range(total_len))
            random.shuffle(indices)
            
            train_indices = indices[:train_len]
            val_indices = indices[train_len:train_len+val_len]
            test_indices = indices[train_len+val_len:]
            
            # 创建数据子集
            from torch.utils.data import Subset
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            test_dataset = Subset(dataset, test_indices)
        else:
            # 没有分割比例，全部用作测试集
            train_dataset = None
            val_dataset = None
            test_dataset = dataset
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    return train_dataset, val_dataset, test_dataset


def get_data_loaders(config):
    """
    获取数据加载器
    
    参数:
        config (dict): 数据集配置
            
    返回:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 获取数据集
    train_dataset, val_dataset, test_dataset = get_dataset(config)
    
    # 批大小
    batch_size = config.get('batch_size', 1)
    
    # 工作进程数
    num_workers = config.get('num_workers', 2)
    
    # 创建数据加载器
    train_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # 测试时通常使用batch_size=1
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader