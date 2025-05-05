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
                 max_disp=192,
                 crop_size=None,
                 resize_size=None):
        """
        初始化数据集
        
        参数:
            root_dir (str): 数据集根目录
            split (str): 数据集分割，'train', 'val', 或 'test'
            transform (callable, optional): 应用于样本的可选变换
            augment (bool): 是否进行数据增强
            max_disp (int): 最大视差值
            crop_size (tuple): 裁剪尺寸，如 (height, width)
            resize_size (tuple): 调整后的尺寸，如 (height, width)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.augment = augment
        self.max_disp = max_disp
        self.crop_size = crop_size
        self.resize_size = resize_size
        
        # 子类需要实现样本列表的生成
        self.samples = []
        
    def _resize_with_aspect_ratio(self, img, target_size, is_disparity=False, scale_factor=None):
        """
        保持宽高比调整图像或视差图大小，并进行中心填充
    
        参数:
            img: numpy.ndarray 类型的图像或视差图
            target_size: 目标尺寸 (height, width)
            is_disparity: 是否为视差图
            scale_factor: 视差值缩放因子，如果为None则自动计算
        
        返回:
            调整大小后的图像或视差图
        """
        # 获取原始尺寸
        if len(img.shape) == 3:  # RGB 图像
            orig_height, orig_width, _ = img.shape
        else:  # 视差图
            orig_height, orig_width = img.shape

        target_height, target_width = target_size
    
        # 计算缩放比例
        width_ratio = target_width / orig_width
        height_ratio = target_height / orig_height
        ratio = min(width_ratio, height_ratio)
        
        # 计算新的尺寸
        new_width = int(orig_width * ratio)
        new_height = int(orig_height * ratio)
        
        # 调整图像/视差图大小
        if is_disparity:
            # 对视差图使用最近邻插值
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            # 调整视差值（视差值需要跟随宽度缩放）
            if scale_factor is None:
                scale_factor = ratio  # 默认使用相同的缩放比例
            resized_img = resized_img * scale_factor
        else:
            # 对普通图像使用双线性插值
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 如果调整后的尺寸与目标尺寸不同，进行填充
        if new_width != target_width or new_height != target_height:
            # 创建目标大小的图像/视差图
            if len(img.shape) == 3:  # RGB 图像
                result_img = np.zeros((target_height, target_width, img.shape[2]), dtype=img.dtype)
            else:  # 视差图
                result_img = np.zeros((target_height, target_width), dtype=img.dtype)
            
            # 计算粘贴位置（居中）
            paste_x = max(0, (target_width - new_width) // 2)
            paste_y = max(0, (target_height - new_height) // 2)
            
            # 粘贴调整大小后的图像/视差图
            if len(img.shape) == 3:  # RGB 图像
                result_img[paste_y:paste_y+new_height, paste_x:paste_x+new_width, :] = resized_img
            else:  # 视差图
                result_img[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = resized_img
            
            return result_img
        else:
            return resized_img

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
            # 根据文件类型加载视差图
            disp_path = sample_paths['disparity']
            if disp_path.endswith('.pfm'):
                disparity = self._read_pfm(disp_path)
            elif disp_path.endswith('.png'):
                disparity = np.array(Image.open(disp_path))
                # 一些数据集中PNG格式的视差图需要缩放
                if disparity.max() > self.max_disp:
                    disparity = disparity / 256.0
            elif disp_path.endswith('.npy'):
                disparity = np.load(disp_path)
            else:
                # 默认处理
                disparity = np.array(Image.open(disp_path))
        
        # 裁剪图像（如果指定了crop_size）
        if self.crop_size is not None:
            left_img, right_img, disparity = self._random_crop(
                left_img, right_img, disparity)
        
        # 调整图像和视差图大小（如果指定了resize_size）
        if self.resize_size is not None:
            # 保存原始宽度（用于计算视差值缩放因子）
            orig_width = left_img.shape[1]
            
            # 调整左右图像大小
            left_img = self._resize_with_aspect_ratio(left_img, self.resize_size)
            right_img = self._resize_with_aspect_ratio(right_img, self.resize_size)
            
            # 调整视差图大小和视差值
            if disparity is not None:
                # 计算视差值缩放因子（基于宽度变化）
                new_width = left_img.shape[1]
                disp_scale_factor = new_width / orig_width
                
                # 调整视差图
                disparity = self._resize_with_aspect_ratio(
                    disparity, self.resize_size, is_disparity=True, scale_factor=disp_scale_factor)
        
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
                disparity = torch.zeros(left_img.shape[1:3], dtype=torch.float32)
            
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
    
    def _random_crop(self, left_img, right_img, disparity=None):
        """
        随机裁剪图像对
        
        参数:
            left_img (numpy.ndarray): 左图像
            right_img (numpy.ndarray): 右图像
            disparity (numpy.ndarray, optional): 视差图
            
        返回:
            tuple: 裁剪后的图像和视差图
        """
        h, w = left_img.shape[:2]
        crop_h, crop_w = self.crop_size
        
        # 确保裁剪尺寸不超过图像尺寸
        crop_h = min(h, crop_h)
        crop_w = min(w, crop_w)
        
        # 随机选择裁剪位置
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        
        # 裁剪图像
        left_img_cropped = left_img[start_h:start_h+crop_h, start_w:start_w+crop_w]
        right_img_cropped = right_img[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # 裁剪视差图（如果有）
        disparity_cropped = None
        if disparity is not None:
            disparity_cropped = disparity[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        return left_img_cropped, right_img_cropped, disparity_cropped
    
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
            if header == 'PF':
                channels = 3
            elif header == 'Pf':
                channels = 1
            else:
                raise Exception('Not a PFM file.')

            dim_match = f.readline().decode('UTF-8')
            width, height = map(int, dim_match.split())
            
            scale = float(f.readline().decode('UTF-8').rstrip())
            little_endian = scale < 0
            scale = abs(scale)
            
            data = np.fromfile(f, np.float32)
            data = data.reshape(height, width, channels) if channels > 1 else data.reshape(height, width)
            
            if little_endian:
                data = data.byteswap()
                
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
                 max_disp=192,
                 crop_size=None,
                 resize_size=None):
        """
        初始化Middlebury数据集
        
        参数与StereoDataset基类相同
        """
        super().__init__(root_dir, split, transform, augment, max_disp, crop_size, resize_size)
        
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


class KITTIDataset(StereoDataset):
    """
    KITTI立体视觉数据集
    
    该类加载KITTI数据集，一个自动驾驶场景的立体视觉数据集。
    """
    
    def __init__(self, 
                 root_dir,
                 split='train',
                 transform=None, 
                 augment=False,
                 max_disp=192,
                 crop_size=None):
        """
        初始化KITTI数据集
        
        参数与StereoDataset基类相同
        """
        super().__init__(root_dir, split, transform, augment, max_disp, crop_size)
        
        # 生成样本列表
        self._generate_samples()
    
    def _generate_samples(self):
        """生成KITTI数据集的样本列表"""
        # 确定图像和视差图的目录
        if self.split == 'train' or self.split == 'val':
            img_dir = os.path.join(self.root_dir, 'training', 'image_2')
            right_img_dir = os.path.join(self.root_dir, 'training', 'image_3')
            disp_dir = os.path.join(self.root_dir, 'training', 'disp_occ_0')
        else:  # test
            img_dir = os.path.join(self.root_dir, 'testing', 'image_2')
            right_img_dir = os.path.join(self.root_dir, 'testing', 'image_3')
            disp_dir = None  # 测试集没有真值视差图
        
        # 确保目录存在
        if not os.path.exists(img_dir) or not os.path.exists(right_img_dir):
            raise ValueError(f"目录不存在: {img_dir} 或 {right_img_dir}")
        
        # 查找所有左图像
        left_imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        # 对于每个左图像，找到对应的右图像和视差图
        for left_img_path in left_imgs:
            # 提取图像ID
            img_id = os.path.basename(left_img_path)
            
            # 右图像
            right_img_path = os.path.join(right_img_dir, img_id)
            if not os.path.exists(right_img_path):
                continue
            
            # 视差图（如果存在）
            disp_path = None
            if disp_dir is not None:
                disp_path = os.path.join(disp_dir, img_id)
                if not os.path.exists(disp_path):
                    disp_path = None
            
            # 添加到样本列表
            self.samples.append({
                'left': left_img_path,
                'right': right_img_path,
                'disparity': disp_path,
                'id': img_id
            })


class ETH3DDataset(StereoDataset):
    """
    ETH3D立体视觉数据集
    
    该类加载ETH3D数据集，一个室内外场景的高分辨率立体视觉数据集。
    """
    
    def __init__(self, 
                 root_dir,
                 split='train',
                 transform=None, 
                 augment=False,
                 max_disp=192,
                 crop_size=None):
        """
        初始化ETH3D数据集
        
        参数与StereoDataset基类相同
        """
        super().__init__(root_dir, split, transform, augment, max_disp, crop_size)
        
        # 生成样本列表
        self._generate_samples()
    
    def _generate_samples(self):
        """生成ETH3D数据集的样本列表"""
        # 确定分割目录
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
            
            # 左右图像目录
            left_dir = os.path.join(scene_dir, 'images', 'left')
            right_dir = os.path.join(scene_dir, 'images', 'right')
            
            # 视差图目录
            disp_dir = os.path.join(scene_dir, 'ground_truth', 'disp_occ')
            
            # 查找所有左图像
            if os.path.exists(left_dir):
                left_imgs = sorted(glob.glob(os.path.join(left_dir, '*.png')))
                
                for left_img_path in left_imgs:
                    # 提取图像ID
                    img_id = os.path.basename(left_img_path)
                    
                    # 右图像
                    right_img_path = os.path.join(right_dir, img_id)
                    if not os.path.exists(right_img_path):
                        continue
                    
                    # 视差图（如果存在）
                    disp_path = None
                    if os.path.exists(disp_dir):
                        disp_path = os.path.join(disp_dir, img_id.replace('.png', '.pfm'))
                        if not os.path.exists(disp_path):
                            disp_path = None
                    
                    # 添加到样本列表
                    self.samples.append({
                        'left': left_img_path,
                        'right': right_img_path,
                        'disparity': disp_path,
                        'scene': scene,
                        'id': img_id
                    })


class SceneFlowDataset(StereoDataset):
    """
    SceneFlow立体视觉数据集
    
    该类加载SceneFlow数据集，一个大规模的合成立体视觉数据集。
    """
    
    def __init__(self, 
                 root_dir,
                 subset='FlyingThings3D',
                 split='train',
                 transform=None, 
                 augment=False,
                 max_disp=192,
                 crop_size=None):
        """
        初始化SceneFlow数据集
        
        参数:
            root_dir (str): 数据集根目录
            subset (str): 子集，'FlyingThings3D', 'Monkaa', 或 'Driving'
            split (str): 数据集分割，'train' 或 'test'
            其他参数与StereoDataset基类相同
        """
        self.subset = subset
        super().__init__(root_dir, split, transform, augment, max_disp, crop_size)
        
        # 生成样本列表
        self._generate_samples()
    
    def _generate_samples(self):
        """生成SceneFlow数据集的样本列表"""
        # 根据子集和分割确定目录
        if self.subset == 'FlyingThings3D':
            subset_dir = os.path.join(self.root_dir, 'FlyingThings3D')
            img_dir = os.path.join(subset_dir, self.split, 'image_clean')
            disp_dir = os.path.join(subset_dir, self.split, 'disparity')
        elif self.subset == 'Monkaa':
            subset_dir = os.path.join(self.root_dir, 'Monkaa')
            img_dir = os.path.join(subset_dir, 'image_clean')
            disp_dir = os.path.join(subset_dir, 'disparity')
        elif self.subset == 'Driving':
            subset_dir = os.path.join(self.root_dir, 'Driving')
            img_dir = os.path.join(subset_dir, self.split, 'image_clean')
            disp_dir = os.path.join(subset_dir, self.split, 'disparity')
        else:
            raise ValueError(f"无效的SceneFlow子集: {self.subset}")
        
        # 确保目录存在
        if not os.path.exists(img_dir) or not os.path.exists(disp_dir):
            raise ValueError(f"目录不存在: {img_dir} 或 {disp_dir}")
        
        # 查找所有场景
        scenes = [d for d in os.listdir(img_dir) 
                 if os.path.isdir(os.path.join(img_dir, d))]
        
        # 对于每个场景，找到左右图像和视差图
        for scene in scenes:
            # 找到左右图像文件夹
            left_dir = os.path.join(img_dir, scene, 'left')
            right_dir = os.path.join(img_dir, scene, 'right')
            
            # 找到视差图文件夹
            disp_left_dir = os.path.join(disp_dir, scene, 'left')
            
            # 检查目录是否存在
            if not all(os.path.exists(d) for d in [left_dir, right_dir, disp_left_dir]):
                continue
            
            # 查找所有左图像
            left_imgs = sorted(glob.glob(os.path.join(left_dir, '*.png')))
            
            for left_img_path in left_imgs:
                # 提取图像ID
                img_id = os.path.basename(left_img_path)
                
                # 右图像
                right_img_path = os.path.join(right_dir, img_id)
                if not os.path.exists(right_img_path):
                    continue
                
                # 视差图
                disp_path = os.path.join(disp_left_dir, img_id.replace('.png', '.pfm'))
                if not os.path.exists(disp_path):
                    continue
                
                # 添加到样本列表
                self.samples.append({
                    'left': left_img_path,
                    'right': right_img_path,
                    'disparity': disp_path,
                    'scene': scene,
                    'id': img_id
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
                 max_disp=192,
                 crop_size=None):
        """
        初始化自定义数据集
        
        参数:
            left_dir (str): 左图像目录
            right_dir (str): 右图像目录
            disp_dir (str, optional): 视差图目录
            transform (callable, optional): 应用于样本的可选变换
            augment (bool): 是否进行数据增强
            max_disp (int): 最大视差值
            crop_size (tuple): 裁剪尺寸，如 (height, width)
        """
        # 不使用StereoDataset的初始化，因为目录结构不同
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.disp_dir = disp_dir
        self.transform = transform
        self.augment = augment
        self.max_disp = max_disp
        self.crop_size = crop_size
        
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
    
    def __len__(self):
        """返回数据集中样本数量"""
        return len(self.samples)
    
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
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # 数据集类型
    dataset_type = config.get('type', 'middlebury')
    
    # 通用参数
    common_params = {
        'transform': transform,
        'augment': config.get('augment', False),
        'max_disp': config.get('max_disp', 192),
        'crop_size': config.get('crop_size', None),
        'resize_size': config.get('resize_size', None)
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
    
    elif dataset_type.lower() == 'kitti':
        train_dataset = KITTIDataset(
            root_dir=config['root_dir'],
            split='train',
            **common_params
        )
        val_dataset = KITTIDataset(
            root_dir=config['root_dir'],
            split='val',
            augment=False,
            **{k: v for k, v in common_params.items() if k != 'augment'}
        )
        test_dataset = KITTIDataset(
            root_dir=config['root_dir'],
            split='test',
            augment=False,
            **{k: v for k, v in common_params.items() if k != 'augment'}
        )
    
    elif dataset_type.lower() == 'eth3d':
        train_dataset = ETH3DDataset(
            root_dir=config['root_dir'],
            split='train',
            **common_params
        )
        val_dataset = ETH3DDataset(
            root_dir=config['root_dir'],
            split='val',
            augment=False,
            **{k: v for k, v in common_params.items() if k != 'augment'}
        )
        test_dataset = ETH3DDataset(
            root_dir=config['root_dir'],
            split='test',
            augment=False,
            **{k: v for k, v in common_params.items() if k != 'augment'}
        )
    
    elif dataset_type.lower() == 'sceneflow':
        subset = config.get('subset', 'FlyingThings3D')
        train_dataset = SceneFlowDataset(
            root_dir=config['root_dir'],
            subset=subset,
            split='train',
            **common_params
        )
        val_dataset = SceneFlowDataset(
            root_dir=config['root_dir'],
            subset=subset,
            split='test',  # SceneFlow没有val分割，使用test
            augment=False,
            **{k: v for k, v in common_params.items() if k != 'augment'}
        )
        test_dataset = val_dataset  # 使用相同的测试集
    
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
    batch_size = config.get('batch_size', 4)
    
    # 工作进程数
    num_workers = config.get('num_workers', 4)
    
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