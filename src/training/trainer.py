"""
模型训练模块

该模块提供了训练BPNN模型的功能。
"""

import os
import time
import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils.metrics import compute_error_metrics
from .loss import StereoLoss


class Trainer:
    """
    BPNN模型训练器
    
    该类负责模型的训练、验证和检查点管理。
    """
    
    def __init__(self, model, train_loader, val_loader, config):
        """
        初始化训练器
        
        参数:
            model (nn.Module): 要训练的模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            config (dict): 训练配置
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 训练设备，从配置中获取
        if 'device' in config:
            self.device = config['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        
        # 损失函数
        loss_config = config.get('loss', {})
        self.criterion = StereoLoss(**loss_config)
        
        # 优化器
        optim_config = config.get('optimizer', {})
        optim_type = optim_config.get('type', 'adam').lower()
        lr = optim_config.get('lr', 0.001)
        weight_decay = optim_config.get('weight_decay', 0.0001)
        
        if optim_type == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                betas=optim_config.get('betas', (0.9, 0.999))
            )
        elif optim_type == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(), 
                lr=lr, 
                momentum=optim_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optim_type == 'rmsprop':
            self.optimizer = optim.RMSprop(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                momentum=optim_config.get('momentum', 0)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optim_type}")
        
        # 学习率调整器
        lr_scheduler_config = config.get('lr_scheduler', {})
        lr_scheduler_type = lr_scheduler_config.get('type', 'step').lower()
        
        if lr_scheduler_type == 'step':
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_scheduler_config.get('step_size', 20),
                gamma=lr_scheduler_config.get('gamma', 0.5)
            )
        elif lr_scheduler_type == 'plateau':
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=lr_scheduler_config.get('factor', 0.5),
                patience=lr_scheduler_config.get('patience', 5)
            )
        elif lr_scheduler_type == 'cosine':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=lr_scheduler_config.get('T_max', 10),
                eta_min=lr_scheduler_config.get('eta_min', 0)
            )
        else:
            self.lr_scheduler = None
        
        # 训练配置
        self.num_epochs = config.get('num_epochs', 100)
        self.save_dir = config.get('save_dir', 'checkpoints')
        self.log_dir = config.get('log_dir', 'logs')
        self.save_freq = config.get('save_freq', 5)
        self.val_freq = config.get('val_freq', 1)
        self.log_freq = config.get('log_freq', 10)
        self.resume = config.get('resume', '')
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志记录
        self.logger = self._setup_logger()
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # 如果需要，从检查点恢复
        if self.resume:
            self._load_checkpoint(self.resume)
    
    def _setup_logger(self):
        """设置日志记录"""
        logger = logging.getLogger('BPNN_Trainer')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理程序
        log_file = os.path.join(self.log_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理程序
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化程序
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理程序到日志记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _save_checkpoint(self, epoch, is_best=False):
        """
        保存检查点
        
        参数:
            epoch (int): 当前轮次
            is_best (bool): 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # 保存最新的检查点
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存到 {checkpoint_path}")
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"最佳模型已保存到 {best_path}")
    
    def _load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        参数:
            checkpoint_path (str): 检查点文件路径
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点不存在: {checkpoint_path}")
            return
        
        self.logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型和优化器状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载学习率调整器状态（如果有）
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # 恢复训练状态
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"恢复训练从轮次 {self.start_epoch}，最佳验证损失: {self.best_val_loss:.4f}")
    
    def train_epoch(self, epoch):
        """
        训练一个轮次
        
        参数:
            epoch (int): 当前轮次
            
        返回:
            float: 本轮平均损失
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {'epe': 0.0, 'bad1': 0.0, 'bad3': 0.0}
        num_batches = len(self.train_loader)
        
        # 创建进度条
        pbar = tqdm(self.train_loader, total=num_batches, desc=f"轮次 {epoch}")
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            left_img = batch['left'].to(self.device)
            right_img = batch['right'].to(self.device)
            target_disp = batch['disparity'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(left_img, right_img)
            
            # 如果模型返回字典，提取视差图
            if isinstance(outputs, dict) and 'disparity' in outputs:
                pred_disp = outputs['disparity']
            else:
                pred_disp = outputs
            
            # 计算损失
            loss = self.criterion(pred_disp, target_disp)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if 'grad_clip' in self.config:
                clip_value = self.config['grad_clip']
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            
            # 优化
            self.optimizer.step()
            
            # 更新进度条和损失
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            # 计算指标
            with torch.no_grad():
                metrics = compute_error_metrics(pred_disp, target_disp)
            
            for k, v in metrics.items():
                epoch_metrics[k] += v
            
            # 更新进度条信息
            pbar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'epe': f"{metrics['epe']:.4f}"
            })
            
            # 记录到TensorBoard
            if (batch_idx + 1) % self.log_freq == 0 or batch_idx == num_batches - 1:
                step = epoch * num_batches + batch_idx
                self.writer.add_scalar('train/batch_loss', batch_loss, step)
                for k, v in metrics.items():
                    self.writer.add_scalar(f'train/batch_{k}', v, step)
        
        # 计算平均损失和指标
        epoch_loss /= num_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        # 记录到TensorBoard
        self.writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        for k, v in epoch_metrics.items():
            self.writer.add_scalar(f'train/epoch_{k}', v, epoch)
        
        # 计算用时
        epoch_time = time.time() - start_time
        
        # 记录日志
        self.logger.info(f"轮次 {epoch} - 训练损失: {epoch_loss:.4f}, "
                        f"EPE: {epoch_metrics['epe']:.4f}, "
                        f"Bad1: {epoch_metrics['bad1']:.4f}, "
                        f"Bad3: {epoch_metrics['bad3']:.4f}, "
                        f"用时: {epoch_time:.2f}s")
        
        return epoch_loss
    
    def validate(self, epoch):
        """
        验证模型
        
        参数:
            epoch (int): 当前轮次
            
        返回:
            float: 验证平均损失
        """
        self.model.eval()
        val_loss = 0.0
        val_metrics = {'epe': 0.0, 'bad1': 0.0, 'bad3': 0.0}
        num_batches = len(self.val_loader)
        
        # 创建进度条
        pbar = tqdm(self.val_loader, total=num_batches, desc=f"验证 {epoch}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # 准备数据
                left_img = batch['left'].to(self.device)
                right_img = batch['right'].to(self.device)
                target_disp = batch['disparity'].to(self.device)
                
                # 前向传播
                outputs = self.model(left_img, right_img)
                
                # 如果模型返回字典，提取视差图
                if isinstance(outputs, dict) and 'disparity' in outputs:
                    pred_disp = outputs['disparity']
                else:
                    pred_disp = outputs
                
                # 计算损失
                loss = self.criterion(pred_disp, target_disp)
                
                # 更新损失
                batch_loss = loss.item()
                val_loss += batch_loss
                
                # 计算指标
                metrics = compute_error_metrics(pred_disp, target_disp)
                
                for k, v in metrics.items():
                    val_metrics[k] += v
                
                # 更新进度条信息
                pbar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'epe': f"{metrics['epe']:.4f}"
                })
        
        # 计算平均损失和指标
        val_loss /= num_batches
        for k in val_metrics:
            val_metrics[k] /= num_batches
        
        # 记录到TensorBoard
        self.writer.add_scalar('val/loss', val_loss, epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f'val/{k}', v, epoch)
        
        # 记录日志
        self.logger.info(f"轮次 {epoch} - 验证损失: {val_loss:.4f}, "
                        f"EPE: {val_metrics['epe']:.4f}, "
                        f"Bad1: {val_metrics['bad1']:.4f}, "
                        f"Bad3: {val_metrics['bad3']:.4f}")
        
        # 如果使用ReduceLROnPlateau，用验证损失更新学习率
        if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(val_loss)
        
        return val_loss
    
    def train(self):
        """训练模型"""
        self.logger.info(f"开始训练，总轮次: {self.num_epochs}")
        self.logger.info(f"训练设备: {self.device}")
        
        # 记录训练开始时间
        train_start_time = time.time()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # 训练一个轮次
            train_loss = self.train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % self.val_freq == 0 or epoch == self.num_epochs - 1:
                val_loss = self.validate(epoch)
                
                # 检查是否是最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.logger.info(f"新的最佳模型，验证损失: {val_loss:.4f}")
            else:
                is_best = False
            
            # 更新学习率（除了ReduceLROnPlateau）
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/learning_rate', current_lr, epoch)
                self.logger.info(f"当前学习率: {current_lr:.6f}")
            
            # 保存检查点
            if (epoch + 1) % self.save_freq == 0 or epoch == self.num_epochs - 1 or is_best:
                self._save_checkpoint(epoch, is_best)
        
        # 计算总训练时间
        total_train_time = time.time() - train_start_time
        hours, rem = divmod(total_train_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        self.logger.info(f"训练完成，总用时: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        self.logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        # 关闭TensorBoard写入器
        self.writer.close()