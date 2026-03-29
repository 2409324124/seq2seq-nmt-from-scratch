import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import time

class BaseTrainingEngine(ABC):
    def __init__(self, device):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.1)
        self.scheduler = None

    @abstractmethod
    def initialize_model(self, input_size, output_size, lr):
        """初始化模型实例、优化器与调度器"""
        pass

    @abstractmethod
    def train_one_epoch(self, train_loader, epoch_idx):
        """执行一轮训练循环 (生成器，用于 UI 进度上报)"""
        pass

    @abstractmethod
    def validate(self, val_loader):
        """执行验证循环并返回平均 Loss"""
        pass

    def save_checkpoint(self, path, epoch_idx, val_loss):
        """通用保存逻辑"""
        torch.save({
            'epoch': epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict() if hasattr(self.optimizer, 'optimizer') else self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, path)
