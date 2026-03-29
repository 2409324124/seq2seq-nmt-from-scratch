import torch
import torch.nn as nn
import torch.optim as optim
import time
from engines.base_engine import BaseTrainingEngine
from models_transformer import TransformerModel

class TransformerEngine(BaseTrainingEngine):
    def initialize_model(self, input_size, output_size, lr):
        self.model = TransformerModel(
            input_size, output_size, 
            d_model=256, nhead=8, nhid=512, nlayers=3, dropout=0.1
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    def train_one_epoch(self, train_loader, epoch_idx):
        self.model.train()
        running_loss = 0.0
        train_count = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            
            # 准备 Mask
            src_padding_mask = (src == 2) # PAD=2
            tgt_padding_mask = (tgt_input == 2)
            
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                output = self.model(
                    src, tgt_input, 
                    src_padding_mask=src_padding_mask, 
                    tgt_padding_mask=tgt_padding_mask, 
                    memory_key_padding_mask=src_padding_mask
                )
                loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            train_count += 1
            
            yield running_loss / train_count

    def validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
                src_mask = (src == 2)
                tgt_mask = (tgt_input == 2)
                
                output = self.model(src, tgt_input, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
                loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                total_val_loss += loss.item()
                val_count += 1
        return total_val_loss / val_count if val_count > 0 else 0.0
