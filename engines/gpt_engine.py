import torch
import torch.nn as nn
import torch.optim as optim
from engines.base_engine import BaseTrainingEngine
from models_gpt import GPTLMHeadModel

class GptEngine(BaseTrainingEngine):
    def initialize_model(self, input_size, output_size, lr):
        # We only use output_size (English vocab) since it's an English auto-regressive LM
        self.model = GPTLMHeadModel(
            vocab_size=output_size, 
            d_model=256, 
            nhead=8, 
            num_layers=6, # Standard GPT has more layers than the encoder-decoder
            dim_feedforward=1024, 
            dropout=0.1
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # CrossEntropyLoss: ignore index 2 (PAD) to avoid meaningless gradient updates
        self.criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.0)
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        self.pad_id = 2

    def train_one_epoch(self, train_loader, epoch_idx):
        self.model.train()
        running_loss = 0.0
        train_count = 0

        # We ignore 'src' entirely. We only care about predicting 'tgt' autoregressively.
        for _, tgt in train_loader:
            tgt = tgt.to(self.device)
            
            # GPT Input is all tokens except the last one
            input_ids = tgt[:, :-1]
            
            # GPT target generation is all tokens except the first one
            labels = tgt[:, 1:]
            
            attention_mask = (input_ids != self.pad_id).float()

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                # Forward pass: shape (batch_size, seq_len, vocab_size)
                logits = self.model(input_ids, attention_mask=attention_mask)
                
                # Reshape for CrossEntropy
                vocab_size = logits.shape[-1]
                loss = self.criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
            for _, tgt in val_loader:
                tgt = tgt.to(self.device)
                input_ids = tgt[:, :-1]
                labels = tgt[:, 1:]
                attention_mask = (input_ids != self.pad_id).float()

                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    logits = self.model(input_ids, attention_mask=attention_mask)
                    vocab_size = logits.shape[-1]
                    loss = self.criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))

                total_val_loss += loss.item()
                val_count += 1

        return total_val_loss / val_count if val_count > 0 else 0.0

    def save_checkpoint(self, path, epoch_idx, val_loss):
        torch.save({
            'epoch': epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, path)
