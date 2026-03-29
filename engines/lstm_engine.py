import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
from engines.base_engine import BaseTrainingEngine
from models_lstm import EncoderLSTM, AttnDecoderLSTM

class LSTMEngine(BaseTrainingEngine):
    def initialize_model(self, input_size, output_size, lr):
        hidden_size = 256
        self.encoder = EncoderLSTM(input_size, hidden_size).to(self.device)
        self.decoder = AttnDecoderLSTM(hidden_size, output_size, dropout=0.4).to(self.device)
        self.model = (self.encoder, self.decoder) # 元组存储
        
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=lr
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    def train_one_epoch(self, train_loader, epoch_idx):
        self.encoder.train()
        self.decoder.train()
        running_loss = 0.0
        train_count = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            # LSTM 需要调整维度 (batch, seq) -> (seq, batch)
            # 或者按照 collate_fn 的输出来处理。通常是 (batch, seq)
            
            self.optimizer.zero_grad()
            
            # Encoder
            encoder_outputs, encoder_hidden = self.encoder(src)
            
            # Decoder
            decoder_input = torch.tensor([[0]] * src.size(0), device=self.device) # SOS_token=0
            decoder_hidden = encoder_hidden
            
            loss = 0
            # Teacher Forcing
            teacher_forcing_ratio = 0.5
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            
            target_length = tgt.size(1)
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                
                loss += self.criterion(decoder_output, tgt[:, di])
                if use_teacher_forcing:
                    decoder_input = tgt[:, di].unsqueeze(1)
                else:
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), 0.5)
            self.optimizer.step()
            
            running_loss += loss.item() / target_length
            train_count += 1
            
            yield running_loss / train_count

    def validate(self, val_loader):
        self.encoder.eval()
        self.decoder.eval()
        total_val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                encoder_outputs, encoder_hidden = self.encoder(src)
                decoder_input = torch.tensor([[0]] * src.size(0), device=self.device)
                decoder_hidden = encoder_hidden
                
                loss = 0
                target_length = tgt.size(1)
                for di in range(target_length):
                    decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    loss += self.criterion(decoder_output, tgt[:, di])
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()
                
                total_val_loss += loss.item() / target_length
                val_count += 1
        return total_val_loss / val_count if val_count > 0 else 0.0

    def save_checkpoint(self, path, epoch_idx, val_loss):
        torch.save({
            'epoch': epoch_idx,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, path)
