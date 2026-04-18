import torch
import torch.nn as nn
import torch.optim as optim
import random
from engines.base_engine import BaseTrainingEngine
from models_lstm import EncoderLSTM, AttnDecoderLSTM


class LSTMEngine(BaseTrainingEngine):
    def initialize_model(self, input_size, output_size, lr):
        hidden_size = 256
        self.encoder = EncoderLSTM(input_size, hidden_size).to(self.device)
        self.decoder = AttnDecoderLSTM(hidden_size, output_size, dropout=0.4).to(self.device)
        self.model = (self.encoder, self.decoder)

        # 参考 train_lstm_standalone.py: 两个独立 optimizer，分别管理 encoder/decoder
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)

        # 同一接口兼容 GUI 中的 scheduler.step()
        self.optimizer = self.encoder_optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode='min', factor=0.5, patience=3
        )
        # 参考 train_lstm_standalone.py: label_smoothing=0.1
        self.criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.1)
        self.max_grad_norm = 0.5
        self._total_epochs = 30  # 默认值，会在 train_one_epoch 中更新

    def train_one_epoch(self, train_loader, epoch_idx, total_epochs=None):
        if total_epochs is not None:
            self._total_epochs = total_epochs

        self.encoder.train()
        self.decoder.train()
        running_loss = 0.0
        train_count = 0

        # 参考 train_lstm_standalone.py: 动态 teacher forcing，随 epoch 推进衰减
        teacher_forcing_ratio = 1.0 - (epoch_idx / self._total_epochs) * 0.5

        for src, tgt in train_loader:
            src, tgt = src.to(self.device), tgt.to(self.device)

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(src)

            # 参考 train_lstm_standalone.py: decoder 起始输入用 tgt[:,0] (SOS)
            decoder_input = tgt[:, 0].unsqueeze(1)
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell

            loss = 0
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            target_length = tgt.size(1) - 1  # 不包含 SOS

            for t in range(1, tgt.size(1)):
                decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs
                )
                loss += self.criterion(decoder_output, tgt[:, t])

                if use_teacher_forcing:
                    decoder_input = tgt[:, t].unsqueeze(1)
                else:
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()

            # 参考 train_lstm_standalone.py: 损失除以实际序列长度（更稳定）
            loss = loss / target_length
            loss.backward()

            # 两个优化器共用一次梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                self.max_grad_norm
            )
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            running_loss += loss.item()
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
                
                encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(src)

                # 参考 train_lstm_standalone.py: 使用 SOS token 作为起始
                decoder_input = tgt[:, 0].unsqueeze(1)
                decoder_hidden = encoder_hidden
                decoder_cell = encoder_cell

                loss = 0
                target_length = tgt.size(1) - 1
                for t in range(1, tgt.size(1)):
                    decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(
                        decoder_input, decoder_hidden, decoder_cell, encoder_outputs
                    )
                    loss += self.criterion(decoder_output, tgt[:, t])
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()

                loss = loss / target_length
                total_val_loss += loss.item()
                val_count += 1

        self.encoder.train()
        self.decoder.train()
        return total_val_loss / val_count if val_count > 0 else 0.0

    def save_checkpoint(self, path, epoch_idx, val_loss):
        torch.save({
            'epoch': epoch_idx,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
            'val_loss': val_loss
        }, path)
