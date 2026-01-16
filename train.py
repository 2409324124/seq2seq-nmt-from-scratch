# train.py - LSTM 版本（完整重写）
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random
import time
from utils import prepare_data, TranslationDataset, collate_fn
from models import EncoderLSTM, AttnDecoderLSTM  # 注意这里改成 LSTM 版本

# ------------------- 参数配置 -------------------
hidden_size = 256
batch_size = 64          # 4060 可调到 96~128，如果 OOM 就降
epochs = 15              # 建议多训一点，LSTM 收敛更快
learning_rate = 0.0005   # 降低学习率，更稳定
teacher_forcing_ratio = 0.75
max_grad_norm = 1.0      # 梯度裁剪阈值
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据准备
input_lang, output_lang, pairs = prepare_data(max_length=25, min_freq=2)
dataset = TranslationDataset(pairs, input_lang, output_lang)

# 分割 train/val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 模型初始化（LSTM 版本）
encoder = EncoderLSTM(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderLSTM(hidden_size, output_lang.n_words).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=2)  # PAD=2

# ------------------- 训练单步 -------------------
def train_step(src_batch, tgt_batch):
    src_batch = src_batch.to(device)
    tgt_batch = tgt_batch.to(device)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    # Encoder 前向（LSTM 返回 (output, (hidden, cell))）
    encoder_outputs, (encoder_hidden, encoder_cell) = encoder(src_batch)
    
    # Decoder 初始化
    decoder_input = tgt_batch[:, 0].unsqueeze(1)  # 第一步 <SOS>
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell  # LSTM 需要 cell
    
    loss = 0
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    
    for t in range(1, tgt_batch.size(1)):
        # Decoder 前向（多传 cell）
        output, decoder_hidden, decoder_cell, _ = decoder(
            decoder_input, decoder_hidden, decoder_cell, encoder_outputs
        )
        
        loss += criterion(output, tgt_batch[:, t])
        
        if use_teacher_forcing:
            decoder_input = tgt_batch[:, t].unsqueeze(1)
        else:
            topv, topi = output.topk(1)
            decoder_input = topi.detach()
    
    loss = loss / (tgt_batch.size(1) - 1)
    loss.backward()
    
    # 梯度裁剪（关键！缓解梯度消失/爆炸）
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()), max_grad_norm
    )
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()

# ------------------- 主训练循环 -------------------
start_time = time.time()
for epoch in range(1, epochs + 1):
    total_loss = 0
    encoder.train()
    decoder.train()
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        loss = train_step(src, tgt)
        total_loss += loss
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss:.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} 完成 | 平均 Loss: {avg_loss:.4f} | 时间: {time.time() - start_time:.0f}s")
    
    # 保存当前 epoch 模型
    torch.save(encoder.state_dict(), f"encoder_lstm_epoch{epoch}.pt")
    torch.save(decoder.state_dict(), f"decoder_lstm_epoch{epoch}.pt")

print("LSTM 训练结束！")