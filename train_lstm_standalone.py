# train.py - LSTM 版本（更稳健版 + 支持从指定 epoch 继续训练）
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random
import time
import matplotlib.pyplot as plt

# 在 import matplotlib.pyplot as plt 后加
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

import numpy as np
from utils import prepare_data, TranslationDataset, collate_fn
from models import EncoderLSTM, AttnDecoderLSTM

# ------------------- 参数配置 -------------------
hidden_size = 256
batch_size = 64
epochs = 60               # 最多 60，总轮数
learning_rate = 0.0001
max_grad_norm = 0.5
patience = 3
start_epoch = 0          # ← 关键！从第 30 轮继续（改成你想继续的 epoch，0 就是从头训）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据准备
input_lang, output_lang, pairs = prepare_data(max_length=25, min_freq=2)
dataset = TranslationDataset(pairs, input_lang, output_lang)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 模型 & 优化器
encoder = EncoderLSTM(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderLSTM(hidden_size, output_lang.n_words, dropout=0.4).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.1)

# 新增：记录 Loss 历史
train_loss_history = []
val_loss_history = []

# ------------------- 从指定 epoch 加载模型（继续训练） -------------------
if start_epoch > 0:
    encoder_path = f"encoder_lstm_epoch{start_epoch}.pt"
    decoder_path = f"decoder_lstm_epoch{start_epoch}.pt"
    
    try:
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
        print(f"成功加载 Epoch {start_epoch} 的模型，继续训练...")
    except FileNotFoundError:
        print(f"警告：未找到 {encoder_path} 或 {decoder_path}，将从头开始训练")
        start_epoch = 0

# ------------------- 验证集评估 -------------------
def validate():
    encoder.eval()
    decoder.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            encoder_outputs, (encoder_hidden, encoder_cell) = encoder(src)
            decoder_input = tgt[:, 0].unsqueeze(1)
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell
            
            loss = 0
            for t in range(1, tgt.size(1)):
                output, decoder_hidden, decoder_cell, _ = decoder(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs
                )
                loss += criterion(output, tgt[:, t])
                topv, topi = output.topk(1)
                decoder_input = topi.detach()
            
            loss = loss / (tgt.size(1) - 1)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    encoder.train()
    decoder.train()
    return avg_val_loss

# ------------------- 训练单步 -------------------
def train_step(src_batch, tgt_batch, epoch, epochs):
    src_batch = src_batch.to(device)
    tgt_batch = tgt_batch.to(device)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    encoder_outputs, (encoder_hidden, encoder_cell) = encoder(src_batch)
    
    decoder_input = tgt_batch[:, 0].unsqueeze(1)
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell
    
    loss = 0
    
    teacher_forcing_ratio = 1.0 - (epoch / epochs) * 0.5
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    
    for t in range(1, tgt_batch.size(1)):
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
    
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()), max_grad_norm
    )
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()

# ------------------- 主训练循环 + 早停 + 画图 -------------------
best_val_loss = float('inf')
patience_counter = 0
start_time = time.time()

for epoch in range(start_epoch + 1, epochs + 1):
    total_train_loss = 0
    encoder.train()
    decoder.train()
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        loss = train_step(src, tgt, epoch, epochs)
        total_train_loss += loss
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss:.4f}")
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    
    avg_val_loss = validate()
    val_loss_history.append(avg_val_loss)
    
    print(f"Epoch {epoch} 完成 | 训练 Loss: {avg_train_loss:.4f} | 验证 Loss: {avg_val_loss:.4f} | 时间: {time.time() - start_time:.0f}s")
    
    # 保存当前 epoch 模型
    torch.save(encoder.state_dict(), f"encoder_lstm_epoch{epoch}.pt")
    torch.save(decoder.state_dict(), f"decoder_lstm_epoch{epoch}.pt")
    
    # 早停
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(f"→ 验证 Loss 改善！保存最佳模型 (Loss: {best_val_loss:.4f})")
        torch.save(encoder.state_dict(), "encoder_lstm_best.pt")
        torch.save(decoder.state_dict(), "decoder_lstm_best.pt")
    else:
        patience_counter += 1
        print(f"验证 Loss 未改善 ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"早停触发！连续 {patience} epoch 验证 Loss 未下降")
            break

# ------------------- 训练结束后自动画 Loss 曲线 -------------------
print("训练结束！最佳验证 Loss:", best_val_loss)

epochs_list = list(range(1, len(train_loss_history) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epochs_list, train_loss_history, 'o-', color='blue', label='训练 Loss')
plt.plot(epochs_list, val_loss_history, 's-', color='red', label='验证 Loss')
plt.title('LSTM 训练 Loss 曲线')
plt.xlabel('Epoch')
plt.ylabel('平均 Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig('loss_curve_lstm.png', dpi=300)
plt.show()

print("Loss 曲线已保存为 loss_curve_lstm.png")