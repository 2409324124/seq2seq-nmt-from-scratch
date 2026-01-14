# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random
import time
import math
from utils import prepare_data, TranslationDataset, collate_fn
from models import EncoderRNN, AttnDecoderRNN

# ------------------- 参数配置 -------------------
hidden_size = 256
batch_size = 64          # 4060 可调到 96~128，如果 OOM 就降
epochs = 10              # 先训 10 epoch，看 loss 下降
learning_rate = 0.001
teacher_forcing_ratio = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据准备（用之前的数据）
input_lang, output_lang, pairs = prepare_data(max_length=25, min_freq=2)
dataset = TranslationDataset(pairs, input_lang, output_lang)

# 分割 train/val（简单 90%/10%）
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 模型初始化
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

# 优化器 + Loss（忽略 PAD=2）
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=2)  # PAD=2

# ------------------- 训练函数 -------------------
def train_step(src_batch, tgt_batch):
    src_batch = src_batch.to(device)
    tgt_batch = tgt_batch.to(device)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    # Encoder
    encoder_outputs, encoder_hidden = encoder(src_batch)
    
    # Decoder 初始化
    decoder_input = tgt_batch[:, 0].unsqueeze(1)  # 第一步输入 <SOS>
    decoder_hidden = encoder_hidden
    
    loss = 0
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    
    for t in range(1, tgt_batch.size(1)):
        output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
        
        # loss 加当前步
        loss += criterion(output, tgt_batch[:, t])
        
        # 决定下一步输入
        if use_teacher_forcing:
            decoder_input = tgt_batch[:, t].unsqueeze(1)  # 用真实词
        else:
            topv, topi = output.topk(1)
            decoder_input = topi.detach()  # 用模型预测
        
    loss = loss / (tgt_batch.size(1) - 1)  # 平均 loss
    loss.backward()
    
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
    
    # 简单保存模型（每 epoch 都存）
    torch.save(encoder.state_dict(), f"encoder_epoch{epoch}.pt")
    torch.save(decoder.state_dict(), f"decoder_epoch{epoch}.pt")

print("训练结束！")