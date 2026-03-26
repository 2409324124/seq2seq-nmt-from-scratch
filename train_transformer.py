import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from utils import prepare_data, TranslationDataset, collate_fn
from models_transformer import TransformerModel

# ------------------- 参数配置 -------------------
d_model = 256
nhead = 8
num_layers = 3
dim_feedforward = 512
dropout = 0.1

batch_size = 64
epochs = 50
learning_rate = 0.0001
patience = 3  # 早停阈值
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
model = TransformerModel(
    input_lang.n_words, 
    output_lang.n_words, 
    d_model, nhead, dim_feedforward, num_layers, dropout
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.1) # 2 是 PAD 的索引

# ------------------- 混合精度训练 -------------------
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

train_loss_history = []
val_loss_history = []

# ------------------- 验证集评估 -------------------
def validate():
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_padding_mask = (src == 2) # PAD 索引
            tgt_padding_mask = (tgt_input == 2)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                output = model(src, tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
                loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

# ------------------- 训练单步 -------------------
def train_step(src, tgt):
    src = src.to(device)
    tgt = tgt.to(device)
    
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    
    # 填充掩码
    src_padding_mask = (src == 2)
    tgt_padding_mask = (tgt_input == 2)
    
    optimizer.zero_grad()
    
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        output = model(src, tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        # CrossEntropyLoss 期望输入维度为 (batch * seq_len, vocab_size)
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
    
    # 混合精度反向传播
    scaler.scale(loss).backward()
    
    # Unscales gradients and clips them
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    # optimizer.step() replaces with scaler.step()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

# ------------------- 主训练循环 -------------------
best_val_loss = float('inf')
patience_counter = 0
start_time = time.time()

for epoch in range(1, epochs + 1):
    total_train_loss = 0
    model.train()
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        loss = train_step(src, tgt)
        total_train_loss += loss
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss:.4f}")
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    
    avg_val_loss = validate()
    val_loss_history.append(avg_val_loss)
    
    print(f"Epoch {epoch} 完成 | 训练 Loss: {avg_train_loss:.4f} | 验证 Loss: {avg_val_loss:.4f} | 时间: {time.time() - start_time:.0f}s")
    
    # 保存滚动 Checkpoint (保留最近3个)
    ckpt_name = f"transformer_epoch_{epoch}.pt"
    torch.save(model.state_dict(), ckpt_name)
    
    # 每 5 轮存一个永久里程碑
    if epoch % 5 == 0:
        import shutil
        milestone = f"transformer_checkpoint_E{epoch}.pt"
        shutil.copy2(ckpt_name, milestone)
        print(f"🏛️ 发现存档点！已保存永久里程碑: {milestone}")

    if epoch > 3:
        old_ckpt = f"transformer_epoch_{epoch-3}.pt"
        if os.path.exists(old_ckpt): os.remove(old_ckpt)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(f"→ 验证 Loss 改善！保存最佳模型 (Loss: {best_val_loss:.4f})")
        torch.save(model.state_dict(), "transformer_best.pt")
    else:
        patience_counter += 1
        print(f"验证 Loss 未改善 ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"早停触发！连续 {patience} 个 Epoch 验证 Loss 未下降，结束训练。")
            break

# ------------------- 训练结束后画图 -------------------
epochs_list = list(range(1, len(train_loss_history) + 1))
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, train_loss_history, 'o-', color='blue', label='训练 Loss')
plt.plot(epochs_list, val_loss_history, 's-', color='red', label='验证 Loss')
plt.title('Transformer 训练 Loss 曲线')
plt.xlabel('Epoch')
plt.ylabel('平均 Loss')
plt.grid(True)
plt.legend()
plt.savefig('loss_curve_transformer.png', dpi=300)
print("Loss 曲线已保存为 loss_curve_transformer.png")
