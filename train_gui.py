import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# ------------------- 视觉风格配置 -------------------
plt.rcParams['font.sans-serif'] = ['SimHei'] # 中文字体支持
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot') # 使用更美观的绘图风格

from utils import prepare_data, TranslationDataset, collate_fn
from models_transformer import TransformerModel
from models import EncoderLSTM, AttnDecoderLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 全局控制变量 ==========
stop_training_flag = False

def request_stop():
    global stop_training_flag
    stop_training_flag = True
    return "🛑 接收到终止指令，正在安全退出循环..."

def create_plot(train_loss, val_loss):
    """ 创建极简且高级的 Matplotlib 曲线图 """
    if not train_loss:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    epochs = range(1, len(train_loss) + 1)
    
    ax.plot(epochs, train_loss, 'o-', color='#3498db', linewidth=2.5, label='训练 Loss (Train)', markersize=6)
    if val_loss:
        val_epochs = range(1, len(val_loss) + 1)
        ax.plot(val_epochs, val_loss, 's-', color='#e74c3c', linewidth=2.5, label='验证 Loss (Val)', markersize=6)
    
    ax.set_title("模型收敛趋势监控 (Seq2Seq)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("迭代轮次 (Epoch)", fontsize=12)
    ax.set_ylabel("平均损失 (Average Loss)", fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, shadow=True)
    
    plt.tight_layout()
    return fig

# ========== 核心训练生成器 (Pro版) ==========
def train_pro(model_choice, batch_size, epochs, learning_rate, patience):
    global stop_training_flag
    stop_training_flag = False
    
    epochs = int(epochs)
    batch_size = int(batch_size)
    
    # 初始化 UI 返回值
    status_msg = "🔄 初始化数据中..."
    log_content = f"[*] 环境检查: {device}\n[*] 加载双语对数据..."
    
    yield (
        status_msg,     # status_text
        0,              # epoch_val
        0.0000,         # train_loss_val
        0.0000,         # val_loss_val
        learning_rate,  # lr_val (NEW)
        "0.0s",         # time_val
        log_content,    # log_box
        None            # plot
    )
    
    # 1. 准备数据
    input_lang, output_lang, pairs = prepare_data(max_length=25, min_freq=2)
    dataset = TranslationDataset(pairs, input_lang, output_lang)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    log_content += f"\n[+] 数据就绪: 训练集 {train_size} / 验证集 {val_size}"
    
    # 2. 模型初始化
    if model_choice == "Transformer":
        model = TransformerModel(
            input_lang.n_words, output_lang.n_words, 
            d_model=256, nhead=8, nhid=512, nlayers=3, dropout=0.1
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.1)
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        encoder = EncoderLSTM(input_lang.n_words, 256).to(device)
        decoder = AttnDecoderLSTM(256, output_lang.n_words, dropout=0.4).to(device)
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.1)
        lstm_model = (encoder, decoder) 
        # LSTM 调度器 (针对验证集 Loss)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.5, patience=3)

    log_content += f"\n[+] {model_choice} 模型已在 {device} 上实例化 (动态 LR 已就绪)。"
    current_lr = learning_rate
    yield "🚀 正在开始训练...", 0, 0.0, 0.0, current_lr, "0s", log_content, None
    
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    # 3. 训练主循环
    for epoch in range(1, epochs + 1):
        if stop_training_flag: break
        
        # --- 训练阶段 ---
        epoch_start = time.time()
        running_loss = 0.0
        
        if model_choice == "Transformer":
            model.train()
            train_count = 0
            for batch_idx, (src, tgt) in enumerate(train_loader):
                if stop_training_flag: break
                src, tgt = src.to(device), tgt.to(device)
                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
                src_padding_mask, tgt_padding_mask = (src == 2), (tgt_input == 2)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    output = model(src, tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
                    loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                
                current_batch_loss = loss.item()
                running_loss += current_batch_loss
                train_count += 1
                
                if batch_idx % 20 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    yield (
                        f"⚡ 正在训练 Epoch {epoch}...", 
                        epoch, 
                        current_batch_loss, 
                        (val_loss_history[-1] if val_loss_history else 0.0),
                        current_lr,
                        f"{time.time() - start_time:.1f}s",
                        log_content + f"\n[Batch {batch_idx}/{len(train_loader)}] Loss: {current_batch_loss:.4f}",
                        create_plot(train_loss_history, val_loss_history)
                    )
            avg_train_loss = running_loss / train_count if train_count > 0 else 0
            
        else: # LSTM
            encoder, decoder = lstm_model
            encoder.train(); decoder.train()
            train_count = 0
            for batch_idx, (src, tgt) in enumerate(train_loader):
                if stop_training_flag: break
                src_batch, tgt_batch = src.to(device), tgt.to(device)
                encoder_optimizer.zero_grad(); decoder_optimizer.zero_grad()
                
                encoder_outputs, (encoder_hidden, encoder_cell) = encoder(src_batch)
                decoder_input = tgt_batch[:, 0].unsqueeze(1)
                decoder_hidden, decoder_cell, loss = encoder_hidden, encoder_cell, 0
                
                use_tf = random.random() < (1.0 - (epoch/epochs)*0.5)
                for t in range(1, tgt_batch.size(1)):
                    out, decoder_hidden, decoder_cell, _ = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
                    loss += criterion(out, tgt_batch[:, t])
                    decoder_input = tgt_batch[:, t].unsqueeze(1) if use_tf else out.topk(1)[1].detach()
                
                loss = loss / (tgt_batch.size(1) - 1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 0.5)
                encoder_optimizer.step(); decoder_optimizer.step()
                
                current_batch_loss = loss.item()
                running_loss += current_batch_loss
                train_count += 1
                
                if batch_idx % 20 == 0:
                    current_lr = decoder_optimizer.param_groups[0]['lr']
                    yield (
                        f"⚡ 正在训练 Epoch {epoch}...", 
                        epoch, 
                        current_batch_loss, 
                        (val_loss_history[-1] if val_loss_history else 0.0),
                        current_lr,
                        f"{time.time() - start_time:.1f}s",
                        log_content + f"\n[Batch {batch_idx}/{len(train_loader)}] Loss: {current_batch_loss:.4f}",
                        create_plot(train_loss_history, val_loss_history)
                    )
            avg_train_loss = running_loss / train_count if train_count > 0 else 0
            
        if stop_training_flag: break
        train_loss_history.append(avg_train_loss)
            
        # --- 验证阶段 ---
        status_msg = f"🔍 正在验证 Epoch {epoch}..."
        yield status_msg, epoch, avg_train_loss, 0.0, current_lr, f"{time.time() - start_time:.1f}s", log_content, create_plot(train_loss_history, val_loss_history)
        
        total_val_loss = 0
        if model_choice == "Transformer":
            model.eval()
            with torch.no_grad():
                for sv, tv in val_loader:
                    sv, tv = sv.to(device), tv.to(device)
                    to_in, to_out = tv[:, :-1], tv[:, 1:]
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        out_v = model(sv, to_in, src_padding_mask=(sv==2), tgt_padding_mask=(to_in==2), memory_key_padding_mask=(sv==2))
                        l_v = criterion(out_v.reshape(-1, out_v.shape[-1]), to_out.reshape(-1))
                    total_val_loss += l_v.item()
        else:
            encoder, decoder = lstm_model
            encoder.eval(); decoder.eval()
            with torch.no_grad():
                for sv, tv in val_loader:
                    sv, tv = sv.to(device), tv.to(device)
                    eo, (eh, ec) = encoder(sv)
                    di, dh, dc, lv = tv[:, 0].unsqueeze(1), eh, ec, 0
                    for t in range(1, tv.size(1)):
                        ov, dh, dc, _ = decoder(di, dh, dc, eo)
                        lv += criterion(ov, tv[:, t])
                        di = ov.topk(1)[1].detach()
                    total_val_loss += (lv / (tv.size(1) - 1)).item()
                    
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # 更新调度器
        old_lr = current_lr
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr'] if model_choice == "Transformer" else decoder_optimizer.param_groups[0]['lr']
        if current_lr < old_lr:
            log_content += f"\n📉 学习率下调至: {current_lr:.6f}"
        
        # 4. 更新状态与保存
        log_content += f"\n[✔] Epoch {epoch} 完成! Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        
        # 保存滚动 Checkpoint (保留最近3个)
        import os
        checkpoint_name = f"{model_choice.lower()}_epoch_{epoch}.pt"
        if model_choice == "Transformer":
            torch.save(model.state_dict(), checkpoint_name)
        else:
            torch.save({'en': encoder.state_dict(), 'de': decoder.state_dict()}, checkpoint_name)
        
        # [NEW] 每 5 轮存一个永久 Milestone
        if epoch % 5 == 0:
            milestone_name = f"{model_choice.lower()}_checkpoint_E{epoch}.pt"
            import shutil
            shutil.copy2(checkpoint_name, milestone_name)
            log_content += f"\n🏛️ 已记录永久里程碑: {milestone_name}"
        
        # 删除滚动旧档 (保留 3 个)
        if epoch > 3:
            old_ckpt = f"{model_choice.lower()}_epoch_{epoch-3}.pt"
            if os.path.exists(old_ckpt): os.remove(old_ckpt)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            suffix = "_best_pro.pt"
            if model_choice == "Transformer":
                torch.save(model.state_dict(), f"transformer{suffix}")
            else:
                torch.save(encoder.state_dict(), f"encoder_lstm{suffix}")
                torch.save(decoder.state_dict(), f"decoder_lstm{suffix}")
            log_content += f" ⭐ 发现更优权重 (Epoch {epoch}) 已经更新 best 副本!"
        else:
            patience_counter += 1
            log_content += f"\n⚠️ 验证 Loss 未改善 ({patience_counter}/{int(patience)})"
            if patience_counter >= patience:
                log_content += f"\n🛑 早停触发！连续 {int(patience)} 轮未改善，及时止损结束训练。"
                stop_training_flag = True
            
        if stop_training_flag:
            # 手动停止或早停时，保存当前最后一份权重
            suffix = "_interrupted.pt" if not (patience_counter >= patience) else "_earlystopped.pt"
            if model_choice == "Transformer":
                torch.save(model.state_dict(), f"transformer{suffix}")
            else:
                torch.save(encoder.state_dict(), f"encoder_lstm{suffix}")
                torch.save(decoder.state_dict(), f"decoder_lstm{suffix}")
            log_content += f"\n💾 已保存当前进度至 {suffix}"
            break
            
        yield (
            f"✅ Epoch {epoch} 迭代完成",
            epoch,
            avg_train_loss,
            avg_val_loss,
            current_lr,
            f"{time.time() - start_time:.0f}s",
            log_content,
            create_plot(train_loss_history, val_loss_history)
        )
    
    final_status = "🎉 训练任务已成功结束" if not stop_training_flag else "⚠️ 训练已被手动中止"
    yield final_status, epochs if not stop_training_flag else epoch, train_loss_history[-1] if train_loss_history else 0, val_loss_history[-1] if val_loss_history else 0, current_lr, f"{time.time() - start_time:.0f}s", log_content + f"\n\n--- {final_status} ---", create_plot(train_loss_history, val_loss_history)

# ========== 构建极简高级界面 ==========
css = """
footer {visibility: hidden}
.metric-card { background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 15px; border: 1px solid rgba(255, 255, 255, 0.1); }
.status-active { color: #3498db; font-weight: bold; }
"""

with gr.Blocks(title="Pro Training Console") as demo:
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("# 🏮 Seq2Seq 专家级训练监控台")
            status_display = gr.Markdown("🟢 系统已就绪，请输入参数点击开始")
        with gr.Column(scale=1):
            gr.Button("开发者: Antigravity", variant="link", interactive=False)

    with gr.Row():
        # --- 侧边栏：参数面板 ---
        with gr.Column(scale=1, min_width=300):
            with gr.Group():
                gr.Markdown("### ⚙️ 核心参数配置")
                model_sel = gr.Dropdown(["Transformer", "LSTM"], value="Transformer", label="架构选择")
                epoch_num = gr.Slider(1, 100, value=30, step=1, label="训练轮次 (Epochs)")
                batch_size_sel = gr.Radio([32, 64, 128], value=64, label="批处理大小")
                lr_val = gr.Number(value=0.0001, label="初始学习率 (Initial LR)")
                patience_val = gr.Number(value=3, label="早停阈值 (Patience)")
                
                with gr.Row():
                    run_btn = gr.Button("🚀 启动引擎", variant="primary")
                    stop_btn = gr.Button("⏹️ 强行停止", variant="stop")
            
            gr.Markdown("---")
            gr.Markdown("### 📋 实时运行日志")
            log_box = gr.Textbox(placeholder="等待任务启动...", lines=10, max_lines=15, show_label=False, interactive=False, autoscroll=True)

        # --- 主面板：数据可视化 ---
        with gr.Column(scale=3):
            # 1. 顶部指标卡片
            with gr.Row():
                with gr.Column(elem_classes="metric-card"):
                    metric_epoch = gr.Number(label="当前 Epoch", value=0, precision=0)
                with gr.Column(elem_classes="metric-card"):
                    metric_train = gr.Number(label="训练 Loss", value=0.000, precision=4)
                with gr.Column(elem_classes="metric-card"):
                    metric_val = gr.Number(label="验证 Loss", value=0.000, precision=4)
                with gr.Column(elem_classes="metric-card"):
                    metric_lr = gr.Number(label="当前 LR", value=0.0000, precision=6)
                with gr.Column(elem_classes="metric-card"):
                    metric_time = gr.Textbox(label="已耗时", value="0s")
            
            # 2. 核心图表区
            with gr.Column(variant="panel"):
                gr.Markdown("### 📉 损失收敛动态追踪")
                plot_box = gr.Plot(show_label=False)

    # 交互绑定
    training_event = run_btn.click(
        fn=train_pro,
        inputs=[model_sel, batch_size_sel, epoch_num, lr_val, patience_val],
        outputs=[status_display, metric_epoch, metric_train, metric_val, metric_lr, metric_time, log_box, plot_box]
    )
    
    stop_btn.click(
        fn=request_stop,
        inputs=None,
        outputs=log_box,
        cancels=[training_event]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft(), css=css)
