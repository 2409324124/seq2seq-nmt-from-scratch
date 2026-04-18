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
import traceback

# ------------------- 视觉风格配置 -------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Tahoma', 'Arial'] # 中文字体支持
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('bmh')

import utils_sys
from utils import prepare_data, TranslationDataset, collate_fn
# 动态加载引擎
from engines.transformer_engine import TransformerEngine
from engines.lstm_engine import LSTMEngine
from engines.bert_engine import BertEngine
from engines.gpt_engine import GptEngine

# 建立引擎映射表
ENGINE_MAP = {
    "Transformer": TransformerEngine,
    "Lstm": LSTMEngine,
    "Bert": BertEngine,
    "Gpt": GptEngine
}

def get_available_archs():
    """自动探测 engines 目录下可用的架构"""
    import glob
    # 也可以根据文件探测，或者直接使用预定义映射
    files = glob.glob(os.path.join("engines", "*_engine.py"))
    archs = []
    for f in files:
        name = os.path.basename(f).replace("_engine.py", "").capitalize()
        if name != "Base":
            archs.append(name)
    return archs if archs else ["Transformer", "LSTM"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 全局控制变量 ==========
stop_training_flag = False

def request_stop():
    global stop_training_flag
    stop_training_flag = True
    return "🛑 接收到终止指令，正在安全退出引擎..."

def create_plot(train_loss, val_loss):
    """ 创建高级 Matplotlib 曲线图 """
    plt.close('all')  # 关闭之前的绘图以释放内存并解决警告
    if not train_loss: return None
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, 'o-', color='#3498db', linewidth=2, label='Train Loss', markersize=4)
    if val_loss:
        val_epochs = range(1, len(val_loss) + 1)
        ax.plot(val_epochs, val_loss, 's-', color='#e74c3c', linewidth=2, label='Val Loss', markersize=4)
    ax.set_title("Training Convergence Monitor", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return fig

# ========== 核心训练生成器 (Modular Pro) ==========
def train_pro(model_choice, batch_size, epochs, learning_rate, patience):
    global stop_training_flag
    stop_training_flag = False
    
    epochs = int(epochs)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    
    # 初始化 UI 返回值
    status_msg = "🔄 初始化数据中..."
    log_content = f"[*] 环境检查: {device}\n[*] 加载双语对数据..."
    
    yield (status_msg, 0, 0.0, 0.0, learning_rate, "0s", log_content, None)
    
    # 1. 准备数据
    try:
        input_lang, output_lang, pairs = prepare_data(max_length=25, min_freq=2)
        dataset = TranslationDataset(pairs, input_lang, output_lang)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
        log_content += f"\n[+] 数据就绪: 训练集 {train_size} / 验证集 {val_size}"
        # 强制中间刷新一次 UI，确保用户看到“数据就绪”
        yield ("🔄 正在初始化引擎...", 0, 0.0, 0.0, learning_rate, "0s", log_content, None)
    except Exception as e:
        yield f"❌ 数据加载失败: {str(e)}", 0, 0.0, 0.0, learning_rate, "0s", log_content, None
        return

    # 2. 引擎初始化
    arch_key = model_choice.capitalize() # 统一标准化 Key
    if arch_key not in ENGINE_MAP:
        yield f"❌ 未知架构: {arch_key}", 0, 0.0, 0.0, learning_rate, "0s", log_content, None
        return

    engine = ENGINE_MAP[arch_key](device)
    engine.initialize_model(input_lang.n_words, output_lang.n_words, learning_rate)
    
    log_content += f"\n[+] {arch_key} 训练引擎已启动。"
    yield ("🔥 引擎就绪，开始训练...", 0, 0.0, 0.0, learning_rate, "0s", log_content, None)
    current_lr = learning_rate
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    # 3. 训练主循环
    try:
        for epoch in range(1, epochs + 1):
            if stop_training_flag: break
            
            # --- 训练阶段 ---
            epoch_start = time.time()
            for i, batch_loss in enumerate(engine.train_one_epoch(train_loader, epoch)):
                if stop_training_flag: break
                
                # 限制 UI 刷新频率（每 30 个 batch 刷新一次），极大幅度提升训练效率！
                if i % 30 == 0:
                    elapsed = time.time() - start_time
                    yield (
                        f"🔥 正在训练 Epoch {epoch}/{epochs} (Batch {i})...", 
                        epoch, 
                        round(batch_loss, 4), 
                        (val_loss_history[-1] if val_loss_history else 0.0),
                        current_lr,
                        f"{int(elapsed)}s",
                        log_content + f"\n[Epoch {epoch}] Batch {i} | Loss: {batch_loss:.4f}",
                        create_plot(train_loss_history + [batch_loss], val_loss_history)
                    )
            
            if stop_training_flag: break
            train_loss_history.append(batch_loss)
            
            # --- 验证阶段 ---
            val_loss = engine.validate(val_loader)
            val_loss_history.append(val_loss)
            
            # 更新调度器
            if engine.scheduler:
                engine.scheduler.step(val_loss)
                current_lr = engine.optimizer.param_groups[0]['lr']
            
            # --- 存档与早停逻辑 ---
            log_content += f"\n[✔] Epoch {epoch} 完成! Train Loss: {batch_loss:.4f} | Val Loss: {val_loss:.4f}"
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = f"{model_choice.lower()}_best_pro.pt"
                engine.save_checkpoint(ckpt_path, epoch, val_loss)
                log_content += f" (🌟 最佳存档已保存: {ckpt_path})"
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log_content += f"\n🛑 早停触发: 连续 {patience} 轮未优化。"
                    break
            
            yield (f"🔍 Epoch {epoch} 验证结束", epoch, batch_loss, val_loss, current_lr, f"{int(time.time()-start_time)}s", log_content, create_plot(train_loss_history, val_loss_history))
    except Exception as e:
        error_trace = traceback.format_exc()
        log_content += f"\n\n❌ 引擎内部崩溃! 错误详情:\n{error_trace}"
        yield "❌ 引擎崩溃", epoch if 'epoch' in locals() else 0, 0.0, 0.0, current_lr, "0s", log_content, None
        return

    final_msg = "✅ 任务顺利完成" if not stop_training_flag else "🛑 任务已手动中断"
    yield final_msg, epoch, train_loss_history[-1], val_loss_history[-1], current_lr, f"{int(time.time()-start_time)}s", log_content, create_plot(train_loss_history, val_loss_history)

# ==================== Windows 2000 Retro UI ====================
css = """
.gradio-container { background-color: #d4d0c8 !important; font-family: 'Tahoma', sans-serif !important; }
.win-window { background-color: #d4d0c8 !important; border: 2px outset #ffffff !important; border-right-color: #404040 !important; border-bottom-color: #404040 !important; padding: 2px !important; margin-bottom: 10px !important; }
.win-titlebar { background: linear-gradient(90deg, #000080 0%, #1084d0 100%) !important; color: white !important; font-weight: bold !important; padding: 2px 8px !important; margin: -2px -2px 5px -2px !important; font-size: 13px !important; }
.win-btn { background-color: #d4d0c8 !important; border: 2px outset #ffffff !important; border-right-color: #404040 !important; border-bottom-color: #404040 !important; border-radius: 0 !important; color: black !important; padding: 2px 12px !important; }
.win-btn:active { border: 2px inset #ffffff !important; border-right-color: #808080 !important; border-bottom-color: #808080 !important; }
.win-inset { background-color: white !important; border: 2px inset #ffffff !important; border-right-color: #dfdfdf !important; border-bottom-color: #dfdfdf !important; padding: 5px !important; }
.stat-card { background-color: #d4d0c8 !important; border: 2px outset #ffffff !important; padding: 5px !important; }
.sys-status { font-family: 'Consolas', monospace; font-size: 11px; color: #000080; background: #c0c0c0; padding: 2px 8px; border: 1px inset #fff; margin-top: 5px; }
"""

with gr.Blocks(title="Seq2Seq AI Pro Hub [Win2k]") as demo:
    # 顶部状态栏
    with gr.Row():
        with gr.Column(elem_classes=["win-window"]):
            gr.Markdown("<div class='win-titlebar'>🖥️ 通用 AI 训练专家控制中心 [Version 6.0.2600.Modular]</div>")
            with gr.Row():
                status_display = gr.Markdown("🟢 **系统状态**: 就绪。就绪。")
                sys_info = gr.Markdown(f"<div class='sys-status'>{utils_sys.get_system_summary()}</div>")
    
    with gr.Row():
        # 侧边栏
        with gr.Column(scale=1, min_width=300, elem_classes=["win-window"]):
            gr.Markdown("<div class='win-titlebar'>🛠️ 架构与超参 (Settings)</div>")
            
            arch_list = get_available_archs()
            model_sel = gr.Dropdown(choices=arch_list, value=arch_list[0], label="架构自动探测")
            
            epoch_num = gr.Slider(1, 100, value=30, step=1, label="训练轮次")
            batch_size_sel = gr.Radio([32, 64, 128], value=64, label="批处理大小")
            lr_val = gr.Dropdown(choices=["0.001", "0.0005", "0.0001", "0.00005", "0.00001"], value="0.001", label="初始学习率 (decimal)", allow_custom_value=False)
            patience_val = gr.Number(value=3, label="早停阈值", minimum=1, precision=0)
            
            with gr.Row():
                run_btn = gr.Button("🚀 启动引擎", variant="primary", elem_classes=["win-btn"])
                stop_btn = gr.Button("⏹️ 终止任务", variant="stop", elem_classes=["win-btn"])
            
            gr.Markdown("<div class='win-titlebar'>📝 运行日志 (Engine Log)</div>")
            log_box = gr.Textbox(placeholder="等待引擎启动...", lines=10, interactive=False, elem_classes=["win-inset"])

        # 主面板
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(elem_classes=["stat-card"]):
                    gr.Markdown("📂 **Epoch**")
                    metric_epoch = gr.Number(value=0, precision=0, show_label=False)
                with gr.Column(elem_classes=["stat-card"]):
                    gr.Markdown("📉 **Train Loss**")
                    metric_train = gr.Number(value=0.000, precision=4, show_label=False)
                with gr.Column(elem_classes=["stat-card"]):
                    gr.Markdown("🧪 **Val Loss**")
                    metric_val = gr.Number(value=0.000, precision=4, show_label=False)
                with gr.Column(elem_classes=["stat-card"]):
                    gr.Markdown("⚡ **LR**")
                    metric_lr = gr.Number(value=0.0000, precision=6, show_label=False)
                with gr.Column(elem_classes=["stat-card"]):
                    gr.Markdown("⏱️ **Time**")
                    metric_time = gr.Textbox(value="0s", show_label=False)
            
            with gr.Column(elem_classes=["win-window"]):
                gr.Markdown("<div class='win-titlebar'>📈 实时指标收敛监控</div>")
                plot_box = gr.Plot(show_label=False, elem_classes=["win-inset"])

    # 事件绑定
    training_event = run_btn.click(
        fn=train_pro, 
        inputs=[model_sel, batch_size_sel, epoch_num, lr_val, patience_val],
        outputs=[status_display, metric_epoch, metric_train, metric_val, metric_lr, metric_time, log_box, plot_box]
    )
    stop_btn.click(fn=request_stop, outputs=[log_box], cancels=[training_event])

if __name__ == "__main__":
    demo.launch(
        share=True, 
        theme=gr.themes.Base(), 
        css=css
    )
