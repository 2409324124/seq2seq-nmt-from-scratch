import gradio as gr
import torch
import torch.nn as nn
import pickle
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import normalize_string, tokenize_de, prepare_data, TranslationDataset, collate_fn
from models_transformer import TransformerModel
from models import EncoderLSTM, AttnDecoderLSTM
from torch.utils.data import DataLoader
import sacrebleu

# ------------------- 环境配置 -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_CACHE = "vocab_cache.pkl"

# 加载词表
if os.path.exists(VOCAB_CACHE):
    with open(VOCAB_CACHE, 'rb') as f:
        input_lang, output_lang = pickle.load(f)
else:
    input_lang, output_lang, _ = prepare_data(max_length=25, min_freq=2)
    with open(VOCAB_CACHE, 'wb') as f:
        pickle.dump((input_lang, output_lang), f)

# ------------------- 全局模型变量 -------------------
current_model = None
current_model_name = ""
current_arch = ""

def get_checkpoints():
    """获取目录下所有的模型权重文件"""
    ckpts = glob.glob("*.pt")
    return sorted(ckpts, key=os.path.getmtime, reverse=True)

def load_selected_model(arch, ckpt_path):
    global current_model, current_model_name, current_arch
    if not ckpt_path:
        return "请选择模型权重文件"
    
    try:
        if arch == "Transformer":
            model = TransformerModel(
                input_lang.n_words, output_lang.n_words, 
                d_model=256, nhead=8, nhid=512, nlayers=3
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            current_model = model
        else:
            # LSTM 需要加载两个
            # 注意：这里假设 LSTM 的保存方式是字典或单独文件
            # 如果是 train_gui 保存的，LSTM 是分开存的，或者是 dict
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            encoder = EncoderLSTM(input_lang.n_words, 256).to(device)
            decoder = AttnDecoderLSTM(256, output_lang.n_words).to(device)
            
            if isinstance(state, dict) and 'en' in state:
                encoder.load_state_dict(state['en'])
                decoder.load_state_dict(state['de'])
            else:
                # 兼容旧格式
                encoder.load_state_dict(state)
                # 尝试寻找对应的 decoder
                dec_path = ckpt_path.replace("encoder", "decoder")
                if os.path.exists(dec_path):
                    decoder.load_state_dict(torch.load(dec_path, map_location=device, weights_only=True))

            current_model = (encoder, decoder)
            
        current_model_name = ckpt_path
        current_arch = arch
        return f"✅ 已成功加载 {arch} 模型: {ckpt_path}"
    except Exception as e:
        return f"❌ 加载失败: {str(e)}"

# ------------------- 核心逻辑 -------------------

def beam_search_transformer(model, src_tensor, beam_size=3, max_length=50):
    model.eval()
    with torch.no_grad():
        src_padding_mask = (src_tensor == 2) # <PAD> 是 2
        memory = model.encode(src_tensor, src_padding_mask=src_padding_mask)
        
        # (score, indices)
        beams = [(0, [output_lang.word2index["<SOS>"]])]
        
        for _ in range(max_length):
            new_beams = []
            for score, indices in beams:
                if indices[-1] == output_lang.word2index["<EOS>"]:
                    new_beams.append((score, indices))
                    continue
                
                tgt_tensor = torch.tensor(indices).unsqueeze(0).to(device)
                tgt_mask = model.generate_square_subsequent_mask(len(indices)).to(device)
                output = model.decode(tgt_tensor, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
                output = model.decoder_out(output)
                
                log_probs = torch.log_softmax(output[0, -1, :], dim=-1)
                top_v, top_i = log_probs.topk(beam_size)
                
                for i in range(beam_size):
                    new_beams.append((score + top_v[i].item(), indices + [top_i[i].item()]))
            
            # 排序并筛选
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]
            
            # 如果所有 beam 都结束了，提前退出
            if all(b[1][-1] == output_lang.word2index["<EOS>"] for b in beams):
                break
                
        return beams[0][1]

def process_translation(text, search_type, beam_size):
    if not current_model:
        return "请先加载模型", None
    if not text.strip():
        return "请输入德语句子", None

    tokens = tokenize_de(normalize_string(text))
    indices = [input_lang.word2index.get(w, 2) for w in tokens]
    src_tensor = torch.tensor([input_lang.word2index["<SOS>"]] + indices + [input_lang.word2index["<EOS>"]]).unsqueeze(0).to(device)
    
    fig = None
    
    if current_arch == "Transformer":
        model = current_model
        if search_type == "Beam Search":
            res_indices = beam_search_transformer(model, src_tensor, beam_size=int(beam_size))
        else:
            # Greedy 模式支持 Heatmap
            model.eval()
            with torch.no_grad():
                src_padding_mask = (src_tensor == 2)
                memory = model.encode(src_tensor, src_padding_mask=src_padding_mask)
                res_indices = [output_lang.word2index["<SOS>"]]
                for _ in range(50):
                    tgt_tensor = torch.tensor(res_indices).unsqueeze(0).to(device)
                    tgt_mask = model.generate_square_subsequent_mask(len(res_indices)).to(device)
                    output = model.decode(tgt_tensor, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
                    output = model.decoder_out(output)
                    idx = output[0, -1, :].argmax().item()
                    res_indices.append(idx)
                    if idx == output_lang.word2index["<EOS>"]: break
            
            # 绘图
            attn = model.get_attention(src_tensor, torch.tensor(res_indices).unsqueeze(0).to(device), src_padding_mask=src_padding_mask)
            if attn is not None:
                attn = attn[0].cpu().numpy() # [tgt_len, src_len]
                src_labels = ["<SOS>"] + tokens + ["<EOS>"]
                tgt_labels = [output_lang.index2word[idx] for idx in res_indices]
                
                plt.close('all')
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(attn, xticklabels=src_labels, yticklabels=tgt_labels, annot=False, cmap='viridis', ax=ax)
                ax.set_title("Transformer Attention Heatmap (Cross-Attention)")
                plt.tight_layout()

        result = " ".join([output_lang.index2word[idx] for idx in res_indices if idx not in [0, 1, 2]])
    else:
        # LSTM 模式 (不支持 Beam Search 演示，仅 Greedy)
        encoder, decoder = current_model
        with torch.no_grad():
            src_rev = torch.flip(src_tensor, dims=[1])
            encoder_outputs, (h, c) = encoder(src_rev)
            decoder_input = torch.tensor([[output_lang.word2index["<SOS>"]]]).to(device)
            res_indices = []
            for _ in range(50):
                output, h, c, _ = decoder(decoder_input, h, c, encoder_outputs)
                idx = output.argmax().item()
                if idx == output_lang.word2index["<EOS>"]: break
                res_indices.append(idx)
                decoder_input = torch.tensor([[idx]]).to(device)
            result = " ".join([output_lang.index2word[idx] for idx in res_indices])

    return result, fig

def run_batch_eval(num_samples):
    if not current_model:
        return "请先加载模型"
    
    # 加载测试数据
    _, _, test_pairs = prepare_data(max_length=25, min_freq=2)
    test_subset = test_pairs[:int(num_samples)]
    
    hypotheses = []
    references = []
    
    # 简单的进度展示回调其实很难在 Blocks 内部直接做，这里直接循环
    for de, en in test_subset:
        trans, _ = process_translation(de, "Greedy", 1)
        hypotheses.append(trans)
        references.append([en])
    
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    result_str = f"📊 评估结果 ({num_samples} 样本):\n"
    result_str += f"- BLEU Score: {bleu.score:.2f}\n"
    result_str += f"- Precisions: {bleu.precisions[0]:.1f} / {bleu.precisions[1]:.1f} / {bleu.precisions[2]:.1f} / {bleu.precisions[3]:.1f}\n"
    result_str += f"- BP (Brevity Penalty): {bleu.bp:.3f}"
    
    return result_str

# ------------------- UI 界面 -------------------

# 动态 CSS
css = """
footer {visibility: hidden}
.gradio-container {background-color: #f8f9fa; font-family: 'Segoe UI', system-ui, sans-serif;}
.pro-card { border-radius: 12px; border: 1px solid #e0e0e0; background: white; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
.status-msg { font-weight: 600; color: #2e7d32; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🚀 NMT 翻译专家工作站 (Pro Edition)")
    gr.Markdown("验证模型质量、观察注意力分配并进行 BLEU 量化测试。")
    
    with gr.Row():
        with gr.Column(scale=1):
            arch_sel = gr.Dropdown(["Transformer", "LSTM"], label="1. 选择架构", value="Transformer")
            ckpt_sel = gr.Dropdown(get_checkpoints(), label="2. 选择权重文件")
            load_btn = gr.Button("🔌 加载/更换模型", variant="primary")
            status_msg = gr.Markdown("⚠️ 等待加载模型...", elem_classes=["status-msg"])
            
            with gr.Accordion("高级解码参数", open=False):
                search_mode = gr.Radio(["Greedy", "Beam Search"], label="搜索策略", value="Greedy")
                beam_sz = gr.Slider(1, 10, value=3, step=1, label="Beam Size")
                
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🎯 交互翻译"):
                    with gr.Group():
                        input_box = gr.Textbox(label="输入德语", placeholder="在此输入德语句子...", lines=3)
                        trans_btn = gr.Button("✨ 开始翻译", variant="secondary")
                    
                    with gr.Row():
                        with gr.Column():
                            output_box = gr.Textbox(label="翻译结果", interactive=False, lines=3)
                        with gr.Column():
                            attn_plot = gr.Plot(label="注意力热图 (仅 Greedy 模式)")
                            
                with gr.TabItem("📊 批量评估"):
                    num_samples = gr.Number(value=100, label="评估样本数 (前 N 条)")
                    eval_btn = gr.Button("📐 启动 BLEU 测评")
                    eval_res = gr.Textbox(label="测评报告", lines=5)

    # 事件处理
    load_btn.click(load_selected_model, [arch_sel, ckpt_sel], status_msg)
    trans_btn.click(process_translation, [input_box, search_mode, beam_sz], [output_box, attn_plot])
    eval_btn.click(run_batch_eval, [num_samples], eval_res)
    
    gr.Examples(
        [["Ein Hund rennt durch den Park.", "Greedy", 1], 
         ["Die Katze schläft auf dem Tisch.", "Beam Search", 3]],
        inputs=[input_box, search_mode, beam_sz]
    )

if __name__ == "__main__":
    demo.launch(share=True)
