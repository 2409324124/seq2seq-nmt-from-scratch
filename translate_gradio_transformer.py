import gradio as gr
import torch
import pickle
import os
from utils import normalize_string, tokenize_de
from models import EncoderLSTM, AttnDecoderLSTM
from models_transformer import TransformerModel

# ------------------- 参数 -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256
VOCAB_CACHE = "vocab_cache.pkl"

# 加载词表
if os.path.exists(VOCAB_CACHE):
    with open(VOCAB_CACHE, 'rb') as f:
        input_lang, output_lang = pickle.load(f)
else:
    from utils import prepare_data
    input_lang, output_lang, _ = prepare_data(max_length=25, min_freq=2)
    with open(VOCAB_CACHE, 'wb') as f:
        pickle.dump((input_lang, output_lang), f)

# ------------------- 加载 LSTM 模型 -------------------
encoder_lstm = EncoderLSTM(input_lang.n_words, hidden_size).to(device)
decoder_lstm = AttnDecoderLSTM(hidden_size, output_lang.n_words).to(device)

try:
    encoder_lstm.load_state_dict(torch.load("encoder_lstm_best.pt", map_location=device, weights_only=True))
    decoder_lstm.load_state_dict(torch.load("decoder_lstm_best.pt", map_location=device, weights_only=True))
    lstm_ready = True
except:
    lstm_ready = False

encoder_lstm.eval()
decoder_lstm.eval()

# ------------------- 加载 Transformer 模型 -------------------
transformer = TransformerModel(
    input_lang.n_words, output_lang.n_words, 
    d_model=256, nhead=8, nhid=512, nlayers=3
).to(device)

try:
    transformer.load_state_dict(torch.load("transformer_best.pt", map_location=device, weights_only=True))
    transformer_ready = True
except:
    transformer_ready = False

transformer.eval()

def translate_lstm(text):
    tokens = tokenize_de(normalize_string(text))
    tokens = list(reversed(tokens))
    indices = [input_lang.word2index.get(w, 2) for w in tokens]
    input_tensor = torch.tensor([input_lang.word2index["<SOS>"]] + indices + [input_lang.word2index["<EOS>"]]).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, (h, c) = encoder_lstm(input_tensor)
        decoder_input = torch.tensor([[output_lang.word2index["<SOS>"]]]).to(device)
        translated = []
        for _ in range(50):
            output, h, c, _ = decoder_lstm(decoder_input, h, c, encoder_outputs)
            idx = output.argmax().item()
            if idx == output_lang.word2index["<EOS>"]: break
            translated.append(output_lang.index2word[idx])
            decoder_input = torch.tensor([[idx]]).to(device)
    return " ".join(translated)

def translate_transformer(text):
    tokens = tokenize_de(normalize_string(text))
    indices = [input_lang.word2index.get(w, 2) for w in tokens]
    src_tensor = torch.tensor([input_lang.word2index["<SOS>"]] + indices + [input_lang.word2index["<EOS>"]]).unsqueeze(0).to(device)
    
    src_padding_mask = (src_tensor == 2)
    memory = transformer.encode(src_tensor, src_padding_mask=src_padding_mask)
    
    tgt_indices = [output_lang.word2index["<SOS>"]]
    for _ in range(50):
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
        tgt_mask = transformer.generate_square_subsequent_mask(len(tgt_indices)).to(device)
        output = transformer.decode(tgt_tensor, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
        output = transformer.decoder_out(output)
        idx = output[0, -1, :].argmax().item()
        if idx == output_lang.word2index["<EOS>"]: break
        tgt_indices.append(idx)
    
    return " ".join([output_lang.index2word[idx] for idx in tgt_indices[1:]])

def translate_all(text, model_type):
    if not text.strip(): return "请输入句子"
    if model_type == "Transformer":
        return translate_transformer(text) if transformer_ready else "Transformer 模型权重未找到"
    else:
        return translate_lstm(text) if lstm_ready else "LSTM 模型权重未找到"

# ------------------- Gradio 界面 -------------------
with gr.Blocks(title="NMT 对比测试 (LSTM vs Transformer)") as demo:
    gr.Markdown("# 德语 → 英语 翻译架构对比")
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="输入德语", placeholder="Ein Mann spielt Gitarre.")
            model_select = gr.Radio(["LSTM", "Transformer"], label="选择模型架构", value="Transformer")
            translate_btn = gr.Button("翻译")
        with gr.Column():
            output_text = gr.Textbox(label="英语翻译结果")
            
    translate_btn.click(fn=translate_all, inputs=[input_text, model_select], outputs=output_text)
    
    gr.Examples(
        [["Ein Hund rennt durch den Park.", "Transformer"], ["Ein kleiner Junge isst einen Apfel.", "LSTM"]],
        inputs=[input_text, model_select]
    )

demo.launch(share=True)
