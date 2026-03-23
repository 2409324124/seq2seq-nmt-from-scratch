# translate.py - LSTM 版本（命令行交互 + 弹出热图 + 源序列反转 + 贪婪解码）

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from utils import Lang, normalize_string, tokenize_de, prepare_data
from models import EncoderLSTM, AttnDecoderLSTM

# 设置 Matplotlib 支持中文（消除字体警告）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ------------------- 参数 -------------------
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载词表
input_lang, output_lang, _ = prepare_data(max_length=25, min_freq=2)

# 加载 LSTM 模型（用 epoch30）
encoder = EncoderLSTM(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderLSTM(hidden_size, output_lang.n_words, dropout=0.4).to(device)

encoder.load_state_dict(torch.load("encoder_lstm_epoch35.pt", map_location=device, weights_only=True))
decoder.load_state_dict(torch.load("decoder_lstm_epoch35.pt", map_location=device, weights_only=True))

encoder.eval()
decoder.eval()

def translate_sentence(sentence, max_len=50, show_attention=True):
    """
    使用贪婪解码进行翻译 + 弹出注意力热图
    """
    sentence = normalize_string(sentence)
    tokens = tokenize_de(sentence)
    
    indices = [input_lang.word2index.get(w, 2) for w in tokens]
    
    input_tensor = torch.tensor([input_lang.word2index["<SOS>"]] + indices + [input_lang.word2index["<EOS>"]]).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, (encoder_hidden, encoder_cell) = encoder(input_tensor)

        decoder_input = torch.tensor([[output_lang.word2index["<SOS>"]]]).to(device)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        translated = []
        attentions = []

        for _ in range(max_len):
            output, decoder_hidden, decoder_cell, attn_weights = decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs
            )

            attentions.append(attn_weights.squeeze(0).cpu().numpy())

            topv, topi = output.topk(1)
            if topi.item() == output_lang.word2index["<EOS>"]:
                break

            translated.append(output_lang.index2word[topi.item()])
            decoder_input = topi.detach()

        translation = " ".join(translated)

    # 弹出注意力热图
    if show_attention and attentions:
        attentions = np.stack(attentions)

        fig, ax = plt.subplots(figsize=(12, 7))
        cax = ax.matshow(attentions, cmap='bone')
        fig.colorbar(cax)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        ax.set_xticklabels([''] + tokens + ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + translated + ['<EOS>'])

        plt.title("Attention Heatmap (Greedy Decoding)")
        plt.xlabel("德语源句")
        plt.ylabel("英语生成句")
        plt.tight_layout()
        plt.show()  # ← 关键：弹出窗口显示热图

    return translation


# ------------------- 交互模式 -------------------
print("\n输入德语句子（输入 q 退出）：")
print("提示：LSTM Epoch 35 + 贪婪解码 + 弹出热图")

while True:
    sent = input("> ")
    if sent.lower() == 'q':
        print("退出翻译")
        break
    if sent.strip() == '':
        continue

    try:
        result = translate_sentence(sent)
        print("翻译结果:", result)
    except Exception as e:
        print("翻译出错:", str(e))