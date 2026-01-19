# translate.py - LSTM 版本（贪婪解码 + 强制热图 + 源序列反转 + OOV 检查）
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from utils import Lang, normalize_string, tokenize_de
from models import EncoderLSTM, BahdanauAttention, AttnDecoderLSTM

# 设置 Matplotlib 支持中文（消除字体警告）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ------------------- 参数 -------------------
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载词表
from utils import prepare_data
input_lang, output_lang, _ = prepare_data(max_length=25, min_freq=2)  # min_freq=1 避免低频词 OOV

# 加载 LSTM 模型（用 epoch37）
encoder = EncoderLSTM(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderLSTM(hidden_size, output_lang.n_words, dropout=0.4).to(device)

encoder.load_state_dict(torch.load("encoder_lstm_epoch30.pt", map_location=device, weights_only=True))
decoder.load_state_dict(torch.load("decoder_lstm_epoch30.pt", map_location=device, weights_only=True))

encoder.eval()
decoder.eval()

def translate_sentence(sentence, max_len=50):
    """
    使用贪婪解码进行翻译 + 强制显示注意力热图
    自动反转源序列（Sutskever 2014 技巧）
    自动检查 OOV（排查 0day 问题）
    """
    sentence = normalize_string(sentence)
    tokens = tokenize_de(sentence)
    
    # 统计并打印 OOV（源句词表未见词）
    missing_words = []
    for w in tokens:
        if w not in input_lang.word2index:
            missing_words.append(w)
    
    if missing_words:
        print(f"警告：源句中有 {len(missing_words)} 个词表未见词（可能导致翻译崩坏）：{missing_words}")
    else:
        print("源句所有词都在词表中 ✓")
    
    # 核心技巧：反转源序列
    tokens_reversed = list(reversed(tokens))
    indices_reversed = [input_lang.word2index.get(w, 2) for w in tokens_reversed]
    
    input_tensor = torch.tensor([input_lang.word2index["<SOS>"]] + indices_reversed + [input_lang.word2index["<EOS>"]]).unsqueeze(0).to(device)

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

    # 强制显示注意力热图（使用反转后的源句）
    if attentions:
        attentions = np.stack(attentions)

        fig, ax = plt.subplots(figsize=(12, 7))
        cax = ax.matshow(attentions, cmap='bone')
        fig.colorbar(cax)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # 用反转后的 tokens 显示横轴
        ax.set_xticklabels([''] + tokens_reversed + ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + translated + ['<EOS>'])

        plt.title("Attention Heatmap (Greedy Decoding + Source Reversed)")
        plt.xlabel("德语源句 (已反转)")
        plt.ylabel("英语生成句")
        plt.tight_layout()
        plt.show()

    return translation


# ------------------- 交互模式 -------------------
print("\n输入德语句子（输入 q 退出）：")
print("提示：LSTM Epoch 37 + 源序列反转 + 贪婪解码 + 强制热图")

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