# translate.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import Lang, normalize_string, tokenize_de
from models import EncoderRNN, AttnDecoderRNN

# 参数（和训练一致）
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载词表（从数据准备获取，或直接用保存的）
# 注意：这里直接重新准备数据获取词表（简单方式）
from utils import prepare_data
input_lang, output_lang, _ = prepare_data(max_length=25)  # 只用词表

# 加载模型（用 epoch 10 的）
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

encoder.load_state_dict(torch.load("encoder_epoch10.pt", map_location=device))
decoder.load_state_dict(torch.load("decoder_epoch10.pt", map_location=device))

encoder.eval()
decoder.eval()

def translate_sentence(sentence):
    sentence = normalize_string(sentence)
    tokens = tokenize_de(sentence)
    indices = [input_lang.word2index.get(w, 2) for w in tokens]  # PAD=2

    input_tensor = torch.tensor([input_lang.word2index["<SOS>"]] + indices + [input_lang.word2index["<EOS>"]]).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        decoder_input = torch.tensor([[output_lang.word2index["<SOS>"]]]).to(device)
        decoder_hidden = encoder_hidden

        translated = []
        attentions = []

        for _ in range(50):  # 最大长度 50
            output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            attentions.append(attn_weights.squeeze(0).cpu().numpy())

            topv, topi = output.topk(1)
            if topi.item() == output_lang.word2index["<EOS>"]:
                break

            translated.append(output_lang.index2word[topi.item()])
            decoder_input = topi.detach()

    # 画注意力热图
    attentions = torch.tensor(attentions).squeeze(1).cpu().numpy()  # (tgt_len, src_len)
    fig, ax = plt.subplots()
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + tokens + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + translated + ['<EOS>'])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

    return ' '.join(translated)

# 交互模式
print("输入德语句子（输入 q 退出）：")
while True:
    sent = input("> ")
    if sent.lower() == 'q':
        break
    if sent.strip() == '':
        continue
    translation = translate_sentence(sent)
    print("翻译:", translation)