# translate_gradio.py - LSTM 版本（完整修复版）
import gradio as gr
import torch
import pickle
import os
from utils import Lang, normalize_string, tokenize_de
from models import EncoderLSTM, AttnDecoderLSTM

# ------------------- 参数 -------------------
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 词表缓存文件（避免每次重新 prepare_data）
VOCAB_CACHE = "vocab_cache.pkl"

if os.path.exists(VOCAB_CACHE):
    with open(VOCAB_CACHE, 'rb') as f:
        input_lang, output_lang = pickle.load(f)
    print("从缓存加载词表")
else:
    from utils import prepare_data
    input_lang, output_lang, _ = prepare_data(max_length=25, min_freq=2)
    with open(VOCAB_CACHE, 'wb') as f:
        pickle.dump((input_lang, output_lang), f)
    print("生成并缓存词表")

# 加载模型（用 epoch30）
encoder = EncoderLSTM(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderLSTM(hidden_size, output_lang.n_words).to(device)

try:
    encoder.load_state_dict(torch.load("encoder_lstm_epoch30.pt", map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load("decoder_lstm_epoch30.pt", map_location=device, weights_only=True))
    print("模型权重加载成功")
except FileNotFoundError:
    print("模型权重文件不存在！请检查 encoder_lstm_epoch30.pt 和 decoder_lstm_epoch30.pt")
    exit(1)
except Exception as e:
    print(f"加载模型失败: {str(e)}")
    exit(1)

encoder.eval()
decoder.eval()

def translate_live(text):
    if not text.strip():
        return "请输入德语句子"

    try:
        sentence = normalize_string(text)
        tokens = tokenize_de(sentence)
        tokens = list(reversed(tokens))
        indices = [input_lang.word2index.get(w, 2) for w in tokens]

        input_tensor = torch.tensor([input_lang.word2index["<SOS>"]] + indices + [input_lang.word2index["<EOS>"]]).unsqueeze(0).to(device)

        with torch.no_grad():
            encoder_outputs, (encoder_hidden, encoder_cell) = encoder(input_tensor)

            decoder_input = torch.tensor([[output_lang.word2index["<SOS>"]]]).to(device)
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell

            translated = []

            for _ in range(50):
                output, decoder_hidden, decoder_cell, _ = decoder(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs
                )

                topv, topi = output.topk(1)
                if topi.item() == output_lang.word2index["<EOS>"]:
                    break

                translated.append(output_lang.index2word[topi.item()])
                decoder_input = topi.detach()

        return " ".join(translated)

    except Exception as e:
        return f"翻译出错: {str(e)}"

# ------------------- Gradio 界面 -------------------
demo = gr.Interface(
    fn=translate_live,
    inputs=gr.Textbox(
        label="输入德语句子",
        placeholder="例如：Ein Roboter tanzt mit einem Menschen auf der Bühne.",
        interactive=True
    ),
    outputs=gr.Textbox(label="实时英语翻译"),
    title="德语 → 英语 实时翻译器（LSTM 版本）",
    description="基于 Seq2Seq + Bahdanau Attention + LSTM，训练 30 epoch，实时翻译",
    examples=[
        ["Ein Mann steht vor einem Haus."],
        ["Die Katze schläft auf dem Sofa."],
        ["Zwei Kinder spielen im Park."],
        ["Ein Junge liest ein Buch."]
    ],
    flagging_mode="never",
)

# 启动 + 公网分享
demo.launch(share=True)