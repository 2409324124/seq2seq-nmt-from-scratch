# translate_gradio.py - LSTM 版本（完整重写）
import gradio as gr
import torch
from utils import Lang, normalize_string, tokenize_de
from models import EncoderLSTM, AttnDecoderLSTM  # 注意：LSTM 版本

# ------------------- 参数 -------------------
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载词表（重新跑 prepare_data 获取）
from utils import prepare_data
input_lang, output_lang, _ = prepare_data(max_length=25)

# 加载模型（用你训练好的 LSTM epoch10 模型）
encoder = EncoderLSTM(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderLSTM(hidden_size, output_lang.n_words).to(device)

# 注意：这里要加载 LSTM 版本的权重（如果你已经训完 LSTM，文件名改成对应）
# 如果还没训 LSTM，先用原来的 epoch10 权重测试（后面再换）
encoder.load_state_dict(torch.load("encoder_lstm_epoch10.pt", map_location=device, weights_only=True))
decoder.load_state_dict(torch.load("decoder_lstm_epoch10.pt", map_location=device, weights_only=True))

encoder.eval()
decoder.eval()

def translate_live(text):
    """
    实时翻译函数：输入德语句子 → 输出英语翻译
    """
    if not text.strip():
        return "请输入德语句子"

    sentence = normalize_string(text)
    tokens = tokenize_de(sentence)
    indices = [input_lang.word2index.get(w, 2) for w in tokens]  # 未知词用 PAD=2

    # 输入 tensor：加 SOS 和 EOS
    input_tensor = torch.tensor([
        input_lang.word2index["<SOS>"]
    ] + indices + [
        input_lang.word2index["<EOS>"]
    ]).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encoder 前向（LSTM 返回 (output, (hidden, cell))）
        encoder_outputs, (encoder_hidden, encoder_cell) = encoder(input_tensor)

        # Decoder 初始化
        decoder_input = torch.tensor([[output_lang.word2index["<SOS>"]]]).to(device)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell  # LSTM 需要 cell

        translated = []

        for _ in range(50):  # 最大长度 50
            # Decoder 前向（多传 cell）
            output, decoder_hidden, decoder_cell, _ = decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs
            )

            topv, topi = output.topk(1)

            if topi.item() == output_lang.word2index["<EOS>"]:
                break

            translated.append(output_lang.index2word[topi.item()])
            decoder_input = topi.detach()

    return " ".join(translated)


# ------------------- Gradio 界面 -------------------
demo = gr.Interface(
    fn=translate_live,
    inputs=gr.Textbox(
        label="输入德语句子",
        placeholder="例如：Ein Roboter tanzt mit einem Menschen auf der Bühne.",
        lines=3,
        interactive=True
    ),
    outputs=gr.Textbox(label="实时英语翻译"),
    title="德语 → 英语 实时翻译器（LSTM 版本）",
    description="基于 Seq2Seq + Bahdanau Attention + LSTM，训练 10+ epoch，实时翻译",
    examples=[
        ["Ein Mann in einem blauen Hemd steht vor einem Gebäude."],
        ["Ein Roboter tanzt mit einem Menschen auf der Bühne."],
        ["Zwei junge Frauen laufen lachend durch den Park."],
        ["Nachdem es geregnet hat, scheint die Sonne wieder und es wird warm."]
    ],
    flagging_mode="never",
    allow_screenshot=True
)

# 启动 + 公网分享
demo.launch(share=True)