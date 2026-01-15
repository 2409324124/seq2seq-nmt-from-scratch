import gradio as gr
import torch
from utils import Lang, normalize_string, tokenize_de
from models import EncoderRNN, AttnDecoderRNN

# 参数（和训练一致）
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载词表（最快的方式：重新跑一次 prepare_data）
from utils import prepare_data
input_lang, output_lang, _ = prepare_data(max_length=25)

# 加载模型（用你最好的那个 epoch）
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

encoder.load_state_dict(torch.load("encoder_epoch10.pt", map_location=device, weights_only=True))
decoder.load_state_dict(torch.load("decoder_epoch10.pt", map_location=device, weights_only=True))

encoder.eval()
decoder.eval()

def translate_live(text):
    if not text.strip():
        return "请输入德语句子"

    sentence = normalize_string(text)
    tokens = tokenize_de(sentence)
    indices = [input_lang.word2index.get(w, 2) for w in tokens]

    input_tensor = torch.tensor([input_lang.word2index["<SOS>"]] + indices + [input_lang.word2index["<EOS>"]]).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        decoder_input = torch.tensor([[output_lang.word2index["<SOS>"]]]).to(device)
        decoder_hidden = encoder_hidden

        translated = []

        for _ in range(50):
            output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = output.topk(1)

            if topi.item() == output_lang.word2index["<EOS>"]:
                break

            translated.append(output_lang.index2word[topi.item()])
            decoder_input = topi.detach()

    return " ".join(translated)


# 创建 Gradio 界面
demo = gr.Interface(
    fn=translate_live,
    inputs=gr.Textbox(
        label="输入德语句子",
        placeholder="例如：Ein Roboter tanzt mit einem Menschen auf der Bühne.",
        lines=3,
        interactive=True
    ),
    outputs=gr.Textbox(label="实时英语翻译"),
    title="德语 → 英语 实时翻译器",
    description="输入德语句子，模型会实时生成英语翻译（基于 Seq2Seq + Attention，训练 10 epoch）",
    examples=[
        ["Ein Mann in einem blauen Hemd steht vor einem Gebäude."],
        ["Ein Roboter tanzt mit einem Menschen auf der Bühne."],
        ["Zwei junge Frauen laufen lachend durch den Park."]
    ],
    flagging_mode="never"   # ← 这里改成这个，新版标准参数
)

# 启动 + 开启公网分享（关键改动在这里！）
demo.launch(share=True)