import torch
import numpy as np
from utils import normalize_string, tokenize_de, prepare_data
from models_transformer import TransformerModel

# ------------------- 参数 -------------------
d_model = 256
nhead = 8
num_layers = 3
dim_feedforward = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载词表
input_lang, output_lang, _ = prepare_data(max_length=25, min_freq=2)

# 加载 Transformer 模型
model = TransformerModel(
    input_lang.n_words, 
    output_lang.n_words, 
    d_model, nhead, dim_feedforward, num_layers
).to(device)

# 尝试加载最佳权重
try:
    model.load_state_dict(torch.load("transformer_best.pt", map_location=device, weights_only=True))
    print("Transformer 最佳权重加载成功 ✓")
except FileNotFoundError:
    print("警告：未找到 transformer_best.pt，将使用随机初始化权重（仅供代码测试）")

model.eval()

def translate_transformer(sentence, max_len=50):
    sentence = normalize_string(sentence)
    tokens = tokenize_de(sentence)
    
    # 注意：Transformer 通常不需要反转源序列，直接使用即可
    indices = [input_lang.word2index.get(w, 2) for w in tokens]
    src_tensor = torch.tensor([input_lang.word2index["<SOS>"]] + indices + [input_lang.word2index["<EOS>"]]).unsqueeze(0).to(device)
    
    # 填充掩码
    src_padding_mask = (src_tensor == 2)
    
    memory = model.encode(src_tensor, src_padding_mask=src_padding_mask)
    
    tgt_indices = [output_lang.word2index["<SOS>"]]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
        
        # 生成掩码
        tgt_mask = model.generate_square_subsequent_mask(len(tgt_indices)).to(device)
        
        output = model.decode(tgt_tensor, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
        output = model.decoder_out(output)
        
        # 取最后一个词的预测
        next_word_idx = output[0, -1, :].argmax().item()
        
        if next_word_idx == output_lang.word2index["<EOS>"]:
            break
            
        tgt_indices.append(next_word_idx)
    
    translated_tokens = [output_lang.index2word[idx] for idx in tgt_indices[1:]]
    return " ".join(translated_tokens)

# ------------------- 交互模式 -------------------
if __name__ == "__main__":
    print("\n[Transformer 翻译模式] 输入德语句子（输入 q 退出）：")
    while True:
        sent = input("> ")
        if sent.lower() == 'q':
            break
        if sent.strip() == '':
            continue

        try:
            result = translate_transformer(sent)
            print("翻译结果:", result)
        except Exception as e:
            print("翻译出错:", str(e))
