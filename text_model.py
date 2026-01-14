# test_models.py
import torch
from models import EncoderRNN, BahdanauAttention, AttnDecoderRNN

# 参数设置（和你的数据匹配）
input_size = 17397   # 德语词汇量
output_size = 9600   # 英语词汇量
hidden_size = 256    # 常用大小，4060 完全跑得动
batch_size = 4       # 小 batch 测试
max_src_len = 20     # 随便设

# 初始化模型
encoder = EncoderRNN(input_size, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_size)

print("模型初始化成功！")

# 模拟输入（随机假数据）
src_tensor = torch.randint(3, input_size, (batch_size, max_src_len))  # 从3开始，避免 SOS/EOS/PAD
tgt_tensor = torch.randint(3, output_size, (batch_size, 1))           # decoder 第一步输入一个词

# Encoder 前向
encoder_outputs, encoder_hidden = encoder(src_tensor)

print("Encoder 输出 shape:", encoder_outputs.shape)  # 预期: (batch, src_len, hidden*2)
print("Encoder 最后 hidden shape:", encoder_hidden.shape)  # 预期: (1, batch, hidden)

# Decoder 前向（单步测试）
# 注意：decoder 的 hidden 需要从 encoder 的 hidden 取（这里简化用 encoder 的）
decoder_input = tgt_tensor  # 第一步输入
decoder_hidden = encoder_hidden  # 初始 hidden

output, new_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)

print("Decoder 输出 shape:", output.shape)          # 预期: (batch, output_size)
print("Attention weights shape:", attn_weights.shape)  # 预期: (batch, src_len)
print("测试完成！模型前向传播正常。")