# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 词嵌入层：把词索引转成向量
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # GRU（比LSTM简单点，速度快，效果差不多；可换成LSTM）
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, 
                          dropout=dropout if num_layers > 1 else 0, 
                          batch_first=True, bidirectional=True)
        
        # 因为 bidirectional=True，输出 hidden 是 2*hidden_size
        self.fc = nn.Linear(hidden_size * 2, hidden_size)  # 压缩成单向大小

    def forward(self, input, hidden=None):
        # input: (batch_size, seq_len)
        embedded = self.embedding(input)  # (batch_size, seq_len, hidden_size)
        
        # GRU 输出: output (batch_size, seq_len, hidden*2), hidden (num_layers*2, batch, hidden)
        output, hidden = self.gru(embedded, hidden)
        
        # 如果 bidirectional，合并前后向 hidden
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # (batch, hidden*2)
            hidden = self.fc(hidden).unsqueeze(0)  # (1, batch, hidden)
        
        return output, hidden  # output 用于 attention，hidden 给 decoder

class BahdanauAttention(nn.Module):
    """加性注意力（Bahdanau Attention），经典版"""
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size * 2, hidden_size)  # 因为 encoder 是 bidirectional
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: decoder 当前 hidden (batch, hidden)
        # keys: encoder outputs (batch, seq_len, hidden*2)
        
        # 计算能量分数
        query = query.unsqueeze(1)  # (batch, 1, hidden)
        energy = torch.tanh(self.Wa(query) + self.Ua(keys))  # (batch, seq_len, hidden)
        scores = self.Va(energy).squeeze(2)  # (batch, seq_len)
        
        # softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        # context vector: 加权求和
        context = torch.bmm(attn_weights.unsqueeze(1), keys)  # (batch, 1, hidden*2)
        context = context.squeeze(1)  # (batch, hidden*2)
        
        return context, attn_weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Attention 层（不变）
        self.attention = BahdanauAttention(hidden_size)
        
        # 这里是关键修正！GRU 的输入尺寸 = embedded (hidden) + context (hidden*2) = hidden*3
        self.gru = nn.GRU(hidden_size * 3, hidden_size, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)
        
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        # input: 当前词索引 (batch, 1)
        # hidden: (1, batch, hidden)
        # encoder_outputs: (batch, src_len, hidden*2)
        
        embedded = self.embedding(input)  # (batch, 1, hidden)
        embedded = self.dropout(embedded)
        
        # 计算 attention
        context, attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        context = context.unsqueeze(1)  # (batch, 1, hidden*2)
        
        # GRU 输入: embedded + context
        gru_input = torch.cat((embedded, context), dim=2)  # (batch, 1, hidden*3? wait no: hidden + hidden*2)
        # 注意：因为 context 是 hidden*2，embedded 是 hidden，所以 cat 后是 hidden*3，但 gru 是 hidden*2 -> hidden
        # 上面 GRU 初始化时 input_size=hidden*2 是错的！修正为：
        # 实际 cat(embedded (hidden), context (hidden*2)) = hidden*3
        # 所以 gru 应该设 input_size = hidden_size + hidden_size * 2 = hidden_size * 3
        
        output, hidden = self.gru(gru_input, hidden)
        output = self.out(output.squeeze(1))  # (batch, output_size)
        
        return output, hidden, attn_weights