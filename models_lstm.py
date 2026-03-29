# models.py - LSTM 版本（替换整个文件）
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    """双向 LSTM Encoder"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # LSTM（双向）
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=True)
        
        # 压缩双向的最后一层 hidden 和 cell 到单向大小
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)  # (batch, seq_len, hidden)
        
        output, (hidden, cell) = self.lstm(embedded)
        
        # 合并双向最后一层
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        cell = torch.tanh(self.fc_cell(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))
        
        hidden = hidden.unsqueeze(0)  # (1, batch, hidden)
        cell = cell.unsqueeze(0)
        
        return output, (hidden, cell)

class BahdanauAttention(nn.Module):
    """Bahdanau 加性注意力（不变）"""
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size * 2, hidden_size)  # encoder 双向
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        query = query.unsqueeze(1)
        energy = torch.tanh(self.Wa(query) + self.Ua(keys))
        scores = self.Va(energy).squeeze(2)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return context, attn_weights

class AttnDecoderLSTM(nn.Module):
    """带 Attention 的 LSTM Decoder"""
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.attention = BahdanauAttention(hidden_size)
        
        # LSTM 输入尺寸：embedded (hidden) + context (hidden*2) = hidden*3
        self.lstm = nn.LSTM(hidden_size * 3, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        
        context, attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        context = context.unsqueeze(1)
        
        lstm_input = torch.cat((embedded, context), dim=2)  # (batch, 1, hidden*3)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.out(output.squeeze(1))
        
        return output, hidden, cell, attn_weights