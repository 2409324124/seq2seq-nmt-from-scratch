import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken_src, ntoken_tgt, d_model, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.encoder_embedding = nn.Embedding(ntoken_src, d_model)
        self.decoder_embedding = nn.Embedding(ntoken_tgt, d_model)
        
        self.transformer = nn.Transformer(
            d_model, nhead, nlayers, nlayers, nhid, dropout, batch_first=True
        )
        
        self.d_model = d_model
        self.decoder_out = nn.Linear(d_model, ntoken_tgt)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_out.bias.data.zero_()
        self.decoder_out.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        if self.src_mask is None or self.src_mask.size(0) != tgt.size(1):
            device = tgt.device
            mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            self.src_mask = mask

        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(
            src, tgt, 
            tgt_mask=self.src_mask, 
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        output = self.decoder_out(output)
        return output

    def encode(self, src, src_padding_mask=None):
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.transformer.encoder(src, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        return self.transformer.decoder(
            tgt, memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
