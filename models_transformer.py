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
        # return Boolean mask: True means 'do not attend'
        return mask == 0

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        if self.src_mask is None or self.src_mask.size(0) != tgt.size(1):
            device = tgt.device
            mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            self.src_mask = mask

        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # 为了获取注意力权重，我们需要手动调用模块或者使用 Hook
        # 这里使用简单的前向传递
        output = self.transformer(
            src, tgt, 
            tgt_mask=self.src_mask, 
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        output = self.decoder_out(output)
        return output

    def get_attention(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        """
        专用方法：用于获取注意力矩阵 (Heatmap)
        由于 nn.Transformer 封装较深，我们通过 Encoder 处理后再对 Decoder 进行单层 Hook 或手动遍历
        """
        self.eval()
        with torch.no_grad():
            src_emb = self.pos_encoder(self.encoder_embedding(src) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
            
            tgt_emb = self.pos_encoder(self.decoder_embedding(tgt) * math.sqrt(self.d_model))
            
            # 提取最后一层 Decoder 的 Cross Attention
            # 用一个简化的方式：手动调用最后一层
            attn_weights = []
            def hook(module, input, output):
                # 这里的 output 是 (tgt, weights)
                if isinstance(output, tuple):
                    attn_weights.append(output[1])

            # 注册临时 Hook 到最后一层解码器的 multihead_attn
            last_decoder_layer = self.transformer.decoder.layers[-1]
            handle = last_decoder_layer.multihead_attn.register_forward_hook(hook)
            
            # 强制 MultiheadAttention 返回权重
            original_need_weights = last_decoder_layer.multihead_attn.need_weights
            last_decoder_layer.multihead_attn.need_weights = True
            
            _ = self.transformer.decoder(tgt_emb, memory, tgt_mask=self.generate_square_subsequent_mask(tgt.size(1)).to(src.device), 
                                        memory_key_padding_mask=src_padding_mask)
            
            handle.remove()
            last_decoder_layer.multihead_attn.need_weights = original_need_weights
            
            return attn_weights[0] if attn_weights else None

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
