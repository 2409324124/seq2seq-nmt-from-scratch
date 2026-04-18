import torch
import torch.nn as nn

class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model=256, max_position_embeddings=512, dropout=0.1):
        super().__init__()
        # Token embeddings
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=2)
        # Learned positional embeddings like GPT (unlike standard Transformer which uses sine/cosine)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class GPTModel(nn.Module):
    """
    A bare-bones GPT model (Decoder-only Transformer).
    We use PyTorch's TransformerEncoder layer but we will pass a causal causal mask 
    to restrict visibility to past tokens only.
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embeddings = GPTEmbeddings(vocab_size, d_model=d_model, dropout=dropout)
        
        # GPT uses GELU activation natively
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True,
            norm_first=True  # Modern GPT uses pre-LN (LayerNorm before attention)
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask=None):
        """
        Since this is an autoregressive model, we must generate a causal mask.
        """
        device = input_ids.device
        seq_len = input_ids.size(1)
        
        # 1. Causal mask (Lower triangular matrix)
        # Prevents token `i` from seeing tokens `i+1` to `seq_len`
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
        # 2. Key padding mask (ignore pad tokens)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        hidden_states = self.embeddings(input_ids)
        
        # In PyTorch, mask is the casual mask, src_key_padding_mask is the padding mask
        hidden_states = self.decoder(
            hidden_states, 
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=True
        )
        
        hidden_states = self.final_layer_norm(hidden_states)
        
        return hidden_states

class GPTLMHeadModel(nn.Module):
    """
    GPT Model with a Language Modeling head on top (Linear layer to vocab size).
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.gpt = GPTModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout)
        
        # Language Modeling Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (typical in GPT-like models)
        self.lm_head.weight = self.gpt.embeddings.word_embeddings.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask=None):
        hidden_states = getattr(self, "gpt")(input_ids, attention_mask)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits
