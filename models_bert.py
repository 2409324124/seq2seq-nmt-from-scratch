import torch
import torch.nn as nn
import math

class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, vocab_size, d_model=256, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=2)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, d_model)

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModel(nn.Module):
    """
    The bare Bert Model transformer outputting raw hidden-states without any specific head on top.
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, d_model=d_model, dropout=dropout)
        
        # Original BERT uses GELU, standard PyTorch TransformerEncoderLayer supports it
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooler layer for the CLS token
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        attention_mask: 1 for tokens that are NOT MASKED, 0 for MASKED tokens (pad).
        However, nn.TransformerEncoder expects src_key_padding_mask where True means IGNORE.
        So we will pass it appropriately.
        """
        embedding_output = getattr(self, "embeddings")(input_ids, token_type_ids=token_type_ids)
        
        # nn.TransformerEncoder expects padding mask where True means ignore.
        # If attention_mask is passed, we assume it's like HuggingFace: 1 for valid, 0 for pad
        # So src_key_padding_mask = (attention_mask == 0)
        src_key_padding_mask = (attention_mask == 0) if attention_mask else None
        
        sequence_output = self.encoder(embedding_output, src_key_padding_mask=src_key_padding_mask)
        
        # Pooler gets the first token CLS/SOS hidden state
        pooled_output = self.pooler(sequence_output[:, 0])
        
        return sequence_output, pooled_output

class BertPreTrainingHeads(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # MLM Head
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12)
        )
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias
        
        # NSP Head
        self.seq_relationship = nn.Linear(d_model, 2)

    def forward(self, sequence_output, pooled_output):
        # MLM prediction
        x = self.transform(sequence_output)
        prediction_scores = self.decoder(x)
        
        # NSP prediction
        seq_relationship_score = self.seq_relationship(pooled_output)
        
        return prediction_scores, seq_relationship_score

class BertForPreTraining(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.bert = BertModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.cls = BertPreTrainingHeads(d_model, vocab_size)
        
        # Tie weights between embedding and MLM decoder
        self.cls.decoder.weight = self.bert.embeddings.word_embeddings.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = getattr(self, "bert")(
            input_ids, token_type_ids, attention_mask
        )
        prediction_scores, seq_relationship_score = getattr(self, "cls")(
            sequence_output, pooled_output
        )
        return prediction_scores, seq_relationship_score
