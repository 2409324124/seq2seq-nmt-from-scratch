import torch
from models_transformer import TransformerModel

def test_shapes():
    ntoken_src = 100
    ntoken_tgt = 120
    d_model = 64
    nhead = 4
    nhid = 128
    nlayers = 2
    
    model = TransformerModel(ntoken_src, ntoken_tgt, d_model, nhead, nhid, nlayers)
    
    batch_size = 8
    src_len = 10
    tgt_len = 12
    
    src = torch.randint(0, ntoken_src, (batch_size, src_len))
    tgt = torch.randint(0, ntoken_tgt, (batch_size, tgt_len))
    
    # Test forward pass
    output = model(src, tgt)
    print(f"Forward output shape: {output.shape}")
    assert output.shape == (batch_size, tgt_len, ntoken_tgt)
    
    # Test masking
    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
    output_masked = model(src, tgt, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
    print(f"Masked output shape: {output_masked.shape}")
    assert output_masked.shape == (batch_size, tgt_len, ntoken_tgt)
    
    print("Shape checks passed!")

if __name__ == "__main__":
    test_shapes()
