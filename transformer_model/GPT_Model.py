import torch
from torch import nn
from transformer_model.Transformer_Block import TransformerBlock
from Layer_Norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #self.device = device
        self.tok_embed = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embed = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        b, seq_len = in_idx.shape
        tok_emb = self.tok_embed(in_idx)
        pos_emb = self.pos_embed(torch.arange(seq_len, device=in_idx.device))
        x = tok_emb + pos_emb
        x = x

        x = self.drop_emb(x)
        x = self.trf_block(x)
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits
    
#print("GPT class success!")