from torch import nn
from utils.AttentionFactory import AttentionFactory
from Feed_Forward import FeedForward
from Layer_Norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg, attention_type="multihead"):
        super().__init__()
        self.attention_type = attention_type
        self.attn = AttentionFactory.get_attention(attention_type)(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            drop_out=cfg["drop_rate"],
            n_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_resid(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut

        return x

#print("Transformer Block class success!")
