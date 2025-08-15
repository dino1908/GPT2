import torch
from torch import nn

#Define a Multi head attention class

class MultiAttn(nn.Module):
    def __init__(self, d_in, d_out, context_length, drop_out, n_heads, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(drop_out)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, n_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_values(x)

        keys = keys.view(b, n_tokens, self.n_heads, self.head_dim)
        queries = queries.view(b, n_tokens, self.n_heads, self.head_dim)
        values = values.view(b, n_tokens, self.n_heads, self.head_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_score = queries@keys.transpose(2,3)
        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]
        attn_score.masked_fill(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_score/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights@values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b,n_tokens,self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec

#print("Mult-Head Attention class success!")