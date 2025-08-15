import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

class FlashAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, context_length, drop_out, n_heads, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = drop_out
        self.out_proj = nn.Linear(d_out, d_out)
        self.context_length = context_length

    def forward(self, x):
        b, n_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_values(x)

        keys = keys.view(b, n_tokens, self.n_heads, self.head_dim).transpose(1,2)
        queries = queries.view(b, n_tokens, self.n_heads, self.head_dim).transpose(1,2)
        values = values.view(b, n_tokens, self.n_heads, self.head_dim).transpose(1,2)

        #Context Manager to select Attention type
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                          SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.MATH]):
            context_vec = F.scaled_dot_product_attention(queries, 
                                                         keys, 
                                                         values,
                                                         dropout_p=self.dropout if self.training else 0.0,
                                                         is_causal=True)
        
        
        context_vec = context_vec.transpose(1,2).contiguous().view(b, n_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
    
print("Flash Attention V2 class success!")