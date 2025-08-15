import torch
from torch import nn

class FlashAttention(nn.Module):
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
        self.context_length = context_length

    def forward(self, x):
        b, n_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_values(x)

        keys = keys.view(b, n_tokens, self.n_heads, self.head_dim).transpose(1,2)
        queries = queries.view(b, n_tokens, self.n_heads, self.head_dim).transpose(1,2)
        values = values.view(b, n_tokens, self.n_heads, self.head_dim).transpose(1,2)

        # Flash Attention: block-wise causal masking
        attn_score = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(n_tokens, n_tokens, device=x.device)).bool()
        attn_score = attn_score.masked_fill(~mask, float('-inf'))

        attn_weights = torch.softmax(attn_score, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = torch.matmul(attn_weights, values)
        context_vec = context_vec.transpose(1,2).contiguous().view(b, n_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec