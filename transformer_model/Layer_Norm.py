import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x-mean)/torch.sqrt(var+self.eps)
        #print(f"x: {torch.get_device(x)} | x_norm: {torch.get_device(x_norm)}| scale: {torch.get_device(self.scale)}|shift: {torch.get_device(self.shift)}")
        return self.scale*x_norm + self.shift
    
#print("Layer Norm class success!")