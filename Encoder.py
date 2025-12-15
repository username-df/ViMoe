import torch
from torch import nn
from Attention import MultiHeadAttention
from MixtureOfExperts import MOE

class EncoderMLP(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.MSA = MultiHeadAttention(embed_dim, num_heads)

        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )

        self.LN1 = nn.LayerNorm(embed_dim)
        self.LN2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x + self.MSA(self.LN1(x))
        x = x + self.dropout(self.MLP(self.LN2(x)))
        return x
    
class EncoderMOE(nn.Module):
    def __init__(self, num_experts, expert_size, k, tau, bias_param, embed_dim, num_heads, bn, pn):
        super().__init__()
        
        self.MSA = MultiHeadAttention(embed_dim, num_heads)
        self.MOE = MOE(num_experts, expert_size, k, tau, bias_param, embed_dim, bn, pn)

        self.LN1 = nn.LayerNorm(embed_dim)
        self.LN2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.MSA(self.LN1(x))
        x = x + (self.MOE(self.LN2(x)))
        return x