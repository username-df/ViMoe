import torch
from torch import nn

# scaled dot-product attention
class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.scale = head_size ** -0.5

    def forward(self, Q, K, V):
        # compute softmax(QK^T / sqrt(dk))V
        # using transpose since .T affects batch dimension
        scaled = (Q @ K.transpose(-2, -1)) * self.scale
        softmax = nn.functional.softmax(scaled, dim=-1)
        return softmax @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert (embed_dim % num_heads == 0), "Embed dimension can not be divided by number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        
        # QKV projections 
        self.Q_proj = nn.Linear(embed_dim, embed_dim)
        self.K_proj = nn.Linear(embed_dim, embed_dim)
        self.V_proj = nn.Linear(embed_dim, embed_dim)
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

        self.HeadsList = nn.ModuleList([
            SelfAttention(self.head_size) for _ in range(num_heads)
        ])

    def forward(self, x):
        batch_size, num_patches, _ = x.shape

        # (batch_size, num_patches, embed_dim)
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)

        # embed_dim -> num_heads * head_size
        Q = Q.view(batch_size, num_patches, self.num_heads, self.head_size)
        K = K.view(batch_size, num_patches, self.num_heads, self.head_size)
        V = V.view(batch_size, num_patches, self.num_heads, self.head_size)

        outputs = []
        for i in range(self.num_heads):
            # (batch_size, num_patches, head_size)
            q = Q[:, :, i, :]
            k = K[:, :, i, :]
            v = V[:, :, i, :]

            output = self.HeadsList[i](q, k, v)
            outputs.append(output)

        attention = torch.cat(outputs, dim=-1)
        attention = self.output_proj(attention)
        attention = self.dropout(attention)
        return attention