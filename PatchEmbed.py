import torch
from torch import nn
from einops.layers.torch import Rearrange

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.get_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.lin_proj = nn.Linear(3*patch_size**2, embed_dim) # linear projection of patches
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True) # learnable CLS token
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, image):
        # image = (batch_size, 3, img_size[0], img_size[1])
        # patches = (batch_size, num_patches, 3*patch_size**2)
        patches = self.get_patches(image)

        # randomly shuffle patches
        idx = torch.randperm(patches.shape[1])
        patches = patches[:, idx, :]

        # (batch_size, num_patches, embed_dim)
        proj = self.lin_proj(patches) 
        
        # account for batch size -> (batch_size, 1, embed_dim)
        cls_tokens = self.cls_token.expand(proj.shape[0], -1, -1) 
        
        # (batch_size, num_patches + 1, embed_dim)
        embeddings = torch.cat((cls_tokens, proj), axis=1) 
        embeddings = self.dropout(embeddings)
        return embeddings