import torch
from torch import nn
import torch.nn.functional as F

class MOE(nn.Module):
    def __init__(self, num_experts, expert_size, k, tau, bias_param, embed_size, bn, pn):
        super().__init__()
        self.num_experts = num_experts
        self.embsize = embed_size
        self.k = k
        self.tau = tau

        self.register_buffer('bias', torch.zeros(bn*pn, num_experts))
        self.bias_param = bias_param

        self.avg_load = bn * pn * k / num_experts

        # self.router = nn.Linear(embed_size, num_experts)
        self.Wlinear = nn.Linear(self.embsize, self.embsize, bias=False).to('cuda:0')
        self.learnedE = nn.Parameter(torch.randn(self.num_experts, self.embsize)).to('cuda:0')

        self.expert_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, expert_size),
                nn.GELU(),
                nn.Linear(expert_size, embed_size),
                nn.Dropout(p=0.1)
            ) for _ in range(num_experts+1) # 1 shared expert
        ])
    
    def forward(self, x):
        # experts will work on the entire image instead of individual patches/tokens 
        # Use CLS token for router according to Vi-MOE paper

        # Linear Router:
        # routed = self.router(x)

        # if self.training:
        #    probabilities = torch.softmax((routed + torch.randn_like(routed) + self.bias[:len(routed)]), dim=1)
        # else:
        #    probabilities = torch.softmax((routed + self.bias[:len(routed)]), dim=1)

        # Using cosine router from GMoE paper
        bn, pn, ps = x.shape
        patches = x.view(bn*pn, ps)

        proj = F.normalize(self.Wlinear(patches), p=2, dim=-1)
        learnedE = F.normalize(self.learnedE, p=2, dim=-1)
        routed = torch.matmul(proj, learnedE.T)
        
        if self.training:
            probabilities = torch.softmax((routed + torch.randn_like(routed) + self.bias[:len(routed)]) / self.tau, dim=1)
        else:
            probabilities = torch.softmax((routed + self.bias[:len(routed)]) / self.tau, dim=1)

        k_probs, k_idx = torch.topk(probabilities, k=self.k, dim=1)

        # Auxiliary Loss-Free Load Balancing -> adjust bias depending on amount of images given to each expert
        if self.training:
            with torch.no_grad():
                update_bias = torch.tensor([((k_idx == e).sum().item()) for e in range(self.num_experts)])
                self.bias += (self.bias_param * (torch.sign(update_bias - self.avg_load)).to('cuda:0'))

        # For each expert, calculate on group of images that expert was selected for
        result = torch.zeros_like(patches)
        for e in range(self.num_experts):
            e_idx = (k_idx == e)
            batch_idx = e_idx.any(dim=1)

            if batch_idx.any():
                selected_images = patches[batch_idx]
                weights = k_probs[e_idx].unsqueeze(1)
                curr_e = self.expert_list[e]
                result[batch_idx] += weights * curr_e(selected_images)

        result += self.expert_list[-1](patches)
        result = result.view(bn, pn, ps)
        return result