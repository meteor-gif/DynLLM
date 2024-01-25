
import torch
import numpy as np
import math

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = torch.nn.Parameter(torch.Tensor(feats))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs):
        scores = torch.mul(node_embs, self.scorer) / self.scorer.norm()
        scores = scores

        vals, topk_indices = torch.topk(scores, self.k, dim=1)   
        tanh = torch.nn.Tanh()
        out = torch.gather(node_embs, dim=1, index=topk_indices) * tanh(torch.gather(scores, dim=1, index=topk_indices))

        #we need to transpose the output
        return out
