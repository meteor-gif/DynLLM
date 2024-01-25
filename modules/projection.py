import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

class TimeProjectionEmbedddings(torch.nn.Module):
    def __init__(self,  users_dim):
        super(TimeProjectionEmbedddings, self).__init__()

        class NormalLinear(nn.Linear):
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer_early = NormalLinear(1, users_dim)
        self.embedding_layer_late = NormalLinear(1, users_dim)
        self.transformer_vector = nn.Parameter(torch.Tensor(users_dim))
        self.transformer_vector.data.normal_(0, 1. / math.sqrt(users_dim))

    def compute_time_projection_embeddings(self, nodes_embeddings, nodes_time_diffs):
        projection_alpha = torch.sum(nodes_embeddings * self.transformer_vector, dim=1)
        expectation_time = torch.sqrt(math.pi / (2 * torch.exp(projection_alpha)))
        expectation_time = (expectation_time - torch.mean(expectation_time)) / torch.std(expectation_time)
        time_diffs_expection = nodes_time_diffs - expectation_time
        time_diffs_expection = torch.unsqueeze(time_diffs_expection, 1)
        nodes_embeddings_early = nodes_embeddings * (1 + self.embedding_layer_early(time_diffs_expection))
        nodes_embeddings_late = nodes_embeddings * (1 + self.embedding_layer_late(time_diffs_expection))
        nodes_embeddings_projection = nodes_embeddings_early.masked_fill_((time_diffs_expection >= 0), 0) + nodes_embeddings_late.masked_fill_((time_diffs_expection < 0), 0)
            

        return nodes_embeddings_projection



class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)
