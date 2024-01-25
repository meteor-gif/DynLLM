from time import time
import torch
from torch import nn
class UsersHistoryUpdater(torch.nn.Module):
    def __init__(self, nodes_dim, time_dim):
        super(UsersHistoryUpdater, self).__init__()

        self.history_updater = nn.GRUCell(input_size=nodes_dim + time_dim, hidden_size=nodes_dim)

    def forward(self, users_embeddings_history, items_embeddings_minus, time_diffs_tensor):
        input_edges_features = torch.cat([items_embeddings_minus, time_diffs_tensor], dim=1)
        users_embeddings = self.history_updater(input_edges_features, users_embeddings_history)

        return users_embeddings

class ItemsHistoryUpdater(torch.nn.Module):
    def __init__(self, nodes_dim):
        super(ItemsHistoryUpdater, self).__init__()

        self.history_updater = nn.GRUCell(input_size=nodes_dim, hidden_size=nodes_dim // 2)

    def forward(self, items_embeddings_history, users_embeddings_minus):
        items_embeddings = self.history_updater(users_embeddings_minus, items_embeddings_history)

        return items_embeddings


