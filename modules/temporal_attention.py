import torch
from torch import nn
import numpy as np
import math

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class TemporalAttentionLayer(torch.nn.Module):
  """
  Temporal attention layer. Return the temporal embedding of a node given the node itself,
   its neighbors and the edge timestamps.
  """

  def __init__(self, nodes_dim, time_dim,
               output_dimension, num_heads=2,
               dropout=0.1):
    super(TemporalAttentionLayer, self).__init__()

    self.num_heads = num_heads

    self.nodes_dim = nodes_dim
    self.time_dim = time_dim

    self.query_dim = nodes_dim + time_dim
    self.key_dim = nodes_dim + time_dim + nodes_dim

    self.merger = MergeLayer(self.query_dim, nodes_dim, nodes_dim, output_dimension)

    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                   kdim=self.key_dim,
                                                   vdim=self.key_dim,
                                                   num_heads=num_heads,
                                                   dropout=dropout)

  def forward(self, source_nodes_features, source_nodes_time_features, neighbors_features,
              neighbors_time_features, neighbors_edges_features, neighbors_padding_mask):
    """
    "Temporal attention model
    :param src_node_features: float Tensor of shape [batch_size, n_node_features]
    :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
    :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
    :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors,
    time_dim]
    :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
    :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
    :return:
    attn_output: float Tensor of shape [1, batch_size, n_node_features]
    attn_output_weights: [batch_size, 1, n_neighbors]
    """

    source_nodes_features_unrolled = torch.unsqueeze(source_nodes_features, dim=1)

    query = torch.cat([source_nodes_features_unrolled, source_nodes_time_features], dim=2)
    key = torch.cat([neighbors_features, neighbors_edges_features, neighbors_time_features], dim=2)

    # print(neighbors_features.shape, edge_features.shape, neighbors_time_features.shape)
    # Reshape tensors so to expected shape by multi head attention
    query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
    key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]

    # Compute mask of which source nodes have no valid neighbors
    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
    # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
    # force the attention to just 'attend' on this neighbor (which has the same features as all
    # the others since they are fake neighbors) and will produce an equivalent result to the
    # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False

    # print(query.shape, key.shape)

    attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
                                                              key_padding_mask=neighbors_padding_mask)

    # mask = torch.unsqueeze(neighbors_padding_mask, dim=2)  # mask [B, N, 1]
    # mask = mask.permute([0, 2, 1])
    # attn_output, attn_output_weights = self.multi_head_target(q=query, k=key, v=key,
    #                                                           mask=mask)

    attn_output = attn_output.squeeze()
    attn_output_weights = attn_output_weights.squeeze()

    # Source nodes with no neighbors have an all zero attention output. The attention output is
    # then added or concatenated to the original source node features and then fed into an MLP.
    # This means that an all zero vector is not used.
    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

    # Skip connection with temporal attention over neighborhood and the features of the node itself
    attn_output = self.merger(attn_output, source_nodes_features)

    return attn_output, attn_output_weights