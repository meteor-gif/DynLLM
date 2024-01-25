import torch
from torch import nn
import numpy as np
from modules.temporal_attention import TemporalAttentionLayer

class NeighborsEmbeddingsAggregation(torch.nn.Module):
    def __init__(self, memory, neighbors_finder, nodes_features, edges_features, device, time_encoder,
                nodes_dim, time_dim, num_layers, num_heads=2, dropout=0.1):
        super(NeighborsEmbeddingsAggregation, self).__init__()

        self.memory = memory
        self.neighbors_finder = neighbors_finder
        self.nodes_features = nodes_features
        self.device = device
        self.time_encoder = time_encoder

        self.nodes_dim = nodes_dim
        self.time_dim = time_dim
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(edges_features.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)


        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            nodes_dim=nodes_dim,
            time_dim=time_dim,
            output_dimension=nodes_dim,
            num_heads=num_heads,
            dropout=dropout)
            for _ in range(num_layers)])

    def compute_embeddings(self, nodes_idxs, timestamps, num_layers, num_neighbors):

        pass

    def neighbors_aggregation(self, num_layers, source_nodes_features, source_nodes_time_embeddings,
                neighbor_embeddings,
                edges_time_embeddings, edges_features, mask):
        attention_model = self.attention_models[num_layers - 1]

        source_embedding, _ = attention_model(source_nodes_features,
                                                source_nodes_time_embeddings,
                                                neighbor_embeddings,
                                                edges_time_embeddings,
                                                edges_features,
                                                mask)

        return source_embedding




class NodesNeighborsEmbeddingsAggregation(NeighborsEmbeddingsAggregation):


    def compute_embeddings(self, nodes_idxs, timestamps, num_layers, num_neighbors):

        assert (num_layers >= 0)

        
        timestamps_tensor = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim = 1)

        # source nodes time span = t - t = 0
        nodes_timestamps_embeddings = self.time_encoder(torch.zeros_like(timestamps_tensor))

        nodes_features = torch.from_numpy(self.nodes_features[nodes_idxs, : self.nodes_dim]).float().to(self.device) + self.memory.get_nodes_memory(nodes_idxs)[:, : self.nodes_dim]

        if num_layers == 0:
            return nodes_features
        else:
            nodes_conv_features = self.compute_embeddings(nodes_idxs, 
                                                                timestamps, 
                                                                num_layers=num_layers - 1, 
                                                                num_neighbors=num_neighbors)
            # print(nodes_idxs.shape)
            # print(timestamps.shape)
            neighbors_idxs, neighbors_edges_idxs, neighbors_edges_timestamps = self.neighbors_finder.get_temporal_neighbors(nodes_idxs,
                                                                                                                    timestamps,
                                                                                                                    num_neighbors)

            neighbors_idxs_tensor = torch.from_numpy(neighbors_idxs).long().to(self.device)

            neighbors_edges_idxs_tensor = torch.from_numpy(neighbors_edges_idxs).long().to(self.device)

            neighbors_edges_time_deltas = timestamps[:, np.newaxis] - neighbors_edges_timestamps

            neighbors_edges_time_deltas_tensor = torch.from_numpy(neighbors_edges_time_deltas).float().to(self.device)

            neighbors_idxs = neighbors_idxs.flatten()
            neighbors_embeddings = self.compute_embeddings(neighbors_idxs,
                                                        np.repeat(timestamps, num_neighbors),
                                                        num_layers=num_layers - 1,
                                                        num_neighbors=num_neighbors)

            effective_num_neighbors = num_neighbors if num_neighbors > 0 else 1
            neighbors_embeddings = neighbors_embeddings.view(len(nodes_idxs), effective_num_neighbors, -1)
            neighbors_edges_time_deltas_embeddings = self.time_encoder(neighbors_edges_time_deltas_tensor)

            #   neighbors_items_features = torch.cat((torch.from_numpy(self.items_static_features[neighbors_items_idxs, :]).float().to(self.device), self.memory.get_items_memory(neighbors_items_idxs)), dim=-1)
            neighbors_edges_features = self.edge_raw_embed(neighbors_edges_idxs_tensor)
            mask = neighbors_idxs_tensor == 0

            nodes_embeddings = self.neighbors_aggregation(num_layers, nodes_conv_features,
                                                nodes_timestamps_embeddings,
                                                neighbors_embeddings,
                                                neighbors_edges_time_deltas_embeddings,
                                                neighbors_edges_features,
                                                mask)

            return nodes_embeddings


