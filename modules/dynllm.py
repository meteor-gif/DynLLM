from time import time
import torch
from torch import nn
import logging
import numpy as np
from modules.neighbors_aggregation import NodesNeighborsEmbeddingsAggregation
from modules.projection import TimeProjectionEmbedddings
from modules.history_updater import UsersHistoryUpdater,  ItemsHistoryUpdater
from modules.memory import Memory
from modules.projection import TimeEncode
from modules.temporal_attention import MergeLayer
from modules.selection_mechanism import TopK
from utils.evaluation import *
import math

class DynLLM(torch.nn.Module):
    def __init__(self, nodes_neighbors_finder, users_num, items_num, edges_features, nodes_features,
                 mean_time_shift_users, std_time_shift_users, prune_k, device,
                 nodes_dim=128, time_dim=128, llm_dim=1536, num_layers=2, num_heads=2, dropout=0.1):
        super(DynLLM, self).__init__()
        
        self.users_num = users_num
        self.items_num = items_num
        self.num_layers = num_layers 
        self.nodes_neighbors_finder = nodes_neighbors_finder
        self.logger = logging.getLogger(__name__)
        self.prune_k = prune_k
        
        self.nodes_dim = nodes_dim
        self.time_dim = time_dim
        self.nodes_features = nodes_features
        self.edges_features = edges_features

        self.mean_time_shift_users = mean_time_shift_users
        self.std_time_shift_users = std_time_shift_users
        
        self.device = device
        self.time_encoder = TimeEncode(expand_dim=time_dim)
        self.memory = Memory(users_num + items_num, nodes_dim, device=device)

        self.crowd_trans = nn.Linear(llm_dim, nodes_dim)
        self.interest_trans = nn.Linear(llm_dim, nodes_dim)
        self.category_trans = nn.Linear(llm_dim, nodes_dim)  
        self.brand_trans = nn.Linear(llm_dim, nodes_dim)  
        self.item_trans = nn.Linear(llm_dim, nodes_dim // 2)  
        nn.init.xavier_uniform_(self.crowd_trans.weight)
        nn.init.xavier_uniform_(self.interest_trans.weight)   
        nn.init.xavier_uniform_(self.category_trans.weight)
        nn.init.xavier_uniform_(self.brand_trans.weight)
        nn.init.xavier_uniform_(self.item_trans.weight)

        self.drop_out = nn.Dropout(dropout)

        self.nodes_neighbors_aggregation = NodesNeighborsEmbeddingsAggregation(self.memory, self.nodes_neighbors_finder, self.nodes_features, self.edges_features, 
                                                                            self.device, self.time_encoder, self.nodes_dim, 
                                                                            self.time_dim, num_layers, num_heads=2, dropout=0.1)

        self.multi_profile_target = nn.MultiheadAttention(embed_dim=self.nodes_dim,
                                                   kdim=self.prune_k,
                                                   vdim=self.prune_k,
                                                   num_heads=num_heads,
                                                   dropout=dropout)

        self.users_history_updater = UsersHistoryUpdater(self.nodes_dim, self.time_dim)
        self.items_history_updater = ItemsHistoryUpdater(self.nodes_dim)
        self.profiles_history_updater = ProfilesHistoryUpdater(self.nodes_dim)
        
        self.users_embeddings_projection = TimeProjectionEmbedddings(self.nodes_dim)
        self.choose_topk = TopK(self.nodes_dim, self.prune_k)

        self.users_merge_layer = nn.Linear(nodes_dim * 3, nodes_dim)
        self.items_merge_layer = nn.Linear(nodes_dim * 3 // 2, nodes_dim // 2)
        self.attn_merge_layer = MergeLayer(nodes_dim, nodes_dim, nodes_dim, nodes_dim)

        self.users_layer_norm = nn.LayerNorm(nodes_dim, elementwise_affine=False)
        self.items_layer_norm = nn.LayerNorm(nodes_dim // 2, elementwise_affine=False)
        self.items_static_layer_norm = nn.LayerNorm(nodes_dim // 2, elementwise_affine=False)
        self.profiles_layer_norm = nn.LayerNorm(nodes_dim, elementwise_affine=False)
        self.projection_layer_norm = nn.LayerNorm(nodes_dim, elementwise_affine=False)


        
    def forward(self, users_idxs_cut, items_idxs_cut, negative_items_idxs_cut, items_static_embeddings, crowds_features_batch, interests_features_batch, categories_features_batch, brands_features_batch, is_train,
                timestamps_cut, user_num_neighbors, item_num_neighbors):
        
        n_samples = len(users_idxs_cut)

        items_static_embeddings = self.items_static_layer_norm(self.drop_out(self.item_trans(torch.from_numpy(items_static_embeddings).float().to(self.device))))
        # memory embeddings
        users_embeddings_cut = self.memory.get_nodes_memory(users_idxs_cut)
        crowds_embeddings_cut, interests_embeddings_cut, categories_embeddings_cut, brands_embeddings_cut = self.memory.get_profiles_memory(users_idxs_cut)

        items_embeddings_cut = self.memory.get_nodes_memory(items_idxs_cut)
        items_dynamic_embeddings_cut = items_embeddings_cut[:, self.nodes_dim // 2:]

        negative_items_embeddings_cut = self.memory.get_nodes_memory(negative_items_idxs_cut)
        negative_items_dynamic_embeddings_cut = negative_items_embeddings_cut[:, self.nodes_dim // 2:]
        
        # memory last update
        users_time_diffs = ((torch.from_numpy(timestamps_cut).float().to(self.device) - self.memory.nodes_last_update[users_idxs_cut]) - self.mean_time_shift_users) / self.std_time_shift_users

        # neighbors aggregation
        users_neighbors = self.nodes_neighbors_aggregation.compute_embeddings(users_idxs_cut, timestamps_cut, self.num_layers, user_num_neighbors)
        items_neighbors = self.nodes_neighbors_aggregation.compute_embeddings(items_idxs_cut, timestamps_cut, 1, item_num_neighbors)
        negative_items_neighbors = self.nodes_neighbors_aggregation.compute_embeddings(negative_items_idxs_cut, timestamps_cut, 1, item_num_neighbors)

        # projection
        users_projection = self.projection_layer_norm(self.users_embeddings_projection.compute_time_projection_embeddings(users_embeddings_cut, users_time_diffs))

        crowds_embeddings_topk = self.choose_topk(crowds_embeddings_cut)
        interests_embeddings_topk = self.choose_topk(interests_embeddings_cut)
        categories_embeddings_topk = self.choose_topk(categories_embeddings_cut)
        brands_embeddings_topk = self.choose_topk(brands_embeddings_cut)

        # embeddings minus
        users_embeddings_aggregation = self.users_merge_layer(torch.cat((users_embeddings_cut, users_neighbors, users_projection), dim=1))

        profiles_embeddings = torch.stack([crowds_embeddings_topk, interests_embeddings_topk, categories_embeddings_topk, brands_embeddings_topk], dim=1)
        users_embeddings_aggregation_unrolled = torch.unsqueeze(users_embeddings_aggregation, dim=1)
        users_embeddings_aggregation_unrolled = users_embeddings_aggregation_unrolled.permute([1, 0, 2])
        profiles_embeddings = profiles_embeddings.permute([1, 0, 2])
        attn_output, _ = self.multi_profile_target(query=users_embeddings_aggregation_unrolled, key=profiles_embeddings, value=profiles_embeddings)
        users_embeddings_minus = self.attn_merge_layer(users_embeddings_aggregation, attn_output.squeeze())

        items_dynamic_embeddings_minus = self.items_merge_layer(torch.cat((items_dynamic_embeddings_cut, items_neighbors), dim=1))
        items_embeddings_minus = torch.cat((items_static_embeddings[items_idxs_cut - self.users_num, :].to(self.device), items_dynamic_embeddings_minus), dim=1) # note

        
        negative_items_dynamic_embeddings_minus = self.items_merge_layer(torch.cat((negative_items_dynamic_embeddings_cut, negative_items_neighbors), dim=1))
        negative_items_embeddings_minus = torch.cat((items_static_embeddings[negative_items_idxs_cut - self.users_num, :].to(self.device), negative_items_dynamic_embeddings_minus), dim=1) # note

        self.memory.update_users_memory(users_idxs_cut, users_embeddings_minus.data.clone())
        self.memory.update_items_memory(items_idxs_cut, items_embeddings_minus.data.clone())

        if is_train:
            result = {}
        else:
            result = test_users_evaluation_metrics(self.memory, users_idxs_cut, items_idxs_cut)
        
        # embeddings update
        crowds_features_batch = self.profiles_layer_norm(self.drop_out(self.crowd_trans(torch.from_numpy(crowds_features_batch).float().to(self.device))))
        interests_features_batch = self.profiles_layer_norm(self.drop_out(self.interest_trans(torch.from_numpy(interests_features_batch).float().to(self.device))))
        categories_features_batch = self.profiles_layer_norm(self.drop_out(self.category_trans(torch.from_numpy(categories_features_batch).float().to(self.device))))
        brands_features_batch = self.profiles_layer_norm(self.drop_out(self.brand_trans(torch.from_numpy(brands_features_batch).float().to(self.device))))
        
        
        users_time_diffs_tensor = torch.squeeze(self.time_encoder(torch.unsqueeze(users_time_diffs, dim=1)), dim=1)

        new_users_embeddings_cut = self.users_layer_norm(self.users_history_updater(users_embeddings_minus, items_embeddings_minus, users_time_diffs_tensor))
        
        new_items_dynamic_embeddings_cut = self.items_layer_norm(self.items_history_updater(items_dynamic_embeddings_cut, users_embeddings_minus))
        
        new_items_embeddings_cut = torch.cat((items_static_embeddings[items_idxs_cut - self.users_num, :].to(self.device), new_items_dynamic_embeddings_cut), dim=1)
    
        # memory
        self.memory.update_users_memory(users_idxs_cut, new_users_embeddings_cut.data.clone())
        self.memory.update_items_memory(items_idxs_cut, new_items_embeddings_cut.data.clone())
        self.memory.update_profiles_memory(users_idxs_cut, crowds_features_batch.data.clone(), interests_features_batch.data.clone(), categories_features_batch.data.clone(), brands_features_batch.data.clone())
        self.memory.nodes_last_update[users_idxs_cut] = torch.from_numpy(timestamps_cut).float().to(self.device).data.clone()
        self.memory.nodes_last_update[items_idxs_cut] = torch.from_numpy(timestamps_cut).float().to(self.device).data.clone()

        
        return users_embeddings_minus, items_embeddings_minus, negative_items_embeddings_minus, result

    