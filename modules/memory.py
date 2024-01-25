import torch
from torch import nn
import math
from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, num_nodes, nodes_embeddings_dimension, device='cpu'):
    super(Memory, self).__init__()
    self.num_nodes = num_nodes
    self.nodes_embeddings_dimension = nodes_embeddings_dimension
    self.device = device
  
    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.nodes_memory = nn.Parameter(torch.zeros((self.num_nodes, self.nodes_embeddings_dimension)).to(self.device),
                               requires_grad=False)
    self.nodes_memory.data.normal_(0, 1 / math.sqrt(self.nodes_embeddings_dimension))

    self.crowds_memory = nn.Parameter(torch.zeros((self.num_nodes, self.nodes_embeddings_dimension)).to(self.device),
                               requires_grad=False)
    self.crowds_memory.data.normal_(0, 1 / math.sqrt(self.nodes_embeddings_dimension))

    self.interests_memory = nn.Parameter(torch.zeros((self.num_nodes, self.nodes_embeddings_dimension)).to(self.device),
                               requires_grad=False)
    self.interests_memory.data.normal_(0, 1 / math.sqrt(self.nodes_embeddings_dimension))

    self.categories_memory = nn.Parameter(torch.zeros((self.num_nodes, self.nodes_embeddings_dimension)).to(self.device),
                               requires_grad=False)
    self.categories_memory.data.normal_(0, 1 / math.sqrt(self.nodes_embeddings_dimension))

    self.brands_memory = nn.Parameter(torch.zeros((self.num_nodes, self.nodes_embeddings_dimension)).to(self.device),
                               requires_grad=False)
    self.brands_memory.data.normal_(0, 1 / math.sqrt(self.nodes_embeddings_dimension))
    
    self.nodes_last_update = nn.Parameter(torch.zeros(self.num_nodes).to(self.device),
                                    requires_grad=False)
    


  def get_nodes_memory(self, nodes_idxs):
    return self.nodes_memory[nodes_idxs, :]

  def get_profiles_memory(self, users_idxs):
    return self.crowds_memory[users_idxs, :], self.interests_memory[users_idxs, :], self.categories_memory[users_idxs, :], self.brands_memory[users_idxs, :]

  def update_users_memory(self, users_idxs, values1):
    self.nodes_memory[users_idxs, :] = values1
    
  def update_profiles_memory(self, users_idxs,  values2, values3, values4, values5):
    self.crowds_memory[users_idxs, :] = values2
    self.interests_memory[users_idxs, :] = values3
    self.categories_memory[users_idxs, :] = values4 
    self.brands_memory[users_idxs, :] = values5

  def update_items_memory(self, items_idxs, values):
    self.nodes_memory[items_idxs, :] = values

  def detach_memory(self):
    self.nodes_memory.detach_()
    self.crowds_memory.detach()
    self.interests_memory.detach()
    self.categories_memory.detach()
    self.brands_memory.detach()
    self.nodes_last_update.detach_()