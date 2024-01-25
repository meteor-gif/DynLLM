import torch
import numpy as np
import torch.nn.functional as F
import math

def bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings, decay):
    pos_scores = torch.sum(torch.mul(user_embeddings, pos_item_embeddings), dim=1)
    neg_scores = torch.sum(torch.mul(user_embeddings, neg_item_embeddings), dim=1)

    regularizer = 1./(2*(user_embeddings**2).sum()+1e-8) + 1./(2*(pos_item_embeddings**2).sum()+1e-8) + 1./(2*(neg_item_embeddings**2).sum()+1e-8)        
    regularizer = regularizer / (user_embeddings.shape[0])

    loss = - F.logsigmoid(pos_scores - neg_scores + 1e-8).mean()
    # mf_loss = - self.prune_loss(maxi, args.prune_loss_drop_rate)

    emb_loss = decay * regularizer
    # reg_loss = 0.0

    return loss + emb_loss

def eval_dynllm_recommendation(model, negative_edge_sampler, users_idxs, items_idxs, timestamps_idxs, edges_idxs, items_static_embeddings, set_name, Ks, user_num_neighbors, item_num_neighbors, batch_size=2048):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    eval_result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'mrr': np.zeros(len(Ks)), 'auc': 0.}
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(users_idxs)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            start_idxs = k * TEST_BATCH_SIZE
            end_idxs = min(num_test_instance, start_idxs + TEST_BATCH_SIZE)
            users_idxs_batch = users_idxs[start_idxs: end_idxs]
            items_idxs_batch = items_idxs[start_idxs: end_idxs]
            timestamps_batch = timestamps_idxs[start_idxs: end_idxs]
            edges_idxs_batch = edges_idxs[start_idxs: end_idxs]

            crowds_features_batch = np.load('./embedding/{}_embedding/crowds_embedding/crowds_embedding_batch_{}.npy'.format(set_name, k))[users_idxs_batch, :]
            interests_features_batch = np.load('./embedding/{}_embedding/interests_embedding/interests_embedding_batch_{}.npy'.format(set_name, k))[users_idxs_batch, :]
            categories_features_batch = np.load('./embedding/{}_embedding/categories_embedding/categories_embedding_batch_{}.npy'.format(set_name, k))[users_idxs_batch, :]
            brands_features_batch = np.load('./embedding/{}_embedding/brands_embedding/brands_embedding_batch_{}.npy'.format(set_name, k))[users_idxs_batch, :]

            size = len(users_idxs_batch)
            _, negative_items_idxs_batch = negative_edge_sampler.sample(size)

            users_embeddings, pos_items_embeddings, neg_items_embeddings, batch_result = model(users_idxs_batch, items_idxs_batch, negative_items_idxs_batch, items_static_embeddings, crowds_features_batch, interests_features_batch, 
                                                                                            categories_features_batch, brands_features_batch, False, timestamps_batch, user_num_neighbors, item_num_neighbors)

           
            eval_result['precision'] += batch_result['precision'] * size
            eval_result['recall'] += batch_result['recall'] * size
            eval_result['ndcg'] += batch_result['ndcg'] * size
            eval_result['hit_ratio'] += batch_result['hit_ratio'] * size
            eval_result['mrr'] += batch_result['mrr'] * size
            # eval_result['auc'] += batch_result['auc'] * size

    return eval_result['precision'] / num_test_instance, eval_result['recall'] / num_test_instance, eval_result['ndcg'] / num_test_instance, eval_result['hit_ratio'] / num_test_instance, eval_result['mrr'] / num_test_instance


class EarlyStopMonitor(object):
  def __init__(self, max_round=20, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round

class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


class NodesNeighborsFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbors(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst