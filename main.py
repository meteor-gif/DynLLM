import math
import logging
import time
import random
import sys
import argparse
from io import BytesIO
import json
import time

import torch
import pandas as pd
import numpy as np
#import numba
from sklearn.preprocessing import scale

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from utils.tools import *
from modules.dynllm import DynLLM

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2020)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for DynLLM experiments on Dynamic Recommendation')
parser.add_argument('--batch_size', type=int, default=2048, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix')
parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--prune_k', type=int, default=80, help='prune k')
parser.add_argument('--user_num_neighbors', type=int, default=3, help='number of user neighbors')
parser.add_argument('--item_num_neighbors', type=int, default=3, help='number of heads used in attention layer')
parser.add_argument('--num_epoches', type=int, default=200, help='number of epochs')
parser.add_argument('--num_layers', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--patience', type=int, default=10, help='Dimensions of the users embedding')
parser.add_argument('--nodes_dim', type=int, default=128, help='Dimensions of the users embedding')
parser.add_argument('--time_dim', type=int, default=128, help='Dimensions of the time embedding')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--Ks', nargs='?', default='[10, 20, 30, 50, 80, 100]', help='K value of ndcg/recall @ k')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.batch_size
NUM_EPOCHES = args.num_epoches
NUM_HEADS = args.num_heads
USER_NUM_NEIGHBORS = args.user_num_neighbors
ITEM_NUM_NEIGHBORS = args.item_num_neighbors
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NUM_LAYERS = args.num_layers
LEARNING_RATE = args.lr
NODES_DIM = args.nodes_dim
TIME_DIM = args.time_dim

current_time = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(int(time.time()))))
MODEL_SAVE_PATH = './saved_models/DynLLM_{}_prune_{}_user_{}_item_{}.pth'.format(args.prefix, args.prune_k, USER_NUM_NEIGHBORS, ITEM_NUM_NEIGHBORS)
checkpoint_path = './saved_checkpoints/DynLLM_{}_prune_{}_user_{}_item_{}_'.format(args.prefix, args.prune_k, USER_NUM_NEIGHBORS, ITEM_NUM_NEIGHBORS)

### set up logger
### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/log_{}_prune_{}_user_{}_item_{}_{}.log'.format(args.prefix, args.prune_k, USER_NUM_NEIGHBORS, ITEM_NUM_NEIGHBORS, current_time))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


### Load data and train val test split
users = np.load('data/users.npy')
max_users_idx = max(users)
items = np.load('data/items.npy')
# reindex items
items = items + max_users_idx + 1
max_items_idx = max(items)
with open('data/timestamp.json', 'r', encoding='utf-8') as f:
    timestamps = json.load(f)

timestamps = np.array([time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S")) for stamp in timestamps])
timestamps = timestamps - timestamps[0]

# Embedding from LLMs
users_features = np.load('embedding/base_embedding.npy')
items_static_embeddings = np.load('embedding/item_static_embedding.npy')
edges_idxs = np.load('data/idx.npy')

# timestamps = scale(timestamps + 1)

# items_features = scale(items_features)
validation_index = int(len(timestamps) * 0.70)
test_index = int(len(timestamps) * 0.85)

users_num = max_users_idx + 1
nodes_num = max(max_users_idx, max_items_idx) + 1
items_num = nodes_num - users_num

random.seed(2020)

# users_distinct_all = set(np.unique(users))
# num_users_distinct_all = len(users_distinct_all)


train_users = users[:validation_index]
train_items = items[:validation_index]
train_timestamps = timestamps[:validation_index]
train_edges_idxs = edges_idxs[:validation_index]

# validation and test with all edges
validation_users = users[validation_index: test_index]
validation_items = items[validation_index: test_index]
validation_timestamps = timestamps[validation_index: test_index]
validation_edges_idxs = edges_idxs[validation_index: test_index]


test_users = users[test_index:]
test_items = items[test_index:]
test_timestamps = timestamps[test_index:]
test_edges_idxs = edges_idxs[test_index:]


candidate_items = [int(item) for item in list(set(validation_items) | set(test_items))]
# print('overall_data_statistics {}'.format({'users_num': int(users_num), 'items_num': int(items_num), 'nodes_num': int(nodes_num), 'Ks': eval(args.Ks), 'candidate_items': candidate_items}))
assert items_num == (max_items_idx - max_users_idx)
# with open('data/overall_data_statistics.json', 'w') as f:
#     json.dump({'users_num': int(users_num), 'items_num': int(items_num), 'nodes_num': int(nodes_num), 'Ks': eval(args.Ks), 'candidate_items': candidate_items}, f)


### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
train_nodes_adj_list = [[] for _ in range(nodes_num)]
for user, item, edge_idx, timestamp in zip(train_users, train_items, train_edges_idxs, train_timestamps):
    train_nodes_adj_list[user].append((item, edge_idx, timestamp))
    train_nodes_adj_list[item].append((user, edge_idx, timestamp))
train_nodes_neighbors_finder = NodesNeighborsFinder(train_nodes_adj_list, uniform=UNIFORM)


# full graph with all the data for the test and validation purpose
full_nodes_adj_list = [[] for _ in range(nodes_num)]
for user, item, edge_idx, timestamp in zip(users, items, edges_idxs, timestamps):
    full_nodes_adj_list[user].append((item, edge_idx, timestamp))
    full_nodes_adj_list[item].append((user, edge_idx, timestamp))
full_nodes_neighbors_finder = NodesNeighborsFinder(full_nodes_adj_list, uniform=UNIFORM)


train_rand_sampler = RandEdgeSampler(train_users, train_items, seed=2020)
val_rand_sampler = RandEdgeSampler(users, items, seed=0)
test_rand_sampler = RandEdgeSampler(users, items, seed=1)


# if nodes_features is None:
#     nodes_features = np.zeros((max_users_idx + 1, USERS_DIM))
    # users_features = np.eye(max_users_idx + 1)
items_static_embeddings = np.load('./embedding/item_static_embedding.npy')
edges_features = np.zeros((len(edges_idxs), NODES_DIM), dtype=float)
nodes_features = np.zeros((nodes_num, NODES_DIM), dtype=float)

mean_time_shift_users, std_time_shift_users, mean_time_shift_items, std_time_shift_items = compute_time_statistics(users, items, timestamps)
## Model initialize
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
dynllm = DynLLM(train_nodes_neighbors_finder, users_num, items_num, edges_features, nodes_features,
                            mean_time_shift_users, std_time_shift_users, args.prune_k,
                            device=device, nodes_dim=NODES_DIM, time_dim=TIME_DIM, llm_dim=1536, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROP_OUT)
dynllm = dynllm.to(device)
optimizer = torch.optim.Adam(dynllm.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()



train_num_instances = len(train_users)
num_batches = math.ceil(train_num_instances / BATCH_SIZE)

logger.info('num of training instances: {}'.format(train_num_instances))
logger.info('num of batches per epoch: {}'.format(num_batches))

# LLM batch
# train_users_llm = []
# for k in range(num_batches):
#     start_idx = k * BATCH_SIZE
#     end_idx = min(train_num_instances - 1, start_idx + BATCH_SIZE)

#     users_idxs_batch, items_idxs_batch = train_users[start_idx: end_idx], train_items[start_idx: end_idx]

early_stopper = EarlyStopMonitor(max_round=args.patience)
for epoch in range(NUM_EPOCHES):
    start_epoch_time = time.time()
    # Training 
    # training use only training graph

    dynllm.memory.__init_memory__()
    dynllm.nodes_neighbors_finder = train_nodes_neighbors_finder
    # acc, ap, f1, auc, m_loss = [], [], [], [], []
    m_loss = []
    result = {'precision': np.zeros(len(eval(args.Ks))), 'recall': np.zeros(len(eval(args.Ks))), 'ndcg': np.zeros(len(eval(args.Ks))),
              'hit_ratio': np.zeros(len(eval(args.Ks)))}
    logger.info('start {} epoch'.format(epoch))
    
    for k in range(num_batches):
         
        start_batch_time = time.time()
        percent = 100 * k / num_batches
        if k % int(0.3 * num_batches) == 0:
            logger.info('progress: {0:10.4f}'.format(percent))

        start_idx = k * BATCH_SIZE
        end_idx = min(train_num_instances - 1, start_idx + BATCH_SIZE)

        users_idxs_batch, items_idxs_batch = train_users[start_idx: end_idx], train_items[start_idx: end_idx]
        timestamps_batch = train_timestamps[start_idx: end_idx]

        crowds_features_batch = np.load('./embedding/train_embedding/crowds_embedding/crowds_embedding_batch_{}.npy'.format(k))[users_idxs_batch, :]
        interests_features_batch = np.load('./embedding/train_embedding/interests_embedding/interests_embedding_batch_{}.npy'.format(k))[users_idxs_batch, :]
        categories_features_batch = np.load('./embedding/train_embedding/categories_embedding/categories_embedding_batch_{}.npy'.format(k))[users_idxs_batch, :]
        brands_features_batch = np.load('./embedding/train_embedding/brands_embedding/brands_embedding_batch_{}.npy'.format(k))[users_idxs_batch, :]

        size = len(users_idxs_batch)
        _, negative_items_idxs_batch = train_rand_sampler.sample(size)
        
        
        # with torch.no_grad():
        #   pos_label = torch.ones(size, dtype=torch.float, device=device)
        #   neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        dynllm = dynllm.train()

        users_embeddings, pos_items_embeddings, neg_items_embeddings, batch_result = dynllm(users_idxs_batch, items_idxs_batch, negative_items_idxs_batch, items_static_embeddings, crowds_features_batch, interests_features_batch, 
                                                                                            categories_features_batch, brands_features_batch, True, timestamps_batch, USER_NUM_NEIGHBORS, ITEM_NUM_NEIGHBORS)

        loss = bpr_loss(users_embeddings, pos_items_embeddings, neg_items_embeddings, args.decay)
        loss.backward()
        optimizer.step()
        dynllm.memory.detach_memory()
        # get training results
        with torch.no_grad():
            dynllm = dynllm.eval()
            m_loss.append(loss.item())
        

    # validation phase use all information
    dynllm.nodes_neighbors_finder = full_nodes_neighbors_finder
    val_precision, val_recall, val_ndcg, val_hit, val_mrr = eval_dynllm_recommendation(dynllm, val_rand_sampler, 
                                                            validation_users, validation_items,
                                                            validation_timestamps, validation_edges_idxs, items_static_embeddings, 'validation', eval(args.Ks),
                                                            USER_NUM_NEIGHBORS, ITEM_NUM_NEIGHBORS, BATCH_SIZE)
    
    epoch_time = time.time() - start_epoch_time

    logger.info('epoch: {} took {:.2f}s'.format(epoch, epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))


    logger.info('val precision@{}: {}, val precision@{}: {}, val precision@{}: {}, val precision@{}: {}, val precision@{}: {}, val precision@{}: {}'.format(eval(args.Ks)[0], val_precision[0], eval(args.Ks)[1], val_precision[1], 
                                                                                                                eval(args.Ks)[2], val_precision[2], eval(args.Ks)[3], val_precision[3], eval(args.Ks)[4], val_precision[4], eval(args.Ks)[-1], val_precision[-1]))
    logger.info('val recall@{}: {}, val recall@{}: {}, val recall@{}: {}, val recall@{}: {}, val recall@{}: {}, val recall@{}: {}'.format(eval(args.Ks)[0], val_recall[0], eval(args.Ks)[1], val_recall[1], 
                                                                                                    eval(args.Ks)[2], val_recall[2], eval(args.Ks)[3], val_recall[3], eval(args.Ks)[4], val_recall[4], eval(args.Ks)[-1], val_recall[-1]))
    logger.info('val ndcg@{}: {}, val ndcg@{}: {}, val ndcg@{}: {},, val ndcg@{}: {}, val ndcg@{}: {}, val ndcg@{}: {}'.format(eval(args.Ks)[0], val_ndcg[0], eval(args.Ks)[1], val_ndcg[1], 
                                                                                            eval(args.Ks)[2], val_ndcg[2], eval(args.Ks)[3], val_ndcg[3], eval(args.Ks)[4], val_ndcg[4], eval(args.Ks)[-1], val_ndcg[-1]))
    logger.info('val hit@{}: {}, val hit@{}: {}, val hit@{}: {},, val hit@{}: {}, val hit@{}: {}, val hit@{}: {}'.format(eval(args.Ks)[0], val_hit[0], eval(args.Ks)[1], val_hit[1], 
                                                                                        eval(args.Ks)[2], val_hit[2], eval(args.Ks)[3], val_hit[3], eval(args.Ks)[4], val_hit[4], eval(args.Ks)[-1], val_hit[-1]))
    logger.info('val mrr@{}: {}, val mrr@{}: {}, val mrr@{}: {}, val mrr@{}: {}, val mrr@{}: {}, val mrr@{}: {}'.format(eval(args.Ks)[0], val_mrr[0], eval(args.Ks)[1], val_mrr[1], 
                                                                                        eval(args.Ks)[2], val_mrr[2], eval(args.Ks)[3], val_mrr[3], eval(args.Ks)[4], val_mrr[4], eval(args.Ks)[-1], val_mrr[-1]))
    



    if early_stopper.early_stop_check(val_recall[0]):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_model_path = checkpoint_path +  '{}.pth'.format(early_stopper.best_epoch)
        dynllm.load_state_dict(torch.load(best_model_path))
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        dynllm.eval()
        break
    else:
        torch.save(dynllm.state_dict(), checkpoint_path + '{}.pth'.format(epoch))


# testing phase use all information
dynllm.nodes_neighbors_finder = full_nodes_neighbors_finder
test_precision, test_recall, test_ndcg, test_hit, test_mrr = eval_dynllm_recommendation(dynllm, val_rand_sampler, 
                                                            validation_users, validation_items,
                                                            validation_timestamps, validation_edges_idxs, items_static_embeddings, 'test', eval(args.Ks),
                                                            USER_NUM_NEIGHBORS, ITEM_NUM_NEIGHBORS, BATCH_SIZE)
# print('Test statistics: -- auc: {}, ap: {}'.format(test_auc, test_ap))
logger.info('test precision@{}: {}, test precision@{}: {}, test precision@{}: {}, test precision@{}: {}, test precision@{}: {}, test precision@{}: {}'.format(eval(args.Ks)[0], test_precision[0], eval(args.Ks)[1], test_precision[1], 
                                                                                                                eval(args.Ks)[2], test_precision[2], eval(args.Ks)[3], test_precision[3], eval(args.Ks)[4], test_precision[4], eval(args.Ks)[-1], test_precision[-1]))
logger.info('test recall@{}: {}, test recall@{}: {}, test recall@{}: {}, test recall@{}: {}, test recall@{}: {}, test recall@{}: {}'.format(eval(args.Ks)[0], test_recall[0], eval(args.Ks)[1], test_recall[1], 
                                                                                                eval(args.Ks)[2], test_recall[2], eval(args.Ks)[3], test_recall[3], eval(args.Ks)[4], test_recall[4], eval(args.Ks)[-1], test_recall[-1]))
logger.info('test ndcg@{}: {}, test ndcg@{}: {}, test ndcg@{}: {}, test ndcg@{}: {}, test ndcg@{}: {}, test ndcg@{}: {}'.format(eval(args.Ks)[0], test_ndcg[0], eval(args.Ks)[1], test_ndcg[1], 
                                                                                        eval(args.Ks)[2], test_ndcg[2], eval(args.Ks)[3], test_ndcg[3], eval(args.Ks)[4], test_ndcg[4], eval(args.Ks)[-1], test_ndcg[-1]))
logger.info('test hit@{}: {}, test hit@{}: {}, test hit@{}: {}, test hit@{}: {}, test hit@{}: {}, test hit@{}: {}'.format(eval(args.Ks)[0], test_hit[0], eval(args.Ks)[1], test_hit[1], 
                                                                                    eval(args.Ks)[2], test_hit[2], eval(args.Ks)[3], test_hit[3], eval(args.Ks)[4], test_hit[4], eval(args.Ks)[-1], test_hit[-1]))
logger.info('test mrr@{}: {}, test mrr@{}: {}, test mrr@{}: {}, test mrr@{}: {}, test mrr@{}: {}, test mrr@{}: {}'.format(eval(args.Ks)[0], test_mrr[0], eval(args.Ks)[1], test_mrr[1], 
                                                                                    eval(args.Ks)[2], test_mrr[2], eval(args.Ks)[3], test_mrr[3], eval(args.Ks)[4], test_mrr[4], eval(args.Ks)[-1], test_mrr[-1]))

logger.info('Saving DynLLM model')
torch.save(dynllm.state_dict(), MODEL_SAVE_PATH)
logger.info('DynLLM models saved')

 




