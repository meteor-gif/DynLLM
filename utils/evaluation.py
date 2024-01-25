import utils.metrics as metrics
import multiprocessing
import heapq
import torch
import pickle
import numpy as np
from time import time
import json

with open('data/overall_data_statistics.json', 'r') as f:
    overall_data = json.load(f)
USER_NUM = overall_data['users_num']
ITEM_NUM = overall_data['items_num']
candidate_items = np.array(overall_data['candidate_items'])
Ks = overall_data['Ks']

cores = multiprocessing.cpu_count() // 5

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i - USER_NUM]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i == user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i == user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i - USER_NUM]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i == user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio, mrr = [], [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, 1))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))
        mrr.append(metrics.mrr_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'mrr': np.array(mrr)}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    user = x[1]
    item = x [-1]
    r, auc = ranklist_by_heapq(item, set(candidate_items), rating, Ks)

    return get_performance(item, r, Ks)


def test_users_evaluation_metrics(memory, users_to_test, items_to_test):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'mrr': np.zeros(len(Ks)), 'auc': 0.}
    pool = multiprocessing.Pool(cores)

    users_to_test = users_to_test.tolist()
    items_to_test = items_to_test.tolist()

    count = 0
    n_test_users = len(users_to_test)
    
    items_batch_idx = range(USER_NUM, USER_NUM + ITEM_NUM)
    users_batch_embeddings = memory.get_nodes_memory(users_to_test)
    items_batch_embeddings = memory.get_nodes_memory(items_batch_idx)
    rate_batch = torch.matmul(users_batch_embeddings, torch.transpose(items_batch_embeddings, 0, 1))

    rate_batch = rate_batch.detach().cpu().numpy()
    user_batch_rating_uid = zip(rate_batch, users_to_test, items_to_test)

    batch_result = pool.map(test_one_user, user_batch_rating_uid)
    count += len(batch_result)

    for re in batch_result:
        result['precision'] += re['precision'] / n_test_users
        result['recall'] += re['recall'] / n_test_users
        result['ndcg'] += re['ndcg'] / n_test_users
        result['hit_ratio'] += re['hit_ratio'] / n_test_users
        result['mrr'] += re['mrr'] / n_test_users
        # result['auc'] += re['auc'] / n_test_users

    assert count == n_test_users
    pool.close()
    return result

