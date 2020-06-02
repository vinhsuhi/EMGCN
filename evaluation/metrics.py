import numpy as np
from evaluation.matcher import top_k, greedy_match
from scipy.sparse import csr_matrix

#Metrics using for benchmark
#greedy match
#- accuracy
# top-k
# - f1_score
# - map (average_precision_score)
# - auc
# - roc

def get_nn_alignment_matrix(alignment_matrix):
    # Sparse
    # return alignment_matrix that is a sparse matrix, at each row, have one entry with value equals to 1
    # the other values of row are zeros
    row = np.arange(len(alignment_matrix))
    col = [np.argmax(alignment_matrix[i]) for i in range(len(alignment_matrix))]
    val = np.ones(len(alignment_matrix))
    result = csr_matrix((val, (row, col)), shape=alignment_matrix.shape)
    return result

def get_statistics(alignment_matrix, groundtruth, groundtruth_matrix=None, use_greedy_match=False, get_all_metric = False):
    # JUST for crosslin
    source_nodes = list(groundtruth.keys())
    target_nodes = list(groundtruth.values())
    sim = (((alignment_matrix[source_nodes]).T)[target_nodes]).T
    if get_all_metric:
        top_k = (1, 10, 50, 100)
    else:
        top_k = [1]
    accs = {}
    acc = None
    for k in top_k:
        topk = get_topk_index(sim, k)
        count = get_hits_from_topk(topk)
        acc = count / len(groundtruth)
        if get_all_metric:
            accs[k] = acc
    if get_all_metric:
        MAP, AUC = compute_MAP_AUC(sim)
        return accs, MAP, AUC
    return acc


def compute_MAP_AUC(sim):
    MAP = 0
    AUC = 0
    for i in range(len(sim)):
        ele_key = sim[i].argsort()[::-1]
        for j in range(len(ele_key)):
            if ele_key[j] == i:
                ra = j + 1
                MAP += 1/ra
                AUC += (sim.shape[1] - ra) / (sim.shape[1] -1)
                break
    n_nodes = len(sim)
    MAP /= n_nodes
    AUC /= n_nodes
    return MAP, AUC


def get_hits_from_topk(topk):
    count = 0
    for i in range(len(topk)):
        if i in topk[i]:
            count += 1
    return count



def compute_precision_k(top_k_matrix, gt):
    n_matched = 0

    if type(gt) == dict:
        for key, value in gt.items():
            if top_k_matrix[key, value] == 1:
                n_matched += 1
        return n_matched/len(gt)

    gt_candidates = np.argmax(gt, axis = 1)
    for i in range(gt.shape[0]):
        if gt[i][gt_candidates[i]] == 1 and top_k_matrix[i][gt_candidates[i]] == 1:
            n_matched += 1

    n_nodes = (gt==1).sum()
    return n_matched/n_nodes


def get_topk_index(sim, k):
    ind = np.argpartition(sim, -k)[:, -k:]
    return ind

