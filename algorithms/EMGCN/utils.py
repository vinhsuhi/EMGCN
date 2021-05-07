from evaluation.metrics import get_statistics
import numpy as np
import torch
import pickle
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


def init_weight(modules, activation):
    """
    Weight initialization
    :param modules: Iterable of modules
    :param activation: Activation function.
    """
    for m in modules:
        if isinstance(m, nn.Linear):
            if activation is None:
                m.weight.data = init.xavier_uniform_(m.weight.data) #, gain=nn.init.calculate_gain(activation.lower()))
            else:
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation.lower()))
            if m.bias is not None:
                m.bias.data = init.constant_(m.bias.data, 0.0)


def get_act_function(activate_function):
    """
    Get activation function by name
    :param activation_fuction: Name of activation function 
    """
    if activate_function == 'sigmoid':
        activate_function = nn.Sigmoid()
    elif activate_function == 'relu':
        activate_function = nn.ReLU()
    elif activate_function == 'tanh':
        activate_function = nn.Tanh()
    else:
        return None
    return activate_function


def cv_coo_sparse(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def get_acc(source_outputs, target_outputs, alphas=None, att_simi=None, value_simi=None, alpha_att_val=[1, 0.5, 0.5], test_dict=None):
    Sf = np.zeros((len(source_outputs[0]), len(target_outputs[0])))
    alphas = np.array(alphas)
    alphas = alphas / alphas.sum()
    accs = ""
    for i in range(0, len(source_outputs)):
        S = torch.matmul(F.normalize(source_outputs[i]), F.normalize(target_outputs[i]).t())
        S_numpy = S.detach().cpu().numpy()
        if test_dict is not None:
            acc = get_statistics(S_numpy, test_dict)
            accs += "Acc layer {} is: {:.4f}, ".format(i, acc)
        if alphas is not None:
            Sf += alphas[i] * S_numpy
        else:
            Sf += S_numpy
    Sori = Sf + 0
    if att_simi is not None:
        Sf = alpha_att_val[0] * Sf + alpha_att_val[1] * att_simi + alpha_att_val[2] * value_simi
    if test_dict is not None:
        acc = get_statistics(Sf, test_dict)
        accs += "Final acc is: {:.4f}".format(acc)
    return accs, Sf, Sori


def Laplacian_graph(A):
    for i in range(len(A)):
        A[i, i] = 1
    A = torch.FloatTensor(A)
    D_ = torch.diag(torch.sum(A, 0)**(-0.5))
    A_hat = torch.matmul(torch.matmul(D_,A),D_)
    A_hat = A_hat.float()
    return A_hat


def update_Laplacian_graph(old_A, new_edges):
    count_updated = 0
    for edge in new_edges:
        if old_A[edge[0], edge[1]] == 0:
            count_updated += 1
        old_A[edge[0], edge[1]] = 1
        old_A[edge[1], edge[0]] = 1
    new_A_hat, new_A = Laplacian_graph(old_A)
    print("Updated {} edges".format(count_updated))
    return new_A_hat, new_A


def save_embeddings(source_outputs, target_outputs):
    print("Saving embeddings")
    for i in range(len(source_outputs)):
        ele_source = source_outputs[i]
        ele_source = ele_source.detach().cpu().numpy()
        ele_target = target_outputs[i]
        ele_target = ele_target.detach().cpu().numpy()
        np.save("numpy_emb/source_layer{}".format(i), ele_source)
        np.save("numpy_emb/target_layer{}".format(i), ele_target)
    print("Done saving embeddings")


def normalize_numpy(matrix):
    denumorator = matrix ** 2
    denumorator = np.sqrt(denumorator.sum(axis=1)).reshape(-1, 1)
    matrix /= denumorator
    where_matrix_nan = isnan(matrix)
    matrix[where_matrix_nan] = 0
    return matrix

def get_numpy_simi_matrix(source, target):
    # normalize
    source = normalize_numpy(source)
    target = normalize_numpy(target)
    return source.dot(target.T)

def get_similarity_matrices(source_outputs, target_outputs):
    """
    Construct Similarity matrix in each layer
    :params source_outputs: List of embedding at each layer of source graph
    :params target_outputs: List of embedding at each layer of target graph
    """
    list_S = []
    for i in range(len(source_outputs)):
        source_output_i = source_outputs[i]
        target_output_i = target_outputs[i]
        S = torch.mm(F.normalize(source_output_i), F.normalize(target_output_i).t())
        S = S.detach().cpu().numpy()
        list_S.append(S)
    return list_S

def get_cosine(vec1, vec2):
    if vec1.sum() == 0 or vec2.sum() == 0:
        return 0
    vec1 = vec1 / np.sqrt((vec1 ** 2).sum())
    vec2 = vec2 / np.sqrt((vec2 ** 2).sum())
    return (vec1 * vec2).sum()

def get_dict_from_S(S, source_id2idx, target_id2idx):
    target = np.argmax(S, axis=1)
    idx2id_source = {v:k for k, v in source_id2idx.items()}
    idx2id_target = {v:k for k, v in target_id2idx.items()}
    dictt = {idx2id_source[i]: idx2id_target[target[i]] for i in range(S.shape[0])}
    return dictt

def linkpred_loss(embedding, A, i, cuda):
    pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())
    if cuda:
        pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]).cuda())), dim = 1)
    else:
        pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim = 1)
    linkpred_losss = (pred_adj - A) ** 2
    linkpred_losss = linkpred_losss.sum() / A.shape[1]
    return linkpred_losss

def linkpred_loss_multiple_layer(outputs, A_hat, cuda):
    count = 0 
    loss = 0
    for i in range(1, len(outputs)):
        loss += linkpred_loss(outputs[i], A_hat, i, cuda)
        count += 1
    loss = loss / count 
    return loss

