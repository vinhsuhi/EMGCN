from input.dataset import Dataset
from time import time
from algorithms import *
from evaluation.metrics import get_statistics
import utils.graph_utils as graph_utils
import random
import numpy as np
import torch
import argparse
import os
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--dataset_name', default="zh_en")
    parser.add_argument('--source_dataset', default="data/networkx/zh_enDI/zh/graphsage/")
    parser.add_argument('--target_dataset', default="data/networkx/zh_enDI/en/offline/graphsage/")
    parser.add_argument('--groundtruth',    default="data/networkx/zh_enDI/dictionaries/groundtruth")
    parser.add_argument('--seed',           default=123,    type=int)
    subparsers = parser.add_subparsers(dest="algorithm", help='Choose 1 of the algorithm from: EMGCN')
    
    # EMGCN
    parser_EMGCN = subparsers.add_parser("EMGCN", help="EMGCN algorithm")
    
    # neverchange args
    parser_EMGCN.add_argument('--lr', default=0.01, type=float)
    parser_EMGCN.add_argument('--act', type=str, default='tanh')
    parser_EMGCN.add_argument('--log', action="store_false", help="Just to print loss")
    parser_EMGCN.add_argument('--cuda',                action="store_true")
    parser_EMGCN.add_argument('--sparse', action="store_true")
    parser_EMGCN.add_argument('--direct_adj', action="store_true")
    parser_EMGCN.add_argument('--num_GCN_blocks', type=int, default=2)
    parser_EMGCN.add_argument('--embedding_dim',       default=200,         type=int)
    parser_EMGCN.add_argument('--emb_epochs',    default=200,        type=int)

    # often change
    parser_EMGCN.add_argument('--refinement_epochs', default=10, type=int)
    parser_EMGCN.add_argument('--threshold_refine', type=float, default=0.8, help="The threshold value to get stable candidates")
    parser_EMGCN.add_argument('--point', type=float, default=1.1)
    parser_EMGCN.add_argument('--rel', type=float, default=1)
    parser_EMGCN.add_argument('--att', type=float, default=0.5)
    parser_EMGCN.add_argument('--attval', type=float, default=0.5) # what is this
    # often change
    parser_EMGCN.add_argument('--num_each_refine', type=int, default=100)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    start_time = time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source_dataset = Dataset(args.source_dataset, args.dataset_name)
    target_dataset = Dataset(args.target_dataset)
    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')

    algorithm = args.algorithm

    if algorithm == "EMGCN":
        model = EMGCN(source_dataset, target_dataset, args)
    else:
        raise Exception("Unsupported algorithm")

    S = model.align()

    for i in range(2):
        if i == 1: 
            print("right to left...")
        else:
            print("left to right...")
        accs, MAP, AUC = get_statistics(S, groundtruth, get_all_metric=True)
        for key, value in accs.items():
            print("Hit_{}: {:.4f}".format(key, value))
        print("MAP: {:.4f}".format(MAP))
        print("AUC: {:.4f}".format(AUC))
        print("Full_time: {:.4f}".format(time() - start_time))

        S = S.T
        groundtruth = {v:k for k, v in groundtruth.items()}

