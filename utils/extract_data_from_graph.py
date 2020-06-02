from input.dataset import Dataset, SynDataset
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
import json
from networkx.readwrite import json_graph
from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="/home/bigdata/thomas/dataspace/graph/flicr_myspace/flickr/graphsage/")
    parser.add_argument('--target_dataset', default="/home/bigdata/thomas/dataspace/graph/flicr_myspace/myspace/graphsage/")
    parser.add_argument('--groundtruth',    default="/home/bigdata/thomas/dataspace/graph/flicr_myspace/dictionaries/groundtruth")
    parser.add_argument('--seed',           default=123,    type=int)
    
    return parser.parse_args()

def filter_nodes(G, nodes_not_fil, thresh_hold_degree):
    still_degree_1 = True 
    num_removed = 0
    iter = 0
    from copy import deepcopy
    while still_degree_1:
        if iter == 1:
            break
        nodes = G.nodes()
        G_degree = deepcopy(G.degree())
        print("Number of nodes: {}".format(len(nodes)))
        for node in nodes:
            if node not in nodes_not_fil:
                if G_degree[node] == 1:
                    G.remove_node(node)
                    num_removed += 1
                    if num_removed % 500 == 0:
                        print("Removed nodes: {}, num removed: {}".format(node, num_removed))
        still_degree_1 = False
        for key in G.degree():
            if G.degree()[key] == 1 and key not in nodes_not_fil:
                still_degree_1 = True
                break
        print("Number of removed node in this iteration: {}".format(num_removed))
        print("Number of nodes left: {}".format(len(G.nodes())))
        iter += 1
    return G
    
def _save_graph(G, output_dir):
    with open(output_dir, "w+") as file:
        res = json_graph.node_link_data(G)
        file.write(json.dumps(res))


if __name__ == "__main__":
    args = parse_args()
    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
    source_groundtruth_nodes = list(groundtruth.keys())
    target_groundtruth_nodes = list(groundtruth.values()) 
    source_idx2id = {v:k for k, v in source_dataset.id2idx.items()}
    target_idx2id = {v:k for k, v in target_dataset.id2idx.items()}
    source_gt_id = [source_idx2id[node] for node in source_groundtruth_nodes]
    target_gt_id = [target_idx2id[node] for node in target_groundtruth_nodes]
    source_care_deg = source_dataset.get_nodes_degrees()[source_groundtruth_nodes]
    target_care_deg = target_dataset.get_nodes_degrees()[target_groundtruth_nodes]

    print("Number of nodes in groundtruth: {}".format(len(groundtruth)))
    print("Num source nodes: {}".format(len(source_dataset.G.nodes())))
    print("Num target nodes: {}".format(len(target_dataset.G.nodes())))
    # print("Source care deg: {}".format(source_care_deg))
    # print("Source care deg: {}".format(target_care_deg))
    G1 = filter_nodes(source_dataset.G, source_gt_id, 1)
    G2 = filter_nodes(target_dataset.G, target_gt_id, 1)
    # save G
    source_node_left = G1.nodes()
    target_node_left = G2.nodes()
    # save id2idx
    source_id2idx = {source_node_left[idx]:idx for idx in range(len(source_node_left))}
    target_id2idx = {target_node_left[idx]:idx for idx in range(len(target_node_left))}
    if not os.path.exists("flickr_myspace/dictionaries"):
        os.makedirs("flickr_myspace")
        os.makedirs("flickr_myspace/flickr")
        os.makedirs("flickr_myspace/flickr/graphsage")
        os.makedirs("flickr_myspace/myspace/graphsage")
        os.makedirs("flickr_myspace/dictionaries")
    in2idx1_file = open("flickr_myspace/flickr/graphsage/id2idx.json", "w")
    json.dump(source_id2idx, in2idx1_file)
    in2idx1_file.close()
    in2idx2_file = open("flickr_myspace/myspace/graphsage/id2idx.json", "w")
    json.dump(target_id2idx, in2idx2_file)
    
    in2idx2_file.close()
    _save_graph(G1, "flickr_myspace/flickr/graphsage/G.json")
    _save_graph(G2, "flickr_myspace/myspace/graphsage/G.json")
    copyfile(args.groundtruth, "flickr_myspace/dictionaries/groundtruth")
    # save feature
    source_feats = source_dataset.features[[source_dataset.id2idx[ele] for ele in source_node_left]]
    target_feats = target_dataset.features[[target_dataset.id2idx[ele] for ele in target_node_left]]
    np.save("flickr_myspace/flickr/graphsage/feats.npy", source_feats)
    np.save("flickr_myspace/myspace/graphsage/feats.npy", target_feats)
    