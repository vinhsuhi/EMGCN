# this script will delete old feature of the dataset.
# it will search all the sub folder to find dictionary and then fix the attribute

import argparse
import os
import pickle
from input.dataset import Dataset
import numpy as np
import shutil
import utils.graph_utils as graph_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--path', default="/home/dataspace/graph/econ-mahindas")
    parser.add_argument('--path2', default="")
    parser.add_argument('--groundtruth', default="")
    parser.add_argument('--num_feat', default=50, type=int)
    parser.add_argument('--keep_old_feats', default="") 
    return parser.parse_args()

def create_onehot_feature(num_feat, num_nodes):
    feats = np.zeros((num_nodes, num_feat))
    feats[:num_feat] = np.eye(num_feat)
    for i in range(len(feats)):
        if feats[i].sum() == 0:
            index = np.random.randint(num_feat)
            feats[i][index] = 1

    nodes = np.arange(num_nodes)
    np.random.shuffle(nodes)
    feats = feats[nodes]
    return feats
            

def remove_exceed_files(path):
    list_dir = [x[0] for x in os.walk(path)]
    for fi in list_dir:
        eles = fi.split("/")
        for ele in eles:
            if "PALE" in ele:
                try:
                    shutil.rmtree(fi)
                    break
                except:
                    continue
            if "seed" in ele:
                if int(ele.split("seed")[-1]) > 5:
                    try:
                        shutil.rmtree(fi)
                        break
                    except:
                        continue
    

def create_feature(num_feat, path, path2, groundtruth):
    source_dataset = Dataset(path + "/graphsage/")
    if path2 != "":
        target_dataset = Dataset(path2 + "/graphsage/")
        groundtruth = graph_utils.load_gt(groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        source_nodes = np.array(list(groundtruth.keys()))
        target_nodes = np.array(list(groundtruth.values()))
        source_feats = create_onehot_feature(num_feat, len(source_dataset.G.nodes()))
        target_feats = np.zeros((len(target_dataset.G.nodes()), num_feat))
        target_feats[target_nodes] = source_feats[source_nodes]
        source_feats2 = np.zeros(source_feats.shape)
        target_feats2 = np.zeros(target_feats.shape)
        source_feats2[:, 0] = 1
        target_feats2[:, 0] = 1
        source_feats2[source_nodes] = source_feats[source_nodes]
        target_feats2[target_nodes] = target_feats[target_nodes]
        np.save(path + "/graphsage/feats.npy", source_feats2)
        np.save(path2 + "/graphsage/feats.npy", target_feats2)
        return 
    print("Remove exceed file")
    remove_exceed_files(path)
    print("Creating features")
    source_id2idx = source_dataset.id2idx
    if args.keep_old_feats != "":
        source_feats = np.load(args.keep_old_feats)
        if source_feats.shape[1] != num_feat:
            print("Number of feat must equal to the old features")
    else:
        source_feats = create_onehot_feature(num_feat, len(source_dataset.G.nodes()))
    print("Saving source feats")
    np.save(path + "/graphsage/feats.npy", source_feats)
    tree_dir = [x[0] for x in os.walk(path)]
    print("Start searching for target dir")
    for dir in tree_dir:
        if "seed" in dir.split("/")[-1]:
            print("Working with {}".format(dir))
            # is a child file
            try:
                target_dataset = Dataset(dir + "/graphsage/")
            except Exception as err:
                print("Error: {}".format(err))
                continue
            target_id2idx = target_dataset.id2idx
            dictionary = graph_utils.load_gt(dir + "/dictionaries/groundtruth", source_id2idx, target_id2idx, 'dict')
            target_feats = np.zeros((len(target_dataset.G.nodes()), num_feat))
            source_nodes = np.array(list(dictionary.keys()))
            target_nodes = np.array(list(dictionary.values()))
            target_feats[target_nodes] = source_feats[source_nodes]
            np.save(dir + "/graphsage/feats.npy", target_feats)
    print("DONE")

if __name__ == "__main__":
    args = parse_args()
    create_feature(args.num_feat, args.path, args.path2, args.groundtruth)
