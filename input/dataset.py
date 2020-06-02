import json
import os
import argparse
from scipy.io import loadmat
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
# from input.data_preprocess import DataPreprocess
import torch
import utils.graph_utils as graph_utils
import scipy.sparse as sp
import inflect
import codecs
import time
from numpy import *

from numpy import unravel_index

class Dataset:
    """
    this class receives input from graphsage format with predefined folder structure, the data folder must contains these files:
    G.json, id2idx.json, features.npy (optional)

    Arguments:
    - data_dir: Data directory which contains files mentioned above.
    """

    def __init__(self, data_dir, dt_name="", demo=False, use_word_embeddings=False, noise="", noise_level=0):
        self.data_dir = data_dir
        self.dt_name = dt_name
        if not demo:
            self.noise = noise
            self.nodes_to_del = []
            self.edges_to_del = []
            self.att_to_del = []
            if noise != "":
                self.get_noise_data(noise, noise_level)
            self._load_G()
            self._load_id2idx()
            self._load_features()
            if noise != "":
                self.id2idx = self.new_id2idx
            # self.load_edge_features()
            print("Dataset info:")
            print("- Nodes: ", len(self.G.nodes()))
            print("- Edges: ", len(self.G.edges()))

        if self.dt_name != "":
            self.dt_name_path = "data/{}".format(self.dt_name)
            self.ent_att_val1 = os.path.join(self.dt_name_path, "ent_att_val_1")
            self.ent_att_val2 = os.path.join(self.dt_name_path, "ent_att_val_2")
            self.index_att_1 = os.path.join(self.dt_name_path, "index_att_1")
            self.index_att_2 = os.path.join(self.dt_name_path, "index_att_2")
            self.glove = "data/sub.glove.300d"
            if use_word_embeddings:
                self.words = self.create_dictionaries()
                self.embedder = self.get_embedder()
        else:
            self.dt_name_path = ""
        

    def get_noise_data(self, noise, noise_level):
        print("Loading noise info")
        if noise == "del_nodes":
            noisex = "del_node"
        else:
            noisex = noise
        data_path = os.path.join("data", noise, "{}_{}_{}".format(self.dt_name, noisex, noise_level))
        if noise == "del_edges":
            with open(data_path, "r", encoding="utf-8") as file:
                for line in file:
                    data_line = line.split()
                    edge = [data_line[0], data_line[-1]]
                    if edge not in self.edges_to_del:
                        self.edges_to_del.append(edge)
        elif noise == "del_nodes":
            with open(data_path, "r", encoding="utf-8") as file:
                for line in file:
                    data_line = line.split()
                    node = data_line[0]
                    if node not in self.nodes_to_del:
                        self.nodes_to_del.append(node)
        print("DONE loading noise info")
                    

    def get_att_dict(self, path):
        att_dict = {}
        with open(path, "r", encoding='utf-8') as file:
            for line in file:
                att = line.split()[1]
                index = line.split()[0]
                att_dict[att] = index
        return att_dict

    def get_raw_att_dicts(self):
        dict1 = self.get_att_dict(self.index_att_1)
        dict2 = self.get_att_dict(self.index_att_2)
        return dict1, dict2


    def _load_G(self):
        G_data = json.load(open(os.path.join(self.data_dir, "G.json")))
        self.G = json_graph.node_link_graph(G_data)
        print("Before: {}".format(len(self.G.nodes())))
        if self.noise == "del_nodes":
            for node in self.nodes_to_del:
                try:
                    self.G.remove_node(node)
                except:
                    continue
        elif self.noise == "del_edges":
            for edge in self.edges_to_del:
                try:
                    self.G.remove_edge(*edge)
                except:
                    continue
        print("after: {}".format(len(self.G.nodes())))
        if type(self.G.nodes()[0]) is int:
            mapping = {k: str(k) for k in self.G.nodes()}
            self.G = nx.relabel_nodes(self.G, mapping)


    def _load_id2idx(self):
        id2idx_file = os.path.join(self.data_dir, 'id2idx.json')
        conversion = type(self.G.nodes()[0])
        self.id2idx = {}
        id2idx = json.load(open(id2idx_file))
        for k, v in id2idx.items():
            self.id2idx[conversion(k)] = v
        if self.noise != "":
            self.new_id2idx = {node: i for i, node in enumerate(self.G.nodes())}
        else:
            self.new_id2idx = self.id2idx


    def _load_features(self):
        self.features = None
        feats_path = os.path.join(self.data_dir, 'feats.npy')
        if os.path.isfile(feats_path):
            self.features = np.load(feats_path)
        else:
            self.features = None

        if len(self.new_id2idx) != len(self.id2idx):
            self.new_features = np.zeros((len(self.new_id2idx), self.features.shape[1]))
            self.new_idx2id = {v:k for k,v in self.new_id2idx.items()}
            for i in range(len(self.new_features)):
                self.new_features[i] = self.features[self.id2idx[self.new_idx2id[i]]]
            self.features = self.new_features
        return self.features


    def load_edge_features(self):
        self.edge_features= None
        feats_path = os.path.join(self.data_dir, 'edge_feats.mat')
        if os.path.isfile(feats_path):
            edge_feats = loadmat(feats_path)['edge_feats']
            self.edge_features = np.zeros((len(edge_feats[0]),
                                           len(self.G.nodes()),
                                           len(self.G.nodes())))
            for idx, matrix in enumerate(edge_feats[0]):
                self.edge_features[idx] = matrix.toarray()
        else:
            self.edge_features = None
        return self.edge_features


    def get_adjacency_matrix(self, sparse=False):
        return self.construct_adjacency(self.G, sparse=False)


    def get_nodes_degrees(self):
        return graph_utils.build_degrees(self.G, self.id2idx)


    def get_nodes_clustering(self):
        return graph_utils.build_clustering(self.G, self.id2idx)


    def get_edges(self):
        return graph_utils.get_edges(self.G, self.id2idx)


    def check_id2idx(self):
        for i, node in enumerate(self.G.nodes()):
            if self.id2idx[node] != i:
                print("Failed at node %s" % str(node))
                return False
        return True


    def normalize_adj(self, adj, rowsum, sparse=False):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        deg_out = np.array(adj.sum(1))
        deg_in = np.array((adj.T).sum(1))
        # rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        if not sparse:
            # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense()
            return (d_mat_inv_sqrt.dot(adj)).dot(d_mat_inv_sqrt).todense(), deg_out, deg_in
        else:
            return (d_mat_inv_sqrt.dot(adj)).dot(d_mat_inv_sqrt).tocoo(), deg_out, deg_in


    def preprocess_adj(self, adj, sparse, rowsum):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized, deg_out, deg_in = self.normalize_adj(adj + sp.eye(adj.shape[0]), rowsum, sparse)
        return adj_normalized


    def construct_adjacency(self, G, sparse=False, direct=False, typee='old'):
        adj = nx.to_scipy_sparse_matrix(G)
        adj_sym = adj + adj.T
        adj_sym[adj_sym > 1] = 1
        if typee == "new":
            rowsum = np.array(adj_sym.sum(axis=1)) + 1
        else:
            rowsum = np.array(adj.sum(axis=1)) + 1
        if not direct:
            adj = adj_sym
        if not sparse:
            adj = adj.todense()
        else:
            adj = adj.tocoo()
        return adj, rowsum


    def construct_laplacian(self, sparse=False, direct=False, typee='old'):
        G = self.G
        adj, rowsum = self.construct_adjacency(G, sparse, direct, typee)
        return self.preprocess_adj(adj, sparse, rowsum)


    def cv_coo_sparse(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def refine_matrix_product(self, A, D):
        """
        A is coo matrix
        D is coo matrix
        """
        return (D.dot(A)).dot(D)


    def clean_string(self, string):
        """
        Just keep alpha and digit string...
        """
        cleaned_string = ""
        for char in string:
            if not char.isalpha():
                if not char.isdigit():
                    cleaned_string += " "
                    continue
            cleaned_string += char
        return cleaned_string


    def words_of_string(self, string):
        """
        Return tokenized list of words.
        pre-requirement: 
            1. having a method for cleaning string.
        """
        p = inflect.engine()
        words_raw = string.split('_')
        words_set = set()
        for word in words_raw:
            if not word.isdigit():
                words_set.add(word)
            else:
                number_splitted_str = p.number_to_words(word)
                number_splitted_str = self.clean_string(number_splitted_str)
                number_splitted_str = number_splitted_str.split()
                words_set.update(set(number_splitted_str))

        return words_set


    def create_dictionaries(self):
        start_time = time.time()
        if self.dt_name_path == "":
            print("There are not path specified! Exitting...")
            exit()
        words = set()

        print("reading att file 1...")

        count_att_1 = 0
        with open(self.index_att_1, "r", encoding="utf-8") as file:
            for line in file:
                count_att_1 += 1
                data_line = line.split()
                words_set = self.words_of_string(data_line[1])
                words.update(words_set)
        print("Number of unique atts in source graph: {}".format(count_att_1))
        
        print("reading att file 2...")

        count_att_2 = 0
        with open(self.index_att_2, "r", encoding="utf-8") as file:
            for line in file:
                count_att_2 += 1
                data_line = line.split()
                words_set = self.words_of_string(data_line[1])
                words.update(words_set)
        print("Number of unique atts in target graph: {}".format(count_att_2))

        print("Number of unique att words: {}".format(len(words)))

        print("reading value file 1...")

        with open(self.ent_att_val1, "r", encoding="utf-8") as file:
            for line in file:
                data_line = line.split()
                list_string = data_line[2:]
                words_set = set()
                for string in list_string:
                    words_set_string = self.words_of_string(string)
                    words_set.update(words_set_string)
                words.update(words_set)
        
        print("reading value file 2...")

        with open(self.ent_att_val2, "r", encoding="utf-8") as file:
            for line in file:
                data_line = line.split()
                list_string = data_line[2:]
                words_set = set()
                for string in list_string:
                    words_set_string = self.words_of_string(string)
                    words_set.update(words_set_string)
                words.update(words_set)
        print("Number of unique words: {}".format(len(words)))
        print("Creating dictionary time: {:.4f}".format(time.time() - start_time))
        return words


    def get_embedder(self):
        # load glove file
        print("Getting word embedder")
        start_time = time.time()
        word_embedder = {}
        with codecs.open(self.glove, 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line.split(" ")) == 2:
                    continue
                info = line.split(' ')
                word = info[0]
                if word in self.words:
                    vec = [float(v) for v in info[1:]]
                    if len(vec) != 300:
                        continue
                    word_embedder[word] = np.array(vec)

        available_words = set(list(word_embedder.keys()))
        for word in self.words:
            if word not in available_words:
                word_embedder[word] = np.random.normal(-0.03, 0.4, 300)
        print("Getting embedder time: {:.4f}".format(time.time() - start_time))
        return word_embedder
    

    def get_embedding_string(self, string):
        """
        Return embeddings of a string
        Pre-requirements:
            1. having a method for getting embedding of a word: self.embedder (init)
            2. having a method for tokenizing words from a string.
            3. 
        """
        words = self.words_of_string(string)
        embeddings = []
        for word in words:
            try:
                embeddings.append(self.embedder[word])
            except:
                continue
        embedding = np.mean(np.array(embeddings), axis=0)
        return embedding


    def get_att_emb(self):
        """
        Return embeddings of all atts as a matrix, att_inndex for both source and target graph.
        Pre-requirements:
            1. having a method for getting embedding from a string
        => YOU CAN DIRECTLY CALL THIS METHOD RIGH AWAY
        """

        att_index1 = dict()
        att_emb_dict1 = dict()
        with open(self.index_att_1, "r", encoding="utf-8") as file:
            for line in file:
                data_line = line.split()
                att_index1[data_line[1]] = int(data_line[0])
                att_emb_dict1[int(data_line[0])] = self.get_embedding_string(data_line[1])
        att_emb_matrix1 = np.zeros((len(att_emb_dict1), 300))
        for key, value in att_emb_dict1.items():
            att_emb_matrix1[int(key)] = value
        
        att_index2 = dict()
        att_emb_dict2 = dict()
        with open(self.index_att_2, "r", encoding="utf-8") as file:
            for line in file:
                data_line = line.split()
                att_index2[data_line[1]] = int(data_line[0])
                att_emb_dict2[int(data_line[0])] = self.get_embedding_string(data_line[1])
        att_emb_matrix2 = np.zeros((len(att_emb_dict2), 300))

        for key, value in att_emb_dict2.items():
            att_emb_matrix2[int(key)] = value

        return att_emb_matrix1, att_emb_matrix2, att_index1, att_index2

    
    def get_emb_entity(self, att_emb, att_keep, path):
        """
        What is this. What the fuck is this
        att_emb: Embedding matrix of att
        att_keep: atts having embeddings
        path: path to the ent_att_val file. 
        """
        embedding_dict = {}
        count_x = 0
        with open(path, "r", encoding="utf-8") as file:
            last_id = None
            not_have_any = True
            for line in file:
                data_line = line.split()

                id = data_line[0]
                if self.noise != "":
                    if id not in self.G.nodes():
                        continue
                att_index = data_line[1]
                value = data_line[2:]

                embedding_id = np.zeros(300)

                for ele in value:
                    embedding_id += self.get_embedding_string(ele)
                # print("Num values: {}".format(len(value)))
                if len(value):
                    embedding_id /= len(value)
                
                if id not in embedding_dict.keys():
                    if last_id and not_have_any:
                        count_x += 1
                        embedding_dict[last_id] = {'value': np.zeros(300), 'att': np.zeros(300)}
                    elif last_id:
                        embedding_dict[last_id]['value'] = np.mean(np.array(embedding_dict[last_id]['value']), axis=0)
                        embedding_dict[last_id]['att'] = np.mean(np.array(embedding_dict[last_id]['att']), axis=0)
                    if int(att_index) not in att_keep:
                        not_have_any = True
                        continue
                    not_have_any = False
                    embedding_dict[id] = {'value': [embedding_id], 'att': [att_emb[int(att_index)]]}
                    
                else:
                    if int(att_index) not in att_keep:
                        continue
                    not_have_any = False
                    embedding_dict[id]['value'].append(embedding_id)
                    embedding_dict[id]['att'].append(att_emb[int(att_index)])
                last_id = id
                

            embedding_dict[last_id]['value'] = np.mean(np.array(embedding_dict[last_id]['value']), axis=0)
            embedding_dict[last_id]['att'] = np.mean(np.array(embedding_dict[last_id]['att']), axis=0)
            last_id = id
        return embedding_dict
    

    def get_emb_entities(self):
        """
        I HOPE THIS WILL BE WHAT I NEED!!!

        """
        # att_emb_matrix1, att_emb_matrix2, att_index1, att_index2 = self.get_att_emb()
        att_emb1, att_emb2, att_index1, att_index2 = self.get_att_emb()
        att_keep1 = set(list(att_index1.values()))
        att_keep2 = set(list(att_index2.values()))
        embedding_dict = self.get_emb_entity(att_emb1, att_keep1, self.ent_att_val1)
        embedding_dict2 = self.get_emb_entity(att_emb2, att_keep2, self.ent_att_val2)
        return embedding_dict, embedding_dict2


    def get_concated_vector_for_entity(self):
        """
        TODO: Return concated vector for all nodes
        """

        print("get embedding for attributes...")
        att_emb1, att_emb2, att_index1, att_index2 = self.get_att_emb()

        print("get embedding for values ")
        embedding_dict1, embedding_dict2 = self.get_emb_entity(att_emb1, att_emb2)

        return embedding_dict1, embedding_dict2
    

    def get_att_pairs(self, att_emb1, att_emb2, att_index1, att_index2):
        att_emb1 = att_emb1 / np.sqrt((att_emb1**2).sum(axis=1)).reshape(-1, 1)
        att_emb2 = att_emb2 / np.sqrt((att_emb2**2).sum(axis=1)).reshape(-1, 1)
        where_att1_nan = isnan(att_emb1)
        where_att2_nan = isnan(att_emb2)
        att_emb1[where_att1_nan] = 0
        att_emb2[where_att2_nan] = 0
        simi_matrix = att_emb1.dot(att_emb2.T)
        index_pairs = {}
        inverse_att_index1 = {int(v):k for k,v in att_index1.items()}
        inverse_att_index2 = {int(v):k for k,v in att_index2.items()}
        

        att1_set = set(list(att_index1.keys()))
        att2_set = set(list(att_index2.keys()))

        common_att = att1_set.intersection(att2_set)

        for ele in common_att:
            index_pairs[att_index1[ele]] = att_index2[ele]
            simi_matrix[att_index1[ele]] -= 10
            simi_matrix = simi_matrix.T
            simi_matrix[att_index2[ele]] -= 10
            simi_matrix = simi_matrix.T
        
        print("Num common att: {}".format(len(common_att)))
        # print(att_index1.keys())
        for i in range(min(simi_matrix.shape) - len(common_att)):
            new_pair = unravel_index(simi_matrix.argmax(), simi_matrix.shape)
            if simi_matrix[new_pair[0], new_pair[1]] < 0.97:
                break
            print(inverse_att_index1[new_pair[0]], inverse_att_index2[new_pair[1]], simi_matrix[new_pair[0], new_pair[1]])
            if new_pair[0] == 540:
                print(new_pair)
                print(simi_matrix[new_pair[0], new_pair[1]])
                exit()
            simi_matrix[new_pair[0]] -= 10
            simi_matrix = simi_matrix.T 
            simi_matrix[new_pair[1]] -= 10
            simi_matrix = simi_matrix.T
            index_pairs[new_pair[0]] = new_pair[1]

        return index_pairs


    def jaccard_simi(self, source_set, target_set):
        if len(source_set) == 0 or len(target_set) == 0:
            return 0
        return len(source_set.intersection(target_set)) / len(source_set.union(target_set))
    

    def preprocess_value(self, value):
        final_val = set()
        for string in value:
            word_set = set(string.split('_'))
            final_val.update(word_set)
        return final_val


    def get_the_raw_datastructure(self, path, att_dict_inverse, index_keeped=None):
        data = {}
        seen_id = set()
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                data_line = line.split()
                id = data_line[0]
                if self.noise != "":
                    if id in self.nodes_to_del:
                        continue
                if not id in seen_id:
                    data[id] = {'att': set(), 'att_value': {}}
                    seen_id.add(id)
                att_index = data_line[1]
                try:
                    att_index = att_dict_inverse[att_index]
                except:
                    continue
                if index_keeped:
                    if att_index not in index_keeped:
                        continue
                data[id]['att'].add(att_index)
                data[id]['att_value'][att_index] = self.preprocess_value(data_line[2:])
        return data


if __name__ == "__main__":


    # print("vinhsuhi")
    dataset = Dataset("", "ja_en", demo=True)
    # att_emb_matrix1, att_emb_matrix2, att_index1, att_index2 = dataset.get_att_emb()
    embedding_dict1, embedding_dict2 = dataset.get_emb_entities()

