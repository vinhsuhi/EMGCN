from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from input.dataset import Dataset
# import utils.graph_utils as graph_utils
# from evaluation.metrics import get_statistics
import argparse
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="../networkAlignment/data/ppi/graphsage")
    parser.add_argument('--target_dataset', default="../networkAlignment/data/ppi/REGAL-d05-seed1/graphsage")
    parser.add_argument('--path', default="ppi-2")
    return parser.parse_args()


def read_embedding_from_text_file(file_path):
    lines = open(file_path).readlines()
    embeddings = []
    for i in range(1, len(lines)):
        line_vector = [float(ele) for ele in lines[i].split()[1:]]
        embeddings.append(line_vector)

    embeddings = np.array(embeddings)
    return embeddings


def read_embedding_from_numpy_file(file_path):
    return np.load(file_path)


def compute_PCA(matrix, n_components=None, pca=None):

    if pca is None:
        assert n_components is not None, "if pca is None, n_components must be specified"
        pca = PCA(n_components=n_components).fit(matrix)

    data_after_transforming = pca.transform(matrix)
    return data_after_transforming, pca


def my_plotter(ax, data1, data2, param_dict, typee='.'):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """

    out = ax.plot(data1, data2, typee, **param_dict)
    # if typee != "-":
    #     for i in range(len(data1)):
    #         if i == 55:
    #             ax.annotate(i, (data1[i], data2[i]))
    # ax.set_xlim(-1.1, 1.1)
    # ax.set_ylim(-1.1, 1.1)
    ax.grid()
    return out



def visualize(data, labels, colors, name, file_name='', edges=None, markersizes=[]):
    """
    1 figure - n Axes : region of many images | 
    1 Axes - 2, 3 Axis: region of image | set_title(), set_xlabel(), set_ylabel()
    Axis - smallest elements: number-like-object | set_xlim(), set_ylim()

    Axes is the region of image with the dataspace. 1 figure - n Axes
    1 Axes - (2 | 3) Axis object
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(data)):
        matrix = data[i] # maxtrix_0 is the target embedding, matrix_1 is the source embedding
        matrix_1 = matrix[:,0]
        matrix_2 = matrix[:,1]

        param_dict = {'markersize': markersizes[i], 'label': labels[i], 'color': colors[i]}
        my_plotter(ax, matrix_1, matrix_2, param_dict)

    # if edges is not None:
    #     for i in range(len(edges)):
    #         matrix = data[0][edges[i]]
    #         matrix_1 = matrix[:,0]
    #         matrix_2 = matrix[:,1]
    #         # param_dict = {''}
    #         param_dict = {'color': 'k', 'linewidth': 0.5}

    #         my_plotter(ax, matrix_1, matrix_2, param_dict, typee='-')
    # if len(data) == 2:
    #     for i in range(len(data[0])):
    #         source_node = data[1][i]
    #         target_node = data[0][i]
    #         ax.plot([source_node[0], target_node[0]], [source_node[1], target_node[1]], color='k', linewidth=1)
    
    plt.legend()
    if not os.path.exists(name):
        os.makedirs(name)
    plt.savefig(name + '/' + file_name)
    print("Figure has been saved in {}".format(name + '/' + file_name))

# def visualize_both(s, name, file_name):
    # fig, ax = plt.subplot(1, 1, dpi=150)
    # for i in range(len(data))


    

if __name__ == "__main__":
    args = parse_args()
    """
    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    source_edges = source_dataset.get_edges()
    target_edges = target_dataset.get_edges()
    source_after_mapping_path = "GUI_sam.npy"
    with open("douban_data", "rb") as f:
        douban_data = pickle.load(f)
    source_hits = douban_data[0]
    target_hits = douban_data[1]
    """
    data_dir = os.listdir("numpy_emb")

    for i in range(1, int(len(data_dir)/2)):
        
        source_path = "numpy_emb/source_layer{}.npy".format(i)
        target_path = "numpy_emb/target_layer{}.npy".format(i)
        try:
            source_embeddings = read_embedding_from_numpy_file(source_path)
        except:
            exit()
        target_embeddings = read_embedding_from_numpy_file(target_path)
        

        source_embeddings = source_embeddings / np.sqrt((source_embeddings ** 2).sum(axis=1)).reshape(len(source_embeddings), -1)
        target_embeddings = target_embeddings / np.sqrt((target_embeddings ** 2).sum(axis=1)).reshape(len(target_embeddings), -1)
        """
        if source_embeddings.shape[1] == 1:
            source_embeddings = np.zeros((source_embeddings.shape[0], 2))
            source_embeddings[:,0] = np.ones((source_embeddings.shape[0]))
            target_embeddings = np.zeros((target_embeddings.shape[0], 2))
            target_embeddings[:,0] = np.ones((target_embeddings.shape[0]))
        print("Align_simi at layer: {}".format(i))
        align_simi = np.sum(source_embeddings[source_hits[i]] * target_embeddings[target_hits[i]], axis=1)
        print(np.mean(align_simi))

        print("Random_simi at layer: {}".format(i))
        print(np.mean(np.sum(source_embeddings[:len(source_hits[i])] * target_embeddings[:len(source_hits[i])], axis=1)))
        """
        colors = ['blue', 'orange']
        labels = ['target', 'source']

        tsne = manifold.TSNE(n_components=2, init='random',
                         random_state=0, perplexity=5)
        tsne.fit(source_embeddings)
        target_visualized_data = tsne.fit_transform(target_embeddings)
        source_visualized_data = tsne.fit_transform(source_embeddings)
        # target_visualized_data, target_pca = compute_PCA(target_embeddings, n_components=2)



        # # target_care_data = target_visualized_data[np.array(target_hits[i])]
        # source_visualized_data, _ = compute_PCA(source_embeddings, n_components=2, pca=target_pca)
        # source_care_data = source_visualized_data[np.array(source_hits[i])]
        data = [source_visualized_data]
        # for 2d
        # data = [source_embeddings]
        colors = ['blue', 'orange']

        # visualize(data, labels, colors, 'GCN_visual/{}'.format(args.path), file_name='source_layer{}.png'.format(i), markersizes=[2.5, 2.5])
        data = [target_visualized_data]
        # for 2d 
        # data = [target_embeddings]

        # colors = ['red', 'orange']

        # visualize(data, labels, colors, 'GCN_visual/{}'.format(args.path), file_name='target_layer{}.png'.format(i), markersizes=[2.5, 2.5])

        # data = [source_embeddings, target_embeddings]
        data = [source_visualized_data, target_visualized_data]
        labels = ["source", "target"]
        colors = ['blue', 'orange']
        visualize(data, labels, colors, 'GCN_visual/{}'.format(args.path), file_name='both_layer{}.png'.format(i), markersizes=[10, 5])
        # visualize()
        # data = [target_care_data, source_care_data]
        # visualize(data, labels, colors, 'GCN_visual/{}'.format(args.path), file_name='source_target_layer{}.png'.format(i))