import time

import numpy as np 
import pandas as pd 

from sklearn.datasets import fetch_openml

from sklearn.manifold import TSNE 

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import seaborn as sns 



# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.0
# mnist = fetch_openml("MNIST original")

# X = mnist.data/255
# y = mnist.target

print(X.shape, y.shape)

feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))

X, y = None, None

print('Size of the dataframe: {}'.format(df.shape))

np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

# plt.gray()
# fig = plt.figure(figsize = (16, 7))
# for i in range(0, 15):
#     ax = fig.add_subplot(3, 5, i+1, title = 'Digit: {}'.format(str(df.loc[rndperm[i], 'label'])))
#     ax.matshow(df.loc[rndperm[i], feat_cols].values.reshape((28, 28)).astype(float))
# plt.show()

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# import pdb 
# pdb.set_trace()
# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=df.loc[rndperm,:]["pca-one"], 
#     ys=df.loc[rndperm,:]["pca-two"], 
#     zs=df.loc[rndperm,:]["pca-three"], 
#     c=df.loc[rndperm,:]["y"].astype(int)
# )
# ax.set_xlabel('pca-one')
# ax.set_ylabel('pca-two')
# ax.set_zlabel('pca-three')
# plt.show()

N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))


sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)

plt.show()



"""
for gcn in 1 2 3 4 5
do

PD=$HOME/dataspace/graph/suhi_allmv_tmdb

python -u network_alignment.py \
--source_dataset ${PD}/allmv/graphsage/ \
--target_dataset ${PD}/tmdb/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
EMGCN \
--embedding_dim 300 \
--emb_epochs 50 \
--lr 0.01 \
--num_GCN_blocks ${gcn} \
--noise_level 0.001 \
--refinement_epoch 50 \
--refine \
--cuda \
--log > output/EMGCN/movie1_gcn${gcn}
done


for gcn in 1 2 3 4 5
do

PD=$HOME/dataspace/graph/flickr_myspace

python -u network_alignment.py \
--source_dataset ${PD}/flickr/graphsage/ \
--target_dataset ${PD}/myspace/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
EMGCN \
--embedding_dim 300 \
--emb_epochs 50 \
--lr 0.01 \
--num_GCN_blocks ${gcn} \
--noise_level 0.001 \
--refinement_epoch 50 \
--refine \
--cuda \
--log > output/EMGCN/flickr_gcn${gcn}
done


"""
