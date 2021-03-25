"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from random import randint
from sklearn.cluster import KMeans


############## Task 5
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    # your code here #
    ##################
    L = nx.laplacian_matrix(G).astype(float)
    eig_val, eig_vect = eigsh(L,k)
    kmeans = KMeans(n_clusters=k).fit_predict(eig_vect)
    
    clustering = {'node': [n for n in G.nodes()], 'cluster': [k for k in kmeans]}
    return clustering



############## Task 6

##################
# your code here #
##################
k_cluster = 50
spec_clusters = spectral_clustering(largest_cc,k_cluster)


############## Task 7
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    modularity = 0
    m = G.number_of_edges()
    nc = np.max(clustering['cluster'])

    for i in range(nc):
        cluster_i = [nd for nd,clstr in zip(clustering['node'],clustering['cluster']) if clstr == i]
        subgraph_i = G.subgraph(cluster_i)
        li = subgraph_i.number_of_edges()
        di = np.sum([subgraph_i.degree(node) for node in subgraph_i.nodes()])
        modularity += li/m -(di/(2*m))**2
        
    return modularity



############## Task 8

##################
# your code here #
##################
rand_clusters = {'node': [n for n in largest_cc.nodes()], 'cluster': [randint(0,k_cluster) for n in largest_cc.nodes()]}

spec_modularity = modularity(largest_cc, spec_clusters)
rand_modularity = modularity(largest_cc, rand_clusters)
print('Modularity - Spectral Clustering: %.3f / Random Clustering: %.3f' %(spec_modularity,rand_modularity))
