"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
##################
path_to_graph = "datasets"
filename = "CA-HepTh.txt"

with open(path_to_graph + '/' + filename, 'rb') as graph_data:
    G = nx.read_edgelist(graph_data, comments = '#', delimiter = '\t')

print('Graph - number of nodes: %d' % G.number_of_nodes())
print('Graph - number of edges: %d' % G.number_of_edges())


############## Task 2

##################
# your code here #
##################
connected_comp = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
print('Number of connected components: %2d' %len(connected_comp))


largest_cc = G.subgraph(connected_comp[0])
print('Largest connected component - number of nodes: %d - number of edges: %d' 
      %(largest_cc.number_of_nodes(),largest_cc.number_of_edges()))

ratio_nodes = largest_cc.number_of_nodes()/G.number_of_nodes()
ratio_edges = largest_cc.number_of_edges()/G.number_of_edges()
print('Largest connected component - proportion of the nodes: %.2f - proportion of the edges %.2f' 
      %(ratio_nodes,ratio_edges)) #majority of the graph


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
##################
# your code here #
##################
max_degree = np.max(degree_sequence)
min_degree = np.min(degree_sequence)
average_degree = np.mean(degree_sequence)

print('Degree - max: %d - min: %d - average: %d' %(max_degree,min_degree,average_degree))

############## Task 4

##################
# your code here #
##################

frequency = nx.degree_histogram(G)

plt.subplots()
plt.subplot(311)
plt.bar(range(len(frequency)),frequency, color='b')
plt.title("degree histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.subplot(313)
plt.bar(range(len(frequency)),frequency, color='r',log = 'True')
plt.title("degree histogram - log scale")
plt.ylabel("Count")
plt.xlabel("Degree")