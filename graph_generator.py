import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



def generate_graph(num_of_nodes,num_of_edges):
    '''
        num_of_nodes: the number of nodes in the network
        num_of_edges: the number of edges in the network
    '''
    
    G = nx.gnm_random_graph(num_of_nodes,num_of_edges)
    k = 0
    while nx.is_connected(G)==False:
        G = nx.gnm_random_graph(num_of_nodes,num_of_edges)
        k += 1
    print("Check if connected: ", nx.is_connected(G))
    draw(G)
    return G

def draw(G):
    nx.draw_networkx(G, pos=nx.spring_layout(G),node_color="pink",node_shape='.',edge_color="skyblue",width=0.7,font_size="7",font_color ="white")
    plt.plot()
    plt.savefig('graph')
    plt.close()

def laplacian(G):
    L = nx.laplacian_matrix(G)
    return L

def adjacency(G):
    A = nx.adjacency_matrix(G)
    return A
