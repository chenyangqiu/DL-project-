import torch
import numpy as np
import networkx as nx
import func_generator as func
import optbaseline as opt
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from copy import deepcopy
from torch_geometric.utils import add_remaining_self_loops
from tqdm import tqdm
import os

def generate_dataset(size, num_of_nodes, num_of_edges, directed=False, dim=None, fix_graph=None, fix_func=None):
    dataset = []

    for i in tqdm(range(size), 'Generating Data'):
        if fix_graph is None:
            data = generate_graph(
                num_of_nodes, num_of_edges, directed=directed)
        else:
            data = deepcopy(fix_graph)

        if fix_func is None:
            func = generate_function(num_of_nodes, dim)
        else:
            func = deepcopy(fix_func)

        # data.A_i = func['A_i']
        data.A = func['A']
        data.b = func['b']
        data.sum_A = func['sum_A']
        data.sum_b = func['sum_b']
        data.x = func['x']
        data.y = func['y']

        dataset.append(data)
    dataset = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset


def generate_graph(num_of_nodes, num_of_edges, directed=False, add_self_loops=True):
    G = nx.gnm_random_graph(num_of_nodes, num_of_edges, directed=directed)
    k = 0
    while (nx.is_strongly_connected(G) if directed else nx.is_connected(G)) == False:
        G = nx.gnm_random_graph(num_of_nodes, num_of_edges, directed=directed)
        k += 1
    # print("Check if connected: ", nx.is_connected(G))
    draw(G)

    data = from_networkx(G)

    if add_self_loops:
        data.edge_index, _ = add_remaining_self_loops(
            data.edge_index, num_nodes=data.num_nodes)

    data.adj = torch.Tensor(nx.adjacency_matrix(G).A)
    # data.laplacian = torch.Tensor(nx.laplacian_matrix(G).A)
    return data

def star_graph(num_of_nodes, fix_func):
    func = deepcopy(fix_func)
    G = nx.star_graph(num_of_nodes-1)
    draw(G)
    data = from_networkx(G)
    data.edge_index, _ = add_remaining_self_loops(
        data.edge_index, num_nodes=data.num_nodes)
    data.adj = torch.Tensor(nx.adjacency_matrix(G).A)

    data.A = func['A']
    data.b = func['b']
    data.sum_A = func['sum_A']
    data.sum_b = func['sum_b']
    data.x = func['x']
    data.y = func['y']
    return DataLoader([data], batch_size=1, shuffle=False)

def path_graph(num_of_nodes, fix_func):
    func = deepcopy(fix_func)
    G = nx.path_graph(num_of_nodes)
    draw(G)
    data = from_networkx(G)
    data.edge_index, _ = add_remaining_self_loops(
        data.edge_index, num_nodes=data.num_nodes)
    data.adj = torch.Tensor(nx.adjacency_matrix(G).A)

    data.A = func['A']
    data.b = func['b']
    data.sum_A = func['sum_A']
    data.sum_b = func['sum_b']
    data.x = func['x']
    data.y = func['y']
    return DataLoader([data], batch_size=1, shuffle=False)

def draw(G):
    nx.draw_networkx(G, pos=nx.spring_layout(G), node_color="pink", node_shape='.',
                     edge_color="skyblue", width=0.7, font_size="7", font_color="white")
    import matplotlib.pyplot as plt
    plt.plot()
    plt.savefig('graph')
    plt.close()


def generate_function(num_of_nodes, dim):
    function = {}
    min_eig = 5
    A, b, sum_A, sum_b = func.problem_parameter(min_eig, num_of_nodes, dim)
    function['A'] = torch.Tensor(A)
    # function['A'] = torch.zeros((num_of_nodes*dim, num_of_nodes*dim))
    # for j in range(num_of_nodes):
    #     function['A'][j*dim:(j+1)*dim, j*dim:(j+1)*dim] = function['A_i'][j]
    function['b'] = torch.Tensor(b)
    function['sum_A'] = torch.Tensor(sum_A)
    function['sum_b'] = torch.Tensor(sum_b)
    # init x & y
    function['x'] = torch.normal(0, 1, size=(num_of_nodes, dim))  # init x
    optimal_solution, optimal_value = opt.opt_solver(sum_A, sum_b, dim)
    function['y'] = torch.Tensor(optimal_solution).view(1, -1)
    return function

def load_function(num_of_nodes):
    pwd=os.getcwd()
    father_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
    file_path_X=os.path.join(father_path,'dl_datasets/data29/data.txt')
    file_path_y=os.path.join(father_path,'dl_datasets/data29/y.txt')
    X_data=np.loadtxt(file_path_X)
    y_data=np.loadtxt(file_path_y)
    dim=X_data.shape[1]
    sample_each_num=int(X_data.shape[0]/num_of_nodes)
    sample_num=sample_each_num*num_of_nodes
    feature=X_data[:sample_num,:]
    target=y_data[:sample_num]

    function={}
    A = np.zeros((num_of_nodes, dim, dim))
    b=np.zeros((num_of_nodes, dim))
    sum_b = np.zeros(dim)
    sum_A = np.zeros((dim,dim))
    for i in range(num_of_nodes):
        At=np.array(feature)[i*sample_each_num:(i+1)*sample_each_num,:]
        A[i]=At.T@At
        bt=np.array(target)[i*sample_each_num:(i+1)*sample_each_num].reshape(-1,)
        b[i]=-2*At.T@bt
        sum_b += b[i]
        sum_A += A[i]
    function['A']=torch.Tensor(A)
    function['b']=torch.Tensor(b)
    function['sum_A']=torch.Tensor(sum_A)
    function['sum_b']=torch.Tensor(sum_b)
        # init x & y
    function['x'] = torch.normal(0, 1, size=(num_of_nodes, dim))  # init x
    optimal_solution, optimal_value = opt.opt_solver(sum_A, sum_b, dim)
    function['y'] = torch.Tensor(optimal_solution).view(1, -1)
    #print(function['A'].shape, function['b'].shape, function['x'].shape, function['y'].shape)
    return function,dim

def load_dataset(size, num_of_nodes, num_of_edges, directed=False, fix_graph=None, fix_func=None):
    dataset = []

    for i in tqdm(range(size), 'Loading Data'):
        if fix_graph is None:
            data = generate_graph(
                num_of_nodes, num_of_edges, directed=directed)
        else:
            data = deepcopy(fix_graph)

        if fix_func is None:
            func = load_function(num_of_nodes)
        else:
            func = deepcopy(fix_func)

        # data.A_i = func['A_i']
        data.A = func['A']
        data.b = func['b']
        data.sum_A = func['sum_A']
        data.sum_b = func['sum_b']
        data.x = func['x']
        data.y = func['y']

        dataset.append(data)
    dataset = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset