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
    # 调用generate_graph()generate_function()和生成图graph和优化函数func，生成size个训练样本，放在dataset里面
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
        data.A = func['A']               # data的属性赋值
        data.b = func['b']
        data.sum_A = func['sum_A']
        data.sum_b = func['sum_b']
        data.x = func['x']
        data.y = func['y']
        # data的维度：(edge_index=[2, 450], num_nodes=50, adj=[50, 50], A=[50, 10, 10], b=[50, 10], sum_A=[10, 10], sum_b=[10],
        #      x=[50, 10], y=[1, 10])
        dataset.append(data)
    # 经过DataLoader加工后的dataset属性的维度可以用.__dict__查看
    dataset = DataLoader(dataset, batch_size=1, shuffle=True)
    ''' 属性{'follow_batch': [], 'exclude_keys': [], 'dataset': [50 * Data(edge_index=[2, 450], num_nodes=50, adj=[50, 50], A=[50, 10, 10], b=[50, 10], sum_A=[10, 10], sum_b=[10], x=[50, 10], y=[1, 10])], 'num_workers': 0, 'prefetch_factor': 2, 'pin_memory': False, 'timeout': 0, 'worker_init_fn': None, '_DataLoader__multiprocessing_context': None, '_dataset_kind': 0, 'batch_size': 1, 'drop_last': False, 'sampler': <torch.utils.data.sampler.RandomSampler object at 0x0000027CA84DEEB0>, 'batch_sampler': <torch.utils.data.sampler.BatchSampler object at 0x0000027CA6B29160>, 'generator': None, 'collate_fn': <torch_geometric.loader.dataloader.Collater object at 0x0000027CE0BC9370>, 'persistent_workers': False, '_DataLoader__initialized': True, '_IterableDataset_len_called': None, '_iterator': None} '''
    return dataset


def generate_graph(num_of_nodes, num_of_edges, directed=False, add_self_loops=True):
    # 生成随机的有向图，
    G = nx.gnm_random_graph(num_of_nodes, num_of_edges, directed=directed)
    k = 0

    # 判断图是不是 强连通（每一个顶点皆可以经由该图上的边抵达其他的每一个点的有向图）如果是connected则返回True；
    while (nx.is_strongly_connected(G) if directed else nx.is_connected(G)) == False:
        G = nx.gnm_random_graph(num_of_nodes, num_of_edges, directed=directed)
        k += 1
    # print("Check if connected: ", nx.is_connected(G))
    draw(G)
    # from_networkx(G)官方定义: Converts a networkx.Graph or networkx.DiGraph to a torch_geometric.data.Data instance
    data = from_networkx(G)
    # 添加剩余的self-loop(自己和自己连接（i，i）),
    if add_self_loops:
        data.edge_index, _ = add_remaining_self_loops(
            data.edge_index, num_nodes=data.num_nodes)
    # print(nx.adjacency_matrix(G).A)

    # 先从图中生成邻接矩阵，然后把nx.adjacency_matrix(G).A 从矩阵变成tensor
    data.adj = torch.Tensor(nx.adjacency_matrix(G).A)
    # data.laplacian = torch.Tensor(nx.laplacian_matrix(G).A)
    #
    return data

def star_graph(num_of_nodes, fix_func):
    func = deepcopy(fix_func)

    # nx.star_graph(num_of_nodes-1) ：返回具有num_of_nodes个节点的Star图:一个中心节点，连接到n个外部节点
    G = nx.star_graph(num_of_nodes-1)
    #画图并保存
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
    # 画图，图保存在graph里面。
    nx.draw_networkx(G, pos=nx.spring_layout(G), node_color="pink", node_shape='.',
                     edge_color="skyblue", width=0.7, font_size="7", font_color="white")
    import matplotlib.pyplot as plt
    plt.plot()
    plt.savefig('graph')
    plt.close()


def generate_function(num_of_nodes, dim):
    function = {}
    min_eig = 5   #最小特征值为5
    # A, b里包含了50个节点上各自的A[i],b[i] , A是(num_of_nodes,dim,dim)维的，b是(num_of_nodes,dim)维的
    # sum_A 是A[i]的和，sum_b同理
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

    # 用 cvxpy 解出最优解optimal_value记为x^*，算法第t步得到的x^t减去x^*得到的一个差，这个差是用来衡量算法收敛速度快慢。即，画的收敛速度图纵坐标是||x^t-x^*||,横坐标是步数t。
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