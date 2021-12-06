import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import *
from baseline import *
from net import Net_GD, Net_GD_Adapt, Net_EXTRA, Net_EXTRA_Adapt, Net_Push_DIGing, Net_Push_DIGing_Adapt
from loss import loss_sup
import optbaseline as opt
import numpy as np
import random

seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def test_model_target(model, test_set, label=None, max_iter=1000):
    my_cost = 0
    with torch.no_grad():
        for data in test_set:
            data = data.to(device)
            init_err = opt.opt_distance(data.y.cpu().numpy(),
                                        data.x.cpu().detach().numpy())
            test_target = init_err*test_target_ratio
            x, cost = model.forward(
                data, num_layers=max_iter, terminal=test_target, recording=False, threshold=att_threshold)
            err = opt.opt_distance(data.y.cpu().numpy(),
                                   x.cpu().detach().numpy())
            print(err)
            if err > test_target:
                cost = float('inf')
            my_cost += cost
    print(label, 'ave. cost:', my_cost/len(test_set))


def test_model_fix_iter(model, test_set, label=None):
    my_error = 0
    err_record = np.zeros(test_layers+1,)
    with torch.no_grad():
        for data in test_set:
            data = data.to(device)
            x, _, record = model.forward(
                data, num_layers=test_layers, recording=True)
            my_error += opt.opt_distance(data.y.cpu().numpy(),
                                         x.cpu().detach().numpy())
            err_record = err_record+np.array(record)
        err_record = err_record/len(test_set)
        plt.plot(err_record, label=label)
    print(label, 'err:', my_error/len(test_set))
    return my_error/len(test_set)


def train_model_fix_iter(model, train_set):
    epoch = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_seq = []
    for epoch in tqdm(range(epoch), 'Training'):
        for data in train_set:
            data = data.to(device)
            optimizer.zero_grad()
            x, _ = model.forward(data, num_layers=train_layers)
            loss = loss_sup(x, data.y)
            loss.backward()
            optimizer.step()
            loss_seq.append(loss.item())
    plt.plot(loss_seq, label='loss')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

train_layers = 100
test_layers = 100

test_target_ratio = 0.2
att_threshold = 0.01

if __name__ == '__main__':

    num_of_nodes = 50
    num_of_edges = 200
    dim = 10

    num_of_train = 50
    num_of_test = 5

    # fixed_function = generate_function(num_of_nodes, dim)
    # train_set = generate_dataset(num_of_train, num_of_nodes,
    #                              num_of_edges, directed=False, dim=dim, fix_func=fixed_function)
    # test_set = generate_dataset(num_of_test, num_of_nodes, num_of_edges,
    #                             directed=False, dim=dim, fix_func=fixed_function)
    # test_set = star_graph(num_of_nodes, fix_func=fixed_function)

    fixed_function,dim = load_function(num_of_nodes)
    train_set = load_dataset(num_of_train, num_of_nodes,
                                 num_of_edges, directed=False, fix_func=fixed_function)
    test_set = load_dataset(num_of_test, num_of_nodes, num_of_edges,
                                directed=False, fix_func=fixed_function)

    # init model
    model_GD = Net_GD(dim=dim, step_size=0.001).to(device)
    model_EXTRA = Net_EXTRA(dim=dim, step_size=0.001).to(device)
    model_PD = Net_Push_DIGing(dim=dim, step_size=0.001).to(device)
    model_GD_adapt = Net_GD_Adapt(dim=dim, step_size=0.001).to(device)
    model_EXTRA_adapt = Net_EXTRA_Adapt(dim=dim, step_size=0.001).to(device)
    model_PD_adapt = Net_Push_DIGing_Adapt(dim=dim, step_size=0.001).to(device)

    train_model_fix_iter(model_GD, train_set)
    train_model_fix_iter(model_EXTRA, train_set)
    train_model_fix_iter(model_PD, train_set)
    train_model_fix_iter(model_GD_adapt, train_set)
    train_model_fix_iter(model_EXTRA_adapt, train_set)
    train_model_fix_iter(model_PD_adapt, train_set)

    plt.legend()
    plt.savefig('loss')
    plt.clf()

    test_model_fix_iter(model_GD, test_set, 'GD')
    test_model_fix_iter(model_EXTRA, test_set, 'EXTRA')
    test_model_fix_iter(model_PD, test_set, 'PD')
    test_model_fix_iter(model_GD_adapt, test_set, 'GD_A')
    test_model_fix_iter(model_EXTRA_adapt, test_set, 'EXTRA_A')
    test_model_fix_iter(model_PD_adapt, test_set, 'PD_A')
    # test_model_target(model_GD, test_set, 'GD')
    # test_model_target(model_EXTRA, test_set, 'EXTRA')
    # test_model_target(model_PD, test_set, 'PD')
    # test_model_target(model_GD_adapt, test_set, 'GD_A')
    # test_model_target(model_EXTRA_adapt, test_set, 'EXTRA_A')
    # test_model_target(model_PD_adapt, test_set, 'PD_A')

    test_origin_fix_iter(test_set, test_layers, method='DGD', step_size=0.001)
    test_origin_fix_iter(test_set, test_layers,
                         method='EXTRA', step_size=0.001)
    test_origin_fix_iter(test_set, test_layers, method='PD', step_size=0.001)
    # test_origin_target(test_set, test_target_ratio, method='DGD', step_size=0.001)
    # test_origin_target(test_set, test_target_ratio,
    #                 method='EXTRA', step_size=0.001)
    # test_origin_target(test_set, test_target_ratio, method='PD', step_size=0.003)

    plt.yscale('log')
    plt.legend()
    plt.savefig('test err')
