import optbaseline as opt
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from net import Net_GD, Net_EXTRA
from loss import loss_sup
from dataset import generate_dataset, generate_graph
from baseline import test_dgd, test_extra


def test(model, test_set):
    max_iter = 10
    my_error = 0
    with torch.no_grad():
        for data in test_set:
            data = data.to(device)
            x = model.forward(data, num_layers=max_iter).cpu()
            my_error += opt.opt_distance(data.y.cpu().numpy(),
                                         x.detach().numpy())
    print('test err:', my_error/num_of_test)


def train(model, train_set):
    epoch = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_seq = []
    for epoch in tqdm(range(epoch)):
        for data in train_set:
            data = data.to(device)
            optimizer.zero_grad()

            x = model.forward(data, num_layers=20)
            loss = loss_sup(x, data.y)
            loss.backward()
            optimizer.step()
            loss_seq.append(loss.item())

    plt.plot(loss_seq, label='loss')
    plt.legend()
    plt.show()
    plt.savefig('loss')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

num_of_nodes = 20
num_of_edges = 50
dim = 5

num_of_train = 20
num_of_test = 5

fixed_graph = generate_graph(num_of_nodes, num_of_edges)
train_set = generate_dataset(size=num_of_train, num_of_nodes=num_of_nodes,
                             num_of_edges=num_of_edges, dim=dim, fix_graph=fixed_graph)
test_set = generate_dataset(size=num_of_test, num_of_nodes=num_of_nodes,
                            num_of_edges=num_of_edges, dim=dim, fix_graph=fixed_graph)

# init model
model_gd = Net_GD(dim=dim).to(device)
model_extra = Net_EXTRA(dim=dim).to(device)

train(model_gd, train_set)
train(model_extra, train_set)
test(model_gd, test_set)
test(model_extra, test_set)
test_dgd(test_set, num_iters=10)
test_extra(test_set, num_iters=10)
