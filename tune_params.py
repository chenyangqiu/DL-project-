import csv
from naive_supervised import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def save_to_csv(l):
    path = "record_N.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = l
        csv_write.writerow(data_row)


ss = [0.001]

models = [Net_GD, Net_GD_Adapt, Net_EXTRA, Net_EXTRA_Adapt,
          Net_Push_DIGing, Net_Push_DIGing_Adapt]

origin_alg = ['DGD', 'EXTRA', 'PD']

dims = [20]

n_nodes = [20, 50, 100]

connectivity = 0.3

num_of_train = 100
num_of_test = 5

# for dim in dims:

#     num_of_nodes = 50
#     num_of_edges = int(num_of_nodes*(num_of_nodes/2)*connectivity)

#     fixed_function = generate_function(num_of_nodes, dim)
#     train_set = generate_dataset(num_of_train, num_of_nodes,
#                                  num_of_edges, directed=False, dim=dim, fix_func=fixed_function)
#     test_set = generate_dataset(num_of_test, num_of_nodes, num_of_edges,
#                                 directed=False, dim=dim, fix_func=fixed_function)

#     for step_size in ss:
#         for M in models:
#             model = M(dim=dim, step_size=step_size).to(device)
#             train_model_fix_iter(model, train_set)
#             err_train = test_model_fix_iter(model, train_set, '')
#             err_test = test_model_fix_iter(model, test_set, '')
#             save_to_csv([dim, num_of_nodes, num_of_edges,
#                          type(model), step_size, err_train, err_test])

#         for ori_M in origin_alg:
#             err_train = test_origin_fix_iter(
#                 train_set, test_layers, method=ori_M, step_size=step_size)
#             err_test = test_origin_fix_iter(
#                 test_set, test_layers, method=ori_M, step_size=step_size)
#             save_to_csv(
#                 [dim, num_of_nodes, num_of_edges, ori_M, step_size, err_train, err_test])


for num_of_nodes in n_nodes:
    dim = 10
    num_of_edges = int(num_of_nodes*(num_of_nodes/2)*connectivity)

    fixed_function = generate_function(num_of_nodes, dim)
    train_set = generate_dataset(num_of_train, num_of_nodes,
                                 num_of_edges, directed=False, dim=dim, fix_func=fixed_function)
    test_set = generate_dataset(num_of_test, num_of_nodes, num_of_edges,
                                directed=False, dim=dim, fix_func=fixed_function)

    for step_size in ss:
        for M in models:
            model = M(dim=dim, step_size=step_size).to(device)
            train_model_fix_iter(model, train_set)
            err_train = test_model_fix_iter(model, train_set, '')
            err_test = test_model_fix_iter(model, test_set, '')
            save_to_csv([dim, num_of_nodes, num_of_edges,
                         type(model), step_size, err_train, err_test])

        for ori_M in origin_alg:
            err_train = test_origin_fix_iter(
                train_set, test_layers, method=ori_M, step_size=step_size)
            err_test = test_origin_fix_iter(
                test_set, test_layers, method=ori_M, step_size=step_size)
            save_to_csv(
                [dim, num_of_nodes, num_of_edges, ori_M, step_size, err_train, err_test])

