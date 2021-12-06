import numpy as np
import copy
import optbaseline as opt
import matplotlib.pyplot as plt


def test_origin_fix_iter(test_set, num_iters, method, step_size):
    err = 0
    err_record = np.zeros(num_iters+1,)
    num_of_test = 0
    for data in test_set:
        num_of_test += 1
        weight_matrix = stochastic_W(data.adj.numpy())
        if method == 'DGD':
            err_seq, _ = DGD(num_iters, step_size, weight_matrix,
                             data.A.numpy(), data.b.numpy(), data.y.numpy(), data.x.numpy())
        if method == 'EXTRA':
            err_seq, _ = EXTRA(num_iters, step_size, weight_matrix,
                               data.A.numpy(), data.b.numpy(), data.y.numpy(), data.x.numpy())
        if method == 'PD':
            err_seq, _ = Push_DIGing(num_iters, step_size, weight_matrix,
                                     data.A.numpy(), data.b.numpy(), data.y.numpy(), data.x.numpy())
        err += err_seq[-1]
        err_record = err_record+np.array(err_seq)
    err_record = err_record/num_of_test
    plt.plot(err_record, label='origin '+method)
    err = err/num_of_test
    print('origin', method, 'err:', err)
    return err


def test_origin_target(test_set, test_target_ratio, method, step_size, max_iter=1000):
    cost = 0
    num_of_test = 0
    for data in test_set:
        num_of_test += 1
        init_err = opt.opt_distance(data.y.cpu().numpy(),
                                    data.x.cpu().detach().numpy())
        test_target = init_err*test_target_ratio
        weight_matrix = stochastic_W(data.adj.numpy())
        cost_per_iter = np.nonzero(weight_matrix)[0].size*data.x.shape[1]
        if method == 'DGD':
            err_seq, k = DGD(max_iter, step_size, weight_matrix,
                             data.A.numpy(), data.b.numpy(), data.y.numpy(), data.x.numpy(), terminal=test_target)
        if method == 'EXTRA':
            err_seq, k = EXTRA(max_iter, step_size, weight_matrix,
                               data.A.numpy(), data.b.numpy(), data.y.numpy(), data.x.numpy(), terminal=test_target)
        if method == 'PD':
            err_seq, k = Push_DIGing(max_iter, step_size, weight_matrix,
                                     data.A.numpy(), data.b.numpy(), data.y.numpy(), data.x.numpy(), terminal=test_target)
        if err_seq[-1] > test_target:
            cost += float('inf')
        else:
            cost += cost_per_iter*k
        print(err_seq[-1])
    cost = cost/num_of_test
    print('origin', method, 'ave. cost:', cost)
    return cost

# def test_push_pull(test_set, num_iters):
#     err = 0
#     step_size = 0.001
#     for data in test_set:
#         weight_C = stochastic_W(data.adj.numpy())
#         weight_R = stochastic_R(data.adj.numpy())
#         err_seq = Push_Pull(num_iters, step_size, weight_C, weight_R,
#                             data.A.numpy(), data.b.numpy(), data.y.numpy(), data.x.numpy())
#         err += err_seq[-1]
#     err = err/len(test_set)
#     print('test push_diging err:', err)
#     return err


def DGD(max_iteration, step_size, weight_matrix, A, b, opt_var, init_x, terminal=0.0):
    (num_of_nodes, dim) = init_x.shape
    x_seq = copy.deepcopy(init_x)
    u_seq = copy.deepcopy(init_x)
    v_seq = np.ones(num_of_nodes)

    error_seq = [opt.opt_distance(opt_var, x_seq)]
    for k in range(1, max_iteration+1):
        alpha = step_size/(k**(0.5))
        u_seq = weight_matrix@x_seq
        v_seq = weight_matrix@v_seq
        V = np.diag(v_seq)
        z_seq = np.linalg.inv(V)@u_seq
        x_seq = u_seq - alpha*grad_qp(A, b, z_seq)

        error = opt.opt_distance(opt_var, z_seq)
        error_seq.append(error)
        if (error < terminal):
            break
        # print(error)
    # print(x_seq)
    return error_seq, k


def EXTRA(max_iteration, step_size, weight_matrix, A, b, optimal_solution, init_x, terminal=0.0):
    # EXTRA parameters generating
    (num_of_nodes, dim) = init_x.shape
    W = weight_matrix
    W_hat = (W+np.identity(num_of_nodes))/2

    # initialization
    k = 1
    x_0 = copy.deepcopy(init_x)
    error_seq = [opt.opt_distance(optimal_solution, x_0)]

    z_0 = copy.deepcopy(init_x)
    x_1 = W@x_0 - step_size * grad_qp(A, b, z_0)
    v_seq = W@np.ones(num_of_nodes)

    error = opt.opt_distance(optimal_solution, x_1)
    error_seq.append(error)

    while (k < max_iteration):
        V = np.diag(v_seq)
        z_seq = np.linalg.inv(V)@x_1
        x_seq = (np.identity(num_of_nodes)+W)@x_1-W_hat@x_0 - \
            step_size*(grad_qp(A, b, z_seq)-grad_qp(A, b, z_0))
        v_seq = W@v_seq

        z_0 = copy.deepcopy(z_seq)
        x_0 = copy.deepcopy(x_1)
        x_1 = copy.deepcopy(x_seq)

        error = opt.opt_distance(optimal_solution, z_seq)
        error_seq.append(error)
        # print(error)

        k += 1
        if (error < terminal):
            break
    return error_seq, k


def Push_DIGing(max_iteration, step_size, weight_matrix, A, b, opt_var, init_x, terminal=0.0):
    num_of_nodes, dim = init_x.shape
    x_seq = copy.deepcopy(init_x)
    y_seq = grad_qp(A, b, x_seq)
    u_seq = copy.deepcopy(init_x)
    v_seq = np.ones(num_of_nodes)
    V = np.diag(v_seq)
    # initializition
    k = 0
    error_seq = [opt.opt_distance(opt_var, x_seq)]
    while k < max_iteration:
        u_seq = weight_matrix@(u_seq - step_size*y_seq)
        v_seq = weight_matrix@v_seq
        V = np.diag(v_seq)
        x_new = np.linalg.inv(V)@u_seq
        grad_1 = grad_qp(A, b, x_new)
        grad_0 = grad_qp(A, b, x_seq)
        y_seq = weight_matrix@y_seq + grad_1 - grad_0

        x_seq = copy.deepcopy(x_new)

        error = opt.opt_distance(opt_var, x_seq)
        error_seq.append(error)
        #print(np.sum(y_seq,axis = 0),np.sum(grad_1,axis = 0))
        k += 1
        if (error < terminal):
            break
    return error_seq, k


def Push_Pull(max_iteration, step_size, weight_C, weight_R, A, b, opt_var, init_x):
    #print('Running with the Push Pull Algorithm...')
    num_of_nodes, dim = init_x.shape
    x_seq = copy.deepcopy(init_x)
    y_seq = grad_qp(A, b, x_seq)

    # initializition
    k = 0
    error_seq = [opt.opt_distance(opt_var, x_seq)]

    while k < max_iteration:

        x_new = weight_R@(x_seq - step_size*y_seq)
        y_seq = weight_C@y_seq + grad_qp(A, b, x_new) - grad_qp(A, b, x_seq)
        x_seq = copy.deepcopy(x_new)

        error = opt.opt_distance(opt_var, x_seq)
        error_seq.append(error)

        k += 1
    return error_seq


def metro_generate(num_of_nodes, adjacency_matrix, L):
    metropolis = np.zeros((num_of_nodes, num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if adjacency_matrix[i, j] == 1:
                metropolis[i, j] = 1/(1+max(L[i, i], L[j, j]))
        metropolis[i, i] = 1-np.sum(metropolis[i, :])
    return metropolis


def stochastic_W(adjacency_matrix):
    N, _ = adjacency_matrix.shape
    W = np.zeros((N, N))
    for i in range(N):
        W[i, i] = np.random.random()
        for j in range(N):
            if adjacency_matrix[i, j] == 1:
                W[j, i] = np.random.random()
        W[:, i] /= np.sum(W[:, i])
    return W


def stochastic_R(adjacency_matrix):
    N, _ = adjacency_matrix.shape
    W = np.zeros((N, N))
    for i in range(N):
        W[i, i] = np.random.random()
        for j in range(i+1, N):
            if adjacency_matrix[i, j] == 1:
                W[i, j] = np.random.random()
                W[j, i] = np.random.random()
        W[i, :] /= np.sum(W[i, :])
    return W


def grad_qp(A, b, x):
    grad_A = np.zeros(x.shape)
    for i in range(x.shape[0]):
        grad_A[i] = 2*x[i]@A[i]+b[i]
    return grad_A


def grad_logistic(M, y, x_seq):
    num_of_nodes, dim = x_seq.shape
    gradient = np.zeros((num_of_nodes, dim))
    for i in range(num_of_nodes):
        temp = np.multiply(y[i], M[i]@x_seq[i])
        temp = 1 - sigmoid(temp)
        temp = np.multiply(-y[i], temp)
        gradient[i] = M[i].T@temp
    return gradient


def sigmoid(x):
    return 1/(1+np.exp(-x))
