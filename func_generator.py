### Function f(x) = \sum_{i=1}^N x^TA_ix+b_i^Tx, where N is the number of nodes.
### Centralized form: let $sum_A = \sum_{i=1}^N A_i$ and $sum_b = \sum_{i=1}^N b_i$.

import numpy as np

def problem_parameter(min_eig,num_of_nodes,dim):
    ''' Given a minimal eigenvalue, generate (num_of_nodes) positive definite (symmetric) matrices A_i
        return:
                A_i: size = (dim, dim)
                sum_A : size = (dim, dim)
    '''
    A = np.zeros((num_of_nodes,dim,dim))
    b = np.random.rand(num_of_nodes,dim)*min_eig
    sum_b = np.zeros(dim)
    sum_A = np.zeros((dim,dim))
    for i in range(num_of_nodes):
        #QR分解
        Q,_ = np.linalg.qr(np.random.rand(dim,dim))
        diag_elem = np.random.rand(dim)+min_eig
        A[i] = Q.T@np.diag(diag_elem)@Q

        sum_A += A[i]
        sum_b += b[i]

    return A,b, sum_A,sum_b