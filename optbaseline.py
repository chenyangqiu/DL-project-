import cvxpy as cp 
import numpy as np
import math


def opt_distance(opt,x):
    error = 0
    num_of_nodes = x.shape[0]
    for i in range(num_of_nodes):
        #print(x[i],opt)
        error += cp.norm(x[i]-opt,2)*2
    return math.sqrt(error.value/num_of_nodes/num_of_nodes)
    
def consensus_error(H,x_bar):
    error = cp.norm(H.T@x_bar,2)
    return error.value

def opt_solver(A,b,dim):
    x = cp.Variable(dim)

    ###regular QP
    #prob=cp.Problem(cp.Minimize(cp.quad_form(x,A)+b.T@x))
    ####QP+norm1  
    prob = cp.Problem(cp.Minimize(cp.quad_form(x,A)+b.T@x))
    prob.solve()

    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)

    return x.value,prob.value