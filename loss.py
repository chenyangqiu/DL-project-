import torch

# supervised loss func
def loss_sup(x, opt):
    error = torch.norm(x-opt, 2)
    return error

def loss_unsup(x, sum_A, sum_b):
    pass