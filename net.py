import torch
from gat_conv import GATConv_Dim as GATConv
from copy import deepcopy
import optbaseline as opt


class Net_GD(torch.nn.Module):
    def __init__(self, dim, step_size):
        super(Net_GD, self).__init__()
        self.dim = dim
        self.step_size = step_size  # torch.nn.Parameter(torch.tensor([0.1]))
        self.conv = GATConv(dim)

    def forward(self, data, num_layers, terminal=0.0, recording=False, threshold=0.0):
        comm_cost = 0
        x = deepcopy(data.x)
        z = deepcopy(data.x)
        u = deepcopy(data.x)
        # v = torch.ones((data.num_nodes, 1)).to(u.device)
        z_last = deepcopy(data.x)
        # func_v = func_value(data.A, data.b, x)
        if recording:
            record = []
            err = opt.opt_distance(data.y.cpu().numpy(),
                                   z.cpu().detach().numpy())
            record.append(err)
        for i in range(num_layers):
            alpha = self.step_size/((i+1)**(0.5))
            # func_v = func_value(data.A, data.b, z)
            u, theta = self.conv(x, data.edge_index,
                          first_iter=(i == 0), threshold=threshold) # 聚合后的每个节点的状态
            comm_cost += self.conv.comm_cost  #通信量，暂时没用
            # x_last = x
            z_last = z
            z = u
            x = u - alpha*grad(data.A, data.b, z)+ theta*(z_last - z)
            #y = grad(data.A, data.b,x)
            # u = self.conv(u-alpha*y, func_v, data.edge_index,
            #              first_iter=(i == 0))
            #v = self.conv(v, func_v, data.edge_index, first_iter=False)
            #inv_v = v**(-1)
            #x = torch.mul(inv_v, u)

            if (terminal > 0):
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       z.cpu().detach().numpy())
                # err = opt.opt_distance(data.y.cpu().numpy(),
                #                        x.cpu().detach().numpy())
                if err < terminal:
                    break
            if recording:
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       z.cpu().detach().numpy())
                # err = opt.opt_distance(data.y.cpu().numpy(),
                #                        x.cpu().detach().numpy())
                record.append(err)
        return (x, comm_cost, record) if recording else (x, comm_cost)


class Net_EXTRA(torch.nn.Module):
    def __init__(self, dim, step_size):
        super(Net_EXTRA, self).__init__()
        self.dim = dim
        self.step_size = step_size  # torch.nn.Parameter(torch.tensor([0.1]))
        self.conv = GATConv(dim)

    def forward(self, data, num_layers, terminal=0.0, recording=False, threshold=0.0):
        comm_cost = 0
        x_1 = deepcopy(data.x)
        z_0 = deepcopy(data.x)
        grad_1 = grad(data.A, data.b, z_0)
        # y_1 = func_value(data.A, data.b, x_1)
        if recording:
            record = []
            err = opt.opt_distance(data.y.cpu().numpy(),
                                   z_0.cpu().detach().numpy())
            record.append(err)

        x = self.conv(z_0, data.edge_index,
                      first_iter=True, threshold=threshold)-self.step_size*grad_1
        comm_cost += self.conv.comm_cost
        v = torch.ones((data.num_nodes, 1)).to(x_1.device)
        x_0 = x_1
        x_1 = x
        if recording:
            err = opt.opt_distance(data.y.cpu().numpy(),
                                   x.cpu().detach().numpy())
            record.append(err)
        for i in range(1, num_layers):
            inv_v = v**(-1)
            z = torch.mul(inv_v, x_1)
            grad_1 = grad(data.A, data.b, z)
            grad_0 = grad(data.A, data.b, z_0)
            # W is not equal to W_hat.
            # y_1 = func_value(data.A, data.b, z)
            # y_0 = func_value(data.A, data.b, z_0)
            x = self.conv(x_1, data.edge_index, first_iter=False, threshold=threshold)+x_1-0.5*self.conv(
                x_0, data.edge_index, first_iter=False, threshold=threshold)-0.5*x_0-self.step_size*(grad_1-grad_0)
            comm_cost += self.conv.comm_cost
            v = self.conv(v, data.edge_index, first_iter=False,
                          threshold=threshold)

            z_0 = z
            x_0 = x_1
            x_1 = x
            if (terminal > 0):
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       z.cpu().detach().numpy())
                if err < terminal:
                    break
            if recording:
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       z.cpu().detach().numpy())
                record.append(err)
        return (z, comm_cost, record) if recording else (z, comm_cost)


'''
class Net_Push_Pull(torch.nn.Module):
    def __init__(self, dim):
        super(Net_Push_DIGing, self).__init__()
        self.dim = dim
        self.step_size = 0.001  # torch.nn.Parameter(torch.tensor([0.1]))
        self.conv = GATConv(dim)

    def forward(self, data, num_layers):
        y = grad(data.A, data.b, data.x)
        x = deepcopy(data.x)
        func_v = func_value(data.A, data.b, x)
        for i in range(num_layers):
            x_new = self.conv(x-self.step_size*y, func_v, data.edge_index,
                          first_iter=(i == 0))
            y = self.conv(y,func_v,data.edge_index,first_iter = (i==0)) + grad(data.A, data.b,x_new) -grad(data.A, data.b,x) 
           
            x = x_new
        return x
'''


class Net_Push_DIGing(torch.nn.Module):
    def __init__(self, dim, step_size):
        super(Net_Push_DIGing, self).__init__()
        self.dim = dim
        self.step_size = step_size  # torch.nn.Parameter(torch.tensor([0.1]))
        self.conv = GATConv(dim)

    def forward(self, data, num_layers, terminal=0.0, recording=False, threshold=0.0):
        comm_cost = 0
        y = grad(data.A, data.b, data.x)
        u = deepcopy(data.x)
        v = torch.ones((data.num_nodes, 1)).to(u.device)
        x_0 = data.x
        # func_v = func_value(data.A, data.b, x_0)
        if recording:
            record = []
            err = opt.opt_distance(data.y.cpu().numpy(),
                                   x_0.cpu().detach().numpy())
            record.append(err)
        for i in range(num_layers):
            u = self.conv(u-self.step_size*y, data.edge_index,
                          first_iter=(i == 0), threshold=threshold)
            comm_cost += self.conv.comm_cost
            v = self.conv(v, data.edge_index, first_iter=False,
                          threshold=threshold)
            inv_v = v**(-1)
            x_1 = torch.mul(inv_v, u)
            grad_1 = grad(data.A, data.b, x_1)
            grad_0 = grad(data.A, data.b, x_0)
            y = self.conv(y, data.edge_index,
                          first_iter=False, threshold=threshold)+grad_1-grad_0
            x_0 = x_1
            if (terminal > 0):
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       x_1.cpu().detach().numpy())
                if err < terminal:
                    break
            if recording:
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       x_1.cpu().detach().numpy())
                record.append(err)
        return (x_1, comm_cost, record) if recording else (x_1, comm_cost)


class Net_GD_Adapt(torch.nn.Module):
    def __init__(self, dim, step_size):
        super(Net_GD_Adapt, self).__init__()
        self.dim = dim
        self.step_size = step_size  # torch.nn.Parameter(torch.tensor([0.1]))
        self.conv = GATConv(dim)

    def forward(self, data, num_layers, terminal=0.0, recording=False, threshold=0.0):
        comm_cost = 0
        x = deepcopy(data.x)
        z = deepcopy(data.x)
        u = deepcopy(data.x)
        v = torch.ones((data.num_nodes, 1)).to(u.device)
        # func_v = func_value(data.A, data.b, x)
        if recording:
            record = []
            err = opt.opt_distance(data.y.cpu().numpy(),
                                   z.cpu().detach().numpy())
            record.append(err)
        for i in range(num_layers):
            alpha = self.step_size/((i+1)**(0.5))
            # func_v = func_value(data.A, data.b, z)
            u = self.conv(x, data.edge_index, threshold=threshold)
            comm_cost += self.conv.comm_cost
            v = self.conv(v, data.edge_index, first_iter=False,
                          threshold=threshold)
            inv_v = v**(-1)
            z = torch.mul(inv_v, u)
            x = u - alpha*grad(data.A, data.b, z)

            #y = grad(data.A, data.b,x)
            # u = self.conv(u-alpha*y, func_v, data.edge_index,
            #              first_iter=(i == 0))
            #v = self.conv(v, func_v, data.edge_index, first_iter=False)
            #inv_v = v**(-1)
            #x = torch.mul(inv_v, u)
            if (terminal > 0):
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       z.cpu().detach().numpy())
                if err < terminal:
                    break
            if recording:
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       z.cpu().detach().numpy())
                record.append(err)
        return (z, comm_cost, record) if recording else (z, comm_cost)


class Net_EXTRA_Adapt(torch.nn.Module):
    def __init__(self, dim, step_size):
        super(Net_EXTRA_Adapt, self).__init__()
        self.dim = dim
        self.step_size = step_size  # torch.nn.Parameter(torch.tensor([0.1]))
        self.conv = GATConv(dim)

    def forward(self, data, num_layers, terminal=0.0, recording=False, threshold=0.0):
        comm_cost = 0
        x_1 = deepcopy(data.x)
        z_0 = deepcopy(data.x)
        grad_1 = grad(data.A, data.b, z_0)
        # y_1 = func_value(data.A, data.b, x_1)
        if recording:
            record = []
            err = opt.opt_distance(data.y.cpu().numpy(),
                                   z_0.cpu().detach().numpy())
            record.append(err)

        x = self.conv(z_0, data.edge_index)-self.step_size*grad_1
        comm_cost += self.conv.comm_cost
        v = torch.ones((data.num_nodes, 1)).to(x_1.device)
        x_0 = x_1
        x_1 = x
        if recording:
            err = opt.opt_distance(data.y.cpu().numpy(),
                                   x.cpu().detach().numpy())
            record.append(err)
        for i in range(1, num_layers):
            inv_v = v**(-1)
            z = torch.mul(inv_v, x_1)
            grad_1 = grad(data.A, data.b, z)
            grad_0 = grad(data.A, data.b, z_0)
            # W is not equal to W_hat.
            # y_1 = func_value(data.A, data.b, z)
            # y_0 = func_value(data.A, data.b, z_0)
            x = self.conv(x_1, data.edge_index, threshold=threshold)+x_1-0.5*self.conv(
                x_0, data.edge_index, first_iter=False, threshold=threshold)-0.5*x_0-self.step_size*(grad_1-grad_0)
            comm_cost += self.conv.comm_cost
            v = self.conv(v, data.edge_index, first_iter=False,
                          threshold=threshold)

            z_0 = z
            x_0 = x_1
            x_1 = x
            if (terminal > 0):
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       z.cpu().detach().numpy())
                if err < terminal:
                    break
            if recording:
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       z.cpu().detach().numpy())
                record.append(err)
        return (z, comm_cost, record) if recording else (z, comm_cost)


class Net_Push_DIGing_Adapt(torch.nn.Module):
    def __init__(self, dim, step_size):
        super(Net_Push_DIGing_Adapt, self).__init__()
        self.dim = dim
        self.step_size = step_size  # torch.nn.Parameter(torch.tensor([0.1]))
        self.conv = GATConv(dim)

    def forward(self, data, num_layers, terminal=0.0, recording=False, threshold=0.0):
        comm_cost = 0
        y = grad(data.A, data.b, data.x)
        u = deepcopy(data.x)
        v = torch.ones((data.num_nodes, 1)).to(u.device)
        x_0 = data.x
        # func_v = func_value(data.A, data.b, x_0)
        if recording:
            record = []
            err = opt.opt_distance(data.y.cpu().numpy(),
                                   x_0.cpu().detach().numpy())
            record.append(err)
        for i in range(num_layers):
            u = self.conv(u-self.step_size*y, data.edge_index,
                          threshold=threshold)
            comm_cost += self.conv.comm_cost
            v = self.conv(v, data.edge_index, first_iter=False,
                          threshold=threshold)
            inv_v = v**(-1)
            x_1 = torch.mul(inv_v, u)
            grad_1 = grad(data.A, data.b, x_1)
            grad_0 = grad(data.A, data.b, x_0)
            y = self.conv(y, data.edge_index,
                          first_iter=False, threshold=threshold)+grad_1-grad_0
            x_0 = x_1
            if (terminal > 0):
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       x_1.cpu().detach().numpy())
                if err < terminal:
                    break
            if recording:
                err = opt.opt_distance(data.y.cpu().numpy(),
                                       x_1.cpu().detach().numpy())
                record.append(err)
        return (x_1, comm_cost, record) if recording else (x_1, comm_cost)


@torch.no_grad()
def grad(A, b, x):
    (N, D) = x.shape
    grad = 2*torch.matmul(A, x.view(N, D, 1)).squeeze()+b
    return grad


@torch.no_grad()
def func_value(A, b, x):
    (N, D) = x.shape
    y = torch.matmul(torch.matmul(x.view(N, 1, D), A), x.view(
        N, D, 1)).squeeze()+torch.mul(b, x).sum(dim=1)
    return y
