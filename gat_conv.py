import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax


class GATConv(MessagePassing):
    def __init__(self, dim, heads=3):
        super(GATConv, self).__init__(aggr='add', node_dim=0)
        self.dim = dim
        self.heads = heads
        self.lin = torch.nn.Linear(dim, heads*dim)
        self.a = torch.nn.Parameter(torch.Tensor(1, heads, dim))
        self.b = torch.nn.Parameter(torch.Tensor(heads, 1))
        self.first_iter = True    # ？？？？？？？？？？
        glorot(self.a)
        glorot(self.b)

    def forward(self, x, edge_index, y=None, first_iter=True, threshold=0.0):
        self.first_iter = first_iter
        self.threshold = threshold
        out = self.propagate(edge_index, x=x, y=y, size=None, node_dim=-1)
        return out.mean(dim=1)

    def message(self, x_i, x_j, y_i, y_j, index, ptr, size_i):
        if self.first_iter:
            hx_i = self.lin(x_i).view(-1, self.heads, self.dim)  # .view() 是reshape函数，-1表示不知道这个的维度，但是确定后面两个的维度是heads和dim。
            hx_j = self.lin(x_j).view(-1, self.heads, self.dim)
            att_i = (hx_i*self.a).sum(dim=-1)
            att_j = (hx_j*self.a).sum(dim=-1)
            att_consensus = torch.matmul(torch.norm(hx_i-hx_j, dim=-1), self.b)
            # att_func_value = (torch.norm(y_i-y_j)*self.b[:, 1]).sum(dim=-1)
            att = att_i+att_j+att_consensus  # +att_func_value
            att = F.leaky_relu(att, 0.2)
            att = softmax(att, index, ptr, size_i)
            if self.threshold > 0.0:
                att[att < self.threshold] = 0
            self.att = att
            self.comm_cost = torch.nonzero(att, as_tuple=False).size(0)
        return x_j.unsqueeze(1)*self.att.unsqueeze(-1)


class GATConv_Dim(MessagePassing):
    def __init__(self, dim, heads=3):
        super(GATConv_Dim, self).__init__(aggr='add', node_dim=0)
        self.dim = dim
        self.heads = heads
        self.lin = torch.nn.Linear(dim, heads*dim)
        self.a = torch.nn.Parameter(torch.Tensor(1, heads, dim))
        self.b = torch.nn.Parameter(torch.Tensor(heads, 1))
        self.theta = torch.nn.Parameter(torch.Tensor(1, dim))
        self.first_iter = True
        glorot(self.a)
        glorot(self.b)
        glorot(self.theta)

    def forward(self, x, edge_index, y=None, first_iter=True, threshold=0.0):
        self.first_iter = first_iter
        self.threshold = threshold
        out, theta = self.propagate(edge_index, x=x, y=y, size=None, node_dim=-1)
        return out.mean(dim=1), theta

    def message(self, x_i, x_j, y_i, y_j, index, ptr, size_i):
        # self.first_iter = True
        if self.first_iter:
            # print(x_i.size())
            hx_i = self.lin(x_i).view(-1, self.heads, self.dim)
            hx_j = self.lin(x_j).view(-1, self.heads, self.dim)
            att_i = (hx_i*self.a)
            att_j = (hx_j*self.a)
            att_consensus = torch.matmul(torch.norm(
                hx_i-hx_j, dim=-1), self.b).unsqueeze(-1)
            # att_func_value = (torch.norm(y_i-y_j)*self.b[:, 1]).sum(dim=-1)
            att = att_i+att_j+att_consensus  # +att_func_value
            att = F.leaky_relu(att, 0.2)
            att = softmax(att, index, ptr, size_i)
            if self.threshold > 0.0:
                att[-size_i:, :, :] += self.threshold
                att[att < self.threshold] = 0
                att = att*(size_i/torch.sum(att, dim=0))
            self.att = att
            self.comm_cost = torch.nonzero(att, as_tuple=False).size(0)
        return x_j.unsqueeze(1)*self.att, self.theta


# unit test code
if __name__ == '__main__':
    conv = GATConv_Dim(5)
    from dataset import generate_graph
    data = generate_graph(50, 200)
    data.x = torch.randn((50, 5))
    data.y = torch.randn((50, 1))
    conv(data.x, data.edge_index, threshold=0.5)
