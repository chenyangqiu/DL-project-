import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax

class GATConv(MessagePassing):
    def __init__(self, dim, heads = 3):  # head 是什么
        super(GATConv, self).__init__(aggr='add', node_dim=0)
        self.dim = dim   # 相当于 in_channels
        self.head = heads
        self.lin = torch.nn.Linear(dim, heads*dim)   # lin被设为了线性变换函数
        # torch.nn.Linear(in_features, out_features, bias=True)
        # 对输入数据进行线性变换, y=xA^T+b
        # in_features: 输入数据的大小。out_features: 输出数据的大小. bias: 是否添加一个可学习的 bias，即上式中的 b
        # 该线性变换，只对输入的 tensor 的最后一维进行：

        self.a = torch.nn.Parameter(torch.Tensor(1, heads, dim))  # 参数初始化，设定a为可训练的参数
        self.b = torch.nn.Parameter(torch.Tensor(heads, 1))       # 参数初始化
        self.first_iter = True   # ???????????????这是在干嘛
        glorot(self.a)
        glorot(self.b)

    def forward(self, x, edge_index, y=None, first_iter = True, threshold = 0.0):
        self.first_iter = first_iter   # ??????????
        self.threshold = threshold
        out = self.propagate(edge_index, x=x, y=y, size=None, node_dim=-1)
        # propagate 函数负责图的信息传递，图神经网络主要是靠图卷积操作来完成的。
        # 图卷积操作是一种将目标节点周围邻居节点的信息进行聚合的一种方法， 具体公式参考 https://zhuanlan.zhihu.com/p/113862170 或https://blog.csdn.net/qq_38729452/article/details/122023844
        # 大致分为三步：
        # A 邻居信息的传播、接受过程，通常都是由权重矩阵进行带bias的线性变换 y=xA^T+b. 对应的函数是self.message
        # B 邻居信息聚合， 原来的代码中并没有这个步骤。 对应的函数是self.aggregate
        # C 聚合后的更新， 这个也没有。self.update

        return out.mean(dim=1)

    def message(self, x_i, x_j, y_i, y_j, index, ptr, size_i):
        if self.first_iter:
            hx_i = self.lin(x_i).view(-1, self.heads, self.dim)  # x_i: torch.Size([72, 10]). lin(x_i):torch.Size([72, 30]). hx_i torch.Size([72, 3, 10])
            # .view() 是reshape函数，-1表示不知道这个的维度，但是确定后面两个的维度是heads和dim
            hx_j = self.lin(x_j).view(-1, self.heads, self.dim)

            att_i = (hx_i*self.a).sum(dim=-1) # dim=0理解为纵向压缩
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
        self.first_iter = True
        glorot(self.a)
        glorot(self.b)

    def forward(self, x, edge_index, y=None, first_iter=True, threshold=0.0):
        self.first_iter = first_iter
        self.threshold = threshold
        out = self.propagate(edge_index, x=x, y=y, size=None, node_dim=-1)
        return out.mean(dim=1)

    def message(self, x_i, x_j, y_i, y_j, index, ptr, size_i):
        if self.first_iter:
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
        return x_j.unsqueeze(1)*self.att


# unit test code
if __name__ == '__main__':
    conv = GATConv_Dim(5)
    from dataset import generate_graph
    data = generate_graph(50, 200)
    data.x = torch.randn((50, 5))
    data.y = torch.randn((50, 1))
    conv(data.x, data.edge_index, threshold=0.5)


