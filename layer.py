import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils.degree import maybe_num_nodes
from math import pi as PI

def B_degree(index, num_nodes, weight, dtype):
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    one = one * weight
    return out.scatter_add_(0, index, one)

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start: float = 0.0, stop: float = 5.0,
                 num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class Scalar_Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, B=False):
        super(Scalar_Conv, self).__init__(aggr="add", node_dim=0)
        self.s_to_s = nn.Linear(2 * in_channels, out_channels)
        self.v_to_s = nn.Linear(2 * in_channels + 50, out_channels)
        self.scalar_linear = nn.Linear(2 * out_channels, out_channels)
        self.activation = nn.SiLU()
        self.rbf = GaussianSmearing()
        self.B = B
        self.out_channels = out_channels

    def forward(self, scalar, vector, position, edge_index, edge_attr):
        if self.B: 
            row, col = edge_index
            deg = B_degree(col, scalar.size(0), edge_attr, dtype=scalar.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0     
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_attr
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=scalar.size(0))   
            row, col = edge_index
            deg = degree(col, scalar.size(0), dtype=scalar.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, scalar=scalar, vector=vector, position=position, norm=norm)

    def message(self, scalar_i, scalar_j, vector_i, vector_j, position_i, position_j, norm):
        # make s to s feature
        s_to_s = (torch.cat([scalar_i, scalar_j], dim=1))
        s_to_s = self.activation(self.s_to_s(s_to_s))

        # get relative postion vector
        position = position_i - position_j

        # get relative dist
        dist = (position).pow(2).sum(dim=-1).sqrt()

        # rbf
        rbf = self.rbf(dist)

        # concat norm of vector features and rbf
        v_to_s = torch.cat([torch.linalg.norm(vector_i, dim=1), torch.linalg.norm(vector_j, dim=1), rbf], dim=-1)

        # make s to v feature
        v_to_s = self.activation(self.v_to_s(v_to_s))

        # concat s_to_s and v_to_s
        scalar_feature = torch.cat([(s_to_s), (v_to_s)], dim=1)
        
        return norm.view(-1, 1) * self.scalar_linear(scalar_feature)


class Vector_Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, B=False):
        super(Vector_Conv, self).__init__(aggr="add", node_dim=0)
        self.s_to_v = nn.Linear(2 * in_channels, out_channels)
        self.v_to_v = nn.Linear(2 * in_channels, out_channels, bias=False)
        self.vector_linear = nn.Linear(2 * out_channels, out_channels, bias=False)
        self.activation = nn.Tanh()
        self.B = B
        self.out_channels = out_channels

    def forward(self, scalar, vector, position, edge_index, edge_attr):
        if self.B: 
            row, col = edge_index
            deg = B_degree(col, scalar.size(0), edge_attr, dtype=scalar.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_attr
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=scalar.size(0))   
            row, col = edge_index
            deg = degree(col, scalar.size(0), dtype=scalar.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 
        return self.propagate(edge_index, scalar=scalar, vector=vector, position=position, norm=norm)

    def message(self, scalar_i, scalar_j, vector_i, vector_j, position_i, position_j, norm):
        # make v to v feature
        v_to_v = torch.cat([vector_i, vector_j], dim=-1)
        v_to_v = self.v_to_v(v_to_v)

        # get relative postion vector
        position = position_i - position_j
        position = position[:, :, None]
        position = position.expand(-1, -1, self.out_channels)

        s_to_v = torch.cat([scalar_i, scalar_j], dim=1)
        s_to_v = self.activation(self.s_to_v(s_to_v))
        s_to_v = s_to_v[:, None, :]
        s_to_v = s_to_v.expand(-1, 3, -1)
        s_to_v = torch.mul(s_to_v, position)

        # double float
        vector_feature = torch.cat([(v_to_v), (s_to_v)], dim=-1)
        vector_feature = self.vector_linear(vector_feature)
        norm = norm[:, None, None]
        norm = norm.expand(-1, 3, self.out_channels)
        return torch.mul(norm, vector_feature)


class MolNet_Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, B=False):
        super(MolNet_Layer, self).__init__()
        self.scalar_conv = Scalar_Conv(in_channels, out_channels, B)
        self.vector_conv = Vector_Conv(in_channels, out_channels, B)
        self.silu = torch.nn.SiLU()

    def forward(self, scalar, vector, position, edge_index, edge_attr):
        scalar_feature = self.silu(self.scalar_conv(scalar, vector, position, edge_index, edge_attr)) + scalar
        vector_feature = self.vector_conv(scalar, vector, position, edge_index, edge_attr) + vector
        return scalar_feature, vector_feature
