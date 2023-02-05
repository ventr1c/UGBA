from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv import MessagePassing

# This works for higher version of torch_gometric, e.g., 2.0.
# from torch_geometric.nn.dense.linear import Linear
from torch.nn import Linear


from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import remove_self_loops, add_self_loops


class MedianConv(MessagePassing):
    r"""Graph convolution with median aggregation function.
    
    Example
    -------
    >>> import torch
    >>> from median_pyg import MedianConv

    >>> edge_index = torch.as_tensor([[0, 1, 2], [2, 0, 1]])
    >>> x = torch.randn(3, 5)
    >>> conv = MedianConv(5, 2)
    >>> conv(x, edge_index)
    tensor([[-0.5138, -1.3301],
            [-0.5138,  0.1693],
            [ 0.2367, -1.3301]], grad_fn=<AddBackward0>)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 add_self_loops: bool = False,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', None)
        super(MedianConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        # This works for higher version of torch_gometric, e.g., 2.0.
        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                   weight_initializer='glorot')
        self.lin = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                               num_nodes=x.size(self.node_dim))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        x = self.lin(x)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def aggregate(self, x_j, index):
        # `to_dense_batch` requires the `index` is sorted
        # TODO: is there any way to avoid `argsort`?
        # print(index.shape)
        ix = torch.argsort(index)
        index = index[ix]
        x_j = x_j[ix]
        dense_x, mask = to_dense_batch(x_j, index)
        out = x_j.new_zeros(dense_x.size(0), dense_x.size(-1))
        deg = mask.sum(dim=1)
        for i in deg.unique():
            deg_mask = deg == i
            # print(deg_mask,deg.unique())
            out[deg_mask] = dense_x[deg_mask, :i].median(dim=1).values
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
