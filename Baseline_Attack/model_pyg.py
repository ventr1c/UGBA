
from math import degrees
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_scatter.scatter import scatter_add
import warnings
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv,GATv2Conv
from torch_sparse.tensor import SparseTensor

from copy import deepcopy
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True):
        super(MLP, self).__init__()
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.convs = torch.nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x, adj_t):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x.log_softmax(dim=-1)

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, weight_decay = 5e-4, lr=0.01, device = None, layer_norm_first=False, use_ln=False):
        super(GCN, self).__init__()
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = device
        self.lr = lr

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x, adj_t, layers=-1):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            print(1,x,adj_t)
            x = conv(x, adj_t)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # obtain output from the i-th layer
            if layers == i+1:
                return x
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

    def con_forward(self,x,adj_t,layers=-1):
        if self.layer_norm_first and layers==1:
            x = self.lns[0](x)
        for i in range(layers-1,len(self.convs)-1):
            x = self.convs[i](x, adj_t)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    def fit(self, features, adj_t, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """
        self.adj_t = adj_t
        # self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
        torch.cuda.empty_cache()

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_t)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_t)
        self.output = output
        torch.cuda.empty_cache()

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_t)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output = self.forward(self.features, self.adj_t)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        torch.cuda.empty_cache()


    def test(self, features, adj_t, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, adj_t)
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)

from torch_geometric.nn import SGConv

class SGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True):
        super(SGCN, self).__init__()
        self.conv = SGConv(in_channels, out_channels, K=num_layers, cached=False)
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        self.dropout = dropout
        self.layer_norm_first = layer_norm_first

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, adj_t):
        if self.layer_norm_first:
            x = self.lns[0](x)
        x= self.conv(x,adj_t)
        return x.log_softmax(dim=-1)

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True):
        super(SAGE, self).__init__()
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()


    def forward(self, x, adj_t):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True, heads=8, att_dropout=0):
        super(GAT, self).__init__()
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels//heads, heads=heads, dropout=att_dropout))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads=heads, dropout=att_dropout))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, dropout=att_dropout))
        
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x, adj_t):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class RGAT(nn.Module):
    """
    Robust GAT inspired by GNNGuard
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True, threshold=0.1, heads=1, att_dropout=0.6, att_cpu=False):
        super(RGAT, self).__init__()
        self.layer_norm_first = layer_norm_first
        if use_ln==False:
            warnings.warn("RGAT has to be accompanied with LN inside")
        self.use_ln = True
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGATConv(in_channels, hidden_channels//heads, heads=heads, threshold=threshold, dropout=att_dropout, att_cpu=att_cpu))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(RGATConv(hidden_channels, hidden_channels//heads, heads=heads, threshold=threshold, dropout=att_dropout, att_cpu=att_cpu))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(RGATConv(hidden_channels, out_channels, dropout=att_dropout, att_cpu=att_cpu))

        self.dropout = dropout
        self.threshold = threshold

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x, adj_t):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, PairTensor,Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_sparse import SparseTensor, set_diag

class RGATConv(GATConv):
    
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, threshold=0.1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0., att_cpu=False,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(RGATConv, self).__init__(in_channels, out_channels, heads,
        concat, negative_slope, dropout, add_self_loops, bias, **kwargs)
        self.threshold = threshold
        self.att_cpu = att_cpu
        # print(self.__dict__)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        # alpha_l: OptTensor = None
        # alpha_r: OptTensor = None

        raw_x = x
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `RGATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            # alpha_l = (x_l * self.att_l).sum(dim=-1)
            # alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `RGATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            # alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                # alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        # assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             raw_x=raw_x,size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, 
                raw_x_i: OptTensor, raw_x_j: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # print(raw_x_i.size(),raw_x_j.size())
        # with torch.no_grad():
        #     alpha_sim = F.cosine_similarity(raw_x_i, raw_x_j,dim=-1).unsqueeze(1)
        #     # alpha_sim[alpha_sim>0.1] = 1.0
        #     alpha_sim[alpha_sim<0.1] = 0
        if self.att_cpu:
            print("move vars to cpu")
            device = raw_x_i.device
            raw_x_i = raw_x_i.cpu()
            raw_x_j = raw_x_j.cpu()
        if raw_x_i.size(1)<=500:
            alpha = F.cosine_similarity(raw_x_i, raw_x_j,dim=-1).unsqueeze(1)
            alpha[alpha<self.threshold] = 1e-6
        else:
            alpha = F.cosine_similarity(x_i.squeeze(1), x_j.squeeze(1),dim=-1).unsqueeze(1)
            alpha[alpha<0.5] = 1e-6
        # att = alpha_j if alpha_i is None else alpha_j + alpha_i
        
        alpha = softmax(alpha.log(), index, ptr, size_i)
        # alpha[alpha<0.5] -= 0.2
        # alpha = F.leaky_relu(alpha,self.negative_slope)
        # alpha = alpha_sim * alpha
        # alpha = softmax(alpha, index, ptr, size_i)
        if self.att_cpu:
            # device = raw_x_i.device
            raw_x_i = raw_x_i.to(device)
            raw_x_j = raw_x_j.to(device)
            alpha = alpha.to(device)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    
"""Torch module for RobustGCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


def GCNAdjNorm(adj, order=-0.5):
    adj = sp.eye(adj.shape[0]) + adj
    # for i in range(len(adj.data)):
    #     if adj.data[i] > 0 and adj.data[i] != 1:
    #         adj.data[i] = 1
    adj.data[np.where((adj.data > 0) * (adj.data == 1))[0]] = 1
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, order).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return adj

# PyG graph normalize
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
def gcn_norm(adj_t, order=-0.5, add_self_loops=True):
    
    # if not adj_t.has_value():
    #     adj_t = adj_t.fill_value(1., dtype=None)
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(order)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

from torch.distributions.multivariate_normal import MultivariateNormal
class RobustGCN(nn.Module):
    r"""
    Description
    -----------
    Robust Graph Convolutional Networks (`RobustGCN <http://pengcui.thumedialab.com/papers/RGCN.pdf>`__)
    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    dropout : bool, optional
        Whether to dropout during training. Default: ``True``.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
    # def __init__(self, in_features, out_features, hidden_features, dropout=True):
        super(RobustGCN, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels

        self.act0 = F.elu
        self.act1 = F.relu

        self.layers = nn.ModuleList()
        self.layers.append(RobustGCNConv(in_channels, hidden_channels, act0=self.act0, act1=self.act1,
                                         initial=True, dropout=dropout))
        for i in range(num_layers - 2):
            self.layers.append(RobustGCNConv(hidden_channels, hidden_channels,
                                             act0=self.act0, act1=self.act1, dropout=dropout))
        self.layers.append(RobustGCNConv(hidden_channels, out_channels, act0=self.act0, act1=self.act1))
        self.dropout = dropout
        self.use_ln = True
        self.gaussian = None
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        r"""
        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : list of torch.SparseTensor
            List of sparse tensor of adjacency matrix.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.
        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).
        """

        adj0, adj1 = gcn_norm(adj), gcn_norm(adj, order=-1.0)
        # adj0, adj1 = normalize_adj(adj), normalize_adj(adj, -1.0)
        mean = x
        var = x
        for layer in self.layers:
            mean, var = layer(mean, var=var, adj0=adj0, adj1=adj1)
        # if self.gaussian == None:
        # self.gaussian = MultivariateNormal(torch.zeros(var.shape),
        #         torch.diag_embed(torch.ones(var.shape)))
        sample = torch.randn(var.shape).to(x.device)
        # sample = self.gaussian.sample().to(x.device)
        output = mean + sample * torch.pow(var, 0.5)

        return output.log_softmax(dim=-1)


class RobustGCNConv(nn.Module):
    r"""
    Description
    -----------
    RobustGCN convolutional layer.
    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    act0 : func of torch.nn.functional, optional
        Activation function. Default: ``F.elu``.
    act1 : func of torch.nn.functional, optional
        Activation function. Default: ``F.relu``.
    initial : bool, optional
        Whether to initialize variance.
    dropout : bool, optional
        Whether to dropout during training. Default: ``False``.
    """

    def __init__(self, in_features, out_features, act0=F.elu, act1=F.relu, initial=False, dropout=0.5):
        super(RobustGCNConv, self).__init__()
        self.mean_conv = nn.Linear(in_features, out_features)
        self.var_conv = nn.Linear(in_features, out_features)
        self.act0 = act0
        self.act1 = act1
        self.initial = initial
        self.dropout = dropout

    def reset_parameters(self):
        self.mean_conv.reset_parameters()
        self.var_conv.reset_parameters()
    
    def forward(self, mean, var=None, adj0=None, adj1=None):
        r"""
        Parameters
        ----------
        mean : torch.Tensor
            Tensor of mean of input features.
        var : torch.Tensor, optional
            Tensor of variance of input features. Default: ``None``.
        adj0 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 0. Default: ``None``.
        adj1 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 1. Default: ``None``.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.
        Returns
        -------
        """
        if self.initial:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var= mean
            mean = self.mean_conv(mean)
            var = self.var_conv(var)
            mean = self.act0(mean)
            var = self.act1(var)
        else:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var= F.dropout(var, p=self.dropout, training=self.training)
            mean = self.mean_conv(mean)
            var = self.var_conv(var)
            mean = self.act0(mean)
            var = self.act1(var)+1e-6 #avoid abnormal gradient
            attention = torch.exp(-var)

            mean = mean * attention
            var = var * attention * attention
            mean = adj0 @ mean
            var = adj1 @ var

        return mean, var

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix
import scipy.sparse as sp
import numpy as np
import torch_geometric.utils as utils

class EGCNGuard(nn.Module):
    """
    Efficient GCNGuard

    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True, attention_drop=True, threshold=0.1):
        super(EGCNGuard, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=False))

        self.dropout = dropout
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        # specific designs from GNNGuard
        self.attention_drop = attention_drop
        # the definition of p0 is confusing comparing the paper and the issue
        # self.p0 = p0
        # https://github.com/mims-harvard/GNNGuard/issues/4
        self.gate = 0. #Parameter(torch.rand(1)) 
        self.prune_edge = True
        self.threshold = threshold

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        

    def forward(self, x, adj):
        if self.layer_norm_first:
            x = self.lns[0](x)
        new_adj = adj
        for i, conv in enumerate(self.convs[:-1]):
            new_adj = self.att_coef(x, new_adj)
            x = conv(x, new_adj)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        new_adj = self.att_coef(x, new_adj)
        x = conv(x, new_adj)
        return x.log_softmax(dim=-1)


    def att_coef(self, features, adj):
        with torch.no_grad():
            row, col = adj.coo()[:2]
            n_total = features.size(0)
            if features.size(1) > 512 or row.size(0)>5e5:
                # an alternative solution to calculate cosine_sim
                # feat_norm = F.normalize(features,p=2)
                batch_size = int(1e8//features.size(1))
                bepoch = row.size(0)//batch_size+(row.size(0)%batch_size>0)
                sims = []
                for i in range(bepoch):
                    st = i*batch_size
                    ed = min((i+1)*batch_size,row.size(0))
                    sims.append(F.cosine_similarity(features[row[st:ed]],features[col[st:ed]]))
                sims = torch.cat(sims,dim=0)
                # sims = [F.cosine_similarity(features[u.item()].unsqueeze(0), features[v.item()].unsqueeze(0)).item() for (u, v) in zip(row, col)]
                # sims = torch.FloatTensor(sims).to(features.device)
            else:
                sims = F.cosine_similarity(features[row],features[col])
            mask = torch.logical_or(sims>=self.threshold,row==col)
            row = row[mask]
            col = col[mask]
            sims = sims[mask]
            has_self_loop = (row==col).sum().item()
            if has_self_loop:
                sims[row==col] = 0

            # normalize sims
            deg = scatter_add(sims, row, dim=0, dim_size=n_total)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            sims = deg_inv_sqrt[row] * sims * deg_inv_sqrt[col]

            # add self-loops
            deg_new = scatter_add(torch.ones(sims.size(),device=sims.device), col, dim=0, dim_size=n_total)+1
            deg_inv_sqrt_new = deg_new.float().pow_(-1.0)
            deg_inv_sqrt_new.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            
            if has_self_loop==0:
                new_idx = torch.arange(n_total,device=row.device)
                row = torch.cat((row,new_idx),dim=0)
                col = torch.cat((col,new_idx),dim=0)
                sims = torch.cat((sims,deg_inv_sqrt_new),dim=0)
            elif has_self_loop < n_total:
                print(f"add {n_total-has_self_loop} remaining self-loops")
                new_idx = torch.ones(n_total,device=row.device).bool()
                new_idx[row[row==col]] = False
                new_idx = torch.nonzero(new_idx,as_tuple=True)[0]
                row = torch.cat((row,new_idx),dim=0)
                col = torch.cat((col,new_idx),dim=0)
                sims = torch.cat((sims,deg_inv_sqrt_new[new_idx]),dim=0)
                sims[row==col]=deg_inv_sqrt_new
            else:
                # print(has_self_loop)
                # print((row==col).sum())
                # print(deg_inv_sqrt_new.size())
                sims[row==col]=deg_inv_sqrt_new
            sims = sims.exp()
            graph_size = torch.Size((n_total,n_total))
            new_adj = SparseTensor(row=row,col=col,value=sims,sparse_sizes=graph_size)
        return new_adj



class GCNGuard(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True, attention_drop=True):
        super(GCNGuard, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=False))

        self.dropout = dropout
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        # specific designs from GNNGuard
        self.attention_drop = attention_drop
        # the definition of p0 is confusing regarding the paper and the issue
        # self.p0 = p0
        # https://github.com/mims-harvard/GNNGuard/issues/4
        self.gate = 0. #Parameter(torch.rand(1)) 
        if self.attention_drop:
            self.drop_learn = torch.nn.Linear(2, 1)
        self.prune_edge = True

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.attention_drop:
            self.drop_learn.reset_parameters()
        # self.gate.weight = 0.0

    def forward(self, x, adj):
        if self.layer_norm_first:
            x = self.lns[0](x)
        adj_memory = None
        for i, conv in enumerate(self.convs[:-1]):
            # print(f"{i} {sum(sum(torch.isnan(x)))} {x.mean()}")
            if self.prune_edge:
                # old_edge_size = adj.coo()[0].size(0)
                new_adj = self.att_coef(x, adj)
                if adj_memory != None and self.gate > 0:
                    # adj_memory makes the performance even worse
                    adj_memory = self.gate * adj_memory.to_dense() + (1 - self.gate) * new_adj.to_dense()
                    row, col = adj_memory.nonzero()[:2]
                    adj_values = adj_memory[row,col]
                else:
                    adj_memory = new_adj
                    row, col, adj_values = adj_memory.coo()[:3]
                # adj_values[torch.isnan(adj_values)] = 0.0
                edge_index = torch.stack((row, col), dim=0)
                # print(f"{sum(torch.isnan(adj_values))} {adj_values.mean()}")
                # adj_values = adj_memory[row, col]
                # print(edge_index,adj_values)
                # print(f"Pruned edges: {i} {old_edge_size-adj.coo()[0].size(0)}")
                adj = new_adj
            x = conv(x, edge_index, edge_weight=adj_values)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.prune_edge:
            # old_edge_size = adj.coo()[0].size(0)
            new_adj = self.att_coef(x, adj)
            if adj_memory != None and self.gate > 0:
                # adj_memory makes the performance even worse
                adj_memory = self.gate * adj_memory.to_dense() + (1 - self.gate) * new_adj.to_dense()
                row, col = adj_memory.nonzero()[:2]
                adj_values = adj_memory[row,col]
            else:
                adj_memory = new_adj
                row, col, adj_values = adj_memory.coo()[:3]
            # adj_values[torch.isnan(adj_values)] = 0.0
            edge_index = torch.stack((row, col), dim=0)
            # print(f"{sum(torch.isnan(adj_values))} {adj_values.mean()}")
            # adj_values = adj_memory[row, col]
            # print(edge_index,adj_values)
            # print(f"Pruned edges: {i} {old_edge_size-adj.coo()[0].size(0)}")
            adj = new_adj
        x = conv(x, edge_index, edge_weight=adj_values)
        # x = self.convs[-1](x, adj)
        # exit()
        return x.log_softmax(dim=-1)


    def att_coef(self, features, adj):
        
        edge_index = adj.coo()[:2]

        n_node = features.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
        features_copy = features.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=features_copy, Y=features_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        """add learnable dropout, make character vector"""
        if self.attention_drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                   att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T).to(features.device)
            drop_score = self.drop_learn(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges
        # print("att", att_dense_norm[0,0])
        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)  # degree +1 is to add itself
            # print(lam.shape)
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        # att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)  # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32).to(features.device)
        # att_adj = torch.tensor(att_adj, dtype=torch.int64).to(features.device)
        shape = (n_node, n_node)
        new_adj = SparseTensor(row=torch.LongTensor(row).to(features.device), 
                            col=torch.LongTensor(col).to(features.device), 
                            value=att_edge_weight, sparse_sizes=torch.Size(shape))
        
        # new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        # print(new_adj)
        return new_adj
