"""Torch module for RobustGCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import utils
from copy import deepcopy


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
    def __init__(self, nfeat, nhid, nclass, num_layers=2, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None):
        # def __init__(self, in_features, out_features, hidden_features, dropout=True):
        super(RobustGCN, self).__init__()
        self.device= device
        self.in_features = nfeat
        self.out_features = nclass

        self.act0 = F.elu
        self.act1 = F.relu

        self.layers = nn.ModuleList()
        self.layers.append(RobustGCNConv(nfeat, nhid, act0=self.act0, act1=self.act1,
                                         initial=True, dropout=dropout))
        for i in range(num_layers - 2):
            self.layers.append(RobustGCNConv(nhid, nhid,
                                             act0=self.act0, act1=self.act1, dropout=dropout))
        self.layers.append(RobustGCNConv(nhid, nclass, act0=self.act0, act1=self.act1))
        self.dropout = dropout
        self.use_ln = True
        self.gaussian = None
        
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj, edge_weights = None):
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
            # print(mean.shape,var.shape)
            mean, var = layer(mean, var=var, adj0=adj0, adj1=adj1)
        # print(mean.shape,var.shape)
        # if self.gaussian == None:
        # self.gaussian = MultivariateNormal(torch.zeros(var.shape),
        #         torch.diag_embed(torch.ones(var.shape)))
        sample = torch.randn(var.shape).to(x.device)
        # sample = self.gaussian.sample().to(x.device)
        output = mean + sample * torch.pow(var, 0.5)

        return output.log_softmax(dim=-1)

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def fit(self, features, edge_index, edge_weights,labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False):
        '''
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        '''
        self.sim = None

        if initialize:
            self.initialize()

        features = features.to(self.device)
        adj = edge_index.to(self.device)
        labels = labels.to(self.device)

        """The normalization gonna be done in the GCNConv"""
        self.adj_norm = adj
        self.features = features
        self.labels = labels

        self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            # print('epoch', i)
            self.train()
            optimizer.zero_grad()
            # print(self.features.shape)
            output = self.forward(self.features, self.adj_norm)
            # print(output.shape)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            # acc_test = utils.accuracy(output[self.idx_test], labels[self.idx_test])

            # if verbose and i % 5 == 0:
            #     print('Epoch {}, training loss: {}, val acc: {}, '.format(i, loss_train.item(), acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        # """my test"""
        # output_ = self.forward(self.features, self.adj_norm)
        # acc_test_ = utils.accuracy(output_[self.idx_test], labels[self.idx_test])
        # print('With best weights, test acc:', acc_test_)

    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        # if edge_weight is not None:
        #     edge_index = edge_index[:,edge_weight>0.5]
        with torch.no_grad():
            output = self.forward(features, edge_index)
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)

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
            # print("adj0 mean",adj0,mean.shape)
            mean = adj0 @ mean
            var = adj1 @ var
            # print("mean1",mean.shape)
        return mean, var