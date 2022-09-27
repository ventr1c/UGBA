#%%
from curses import noraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn import GINConv,GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from torch.nn import Sequential, Linear, ReLU
from models.GCN import GCN

class GIN(GCN,nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4 ,device=None):
        """
        """
        nn.Module.__init__(self)

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.gc1 = GCNConv(nfeat, nhid, bias=True,add_self_loops=True,normalize=False)
        self.h1 = Linear(nhid,nhid)
        self.gc2 = GCNConv(nhid, nhid, bias=True,add_self_loops=True,normalize=False)
        self.h2 = Linear(nhid,nclass)

        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.gc1(x, edge_index))
        x = self.h1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.bn1(x)
        x = self.gc2(x, edge_index)
        x = self.h2(x)
        return F.log_softmax(x,dim=1)
# %%
