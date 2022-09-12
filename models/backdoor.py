import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.05):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*nfeat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device

    def forward(self, input, thrd, binaryfeat=False):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight


import numpy as np
import torch.optim as optim
from models.GCN import GCN
def obtain_attach_nodes(node_idxs, size):
    ### current random to implement
    size = min(len(node_idxs),size)
    return node_idxs[np.random.choice(len(node_idxs),size,replace=False)]

from torch_geometric.utils import to_undirected
class Backdoor:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.poison_x = None
        self.poison_edge_index = None
        self.poison_edge_weights = None

    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for i,idx in enumerate(idx_attach):
            edge_list.append([idx,start+i*trigger_size])
            for j in range(trigger_size):
                for k in range(j):
                    edge_list.append([start+i*trigger_size+j,start+i*trigger_size+k])
        
        edge_index = torch.tensor(edge_list).long().T

        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index
        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight):

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(self.device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        return update_feat, update_edge_index, update_edge_weights


    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach):

        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)

        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=self.args.dropout, device=self.device).to(self.device)

        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        labels = labels.clone()
        labels[idx_attach] = args.target_class
        self.labels = labels

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)
        self.poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        for i in range(args.epochs): 
            optimizer_shadow.zero_grad()
            optimizer_trigger.zero_grad()

            trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd) # may revise the process of generate
            trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)

            trojan_weights = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1,features.shape[1]])
            self.poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]) # repeat trojan weights beacuse of undirected edge
            self.poison_x = torch.cat([features,trojan_feat])

            output = self.shadow_model(self.poison_x, self.poison_edge_index, self.poison_edge_weights)
            loss_train = F.nll_loss(output[torch.cat([idx_train,idx_attach])], labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
            loss_train.backward()

            optimizer_shadow.step()
            optimizer_trigger.step()

            acc_train_clean = utils.accuracy(output[idx_train], labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], labels[idx_attach])
            

            if args.debug and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_train_clean: {:.4f}, acc_train_attach: {:.4f}".format(acc_train_clean,acc_train_attach))
    
    # def test_atk(self, features, edge_index, edge_weight, idx_atk):
    #     output = 


