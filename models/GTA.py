#%%
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from models.GCN import GCN

#%%
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
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
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

    def forward(self, input, thrd):

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
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight

class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

#%%
import numpy as np
class Backdoor:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size)
    
    def get_trigger_index(self,trigger_size):
        edge_list = []
        edge_list.append([0,0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        return edge_index

    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index
        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights
    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled):

        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        
        # initial a shadow model
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=0.0, device=self.device).to(self.device)
        # initalize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
        # change the labels of the poisoned node to the target class
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        
        loss_best = 1e8
        for i in range(args.trojan_epochs):
            self.trojan.train()
            for j in range(self.args.inner):

                optimizer_shadow.zero_grad()
                optimizer_trigger.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd) # may revise the process of generate
                trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1,features.shape[1]])
                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]) # repeat trojan weights beacuse of undirected edge
                poison_x = torch.cat([features,trojan_feat])

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                
                loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                
                loss_inner.backward()
                optimizer_shadow.step()
                optimizer_trigger.step()

        self.trojan.eval()

    # def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled):

    #     args = self.args
    #     if edge_weight is None:
    #         edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
    #     self.idx_attach = idx_attach
    #     self.features = features
    #     self.edge_index = edge_index
    #     self.edge_weights = edge_weight
        
    #     # initial a shadow model
    #     self.shadow_model = GCN(nfeat=features.shape[1],
    #                      nhid=self.args.hidden,
    #                      nclass=labels.max().item() + 1,
    #                      dropout=0.0, device=self.device).to(self.device)
    #     # initalize a trojanNet to generate trigger
    #     self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)
    #     self.homo_loss = HomoLoss(self.args,self.device)

    #     optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    #     # change the labels of the poisoned node to the target class
    #     self.labels = labels.clone()
    #     self.labels[idx_attach] = args.target_class

    #     # get the trojan edges, which include the target-trigger edge and the edges among trigger
    #     trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)

    #     # update the poisoned graph's edge index
    #     poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)


    #     # furture change it to bilevel optimization
        
    #     loss_best = 1e8
    #     for i in range(args.trojan_epochs):
    #         self.trojan.train()
    #         for j in range(self.args.inner):

    #             optimizer_shadow.zero_grad()
    #             trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd) # may revise the process of generate
    #             trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
    #             trojan_weights = trojan_weights.flatten()
    #             trojan_feat = trojan_feat.view([-1,features.shape[1]])
    #             poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
    #             poison_x = torch.cat([features,trojan_feat]).detach()

    #             output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                
    #             loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                
    #             loss_inner.backward()
    #             optimizer_shadow.step()

            
    #         acc_train_clean = utils.accuracy(output[idx_train], self.labels[idx_train])
    #         acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])
            
    #         # involve unlabeled nodes in outter optimization
    #         self.trojan.eval()
    #         optimizer_trigger.zero_grad()

    #         rs = np.random.RandomState(self.args.seed)
    #         # idx_outter = torch.cat([idx_attach,idx_unlabeled[rs.choice(len(idx_unlabeled),size=512,replace=False)]])
    #         idx_outter = idx_attach

    #         trojan_feat, trojan_weights = self.trojan(features[idx_outter],self.args.thrd) # may revise the process of generate
        
    #         trojan_weights = torch.cat([torch.ones([len(idx_outter),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
    #         trojan_weights = trojan_weights.flatten()

    #         trojan_feat = trojan_feat.view([-1,features.shape[1]])

    #         trojan_edge = self.get_trojan_edge(len(features),idx_outter,self.args.trigger_size).to(self.device)

    #         update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
    #         update_feat = torch.cat([features,trojan_feat])
    #         update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

    #         output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)

    #         labels_outter = labels.clone()
    #         labels_outter[idx_outter] = args.target_class
    #         loss_target = F.nll_loss(output[torch.cat([idx_train,idx_outter])],
    #                                 labels_outter[torch.cat([idx_train,idx_outter])])

            
    #         loss_outter = loss_target

    #         loss_outter.backward()
    #         optimizer_trigger.step()
    #         acc_train_outter =(output[idx_outter].argmax(dim=1)==args.target_class).float().mean()

    #         if loss_outter<loss_best:
    #             self.weights = deepcopy(self.trojan.state_dict())
    #             loss_best = float(loss_outter)

    #         if args.debug and i % 10 == 0:
    #             print('Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f} '\
    #                     .format(i, loss_inner, loss_target))
    #             print("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outter: {:.4f}"\
    #                     .format(acc_train_clean,acc_train_attach,acc_train_outter))
    #     if args.debug:
    #         print("load best weight based on the loss outter")
    #     self.trojan.load_state_dict(self.weights)
    #     self.trojan.eval()

        # torch.cuda.empty_cache()

    def get_poisoned(self):

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

# %%
