
from Baseline_Attack.attacks.injection import random_injection, tdgia_injection
from Baseline_Attack.attacks.attack import gcn_norm, gia_update_features, init_feat, node_sim_estimate, update_features
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_sparse

import attacks.metric as metric
import attacks.utils as utils
from attacks.utils import EarlyStop
import torch.nn as nn

from torch_sparse import SparseTensor


class AGIA(object):
    """
    Adaptive Graph Injection Attacks with gradients
    """
    def __init__(self,
                 epsilon,
                 n_epoch,
                 a_epoch,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 device='cpu',
                 early_stop=False,
                 verbose=True,
                 disguise_coe=1.0,
                 opt="a",
                 iter_epoch=2):
        self.adj_atk = None
        self.device = device
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.a_epoch = a_epoch
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.verbose = verbose

        self.old_reg = False
        self.iter_epoch = iter_epoch

        self.opt = "adam"
        
        # Early stop
        if early_stop:
            self.early_stop = EarlyStop(patience=early_stop, epsilon=1e-4)
        else:
            self.early_stop = early_stop

        # threshold to select the matrix
        self.threshold = 0.5
        self.disguise_coe = disguise_coe

    def attack(self, model, adj, features, target_idx, labels=None):
        model.to(self.device)
        model.eval()
        
        if labels == None:
            with torch.no_grad():
                pred_orig = model(features, adj)
            origin_labels = torch.argmax(pred_orig, dim=1)
        else:
            origin_labels = labels.view(-1)

        features_attack = init_feat(self.n_inject_max, features, self.device, style="random", 
                                    feat_lim_min=self.feat_lim_min, feat_lim_max=self.feat_lim_max)
        features_h = node_sim_estimate(features,adj,self.n_inject_max)
        adj_attack = adj

        for i in range(self.iter_epoch):
            if i == 0:
                adj_attack = random_injection(adj,self.n_inject_max, self.n_edge_max, target_idx, self.device)
            else:
                adj_attack = self.opt_adj_attack(model, adj_attack, features, features_attack, target_idx, origin_labels, 499)
            features_attack = init_feat(self.n_inject_max, features, self.device, style="random", 
                                    feat_lim_min=self.feat_lim_min, feat_lim_max=self.feat_lim_max)
            features_attack = gia_update_features(self,model,adj_attack,features,features_attack,origin_labels,target_idx,features_h)
        return adj_attack, features_attack
    
    def opt_adj_attack(self, model, adj, features, features_attack, target_idx, origin_labels, n_epoch=1e9, homophily=None):
        model.to(self.device)
        model.eval()
        n_epoch = min(n_epoch,self.n_epoch)
        n_total = features.size(0)
        # setup the edge entries for optimization
        new_x = torch.cat([torch.LongTensor([i+n_total]).repeat(target_idx.size(0))
                          for i in range(self.n_inject_max)]).to(self.device)
        new_y = target_idx.repeat(self.n_inject_max).to(self.device)
        assert new_x.size() == new_y.size()
        vals = torch.zeros(new_x.size(0)).to(self.device)
        print(f"#original edges {adj.nnz()}, #target idx {len(target_idx)}, #init edges {vals.size(0)}")

        # jointly update adjecency matrix & features

        # there have been some injected nodes
        if adj.size(0)>n_total:
            print("init edge weights from the previous results")
            # that's a attacked adj
            orig_adj = adj[:n_total,:n_total]
            x, y, z = orig_adj.coo()
            # now we init val with the attacked graph
            vals[:] = 0
            x_inj, y_inj, _ = adj[n_total:,:].coo()
            idx_map = {}
            for (i, idx) in enumerate(target_idx):
                idx_map[idx.item()] = i 

            for i in range(self.n_inject_max*self.n_edge_max):
                xx, yy = x_inj[i], y_inj[i]
                pos = xx*len(target_idx)+idx_map[yy.cpu().item()]
                vals[pos] = 1
            old_vals = vals.clone()
        else:
            old_vals = None
            x, y, z = adj.coo()
        
        z = torch.ones(x.size(0)).to(self.device) if z == None else z
        isolate_idx = torch.nonzero((adj.sum(-1)==0)[:n_total].long(),as_tuple=True)[0].cpu()

        makeup_x = []
        makeup_y = []
        makeup_z = []
        for iidx in isolate_idx:
            makeup_x.append(iidx)
            makeup_y.append(iidx)
            makeup_z.append(1)
        x = torch.cat((x,torch.LongTensor(makeup_x).to(self.device)),dim=0)
        y = torch.cat((y,torch.LongTensor(makeup_y).to(self.device)),dim=0)
        z = torch.cat((z,torch.LongTensor(makeup_z).to(self.device)),dim=0)
        print(f"add self-con for {len(isolate_idx)} nodes")
        new_row = torch.cat((x, new_x, new_y), dim=0)
        new_col = torch.cat((y, new_y, new_x), dim=0)
        vals.requires_grad_(True)
        vals.retain_grad()
        adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.cat((z, vals, vals), dim=0))
        
        optimizer_adj = torch.optim.Adam([vals],self.epsilon)

        features_concat = torch.cat((features, features_attack), dim=0)

        old_layer_output = None
        orig_layer_output = None
        beta = 0.01 if self.n_edge_max >= 100 else 1
        for i in range(self.a_epoch):
            pred = model(features_concat, adj_attack)
            pred_loss = self.loss(pred[:n_total][target_idx],
                                  origin_labels[target_idx]).to(self.device)
            # sparsity loss for the adjacency matrix, based on L1 norm
            sparsity_loss = beta*(self.n_edge_max*self.n_inject_max-torch.norm(vals,p=1))
            pred_loss = -pred_loss-sparsity_loss
            
            
            optimizer_adj.zero_grad()
            pred_loss.backward(retain_graph=True)
            optimizer_adj.step()


            test_score = self.eval_metric(pred[:n_total][target_idx],
                                          origin_labels[target_idx])
            if self.verbose:
                print("Attacking Edges: Epoch {}, Loss: {:.2f}, Surrogate test score: {:.2f}, injected edge {:}".format(
                        i, pred_loss, test_score, vals[:len(target_idx)].sum()),end='\r' if i != self.n_epoch - 1 else '\n')


        # select edges with higher weights as the final injection matrix
        tmp_vals = -vals.detach().view(self.n_inject_max, -1)
        sel_idx = tmp_vals.argsort(dim=-1)[:, :self.n_edge_max]
        sel_mask = torch.zeros(tmp_vals.size()).bool()
        for i in range(sel_idx.size(0)):
            sel_mask[i, sel_idx[i]] = True
        sel_idx = torch.nonzero(sel_mask.view(-1)).squeeze()

        new_x = new_x[sel_idx]
        new_y = new_y[sel_idx]
        print(f"Finally injected edges {len(new_x)}, minimum vals {vals[sel_idx].min()}")
        new_row = torch.cat((x, new_x, new_y), dim=0)
        new_col = torch.cat((y, new_y, new_x), dim=0)
        adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.ones(new_row.size(0),device=self.device))
        if old_vals!=None:
            new_vals = torch.zeros(old_vals.size()).to(old_vals.device)
            new_vals[sel_idx] = 1
            print(f"number of modifications: {(old_vals-new_vals).abs().sum()}")
            print(f"added: {((-old_vals+new_vals)>0).sum()}")
            print(f"removed: {((old_vals-new_vals)>0).sum()}")
        return adj_attack
